import kagglehub, os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from models import *

# Initialize full train dataset (main classes train)
dataset_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
img_dir = os.path.join(dataset_path,"images")

class FashionDataset(Dataset):
    def __init__(self, csv_file, img_dir,column_class="articleTypeId", transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with labels.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)  # load CSV file
        self.img_dir = img_dir  # image folder path
        self.transform = transform  # image transformations
        self.targets = list(self.df[column_class].values)


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.df.iloc[idx]['imageId']}.jpg")  # Get image filename
        image = Image.open(img_name).convert("RGB")  # Load image

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, self.targets[idx]

transform = transforms.Compose([
    transforms.Resize((80, 60)),
    transforms.ToTensor()
])

support_dataset = FashionDataset('dataset/main_support.csv', img_dir, transform=transform)
support_loader = DataLoader(support_dataset, batch_size=512, shuffle=False)
query_dataset = FashionDataset('dataset/main_test.csv', img_dir, transform=transform)
query_loader = DataLoader(query_dataset, batch_size=512, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
protonet = AdvancedCNN(num_classes=39)
protonet.load_state_dict(torch.load("saved_models/Task2_protonet.pth"))
protonet.to(device)


def compute_confidence(query_embedding, recommended_embeddings):
    cosine_similarities = nn.functional.cosine_similarity(query_embedding.unsqueeze(0), recommended_embeddings, dim=1)
    return cosine_similarities

def filter_recommendations(query_recommendations, threshold):
    filtered = []
    for labels, confidences in query_recommendations:
        mask = confidences > threshold
        filtered_labels = labels[mask]
        filtered.append(filtered_labels)
    return filtered


def get_recommendations(query_loader, support_loader, model, threshold, top_k=3):
    # Get support_embeddings
    model.eval()
    support_embeddings = []
    support_labels = []

    with torch.no_grad():
        for images, labels in support_loader:
            images = images.to(device)
            emb = model(images)
            support_embeddings.append(emb.cpu())
            support_labels.append(labels)

    support_embeddings = torch.cat(support_embeddings, dim=0)
    support_labels = torch.cat(support_labels, dim=0)
    
    # Get recommendations for queries
    query_recommendations = []
    query_labels = []
    with torch.no_grad():
        for images, labels in query_loader:
            images = images.to(device)
            query_embeddings = model(images).cpu()
            query_labels.append(labels)
            dists = torch.cdist(query_embeddings, support_embeddings)

            # Loop over each query in this batch
            for i in range(query_embeddings.size(0)):
                distances = dists[i]
                topk_indices = torch.topk(-distances, top_k).indices

                # Store label of recommendation and confidence per query item
                recommendation_labels = support_labels[topk_indices]
                recommendation_embeddings = support_embeddings[topk_indices]
                query_embedding = query_embeddings[i]

                confidences = compute_confidence(query_embedding, recommendation_embeddings)
                query_recommendations.append((recommendation_labels, confidences))
    
    query_labels = torch.cat(query_labels, dim=0).view(-1)

    # Filter recommendations based on confidence and threshold
    filtered_recommendations = filter_recommendations(query_recommendations, threshold)

    assert len(filtered_recommendations) == len(query_labels)

    success_count = 0
    error_count = 0
    total_queries = len(query_labels)

    for true_label, preds in zip(query_labels, filtered_recommendations):
        if len(preds)==0:
            continue  # No recommendation is not wrong nor correct
        elif true_label in preds:
            success_count += 1
        else:
            error_count += 1
    
    success_rate = success_count / total_queries
    error_rate = error_count / total_queries

    return success_rate, error_rate

sr, er = get_recommendations(query_loader, support_loader, protonet, threshold=0.5, top_k=3)
print(round(sr,4), round(er,4))