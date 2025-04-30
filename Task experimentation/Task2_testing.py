from models import *
import kagglehub
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
protonet = AdvancedCNN(num_classes=39)
protonet.load_state_dict(torch.load("saved_models/Task2_protonet_trial.pth"))
protonet.to(device)

# Initialize support and query datasets and loaders
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

support_dataset = FashionDataset('dataset/new_support.csv', img_dir, transform=transform)
support_loader = DataLoader(support_dataset, batch_size=512, shuffle=False)
query_dataset = FashionDataset('dataset/new_test.csv', img_dir, transform=transform)
query_loader = DataLoader(query_dataset, batch_size=512, shuffle=False)


# Compute prototypes
def get_prototypes(model, support_loader, device):
    model.eval()
    support_embeddings = []
    support_labels = []

    with torch.no_grad():
        for images, labels in support_loader:
            images, labels = images.to(device), labels.to(device)
            emb = model(images)
            support_embeddings.append(emb)
            support_labels.append(labels)

    support_embeddings = torch.cat(support_embeddings)
    support_labels = torch.cat(support_labels)

    prototypes = []
    prototype_labels = []
    for c in torch.unique(support_labels):
        class_mask = support_labels == c
        proto = support_embeddings[class_mask].mean(dim=0)
        prototypes.append(proto)
        prototype_labels.append(c)

    return torch.stack(prototypes), torch.tensor(prototype_labels)

prototypes, proto_labels = get_prototypes(protonet, support_loader, device)


# Classify query set
def classify_queries(model, query_loader, prototypes, prototype_labels, device):
    model.eval()
    predictions = []
    true_labels = []
    prototype_labels = prototype_labels.to(device)

    with torch.no_grad():
        for images, labels in query_loader:
            images = images.to(device)
            emb = model(images)
            dists = torch.cdist(emb, prototypes)  # [B, N]
            pred_indices = torch.argmin(dists, dim=1)
            preds = prototype_labels[pred_indices]

            predictions.append(preds.cpu())
            true_labels.append(labels)

    predictions = torch.cat(predictions)
    true_labels = torch.cat(true_labels)
    return predictions, true_labels

preds, labels = classify_queries(protonet, query_loader, prototypes, proto_labels, device)


# Evaluate predictions
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
acc = accuracy_score(labels, preds)
balanced_acc = balanced_accuracy_score(labels, preds)
print(
    f"Accuracy: {round(acc, 4)}",
    f"\nBallanced Accuracy: {round(balanced_acc, 4)}"
)