from collections import defaultdict
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from models import *
import kagglehub
import os
import pandas as pd
from PIL import Image
import wandb
from torchvision import transforms


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

train_dataset = FashionDataset('dataset/train.csv', img_dir, transform=transform)


# Initialize episodic sampler
class EpisodicSampler(Sampler):
    def __init__(self, labels, num_episodes, n_way, k_shot, q_queries):
        self.labels = labels
        self.num_episodes = num_episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries

        self.class_to_idx = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_idx[label].append(idx)
    
    def __iter__(self):
        for ep in range(self.num_episodes):
            episode_idx = []
            classes = random.sample(list(self.class_to_idx.keys()), self.n_way)
            for cls in classes:
                indices = random.sample(self.class_to_idx[cls], self.k_shot + self.q_queries)
                episode_idx.extend(indices)
            yield episode_idx
    
    def __len__(self):
        return self.num_episodes

# Set sampler parameters
all_labels = [label for _, label in train_dataset]
episodes_per_epoch = 100
n_way = 5
k_shot = 10
q_queries= 10
sampler = EpisodicSampler(labels=all_labels, 
                          num_episodes=episodes_per_epoch,
                          n_way = n_way,
                          k_shot = k_shot,
                          q_queries= q_queries
                          )

train_loader = DataLoader(train_dataset, batch_sampler=sampler)


# Define prototypical loss
def prototypical_loss(model, support, query, support_labels, query_labels):
    # Encode
    support_embeddings = model(support)
    query_embeddings = model(query)

    # Compute prototypes
    prototypes = []
    for c in torch.unique(support_labels):
        class_mask = support_labels == c
        class_prototype = support_embeddings[class_mask].mean(dim=0)
        prototypes.append(class_prototype)
    prototypes = torch.stack(prototypes)  # (N, D)

    # Compute distances and loss
    dists = torch.cdist(query_embeddings, prototypes)  # (Q*N, N)
    logits = -dists
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, query_labels)
    return loss


# Train
# Set training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedCNN(num_classes=39).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 100
patience = 5

wandb.init(project="ProtoNetTraining1")
best_loss = float('inf')
wait = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for episode in train_loader:
        images, labels = episode
        images, labels = images.to(device), labels.to(device)

        # Split into support and query sets
        total_per_class = k_shot + q_queries
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        unique_labels = torch.unique(labels)
        label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}

        for cls in unique_labels:
            cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
            cls_indices = cls_indices[torch.randperm(len(cls_indices))]  # shuffle
            support_idx = cls_indices[:k_shot]
            query_idx = cls_indices[k_shot:k_shot + q_queries]

            support_images.append(images[support_idx])
            support_labels.extend([label_map[cls.item()]] * k_shot)
            query_images.append(images[query_idx])
            query_labels.extend([label_map[cls.item()]] * q_queries)
        
        # Prep sets for actual training
        support_images = torch.cat(support_images)
        query_images = torch.cat(query_images)
        support_labels = torch.tensor(support_labels, device=device)
        query_labels = torch.tensor(query_labels, device=device)

        # Train step
        optimizer.zero_grad()
        loss = prototypical_loss(model, support_images, query_images, support_labels, query_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    epoch_loss = total_loss/episodes_per_epoch
    wandb.log({"loss": epoch_loss})

    if epoch_loss < best_loss:
        wait = 0
        best_loss = epoch_loss
        torch.save(model.state_dict(), f'saved_models/Task2_testtraining.pth')  #TODO edit for final training!
    else:
        wait += 1
        if wait >= patience:
            print('Early stopping triggered.')
            break

wandb.finish()