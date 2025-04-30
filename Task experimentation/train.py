# For train parsing
import argparse, os
from types import SimpleNamespace
# For data prep
import kagglehub
import pandas as pd
from PIL import Image
# For model/train functionality
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score
import wandb
import time
from tqdm import tqdm
from models import *

num_classes = 39

# Retrieves data from local cache or downloads if not cached yet
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
        img_name = os.path.join(self.img_dir, f"{self.df.loc[idx,'imageId']}.jpg")  # Get image filename
        image = Image.open(img_name).convert("RGB")  # Load image

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, self.targets[idx]

transform = transforms.Compose([
    transforms.Resize((80, 60)),
    transforms.ToTensor()
])

train_dataset = FashionDataset('dataset/presplit_train/trainsplit.csv', img_dir, transform=transform)
val_dataset = FashionDataset('dataset/presplit_train/valsplit.csv', img_dir, transform=transform)


def train(config):
    with wandb.init(project="FinalTest", config=config):
        config = wandb.config

        # Prepare dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Initialize model, optimizer, and loss function (criterion)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AdvancedCNN(nr_filters=config.nr_filters, activation=nn.ReLU, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

        # Determine optimizer from config
        if config.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=1e-4, momentum=0.9)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',              # 'min' because we want to minimize val_loss
                factor=0.5,              # LR reduced by a factor of 0.5
                patience=4               # Wait 4 epochs before reducing
            )
        elif config.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

        # Initialize patience variables
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(config.epochs):

            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            all_train_preds, all_train_labels = [], []

            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} - Training', leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                train_correct += preds.eq(labels).sum().item()
                train_total += labels.size(0)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            balanced_train_acc = balanced_accuracy_score(all_train_labels, all_train_preds)

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_val_preds, all_val_labels = [], []

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} - Validation', leave=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(labels).sum().item()
                    val_total += labels.size(0)
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            balanced_val_acc = balanced_accuracy_score(all_val_labels, all_val_preds)

            if config.optimizer == "SGD":
                scheduler.step(val_loss)

            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_balanced_accuracy': balanced_train_acc,
                'validation_loss': val_loss,
                'validation_accuracy': val_acc,
                'validation_balanced_accuracy': balanced_val_acc
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'saved_models/Task1_testtraining.pth')  #TODO edit for final training!
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print('Early stopping triggered.')
                    break

    wandb.finish()


# Initialize default hyperparameters and parser functionality
default_config = SimpleNamespace(
    batch_size=1024,
    nr_filters=64,
    optimizer="SGD",
    learning_rate=5e-2,
    epochs=500,
    patience=10
)


def parse_args():
    argparser = argparse.ArgumentParser(description='Process hyperparameters')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='The batch size for the data loaders')
    argparser.add_argument('--nr_filters', type=int, default=default_config.nr_filters, help='The number of output channels of the first ConvBNlayer of the model')
    argparser.add_argument('--optimizer', type=str, default=default_config.optimizer, help='The optimizer to use for training')
    argparser.add_argument('--learning_rate', type=float, default=default_config.learning_rate, help='The learning rate for the Adam optimizer')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='The maximum number of epochs to run the training for')
    argparser.add_argument('--patience', type=int, default=default_config.patience, help='The number of epochs after which training is stopped if validation loss does not improve')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


if __name__ == '__main__':
    if wandb.run is not None:
        # Meaning a sweep is active
        train(wandb.config)  # ensure to declare all arguments in the yaml
    else:
        # Meaning a 'manual' run is done
        parse_args()
        train(default_config)