# For train parsing
import argparse, os
from types import SimpleNamespace
# For data prep
import kagglehub
import pandas as pd
from PIL import Image
# For model/train functionality
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import wandb
import tqdm
from models import *

num_classes = 39
model_dict = {
    "BasicCNN": BasicCNN(num_classes=num_classes),
    "IntermediateCNN": IntermediateCNN(num_classes=num_classes),
    "AdvancedCNN": AdvancedCNN(num_classes=num_classes),
    "MobileNetV3": MobileNetV3(num_classes=num_classes)
}

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
    transforms.ToTensor()
])

train_dataset = FashionDataset('dataset/presplit_train/trainsplit.csv', img_dir, transform=transform)
val_dataset = FashionDataset('dataset/presplit_train/valsplit.csv', img_dir, transform=transform)


def train(config):
    with wandb.init(project=config.model_name, config=config):
        config = wandb.config

        # Prepare dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Initialize model, optimizer, and loss function (criterion)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_dict[config.model_name]
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Initialize patience variables
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(config.epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

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
            
            train_loss /= train_total
            train_acc = train_correct / train_total

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} - Validation', leave=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss /= val_total
            val_acc = val_correct / val_total

            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'validation_loss': val_loss,
                'validation_accuracy': val_acc
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model, f'saved_models/{config.model_name}.pth')
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print('Early stopping triggered.')
                    break

    wandb.finish()


# Initialize default hyperparameters and parser functionality
default_config = SimpleNamespace(
    model_name="MobileNetV3",
    batch_size=64,
    learning_rate=5e-4,
    epochs=500,
    patience=5
)


def parse_args():
    argparser = argparse.ArgumentParser(description='Process hyperparameters')
    argparser.add_argument('--model_name', type=str, default=default_config.model_name, help='The name of the preferred model as specified in models.py')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='The batch size for the data loaders')
    argparser.add_argument('--learning_rate', type=float, default=default_config.learning_rate, help='The learning rate for the Adam optimizer')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='The maximum number of epochs to run the training for')
    argparser.add_argument('--patience', type=int, default=default_config.patience, help='The number of epochs after which training is stopped if validation loss does not improve')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


if __name__ == '__main__':
    parse_args()
    train(default_config)