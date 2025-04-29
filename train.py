import argparse, os
from types import SimpleNamespace
import torch
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import wandb
import tqdm
from models import *

num_classes = ...
model_dict = {
    "BasicCNN": BasicCNN(num_classes=num_classes),
    "IntermediateCNN": IntermediateCNN(num_classes=num_classes),
    "AdvancedCNN": AdvancedCNN(num_classes=num_classes),
    "MobileNetV3": MobileNetV3(num_classes=num_classes)
}

def train(config):
    with wandb.init(project=config.model_name, config=config):
        config = wandb.config

        #TODO prep data in DataLoader, using config.batch_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_dict[config.model_name]
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

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