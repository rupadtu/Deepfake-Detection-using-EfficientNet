from config import Config
from models.efficientnet_model import EfficientNetModel
from utils.data_loader import DeepfakeDataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(Config.device), labels.to(Config.device)
        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels.bool()).sum().item()

    return running_loss / len(loader.dataset), correct / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(Config.device), labels.to(Config.device)
            outputs = model(images)
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels.bool()).sum().item()

    return running_loss / len(loader.dataset), correct / len(loader.dataset)

def train_model():
    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = DeepfakeDataset(Config.train_path, transform=transform)
    skf = StratifiedKFold(n_splits=Config.k_folds, shuffle=True, random_state=Config.seed)
    targets = dataset.labels

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n==== Fold {fold} ====")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=Config.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=Config.batch_size, shuffle=False)

        model = EfficientNetModel().to(Config.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        criterion = nn.BCEWithLogitsLoss()

        best_acc = 0.0
        for epoch in range(Config.epochs):
            print(f"\nEpoch {epoch + 1}/{Config.epochs}")
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = validate(model, val_loader, criterion)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'best_model_fold{fold}.pth')