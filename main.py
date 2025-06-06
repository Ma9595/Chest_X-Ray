import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score


# Dataset class
class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# Load data
def load_data(data_dir, labels_file):
    df = pd.read_csv(labels_file)
    df['Path'] = df['Image Index'].apply(lambda x: os.path.join(data_dir, x))
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['Finding Labels'].str.split('|'))
    return df['Path'].tolist(), labels, mlb.classes_


# Handle class imbalance
def compute_pos_weights(labels):
    labels_np = np.array(labels)
    label_counts = labels_np.sum(axis=0)
    total = labels_np.shape[0]
    pos_weight = (total - label_counts) / (label_counts + 1e-6)
    return torch.tensor(np.clip(pos_weight, 1.0, 100.0), dtype=torch.float32)


# Evaluation
def evaluate_model(model, dataloader, device, class_names, threshold=0.5):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    y_bin = (y_pred >= threshold).astype(int)

    print("\n📊 Class-wise Performance:")
    for i, name in enumerate(class_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = float('nan')
        f1 = f1_score(y_true[:, i], y_bin[:, i], zero_division=0)
        print(f"{name:20s} | F1: {f1:.3f} | AUC: {auc:.3f}")

    print(f"\n✅ Macro F1: {f1_score(y_true, y_bin, average='macro', zero_division=0):.4f}")
    print(f"✅ Micro F1: {f1_score(y_true, y_bin, average='micro', zero_division=0):.4f}")


# Training loop
def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=50, patience=7):
    best_loss = float('inf')
    best_model_wts = model.state_dict()
    patience_counter = 0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"Epoch {epoch+1}/{num_epochs} - {phase} Loss: {epoch_loss:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(), 'best_model.pt')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        model.load_state_dict(best_model_wts)
                        return model
    model.load_state_dict(best_model_wts)
    return model


# MAIN PIPELINE
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Paths
    data_dir = 'full_data'
    labels_file = 'Data_Entry_2017.csv'

    # Load data
    image_paths, labels, class_names = load_data(data_dir, labels_file)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Datasets & Loaders
    split_idx = int(0.9 * len(image_paths))
    train_dataset = ChestXrayDataset(image_paths[:split_idx], labels[:split_idx], transform)
    val_dataset = ChestXrayDataset(image_paths[split_idx:], labels[split_idx:], transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # Loss and optimizer
    pos_weight = compute_pos_weights(labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Train
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, device)

    # Evaluate
    evaluate_model(model, val_loader, device, class_names)
