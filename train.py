import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

class AgeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['image_path']
        age = self.df.iloc[idx]['age']
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([age], dtype=torch.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv("aaf_age_labels.csv")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    df_train['age_bin'] = pd.cut(df_train['age'], bins=bins, labels=False)
    bin_counts = df_train['age_bin'].value_counts().to_dict()
    weights = df_train['age_bin'].apply(lambda x: 1.0 / bin_counts.get(x, 1)).values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    BATCH_SIZE = 8
    train_loader = DataLoader(AgeDataset(df_train, train_transform), batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(AgeDataset(df_test, test_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    EPOCHS = 5
    best_mae = float('inf')
    metrics_per_epoch = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, targets in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * imgs.size(0)
                all_preds.extend(outputs.view(-1).cpu().numpy())
                all_targets.extend(targets.view(-1).cpu().numpy())

        avg_test_loss = test_loss / len(test_loader.dataset)
        mae_score = mean_absolute_error(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        rmse_score = mse ** 0.5
        r2 = r2_score(all_targets, all_preds)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} / Test Loss = {avg_test_loss:.4f} / "
              f"MAE = {mae_score:.2f}years / RMSE = {rmse_score:.2f}years / R² = {r2:.3f}")

        metrics_per_epoch.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'mae': mae_score,
            'rmse': rmse_score,
            'r2': r2
        })

        if mae_score < best_mae:
            best_mae = mae_score
            torch.save(model.state_dict(), "best_model.pth")
            print(f"[Epoch {epoch+1}] Best model saved with MAE = {best_mae:.2f}세")

    pd.DataFrame(metrics_per_epoch).to_csv("training_metrics.csv", index=False)
    print("Metrics saved to training_metrics.csv")

if __name__ == "__main__":
    main()
