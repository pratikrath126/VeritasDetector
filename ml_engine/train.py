import os
import sys
import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report
import time

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "real_dir": "../dataset/real",
    "fake_dir": "../dataset/fake",
    "model_path": "model.pth",
    "image_size": 224,
    "batch_size": 32,
    "epochs": 12,
    "learning_rate": 0.0001,
    "val_split": 0.2,
    "num_workers": 0,
    "early_stopping_patience": 3,
}

# ============================================================
# DEVICE SETUP — Apple M4 / NVIDIA / CPU
# ============================================================
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple Silicon — M4 Accelerated)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: cpu (No GPU found — training will be slow)")
    return device

# ============================================================
# DATASET CLASS
# ============================================================
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.ImageCompression(
                quality_lower=70,
                quality_upper=100,
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.RandomShadow(p=0.1),
            A.Rotate(limit=10, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((CONFIG["image_size"], CONFIG["image_size"]), Image.LANCZOS)
            img_np = np.array(img)

            if self.augment:
                result = self.aug_transform(image=img_np)
            else:
                result = self.val_transform(image=img_np)

            return result["image"], label

        except Exception:
            img_np = np.zeros((CONFIG["image_size"], CONFIG["image_size"], 3), dtype=np.uint8)
            result = self.val_transform(image=img_np)
            return result["image"], label

# ============================================================
# LOAD ALL IMAGE PATHS
# ============================================================
def load_dataset():
    image_paths = []
    labels = []

    real_dir = CONFIG["real_dir"]
    fake_dir = CONFIG["fake_dir"]

    if not os.path.exists(real_dir):
        print(f"ERROR: Real directory not found: {real_dir}")
        print("Please download dataset first. See README.md")
        sys.exit(1)

    if not os.path.exists(fake_dir):
        print(f"ERROR: Fake directory not found: {fake_dir}")
        print("Please download dataset first. See README.md")
        sys.exit(1)

    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')

    real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(valid_ext)]
    for fname in real_files:
        image_paths.append(os.path.join(real_dir, fname))
        labels.append(0)

    fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(valid_ext)]
    for fname in fake_files:
        image_paths.append(os.path.join(fake_dir, fname))
        labels.append(1)

    print(f"Loaded {len(real_files)} real images")
    print(f"Loaded {len(fake_files)} fake images")
    print(f"Total: {len(image_paths)} images")

    if len(real_files) == 0 or len(fake_files) == 0:
        print("ERROR: Dataset folders are empty. Fill dataset/real and dataset/fake first.")
        sys.exit(1)

    if len(image_paths) < 1000:
        print("ERROR: Too few images. Please download dataset. See README.md")
        sys.exit(1)

    return image_paths, labels

# ============================================================
# BUILD MODEL
# ============================================================
def build_model(device):
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    model = model.to(device)
    print("Model: EfficientNet-B0 (pretrained ImageNet)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

# ============================================================
# TRAINING LOOP
# ============================================================
def train():
    print("\n" + "=" * 60)
    print("VERITAS DEEPFAKE DETECTOR — TRAINING")
    print("=" * 60)

    device = get_device()

    print("\nLoading dataset...")
    image_paths, labels = load_dataset()

    total = len(image_paths)
    val_size = int(total * CONFIG["val_split"])
    train_size = total - val_size

    indices = list(range(total))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    train_dataset = FaceDataset(train_paths, train_labels, augment=True)
    val_dataset = FaceDataset(val_paths, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    print("\nBuilding model...")
    model = build_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_val_acc = 0.0
    patience_counter = 0
    all_preds = []
    all_labels = []

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, batch_labels) in enumerate(train_loader):
            images = images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()

            if batch_idx % 50 == 0:
                print(
                    f"  Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}",
                    end='\r'
                )

        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, batch_labels in val_loader:
                images = images.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(images)
                _, predicted = outputs.max(1)

                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        avg_loss = train_loss / len(train_loader)
        epoch_time = time.time() - start_time

        print(
            f"\nEpoch {epoch + 1}/{CONFIG['epochs']} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'config': CONFIG,
            }, CONFIG["model_path"])
            print(f"  ✓ New best model saved (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        scheduler.step()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {CONFIG['model_path']}")
    print("=" * 60)

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

if __name__ == "__main__":
    train()
