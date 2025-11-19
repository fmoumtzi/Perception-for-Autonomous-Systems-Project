import os
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image


KITTI_ROOT = "/path/to/KITTI"

IMAGE_DIR = Path(KITTI_ROOT) / "training" / "image_2"
LABEL_DIR = Path(KITTI_ROOT) / "training" / "label_2"

OCCLUSION_IMG = "occlusion.png"

# How many occlusion samples to create (you can tune this)
NUM_OCCLUSION_SAMPLES = 1000

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classes we care about in KITTI labels
KITTICLASSES = ["Pedestrian", "Cyclist", "Car"]

# We map them to 4 classes including occlusion
CLASS_TO_IDX = {
    "Pedestrian": 0,
    "Cyclist": 1,
    "Car": 2,
    "Occlusion": 3,
}

# ---------------- DATASET BUILDING ---------------- #

def parse_kitti_labels():
    """
    Parse KITTI label_2 txt files and collect object instances
    for classes Pedestrian, Cyclist, Car.

    Returns:
        objects: list of dicts with keys:
            - img_path: str
            - bbox: (x1, y1, x2, y2)
            - label_name: str (Pedestrian / Cyclist / Car)
    """
    objects = []

    label_files = sorted(LABEL_DIR.glob("*.txt"))
    for lf in label_files:
        img_id = lf.stem  # '000000'
        img_path = IMAGE_DIR / f"{img_id}.png"
        if not img_path.exists():
            continue

        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                obj_type = parts[0]
                if obj_type not in KITTICLASSES:
                    continue

                # bbox: left, top, right, bottom (floats)
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])

                objects.append(
                    {
                        "img_path": str(img_path),
                        "bbox": (x1, y1, x2, y2),
                        "label_name": obj_type,
                    }
                )

    return objects


class KittiObjectDataset(Dataset):
    """
    Dataset of cropped KITTI objects + occlusion samples.
    Each item returns (image_tensor, label_index).
    """

    def __init__(self, objects, occlusion_img_path, num_occlusion_samples,
                 transform=None):
        self.objects = objects
        self.transform = transform

        self.occlusion_img_path = occlusion_img_path
        self.num_occlusion_samples = num_occlusion_samples

        # Pre-load occlusion image (once)
        self.occlusion_img = Image.open(self.occlusion_img_path).convert("RGB")

        # For indexing:
        self.num_real = len(self.objects)
        self.num_total = self.num_real + self.num_occlusion_samples

    def __len__(self):
        return self.num_total

    def __getitem__(self, idx):
        if idx < self.num_real:
            # Real KITTI object
            obj = self.objects[idx]
            img = Image.open(obj["img_path"]).convert("RGB")
            x1, y1, x2, y2 = obj["bbox"]
            w, h = img.size

            # Clamp bbox to image bounds
            x1 = max(0, min(int(x1), w - 1))
            x2 = max(0, min(int(x2), w))
            y1 = max(0, min(int(y1), h - 1))
            y2 = max(0, min(int(y2), h))

            if x2 <= x1 or y2 <= y1:
                # Degenerate box; just return a center crop
                img_crop = img
            else:
                img_crop = img.crop((x1, y1, x2, y2))

            label_name = obj["label_name"]
            label = CLASS_TO_IDX[label_name]
        else:
            # Occlusion sample
            img_crop = self.occlusion_img.copy()
            label = CLASS_TO_IDX["Occlusion"]

        if self.transform is not None:
            img_crop = self.transform(img_crop)

        return img_crop, label


# ---------------- MODEL ---------------- #

def make_resnet_classifier(num_classes=4):
    """
    ResNet-18 pre-trained on ImageNet, last FC replaced by 4-class head:
      [Pedestrian, Cyclist, Car, Occlusion]
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


# ---------------- TRAIN/VAL LOOP ---------------- #

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


# ---------------- MAIN ---------------- #

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Parsing KITTI labels...")
    objects = parse_kitti_labels()
    print(f"Found {len(objects)} objects of classes {KITTICLASSES}")

    if len(objects) == 0:
        print("No objects found. Check KITTI_ROOT and folder structure.")
        return

    # Transforms: resize to 224x224, augment for train
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_dataset = KittiObjectDataset(
        objects=objects,
        occlusion_img_path=OCCLUSION_IMG,
        num_occlusion_samples=NUM_OCCLUSION_SAMPLES,
        transform=train_transform,  # will override for val split later
    )

    # Train/val split
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    # Set proper transform for val_dataset
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Model, loss, optimizer
    model = make_resnet_classifier(num_classes=len(CLASS_TO_IDX)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {DEVICE} for {NUM_EPOCHS} epochs...")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

    # Save model
    out_path = "resnet18_kitti_ped_cyc_car_occ.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved trained model to {out_path}")


if __name__ == "__main__":
    main()
