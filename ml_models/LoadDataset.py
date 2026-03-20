from pydicom import Dataset
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch
from app.config.settings import DATASET_PATH_OPEN, DATASET_PATH_BLIND, label_map
from app.services.preprocess import base_transform

# -----------------------------
# Custom Dataset Loader for CT scans
# -----------------------------

class CTDataset(Dataset):
    def __init__(self, root_dir, transform, label_map):
        self.files = []
        self.labels = []
        self.transform = transform

        for cls in os.listdir(root_dir):   # TB, TM, FB, FM
            cls_path = os.path.join(root_dir, cls)
            for scan in os.listdir(cls_path):   # scan folders
                scan_path = os.path.join(cls_path, scan)
                for f in os.listdir(scan_path):
                    if f.endswith(".dcm"):
                        self.files.append(os.path.join(scan_path, f))
                        self.labels.append(label_map[cls])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]

        image = self.transform(image_path)

        # extra safety conversion
        # if not isinstance(image, torch.Tensor):
        #     image = torch.as_tensor(image)
        image = torch.as_tensor(image).float()

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return image, label
    
# -----------------------------
# Load Open Set dataset -- 80% train, 20% validation
# -----------------------------

open_dataset = CTDataset(
    DATASET_PATH_OPEN,
    base_transform,
    label_map
)

train_size = int(0.8 * len(open_dataset))
validation_size   = len(open_dataset) - train_size

print(f"Total samples: {len(open_dataset)}")
print(f"Training samples: {train_size}")
print(f"Validation samples: {validation_size}")

train_set, val_set = random_split(open_dataset, [train_size, validation_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
validation_loader   = DataLoader(val_set, batch_size=8)

print("✅ Datasets loaded and split into train/validation.")


# ------------------------------
# Laod Blind Set dataset -- only for Testing
# ------------------------------
blind_dataset = CTDataset(
    DATASET_PATH_BLIND,
    base_transform,
    label_map
)
blind_loader = DataLoader(blind_dataset, batch_size=8)
print("✅ Blind test dataset loaded.")