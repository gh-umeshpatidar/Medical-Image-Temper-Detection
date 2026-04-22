import os

import torch

import numpy as np

import pydicom

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split

from collections import Counter

from app.config.settings import DATASET_PATH, folders, BATCH_SIZE

from app.services.preprocess import train_transform, val_transform, apply_windowing

data, labels = [], []

for folder, label in folders.items():

    folder_path = os.path.join(DATASET_PATH, folder)

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            full_path = os.path.join(root, file)

            if not os.path.isfile(full_path):

                continue

            data.append(full_path)

            labels.append(label)

print("Full dataset distribution:", Counter(labels))

train_data, temp_data, train_labels, temp_labels = train_test_split(
    data, labels, test_size=0.3, stratify=labels, random_state=42
)

val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

print(f"Train: {Counter(train_labels)}")

print(f"Val:   {Counter(val_labels)}")

print(f"Test:  {Counter(test_labels)}")

label_counts = Counter(train_labels)

class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}

sample_weights = [class_weights[l] for l in train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)


class DicomDataset(Dataset):

    def __init__(self, file_paths, labels, transform=None):

        self.file_paths = file_paths

        self.labels = labels

        self.transform = transform

    def __len__(self):

        return len(self.file_paths)

    def __getitem__(self, idx):

        path = self.file_paths[idx]

        try:

            ds = pydicom.dcmread(path)

            img = apply_windowing(ds.pixel_array, ds)

            if len(img.shape) == 2:

                img = np.stack([img] * 3, axis=-1)

            elif img.shape[2] == 1:

                img = np.concatenate([img] * 3, axis=-1)

        except Exception as e:

            print(f"Error reading {path}: {e}")

            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:

            img = self.transform(image=img)["image"]

        label = torch.tensor(self.labels[idx]).long()

        return img, label, path


_workers = 0

train_loader = DataLoader(
    DicomDataset(train_data, train_labels, train_transform),
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=_workers,
    pin_memory=True,
)

val_loader = DataLoader(
    DicomDataset(val_data, val_labels, val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=_workers,
    pin_memory=True,
)

test_loader = DataLoader(
    DicomDataset(test_data, test_labels, val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=_workers,
    pin_memory=True,
)
