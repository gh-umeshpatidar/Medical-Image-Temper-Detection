import numpy as np
import cv2
import torch
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Resize,
    EnsureType,
    Compose
)

# ------------------------------------
# CLAHE (Contrast Enhancement)
# ------------------------------------
def apply_clahe(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# ------------------------------------
# Denoising
# ------------------------------------
def denoise(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

# ------------------------------------
# Base transform (only loading + resize)
# ------------------------------------
base_transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Resize((224, 224)),   # ✅ FIXED for EfficientNet-B0
    EnsureType()
])

# ------------------------------------
# MAIN preprocessing
# ------------------------------------
def preprocess_image(image_path: str):

    image = base_transform(image_path)  # (C, H, W)
    image = image.numpy()

    # Handle grayscale medical images
    if image.shape[0] == 1:
        image = image[0]
        image = apply_clahe(image)
        image = denoise(image)
        image = np.expand_dims(image, axis=0)

    # Convert to 3-channel
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)

    # Normalize (VERY IMPORTANT)
    image = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    image = (image - mean) / std

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return torch.tensor(image, dtype=torch.float32)