import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pydicom

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(translate_percent=(-0.1, 0.1), p=0.3),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])


def apply_windowing(pixel_array, ds):

    img = pixel_array.astype(np.float32)

    # Step 1: Rescale to Hounsfield Units
    slope     = float(getattr(ds, "RescaleSlope",     1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    img = img * slope + intercept

    # Step 2: Apply windowing
    level = getattr(ds, "WindowCenter", 40)
    width = getattr(ds, "WindowWidth",  400)

    # Some scanners store these as lists — take first element
    if hasattr(level, "__iter__"): level = float(list(level)[0])
    if hasattr(width, "__iter__"): width = float(list(width)[0])

    level, width = float(level), float(width)
    lo = level - width / 2
    hi = level + width / 2

    img = np.clip(img, lo, hi)
    img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)

    # Step 3: Ensure RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img  # uint8, shape (H, W, 3)


def load_dicom(file_path):
    try:
        ds = pydicom.dcmread(file_path)
        return apply_windowing(ds.pixel_array, ds)
    except Exception as e:
        raise ValueError(f"Failed to load DICOM file: {str(e)}")


def load_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Failed to load image: {file_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def transform(file_path):
    img = load_dicom(file_path) if file_path.lower().endswith('.dcm') else load_image(file_path)
    transformed = val_transform(image=img)
    return transformed['image'].unsqueeze(0)