import albumentations as A

from albumentations.pytorch import ToTensorV2

import cv2

import numpy as np

import pydicom

train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.Affine(translate_percent=(-0.1, 0.1), p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ]
)

val_transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])

_MODALITY_DEFAULTS = {
    "CT": (40, 400),
    "MR": (500, 1000),
    "CR": (2048, 4096),
    "DX": (2048, 4096),
    "PT": (3000, 6000),
    "NM": (3000, 6000),
}

_DEFAULT_WINDOW = (40, 400)


def apply_windowing(pixel_array, ds):

    img = pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1))

    intercept = float(getattr(ds, "RescaleIntercept", 0))

    img = img * slope + intercept

    level = getattr(ds, "WindowCenter", None)

    width = getattr(ds, "WindowWidth", None)

    if level is None or width is None:

        modality = str(getattr(ds, "Modality", "CT")).upper().strip()

        level, width = _MODALITY_DEFAULTS.get(modality, _DEFAULT_WINDOW)

    else:

        if hasattr(level, "__iter__") and not isinstance(level, str):

            level = float(list(level)[0])

        if hasattr(width, "__iter__") and not isinstance(width, str):

            width = float(list(width)[0])

        level, width = float(level), float(width)

    lo = level - width / 2

    hi = level + width / 2

    img = np.clip(img, lo, hi)

    img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)

    if len(img.shape) == 2:

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


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

    img = (
        load_dicom(file_path)
        if file_path.lower().endswith(".dcm")
        else load_image(file_path)
    )

    transformed = val_transform(image=img)

    return transformed["image"].unsqueeze(0)
