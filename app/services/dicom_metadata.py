import os
import numpy as np
import pydicom
from tqdm import tqdm
from app.config.settings import DATASET_PATH_OPEN, DATASET_PATH_BLIND, label_map

# Dataset path
DATASET_PATH = DATASET_PATH_OPEN


# Store features and labels
X = []
Y = []

# Extract metadata features for ML model in Numeric format
def extract_metadata(ds):

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except:
            return default

    def safe_get_first(value, default=0.0):
        try:
            if isinstance(value, (list, tuple)):
                return float(value[0])
            return float(value)
        except:
            return default

    # Pixel spacing safe handling
    pixel_spacing_x = 0
    pixel_spacing_y = 0

    if hasattr(ds, "PixelSpacing"):
        try:
            spacing = ds.PixelSpacing
            if isinstance(spacing, (list, tuple)):
                pixel_spacing_x = safe_float(spacing[0])
                pixel_spacing_y = safe_float(spacing[1])
            else:
                pixel_spacing_x = safe_float(spacing)
                pixel_spacing_y = 0
        except:
            pass

    # Compression flag safe
    compressed = 0
    try:
        compressed = 1 if ds.file_meta.TransferSyntaxUID.is_compressed else 0
    except:
        compressed = 0

    features = [

        # Image size
        safe_float(getattr(ds, "Rows", 0)),
        safe_float(getattr(ds, "Columns", 0)),

        # Pixel spacing
        pixel_spacing_x,
        pixel_spacing_y,

        # Intensity info
        safe_float(getattr(ds, "BitsAllocated", 0)),
        safe_float(getattr(ds, "BitsStored", 0)),

        # Scanner info
        safe_float(getattr(ds, "KVP", 0)),

        # Window settings
        safe_get_first(getattr(ds, "WindowCenter", 0)),
        safe_get_first(getattr(ds, "WindowWidth", 0)),

        # Slice thickness
        safe_get_first(getattr(ds, "SliceThickness", 0)),

        # Rescale info
        safe_float(getattr(ds, "RescaleSlope", 0)),
        safe_float(getattr(ds, "RescaleIntercept", 0)),
        
        # UID lengths
        len(str(getattr(ds, "SeriesInstanceUID", ""))),
        # Compression flag
        compressed,

        # Modality
        1 if getattr(ds, "Modality", "") == "CT" else 0,

        # Patient position
        1 if getattr(ds, "PatientPosition", "") == "HFS" else 0,

        # Manufacturer present
        1 if hasattr(ds, "Manufacturer") else 0,

        # Software version present
        1 if hasattr(ds, "SoftwareVersions") else 0,

    ]

    return features
