import os

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "best_model.pth")

DATASET_PATH = os.path.join(BASE_DIR, "DataSet")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UPLOAD_DIR = os.path.join(BASE_DIR, "Uploads", "Images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(UPLOAD_DIR, exist_ok=True)

BATCH_SIZE = 32

folders = {"TB": 0, "FM": 1, "TM": 0, "FB": 1}
