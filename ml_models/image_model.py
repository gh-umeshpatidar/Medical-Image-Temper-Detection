import torch

import torchvision.models as models

import torch.nn as nn

from app.config.settings import DEVICE, MODEL_PATH


def load_model():

    m = models.resnet50(weights=None)

    m.fc = nn.Linear(m.fc.in_features, 2)

    m = m.to(DEVICE)

    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    m.eval()

    print("Model loaded successfully.")

    return m


model = None

def get_model():
    global model

    if model is None:
        model = load_model()

    return model
