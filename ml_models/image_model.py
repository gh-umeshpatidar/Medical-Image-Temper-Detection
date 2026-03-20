import torch
import timm
import os

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "efficientnet_b0"   # change dynamically if needed
MODEL_PATH = f"ml_models/image_model_{MODEL_NAME}.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load checkpoint
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ {MODEL_PATH} not found. Train the model first.")

checkpoint = torch.load(MODEL_PATH, map_location=device)

# -----------------------------
# Create SAME model
# -----------------------------
image_model = timm.create_model(
    checkpoint['model_name'],   # 🔥 VERY IMPORTANT
    pretrained=False,
    num_classes=2
)

# -----------------------------
# Load weights
# -----------------------------
image_model.load_state_dict(checkpoint['model_state_dict'])

image_model.to(device)
image_model.eval()

print(f"✅ {checkpoint['model_name']} loaded successfully")
