import torch
import timm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from ml_models.LoadDataset import validation_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Models to evaluate
# -----------------------------
MODEL_LIST = [
    "efficientnet_b0",
    "resnet18",
    "mobilenetv2_100"
]

BASE_MODEL_PATH = "ml_models/"  # update this

# -----------------------------
# Evaluation Loop
# -----------------------------
for model_name in MODEL_LIST:

    print(f"\n🔍 Evaluating Model: {model_name}")

    model = timm.create_model(model_name, pretrained=False, num_classes=2)

    checkpoint_path = BASE_MODEL_PATH + f"_{model_name}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # -----------------------------
    # Metrics
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔁 Recall: {recall:.4f}")
    print(f"📊 F1 Score: {f1:.4f}")