import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from ml_models.LoadDataset import train_loader, val_loader, train_labels
from app.config.settings import DEVICE, OUTPUT_DIR

EPOCHS = 15
LR     = 1e-4

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# ---------------------------------------------------------------------------
# Class-weighted loss (7k real vs 15k tampered)
# ---------------------------------------------------------------------------
class_weights = compute_class_weight(
    class_weight = 'balanced',
    classes      = np.array([0, 1]),
    y            = train_labels
)
print(f"Class weights → real: {class_weights[0]:.3f}, tampered: {class_weights[1]:.3f}")

weights   = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels, _ in tqdm(loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds       = outputs.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total


best_val_loss = float("inf")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"  Train → Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
    print(f"  Val   → Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
        print("  ✅ Saved best model")

# ---------------------------------------------------------------------------
# Save plots
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(train_losses, label="Train Loss")
ax.plot(val_losses,   label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss Curve")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/loss_curve.png")
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(train_accs, label="Train Acc")
ax.plot(val_accs,   label="Val Acc")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Curve")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/accuracy_curve.png")
plt.close(fig)

print("\n✅ Training complete.")