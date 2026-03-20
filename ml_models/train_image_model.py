import torch
import timm
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from app.config.settings import MODEL_PATH
from ml_models.LoadDataset import train_loader, validation_loader  # <-- add validation_loader

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Models to Compare
# -----------------------------
model_name = "efficientnet_b0"   # change dynamically if needed

# -----------------------------
# Training Config
# -----------------------------
EPOCHS = 10
LR = 1e-4

train_losses = []
val_accuracies = []
# -----------------------------
# Loop over models
# -----------------------------
model = timm.create_model(model_name, pretrained=True, num_classes=2)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(running_loss)
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

# -----------------------------
# Validation
# -----------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    val_accuracies.append(acc)

    print(f"📊 Val Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # Save best model
    if acc > best_val_acc:
        best_val_acc = acc

        save_path = MODEL_PATH.replace(".pth", f"_{model_name}.pth")

        torch.save({
            'model_name': model_name,
            'model_state_dict': model.state_dict(),
            'accuracy': acc
        }, save_path)

        print(f"✅ Best model saved: {model_name}")

# Loss Plot
plt.figure()
plt.plot(train_losses)
plt.title(f"{model_name} - Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{model_name}_loss.png")
# -----------------------------
# Final Comparison
# -----------------------------
print("\n📊 FINAL MODEL COMPARISON")

plt.figure()
plt.plot(val_accuracies)
plt.title(f"{model_name} - Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(f"{model_name}_accuracy.png")