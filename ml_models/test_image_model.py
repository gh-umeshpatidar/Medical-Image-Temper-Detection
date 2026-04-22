import os

import torch

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cv2

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from ml_models.LoadDataset import test_loader

from ml_models.image_model import model

from app.config.settings import OUTPUT_DIR, DEVICE

from app.services.gradcam import gradcam

all_preds, all_probs, all_labels, all_paths = [], [], [], []

all_imgs_for_cam = []

with torch.no_grad():

    for imgs, labels, paths in test_loader:

        imgs = imgs.to(DEVICE)

        outputs = model(imgs)

        probs = torch.softmax(outputs, dim=1)[:, 1]

        preds = (probs > 0.5).int()

        all_preds.extend(preds.cpu().numpy())

        all_probs.extend(probs.cpu().numpy())

        all_labels.extend(labels.numpy())

        all_paths.extend(paths)

        all_imgs_for_cam.append(imgs.cpu())

df = pd.DataFrame(
    {
        "image": all_paths,
        "true_label": all_labels,
        "pred_label": all_preds,
        "probability": all_probs,
    }
)

df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots()

ax.imshow(cm, cmap="Blues")

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(
            j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12
        )

ax.set_xlabel("Predicted")

ax.set_ylabel("Actual")

ax.set_title("Confusion Matrix")

fig.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")

plt.close(fig)

print("\nClassification Report:")

print(
    classification_report(all_labels, all_preds, target_names=["Authentic", "Tampered"])
)

auc_score = roc_auc_score(all_labels, all_probs)

print("ROC-AUC:", auc_score)

fpr, tpr, _ = roc_curve(all_labels, all_probs)

fig, ax = plt.subplots()

ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")

ax.plot([0, 1], [0, 1], linestyle="--")

ax.set_xlabel("False Positive Rate")

ax.set_ylabel("True Positive Rate")

ax.set_title("ROC Curve")

ax.legend()

fig.savefig(f"{OUTPUT_DIR}/roc_curve.png")

plt.close(fig)

precision, recall, _ = precision_recall_curve(all_labels, all_probs)

fig, ax = plt.subplots()

ax.plot(recall, precision)

ax.set_xlabel("Recall")

ax.set_ylabel("Precision")

ax.set_title("Precision-Recall Curve")

ax.grid()

fig.savefig(f"{OUTPUT_DIR}/pr_curve.png")

plt.close(fig)

for i, imgs in enumerate(all_imgs_for_cam):

    imgs = imgs.to(DEVICE)

    outputs = model(imgs)

    preds = outputs.argmax(dim=1)

    for j in range(imgs.size(0)):

        img_single = imgs[j].unsqueeze(0)

        pred_class = preds[j].item()

        cam = gradcam.generate(img_single, pred_class, use_smoothing=True)

        original = imgs[j].cpu().permute(1, 2, 0).numpy()

        original = (original - original.min()) / (
            original.max() - original.min() + 1e-8
        )

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        heatmap = heatmap[:, :, ::-1] / 255.0

        overlay = np.clip(heatmap * 0.4 + original, 0, 1)

        label_tag = "tampered" if pred_class == 1 else "authentic"

        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{i}_{j}_{label_tag}.png")

        cv2.imwrite(save_path, np.uint8(255 * overlay[:, :, ::-1]))
