import os
import torch
import cv2
import numpy as np
import pydicom
from ml_models.image_model import model
from app.services.preprocess import transform
from app.services.gradcam import gradcam
from app.config.settings import OUTPUT_DIR, DEVICE


def run_inference(file_path):
    try:
        img_tensor = transform(file_path).to(DEVICE)

        with torch.no_grad():
            outputs       = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds      = torch.max(probabilities, 1)

        pred_class = preds.item()

        original = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        original = (original - original.min()) / (original.max() - original.min() + 1e-8)

        response = {
            "classification":        "Tampered" if pred_class == 1 else "Authentic",
            "tampered_probability":  float(round(probabilities[0][1].item(), 4)),
            "authentic_probability": float(round(probabilities[0][0].item(), 4)),
            "heatmap":               None,
            "heatmap_path":          None
        }

        if pred_class == 1:
            cam     = gradcam.generate(img_tensor, pred_class, use_smoothing=False)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = heatmap[:, :, ::-1] / 255.0   # BGR → RGB

            overlay = np.clip(heatmap * 0.4 + original, 0, 1)

            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            gradmap_path  = os.path.join(OUTPUT_DIR, f"tamper_localization_{base_filename}.png")
            cv2.imwrite(gradmap_path, np.uint8(255 * overlay[:, :, ::-1]))  # RGB → BGR for cv2

            response["heatmap"]      = overlay.tolist()
            response["heatmap_path"] = f"/output/{os.path.basename(gradmap_path)}"

        return response

    except Exception as e:
        raise Exception(f"Inference error: {str(e)}")