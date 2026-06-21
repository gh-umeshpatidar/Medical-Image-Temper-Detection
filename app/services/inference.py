import os

import torch

import cv2

import numpy as np

from app.services import gradcam
from ml_models.image_model import get_model

from app.services.preprocess import transform

from app.services.gradcam import get_gradcam

from app.config.settings import OUTPUT_DIR, DEVICE

import time


def run_inference(file_path):

    try:

        start_time = time.time()

        img_tensor = transform(file_path).to(DEVICE)

        with torch.no_grad():
            model = get_model()
            outputs = model(img_tensor)

            probabilities = torch.softmax(outputs, dim=1)

            _, preds = torch.max(probabilities, 1)

        pred_class = preds.item()

        original_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()

        original_np = (original_np - original_np.min()) / (
            original_np.max() - original_np.min() + 1e-8
        )

        response = {
            "classification": "Tampered" if pred_class == 1 else "Authentic",
            "tampered_probability": float(round(probabilities[0][1].item(), 4)),
            "authentic_probability": float(round(probabilities[0][0].item(), 4)),
            "heatmap": None,
            "heatmap_path": None,
        }

        if pred_class == 1:

            gradcam = get_gradcam()
            cam = gradcam.generate(img_tensor, pred_class, use_smoothing=True)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

            heatmap = heatmap[:, :, ::-1] / 255.0

            overlay = np.clip(heatmap * 0.4 + original_np, 0, 1)

            base_filename = os.path.splitext(os.path.basename(file_path))[0]

            gradmap_path = os.path.join(
                OUTPUT_DIR, f"tamper_localization_{base_filename}_tampered.png"
            )

            cv2.imwrite(gradmap_path, np.uint8(255 * overlay[:, :, ::-1]))

            response["heatmap"] = overlay.tolist()

            response["heatmap_path"] = f"/output/{os.path.basename(gradmap_path)}"

        elapsed = time.time() - start_time

        label = "tampered" if pred_class == 1 else "real"

        print(f"Execution time for {label} image: {elapsed:.2f}s")

        return response

    except Exception as e:

        raise Exception(f"Inference error: {str(e)}")
