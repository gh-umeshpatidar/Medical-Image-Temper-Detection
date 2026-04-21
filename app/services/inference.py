import os
import torch
import cv2
import numpy as np
import pydicom
from ml_models.image_model import model
from app.services.preprocess import transform
from app.services.gradcam import gradcam
from app.config.settings import OUTPUT_DIR, DEVICE
import time


def save_as_dicom(img_np, output_path, reference_path):
    """Save a 2-D uint16 array as DICOM, copying metadata from reference."""
    if img_np.ndim != 2:
        raise ValueError(f"save_as_dicom expects a 2-D array, got shape {img_np.shape}")

    ds                    = pydicom.dcmread(reference_path)
    ds.PixelData          = img_np.tobytes()
    ds.Rows, ds.Columns   = img_np.shape
    ds.BitsAllocated      = 16
    ds.BitsStored         = 12
    ds.HighBit            = 11
    ds.PixelRepresentation = 0
    ds.WindowCenter       = 40
    ds.WindowWidth        = 400
    ds.RescaleIntercept   = 0
    ds.RescaleSlope       = 1
    ds.save_as(output_path)


def run_inference(file_path):
    try:
        start_time = time.time()

        # ── Load & save original DICOM ────────────────────────────────
        dicom        = pydicom.dcmread(file_path)
        original_img = dicom.pixel_array.astype(np.uint16)
        if original_img.ndim != 2:
            original_img = original_img[0]          # take first slice if 3-D
        save_as_dicom(original_img, "output/before.dcm", file_path)

        # ── Preprocess & save transformed DICOM ──────────────────────
        img_tensor = transform(file_path).to(DEVICE)

        img_np = img_tensor.squeeze().cpu().numpy()
        if img_np.ndim == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        img_np = np.clip(img_np * 0.5 + 0.5, 0, 1)
        img_np = (img_np * 4095).astype(np.uint16)
        save_as_dicom(img_np, "output/after.dcm", file_path)

        # ── Inference ─────────────────────────────────────────────────
        with torch.no_grad():
            outputs       = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds      = torch.max(probabilities, 1)

        pred_class = preds.item()

        # Normalised original for overlay blending
        original_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        original_np = (
            (original_np - original_np.min())
            / (original_np.max() - original_np.min() + 1e-8)
        )

        # ── Build initial response ────────────────────────────────────
        response = {
            "classification":        "Tampered" if pred_class == 1 else "Authentic",
            "tampered_probability":  float(round(probabilities[0][1].item(), 4)),
            "authentic_probability": float(round(probabilities[0][0].item(), 4)),
            "heatmap":               None,
            "heatmap_path":          None,
        }

        # ── GradCAM — generated for ALL predictions ───────────────────
        # Previously this block ran only when pred_class == 1.
        # We now generate it unconditionally so that:
        #   • Tampered images show WHERE the forgery is localised.
        #   • Authentic images show which regions the model found clean
        #     (useful for debugging and transparency).
        # use_smoothing=True applies bilateral filtering to reduce
        # speckle noise, producing a cleaner heatmap.
        cam     = gradcam.generate(img_tensor, pred_class, use_smoothing=True)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = heatmap[:, :, ::-1] / 255.0   # BGR → RGB

        overlay = np.clip(heatmap * 0.4 + original_np, 0, 1)

        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        label_suffix  = "tampered" if pred_class == 1 else "authentic"
        gradmap_path  = os.path.join(
            OUTPUT_DIR,
            f"tamper_localization_{base_filename}_{label_suffix}.png"
        )
        # cv2.imwrite expects BGR
        cv2.imwrite(gradmap_path, np.uint8(255 * overlay[:, :, ::-1]))

        response["heatmap"]      = overlay.tolist()
        response["heatmap_path"] = f"/output/{os.path.basename(gradmap_path)}"

        # ── Timing ────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        label   = "tampered" if pred_class == 1 else "real"
        print(f"Execution time for {label} image: {elapsed:.2f}s")

        return response

    except Exception as e:
        raise Exception(f"Inference error: {str(e)}")
