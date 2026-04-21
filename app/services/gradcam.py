import torch
import cv2
import numpy as np
from app.config.settings import DEVICE
from ml_models.image_model import model


class GradCAM:

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _get_body_mask(self, original_img_np):
        """
        Create a binary mask that isolates the body/anatomy region from
        the black background in a CT/MRI scan.

        Steps:
          1. Convert the normalised [0,1] RGB array to uint8 grayscale.
          2. Threshold: any pixel brighter than 10/255 belongs to the body.
          3. Morphological close  → fill small holes inside the body.
          4. Morphological open   → remove thin noise speckles outside.
          5. Keep only the largest connected component (the body itself).

        Returns a float32 mask of shape (224, 224) with values in {0, 1}.
        """
        gray = cv2.cvtColor(
            (original_img_np * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )

        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        kernel = np.ones((15, 15), np.uint8)
        mask   = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel)

        # Keep largest connected component only
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            # component 0 is always the background; skip it
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask    = np.where(labels == largest, 255, 0).astype(np.uint8)

        return mask.astype(np.float32) / 255.0   # → [0, 1]

    def generate(self, input_tensor, class_idx,
                 use_smoothing=True, percentile_clip=95):
        """
        Generate a GradCAM heatmap for `class_idx`.

        Key changes vs original:
          * Body mask applied BEFORE percentile clipping so the clip
            threshold is computed only from within-body pixels.
          * use_smoothing defaults to True for cleaner localisation.

        Returns a numpy array of shape (224, 224) normalised to [0, 1].
        """
        self.model.eval()

        # Forward pass — need gradients, so no torch.no_grad()
        output = self.model(input_tensor)

        self.model.zero_grad()
        output[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        gradients   = self.gradients[0]    # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global-average-pool the gradients → per-channel weights
        weights = gradients.mean(dim=(1, 2))          # (C,)

        # Weighted sum of activation maps
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        # Resize to input resolution
        cam = cv2.resize(cam, (224, 224))

        # ── Body-mask: zero-out background activations ──────────────
        original_np = input_tensor[0].cpu().permute(1, 2, 0).numpy()
        original_np = (
            (original_np - original_np.min())
            / (original_np.max() - original_np.min() + 1e-8)
        )
        body_mask = self._get_body_mask(original_np)
        cam       = cam * body_mask   # background pixels → 0
        # ─────────────────────────────────────────────────────────────

        # Percentile clip computed only over body pixels to avoid
        # background values pulling the threshold down
        body_pixels = cam[body_mask > 0]
        if body_pixels.size > 0:
            threshold = np.percentile(body_pixels, percentile_clip)
        else:
            threshold = cam.max()
        cam = np.minimum(cam, threshold)

        # Optional bilateral smoothing (default ON)
        if use_smoothing:
            cam = cv2.bilateralFilter(cam.astype(np.float32), 9, 75, 75)

        # Normalise to [0, 1]
        cam_max = cam.max()
        cam     = cam / cam_max if cam_max > 1e-8 else np.zeros_like(cam)

        return cam


gradcam = GradCAM(model, model.layer4[-1])
