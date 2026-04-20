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

    def generate(self, input_tensor, class_idx,
                 use_smoothing=False, percentile_clip=95):
        self.model.eval()

        # Forward — needs grad, so no torch.no_grad() here
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Guard against hooks not firing
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        gradients   = self.gradients[0]    # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global-average-pool gradients → per-channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted sum of activation maps
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        cam = cv2.resize(cam, (224, 224))

        # Clip extreme values to reduce noise
        threshold = np.percentile(cam, percentile_clip)
        cam = np.minimum(cam, threshold)

        if use_smoothing:
            cam = cv2.bilateralFilter(cam.astype(np.float32), 5, 50, 50)

        cam_max = cam.max()
        cam = cam / cam_max if cam_max > 1e-8 else np.zeros_like(cam)

        return cam


gradcam = GradCAM(model, model.layer4[-1])