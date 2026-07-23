"""
Grad-CAM for the multi-label CXR model.

For a chosen target class, Grad-CAM weights the last conv layer's feature
maps by the gradient of that class's logit w.r.t. those feature maps, then
does a weighted sum + ReLU. The result is a coarse heatmap over the image
showing which regions pushed the prediction for that specific finding --
this is the piece that makes a black-box prediction clinically inspectable
(e.g. "the model called Cardiomegaly because of the heart silhouette, not
because of an artifact in the corner of the film").

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (2017).
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or model.target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        image: (1, 3, H, W) single preprocessed image, requires no grad
            from the caller -- gradients are handled internally.
        class_idx: index into NIH_CLASSES for the finding to explain.

        Returns: (H, W) heatmap in [0, 1], resized to the input resolution.
        """
        self.model.eval()
        image = image.clone().requires_grad_(True)

        logits = self.model(image)
        score = logits[0, class_idx]

        self.model.zero_grad()
        score.backward()

        # global-average-pool the gradients over spatial dims -> per-channel weight
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """
    image_np: (H, W, 3) uint8 or float [0,1] original image (unnormalized)
    heatmap: (H, W) in [0, 1] from GradCAM.__call__
    Returns: (H, W, 3) uint8 blended image, red = high attention.
    """
    import matplotlib as mpl

    if image_np.dtype != np.uint8:
        image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)

    colored = (mpl.colormaps["jet"](heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * colored + (1 - alpha) * image_np).astype(np.uint8)
    return blended


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from model import MultiLabelCXRModel, NIH_CLASSES

    torch.manual_seed(0)
    model = MultiLabelCXRModel(pretrained=False)
    cam_engine = GradCAM(model)

    dummy = torch.randn(1, 3, 224, 224)
    class_idx = NIH_CLASSES.index("Cardiomegaly")
    heatmap = cam_engine(dummy, class_idx)

    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0

    fake_image = ((dummy[0].permute(1, 2, 0).numpy() - dummy.min().item()))
    fake_image = fake_image / fake_image.max()
    blended = overlay_heatmap(fake_image, heatmap)
    assert blended.shape == (224, 224, 3) and blended.dtype == np.uint8

    cam_engine.close()
    print(f"[ok] gradcam heatmap shape={heatmap.shape}, range=({heatmap.min():.3f},{heatmap.max():.3f}), overlay shape={blended.shape}")
