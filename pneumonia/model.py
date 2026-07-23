"""
Multi-label chest X-ray classifier.

ResNet-50 backbone (ImageNet-pretrained) with the final FC layer replaced
by a 14-way linear head, one logit per NIH ChestX-14 finding. No sigmoid
inside the model -- keep raw logits and apply sigmoid at inference time
so the loss function (BCEWithLogits / focal loss) stays numerically stable.
"""
import torch
import torch.nn as nn
from torchvision import models

NIH_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


class MultiLabelCXRModel(nn.Module):
    def __init__(self, num_classes: int = len(NIH_CLASSES), pretrained: bool = True,
                 freeze_backbone: bool = False, dropout: float = 0.2):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # replace the classification head; keep it trainable even if the
        # rest of the backbone is frozen
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # kept for Grad-CAM: last conv block before global average pooling
        self.target_layer = self.backbone.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # raw logits, shape (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def unfreeze_backbone(self, from_layer: str = "layer3"):
        """
        Progressive unfreezing: start training only the head, then unfreeze
        deeper layers once the head has stabilized. from_layer in
        {'layer1','layer2','layer3','layer4'} unfreezes that block onward.
        """
        order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
        start = order.index(from_layer)
        unfreeze = False
        for name, child in self.backbone.named_children():
            if name in order and order.index(name) >= start:
                unfreeze = True
            if unfreeze:
                for p in child.parameters():
                    p.requires_grad = True


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MultiLabelCXRModel(pretrained=False)  # skip downloading ImageNet weights for the smoke test
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (2, len(NIH_CLASSES)), out.shape
    probs = model.predict_proba(dummy)
    assert probs.shape == out.shape and (probs >= 0).all() and (probs <= 1).all()

    model.unfreeze_backbone("layer3")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[ok] forward shape={tuple(out.shape)}, trainable params after unfreeze(layer3)={trainable:,}/{total:,}")
