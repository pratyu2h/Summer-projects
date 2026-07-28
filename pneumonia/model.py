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

        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Grad-CAM hooks
        self.target_layer = self.backbone.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.backbone.fc(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def unfreeze_backbone(self, from_layer: str = "layer3"):
        """Unfreeze the backbone starting from a specific layer."""
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
