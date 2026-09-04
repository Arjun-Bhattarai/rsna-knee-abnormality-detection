import torch

from src.models.dinov2 import DINOv2Backbone
from src.models.backbone import VisionBackbone
from src.models.model import KneeModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# Real DINOv2
dinov2 = DINOv2Backbone(
    model_name="facebook/dinov2-base",
    freeze=True,
)

# Adapt DINOv2 to our existing architecture
backbone = VisionBackbone(
    backbone=dinov2,
    feature_dim=dinov2.feature_dim,
    freeze=True,
).to(device)

model = KneeModel(
    backbone=backbone,
    feature_dim=dinov2.feature_dim,
    num_targets=12,
).to(device)

model.eval()

# Fake MRI:
# B=1, S=4, C=1, H=224, W=224
sagittal = torch.randn(1, 4, 1, 224, 224, device=device)
coronal = torch.randn(1, 4, 1, 224, 224, device=device)
axial = torch.randn(1, 4, 1, 224, 224, device=device)

with torch.no_grad():
    logits = model(
        sagittal,
        coronal,
        axial,
    )

print("Sagittal:", sagittal.shape)
print("Coronal:", coronal.shape)
print("Axial:", axial.shape)
print("Output logits:", logits.shape)
print("Output:", logits)