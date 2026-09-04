import torch

from src.models.dinov2 import DINOv2Backbone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

model = DINOv2Backbone(
    model_name="facebook/dinov2-base",
    freeze=True,
).to(device)

model.eval()

x = torch.randn(
    1, 1, 224, 224,
    device=device,
)

with torch.no_grad():
    features = model(x)

print("Input shape:", x.shape)
print("Feature shape:", features.shape)
print("Feature dimension:", model.feature_dim)