# NOTE: This script is a planned extension for embedded deployment.
# It has not been tested or benchmarked in this certification phase.

import torch
import torch.nn as nn
import torch.optim as optim
import scripts.config as config
from torchvision import transforms
from torch.utils.data import DataLoader
from scripts.Segmentation.segDS import SegmentationDataset

# ----------------------
# Compact U-Net-like model defined explicitly
# ----------------------
class EmbeddedUNet(nn.Module):
    def __init__(self):
        super().init()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 128 -> 64
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64 -> 32
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )

        # Output
        self.out = nn.Conv2d(8, 2, kernel_size=1)  # 2 classes: background + object

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x = self.bottleneck(x2)
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        return self.out(x)

# ----------------------
# Configuration
# ----------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------
# Data Loading
# ----------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

train_ds = SegmentationDataset(config.images,
                                config.masks,
                                transform)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------
# Training
# ----------------------
model = EmbeddedUNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, masks in train_dl:
        imgs, masks = imgs.to(DEVICE), masks.long().to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# ----------------------
# Save weights only
# ----------------------
torch.save(model.state_dict(), 'embedded_model_weights.pth')

# ----------------------
# Export to ONNX (float16)
# ----------------------
model.eval()
dummy_input = torch.randn(1, 1, *IMG_SIZE).to(DEVICE)
torch.onnx.export(
    model.half(),                  # Convert model to float16
    dummy_input.half(),            # Dummy input in float16
    "embedded_model_fp16.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    do_constant_folding=True,
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

print("✅ Model exported as embedded_model_fp16.onnx")
