import torch
import torch.nn as nn

from src.config import Config


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = Config.NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BaselineCNN()
    n_params = count_parameters(model)
    print(f"BaselineCNN trainable parameters: {n_params:,}")

    dummy = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (2, Config.NUM_CLASSES)
    print("Forward pass OK")