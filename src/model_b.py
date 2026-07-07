import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.config import Config


def get_model_b(num_classes: int = Config.NUM_CLASSES):
    model = AutoModelForImageClassification.from_pretrained(
        Config.SWIN_CHECKPOINT,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    processor = AutoImageProcessor.from_pretrained(Config.SWIN_CHECKPOINT)
    return model, processor


def freeze_backbone(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model) -> None:
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model, processor = get_model_b()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:      {total:,}")

    freeze_backbone(model)
    print(f"Trainable (frozen):    {count_trainable(model):,}")

    unfreeze_all(model)
    print(f"Trainable (unfrozen):  {count_trainable(model):,}")

    dummy = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    with torch.no_grad():
        out = model(pixel_values=dummy).logits
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (2, Config.NUM_CLASSES)
    print("Forward pass OK")
