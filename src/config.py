from pathlib import Path

import torch

class Config:
    SEED: int = 42

    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_CLASSES: int = 40

    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.15
    TEST_FRAC: float = 0.15

    NUM_EPOCHS_A: int = 30
    LR_A: float = 1e-3

    NUM_EPOCHS_B: int = 15
    LR_B: float = 3e-5
    SWIN_CHECKPOINT: str = "microsoft/swin-tiny-patch4-window7-224"
    FREEZE_EPOCHS_B: int = 3

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 4

    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = ROOT_DIR / "data"
    MODELS_DIR: Path = ROOT_DIR / "models"
    OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
    CHECKPOINTS_DIR: Path = ROOT_DIR / "checkpoints"