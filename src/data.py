import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import Config
from src.utils import seed_worker

FOOD40 = sorted([
    "pizza", "spaghetti_bolognese", "spaghetti_carbonara", "lasagna",
    "ravioli", "gnocchi", "risotto", "paella",
    "caesar_salad", "greek_salad", "caprese_salad", "beet_salad",
    "grilled_salmon", "steak", "filet_mignon", "pork_chop",
    "chicken_wings", "fish_and_chips", "mussels",
    "hamburger", "club_sandwich", "grilled_cheese_sandwich", "hot_dog",
    "french_fries", "french_onion_soup", "lobster_bisque", "clam_chowder",
    "omelette", "pancakes", "waffles", "french_toast", "eggs_benedict",
    "croque_madame", "bruschetta", "hummus", "falafel",
    "tiramisu", "cheesecake", "chocolate_cake", "creme_brulee",
])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(Config.IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(Config.IMAGE_SIZE),
    transforms.CenterCrop(Config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_food40_splits():
    ds = load_dataset("ethz/food101", cache_dir=str(Config.DATA_DIR))
    full = concatenate_datasets([ds["train"], ds["validation"]])

    label_feature = full.features["label"]
    keep_ids = set(label_feature.str2int(name) for name in FOOD40)
    subset = full.filter(lambda ex: ex["label"] in keep_ids)

    label_map = {label_feature.str2int(name): i for i, name in enumerate(FOOD40)}

    split1 = subset.train_test_split(
        test_size=Config.TEST_FRAC,
        seed=Config.SEED,
        stratify_by_column="label",
    )
    test_ds = split1["test"]

    val_frac_of_rest = Config.VAL_FRAC / (Config.TRAIN_FRAC + Config.VAL_FRAC)
    split2 = split1["train"].train_test_split(
        test_size=val_frac_of_rest,
        seed=Config.SEED,
        stratify_by_column="label",
    )
    train_ds, val_ds = split2["train"], split2["test"]

    return train_ds, val_ds, test_ds, label_map


class Food40Dataset(Dataset):
    def __init__(self, hf_dataset, transform, label_map):
        self.ds = hf_dataset
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        return self.transform(image), self.label_map[item["label"]]


def get_dataloaders():
    train_ds, val_ds, test_ds, label_map = load_food40_splits()

    train_set = Food40Dataset(train_ds, TRAIN_TRANSFORM, label_map)
    val_set = Food40Dataset(val_ds, EVAL_TRANSFORM, label_map)
    test_set = Food40Dataset(test_ds, EVAL_TRANSFORM, label_map)

    g = torch.Generator()
    g.manual_seed(Config.SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
