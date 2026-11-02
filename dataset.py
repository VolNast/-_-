import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, augment: bool = True) -> A.Compose:
    """Аугментации для train и честная ресайз-обработка для val/test."""
    if augment:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class FoodCaloriesDataset(Dataset):
    """Фото блюда + ингредиенты текстом + целевая переменная (total_calories)."""

    def __init__(self, df: pd.DataFrame,
                 ingredients_csv: str = "data/ingredients.csv",
                 image_root: str = "data/images",
                 transform: A.Compose = None):
        self.image_root = image_root
        self.transform = transform

        ingr = pd.read_csv(ingredients_csv)
        id2name = dict(zip(ingr["id"].astype(int), ingr["ingr"].astype(str)))

        def ids_to_names(s: str) -> str:
            names = []
            for token in str(s).split(";"):
                try:
                    idx = int(token.replace("ingr_", "").lstrip("0") or 0)
                    names.append(id2name.get(idx, "unknown ingredient"))
                except ValueError:
                    names.append(token)
            return ", ".join(names)

        self.df = df.reset_index(drop=True).copy()
        self.df["text"] = self.df["ingredients"].apply(ids_to_names)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        dish_id = str(row["dish_id"])
        img_path = os.path.join(self.image_root, dish_id, "rgb.png")
        image = np.array(Image.open(img_path).convert("RGB"))
        image = self.transform(image=image)["image"]
        return {
            "image": image,
            "text": row["text"],
            "target": float(row["total_calories"]) / max(float(row["total_mass"]), 1.0),
            "mass": float(row["total_mass"]),
            "dish_id": dish_id,
        }