import os
import random
import math
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
from functools import partial
from sklearn.model_selection import train_test_split

from dataset import FoodCaloriesDataset, build_transforms


# ---------------------- воспроизводимость ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------- конфиг ----------------------
def load_cfg(source) -> dict:
    """cfg может быть путем к yaml-файлу или словарем."""
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return dict(source)


# ---------------------- модель ----------------------
class CalorieModel(nn.Module):
    """Текст (ингредиенты) + фото -> калорийность."""

    def __init__(self, text_model_name: str = "distilbert-base-uncased",
                 image_model_name: str = "resnet18",
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=0)

        text_dim = self.text_encoder.config.hidden_size
        image_dim = self.image_encoder.num_features

        # проекционные слои приводят оба энкодера к общей размерности
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        # финальная голова-регрессор
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, input_ids, attention_mask, images):
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = t_out.last_hidden_state[:, 0]          # [CLS]-токен
        img_feat = self.image_encoder(images)              # pooled-вектор
        z = torch.cat([self.text_projection(text_feat),
                       self.image_projection(img_feat)], dim=1)
        return self.classifier(z).squeeze(1)               # предсказание log1p(calories)


# ---------------------- коллатор ----------------------
def make_collate(tokenizer, max_len: int = 64):
    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        targets = torch.tensor([b["target"] for b in batch], dtype=torch.float32)
        masses = torch.tensor([b["mass"] for b in batch], dtype=torch.float32)
        enc = tokenizer([b["text"] for b in batch], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "images": images,
                "targets": targets,
                "masses": masses}
    return collate



# ---------------------- одна эпоха ----------------------
def run_epoch(model, loader, optimizer, criterion, device, mae, rmse):
    model.train() if optimizer is not None else model.eval()
    total_loss, n_batches = 0.0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)      # калорийность на грамм
            masses = batch["masses"].to(device)

            preds = model(input_ids, attention_mask, images)   # предсказание плотности
            loss = criterion(preds, targets)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # переводим обратно в калории и считаем метрики
            preds_cal = preds * masses
            targets_cal = targets * masses
            mae.update(preds_cal.detach().cpu(), targets_cal.cpu())
            rmse.update(preds_cal.detach().cpu(), targets_cal.cpu())

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1), mae.compute().item(), rmse.compute().item()

# ---------------------- обучение ----------------------
def train(cfg):
    """Точка входа: train("config.yaml") или train(dict)."""
    cfg = load_cfg(cfg)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ----- данные -----
    dish = pd.read_csv(cfg["dish_csv"])
    df_train_full = dish[dish["split"] == "train"].reset_index(drop=True)

    # стратифицированный сплит по бинам калорийности
    bins = pd.qcut(df_train_full["total_calories"], q=10, labels=False)
    tr_idx, val_idx = train_test_split(
        np.arange(len(df_train_full)),
        test_size=cfg.get("val_size", 0.1),
        stratify=bins,
        random_state=seed)

    train_ds = FoodCaloriesDataset(
        df_train_full.iloc[tr_idx], cfg["ingredients_csv"], cfg["image_root"],
        build_transforms(cfg.get("image_size", 224), augment=True))
    val_ds = FoodCaloriesDataset(
        df_train_full.iloc[val_idx], cfg["ingredients_csv"], cfg["image_root"],
        build_transforms(cfg.get("image_size", 224), augment=False))

    tokenizer = AutoTokenizer.from_pretrained(cfg.get("text_model", "distilbert-base-uncased"))
    collate = make_collate(tokenizer, cfg.get("max_len", 64))

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32), shuffle=True,
                              num_workers=cfg.get("num_workers", 2), collate_fn=collate,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 32), shuffle=False,
                            num_workers=cfg.get("num_workers", 2), collate_fn=collate,
                            pin_memory=True)

    # ----- модель и оптимизатор -----
    model = CalorieModel(
        text_model_name=cfg.get("text_model", "distilbert-base-uncased"),
        image_model_name=cfg.get("image_model", "resnet18"),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.2)).to(device)

    optimizer = AdamW([
        {"params": model.text_encoder.parameters(), "lr": cfg["TEXT_LR"]},
        {"params": model.image_encoder.parameters(), "lr": cfg["IMAGE_LR"]},
        {"params": model.text_projection.parameters(), "lr": cfg["CLASSIFIER_LR"]},
        {"params": model.image_projection.parameters(), "lr": cfg["CLASSIFIER_LR"]},
        {"params": model.classifier.parameters(), "lr": cfg["CLASSIFIER_LR"]},
    ], weight_decay=cfg.get("weight_decay", 0.01))

    loss_name = cfg.get("loss", "smooth_l1")
    criterion = nn.L1Loss()
    use_log_target = False

    # ----- цикл обучения -----
    best_mae = float("inf")
    for epoch in range(1, cfg.get("epochs", 5) + 1):
        train_mae = torchmetrics.regression.MeanAbsoluteError()
        train_rmse = torchmetrics.regression.MeanSquaredError(squared=False)
        val_mae = torchmetrics.regression.MeanAbsoluteError()
        val_rmse = torchmetrics.regression.MeanSquaredError(squared=False)

        tr_loss, tr_mae, tr_rmse = run_epoch(model, train_loader, optimizer, criterion,
                                             device, train_mae, train_rmse)
        va_loss, va_mae, va_rmse = run_epoch(model, val_loader, None, criterion,
                                             device, val_mae, val_rmse)


        print(f"[эпоха {epoch:02d}] "
              f"train loss={tr_loss:.4f} MAE={tr_mae:.1f} RMSE={tr_rmse:.1f} | "
              f"val loss={va_loss:.4f} MAE={va_mae:.1f} RMSE={va_rmse:.1f}")

        if va_mae < best_mae:
            best_mae = va_mae
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch},
                       cfg.get("save_path", "best_model.pt"))
            print(f"  -> сохранён лучший чекпоинт (val MAE={va_mae:.1f} кал)")

    # ----- финальная оценка на тесте -----
    final_metrics = {"best_val_mae": best_mae}
    if cfg.get("evaluate_test", False):
        df_test = dish[dish["split"] == "test"].reset_index(drop=True)
        test_ds = FoodCaloriesDataset(df_test, cfg["ingredients_csv"], cfg["image_root"],
                                      build_transforms(cfg.get("image_size", 224), augment=False))
        test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 32), shuffle=False,
                                 num_workers=cfg.get("num_workers", 2), collate_fn=collate,
                                 pin_memory=True)
        test_mae = torchmetrics.regression.MeanAbsoluteError()
        test_rmse = torchmetrics.regression.MeanSquaredError(squared=False)
        _, te_mae, te_rmse = run_epoch(model, test_loader, None, criterion,
                                       device, test_mae, test_rmse)
        final_metrics.update({"test_mae": te_mae, "test_rmse": te_rmse})
        print(f"ТЕСТ: MAE={te_mae:.1f} кал, RMSE={te_rmse:.1f} кал")

    return final_metrics
