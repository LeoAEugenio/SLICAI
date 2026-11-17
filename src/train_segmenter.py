from pathlib import Path
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as transform_functional
from tqdm import tqdm


# ============================================================
# Configurações
# ============================================================

QUICK_TEST = False

IMAGE_SIZE = 256
BATCH_SIZE = 8
FULL_TRAINING_EPOCHS = 15
QUICK_TEST_EPOCHS = 2
LEARNING_RATE = 0.0005
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "isic2016_segmentation_splits.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

if QUICK_TEST:
    MODEL_PATH = MODELS_DIR / "segmenter_unet_smoke_test.pth"
    HISTORY_PATH = REPORTS_DIR / "segmentation_smoke_test_history.csv"
    EPOCHS = QUICK_TEST_EPOCHS
else:
    MODEL_PATH = MODELS_DIR / "segmenter_unet_isic2016.pth"
    HISTORY_PATH = REPORTS_DIR / "segmentation_training_history.csv"
    EPOCHS = FULL_TRAINING_EPOCHS


# ============================================================
# Utilitários
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Dataset
# ============================================================

class ISICSegmentationDataset(Dataset):
    def __init__(self, dataframe, image_size):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )

        image_tensor = transform_functional.to_tensor(image)
        image_tensor = transform_functional.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask_array = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(
            (mask_array > 0).astype(np.float32)
        ).unsqueeze(0)

        return image_tensor, mask_tensor


# ============================================================
# Arquitetura U-Net leve
# ============================================================

class DoubleConvolution(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = DoubleConvolution(3, 16)
        self.encoder_2 = DoubleConvolution(16, 32)
        self.encoder_3 = DoubleConvolution(32, 64)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConvolution(64, 128)

        self.up_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_3 = DoubleConvolution(128, 64)

        self.up_2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_2 = DoubleConvolution(64, 32)

        self.up_1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder_1 = DoubleConvolution(32, 16)

        self.output_layer = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        encoder_1 = self.encoder_1(x)

        encoder_2 = self.encoder_2(self.pool(encoder_1))

        encoder_3 = self.encoder_3(self.pool(encoder_2))

        bottleneck = self.bottleneck(self.pool(encoder_3))

        decoder_3 = self.up_3(bottleneck)
        decoder_3 = self.decoder_3(
            torch.cat([decoder_3, encoder_3], dim=1)
        )

        decoder_2 = self.up_2(decoder_3)
        decoder_2 = self.decoder_2(
            torch.cat([decoder_2, encoder_2], dim=1)
        )

        decoder_1 = self.up_1(decoder_2)
        decoder_1 = self.decoder_1(
            torch.cat([decoder_1, encoder_1], dim=1)
        )

        return self.output_layer(decoder_1)


# ============================================================
# Loss e métricas
# ============================================================

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.binary_cross_entropy = nn.BCEWithLogitsLoss()

    def forward(self, logits, masks):
        bce_loss = self.binary_cross_entropy(logits, masks)

        probabilities = torch.sigmoid(logits)

        intersection = (probabilities * masks).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

        dice = (2 * intersection + 1.0) / (denominator + 1.0)
        dice_loss = 1 - dice.mean()

        return bce_loss + dice_loss


def calculate_metrics(logits, masks):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

    intersection = (predictions * masks).sum(dim=(1, 2, 3))
    prediction_area = predictions.sum(dim=(1, 2, 3))
    mask_area = masks.sum(dim=(1, 2, 3))

    dice = (2 * intersection + 1.0) / (
        prediction_area + mask_area + 1.0
    )

    union = prediction_area + mask_area - intersection

    iou = (intersection + 1.0) / (union + 1.0)

    return dice.mean().item(), iou.mean().item()


# ============================================================
# Treino e validação
# ============================================================

def run_epoch(model, data_loader, loss_function, device, optimizer=None):
    is_training = optimizer is not None

    if is_training:
        model.train()
        description = "Treinamento"
    else:
        model.eval()
        description = "Validação"

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_images = 0

    progress_bar = tqdm(data_loader, desc=description, leave=False)

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        if is_training:
            optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, masks)

            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = loss_function(logits, masks)

        dice, iou = calculate_metrics(logits, masks)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        total_images += batch_size

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / total_images,
        "dice": total_dice / total_images,
        "iou": total_iou / total_images,
    }


# ============================================================
# Execução principal
# ============================================================

def main():
    set_seed(RANDOM_SEED)

    torch.set_num_threads(min(6, os.cpu_count() or 1))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("SLICAI — Treinamento do segmentador U-Net")
    print("=" * 60)
    print(f"Modo de teste rápido: {QUICK_TEST}")
    print(f"Dispositivo utilizado: {device}")
    print(f"Tamanho das imagens: {IMAGE_SIZE}x{IMAGE_SIZE}")

    dataframe = pd.read_csv(SPLITS_PATH)

    train_dataframe = dataframe[dataframe["split"] == "train"].copy()
    validation_dataframe = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    if QUICK_TEST:
        train_dataframe = train_dataframe.sample(
            n=min(96, len(train_dataframe)),
            random_state=RANDOM_SEED,
        )

        validation_dataframe = validation_dataframe.sample(
            n=min(32, len(validation_dataframe)),
            random_state=RANDOM_SEED,
        )

    print(f"Imagens de treinamento: {len(train_dataframe)}")
    print(f"Imagens de validação: {len(validation_dataframe)}")
    print(f"Épocas: {EPOCHS}")

    train_dataset = ISICSegmentationDataset(
        train_dataframe,
        IMAGE_SIZE,
    )

    validation_dataset = ISICSegmentationDataset(
        validation_dataframe,
        IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNet().to(device)

    loss_function = BCEDiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.00001,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best_validation_dice = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nÉpoca {epoch}/{EPOCHS}")

        training_results = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )

        validation_results = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )

        print(
            f"Treino — Loss: {training_results['loss']:.4f} | "
            f"Dice: {training_results['dice']:.4f} | "
            f"IoU: {training_results['iou']:.4f}"
        )

        print(
            f"Validação — Loss: {validation_results['loss']:.4f} | "
            f"Dice: {validation_results['dice']:.4f} | "
            f"IoU: {validation_results['iou']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_results["loss"],
                "training_dice": training_results["dice"],
                "training_iou": training_results["iou"],
                "validation_loss": validation_results["loss"],
                "validation_dice": validation_results["dice"],
                "validation_iou": validation_results["iou"],
            }
        )

        if validation_results["dice"] > best_validation_dice:
            best_validation_dice = validation_results["dice"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "U-Net",
                    "image_size": IMAGE_SIZE,
                    "threshold": 0.5,
                    "best_validation_dice": best_validation_dice,
                    "quick_test": QUICK_TEST,
                },
                MODEL_PATH,
            )

            print("Novo melhor modelo salvo.")

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    print("\nTreinamento concluído.")
    print(f"Melhor Dice de validação: {best_validation_dice:.4f}")
    print(f"Modelo salvo em:\n{MODEL_PATH}")
    print(f"Histórico salvo em:\n{HISTORY_PATH}")


if __name__ == "__main__":
    main()