import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_segmenter import (
    IMAGE_SIZE,
    ISICSegmentationDataset,
    PROJECT_ROOT,
    UNet,
)


SPLITS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "isic2016_segmentation_splits.csv"
)

MODEL_PATH = PROJECT_ROOT / "models" / "segmenter_unet_isic2016.pth"

REPORTS_DIR = PROJECT_ROOT / "reports"

METRICS_PATH = REPORTS_DIR / "segmentation_test_metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "segmentation_test_predictions.csv"

BATCH_SIZE = 8


def main():
    print("=" * 60)
    print("SLICAI — Avaliação do segmentador no teste")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataframe = pd.read_csv(SPLITS_PATH)

    test_dataframe = dataframe[
        dataframe["split"] == "test"
    ].copy().reset_index(drop=True)

    print(f"Imagens no teste: {len(test_dataframe)}")
    print(f"Dispositivo utilizado: {device}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    threshold = float(checkpoint.get("threshold", 0.5))
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))

    print(f"Limiar utilizado: {threshold}")
    print(f"Tamanho das imagens: {image_size}x{image_size}")

    test_dataset = ISICSegmentationDataset(
        test_dataframe,
        image_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records = []
    processed_images = 0

    with torch.no_grad():
        progress_bar = tqdm(
            test_loader,
            desc="Avaliação no teste",
        )

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()

            intersection = (predictions * masks).sum(dim=(1, 2, 3))
            prediction_area = predictions.sum(dim=(1, 2, 3))
            mask_area = masks.sum(dim=(1, 2, 3))

            dice_values = (2 * intersection + 1.0) / (
                prediction_area + mask_area + 1.0
            )

            union = prediction_area + mask_area - intersection

            iou_values = (intersection + 1.0) / (union + 1.0)

            batch_size = images.size(0)
            batch_rows = test_dataframe.iloc[
                processed_images: processed_images + batch_size
            ]

            for index_in_batch, (_, row) in enumerate(
                batch_rows.iterrows()
            ):
                records.append(
                    {
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "mask_path": row["mask_path"],
                        "dice": float(dice_values[index_in_batch].item()),
                        "iou": float(iou_values[index_in_batch].item()),
                        "predicted_lesion_fraction": float(
                            predictions[index_in_batch].mean().item()
                        ),
                        "ground_truth_lesion_fraction": float(
                            masks[index_in_batch].mean().item()
                        ),
                    }
                )

            processed_images += batch_size

    results_dataframe = pd.DataFrame(records)

    mean_dice = float(results_dataframe["dice"].mean())
    mean_iou = float(results_dataframe["iou"].mean())

    metrics = {
        "dataset": "ISIC 2016 segmentation test",
        "total_images": int(len(results_dataframe)),
        "image_size": image_size,
        "threshold": threshold,
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "dice_standard_deviation": float(
            results_dataframe["dice"].std()
        ),
        "iou_standard_deviation": float(
            results_dataframe["iou"].std()
        ),
        "best_validation_dice": float(
            checkpoint.get("best_validation_dice", 0.0)
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    results_dataframe.to_csv(PREDICTIONS_PATH, index=False)

    print("\nResultados do teste:")
    print(f"Dice médio: {mean_dice:.4f}")
    print(f"IoU médio: {mean_iou:.4f}")

    print("\nArquivos gerados:")
    print(METRICS_PATH)
    print(PREDICTIONS_PATH)


if __name__ == "__main__":
    main()