"""Avalia o U-Net salvo no conjunto de teste oficial da ISIC 2016.

Coloque este arquivo em C:\\SLICAI\\src e execute a partir de C:\\SLICAI:
    .\\.venv\\Scripts\\python.exe .\\src\\evaluate_isic2016_official.py
"""

import csv
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_segmenter import ISICSegmentationDataset, PROJECT_ROOT, UNet


TEST_ROOT = PROJECT_ROOT / "data" / "raw" / "isic2016_official_test"
IMAGE_DIR = TEST_ROOT / "images"
MASK_DIR = TEST_ROOT / "masks"
MODEL_PATH = PROJECT_ROOT / "models" / "segmenter_unet_isic2016.pth"
REPORTS_DIR = PROJECT_ROOT / "reports"
CSV_PATH = REPORTS_DIR / "segmentation_official_test_predictions.csv"
JSON_PATH = REPORTS_DIR / "segmentation_official_test_metrics.json"
EXPECTED_PAIRS = 379


def collect_pairs():
    images = sorted(IMAGE_DIR.rglob("*.jpg"))
    masks = sorted(MASK_DIR.rglob("*.png"))
    if len(images) != EXPECTED_PAIRS or len(masks) != EXPECTED_PAIRS:
        raise ValueError(
            f"Esperados {EXPECTED_PAIRS} JPG e {EXPECTED_PAIRS} PNG; "
            f"encontrados {len(images)} JPG e {len(masks)} PNG."
        )

    image_by_id = {path.stem: path for path in images}
    suffix = "_Segmentation"
    mask_by_id = {
        path.stem[: -len(suffix)]: path
        for path in masks
        if path.stem.endswith(suffix)
    }
    if (len(image_by_id) != EXPECTED_PAIRS
            or len(mask_by_id) != EXPECTED_PAIRS
            or set(image_by_id) != set(mask_by_id)):
        missing_masks = sorted(set(image_by_id) - set(mask_by_id))
        missing_images = sorted(set(mask_by_id) - set(image_by_id))
        raise ValueError(
            "Imagens e máscaras não formam 379 pares únicos. "
            f"Sem máscara: {missing_masks[:5]}; sem imagem: {missing_images[:5]}"
        )

    # Evita avaliar como 'externas' imagens eventualmente presentes no treino.
    development_images = PROJECT_ROOT / "data" / "raw" / "isic2016_segmentation" / "images"
    if development_images.is_dir():
        development_ids = {path.stem for path in development_images.rglob("*.jpg")}
        overlap = sorted(set(image_by_id) & development_ids)
        if overlap:
            raise ValueError(
                f"{len(overlap)} IDs também estão no conjunto de desenvolvimento: "
                f"{overlap[:5]}. Verifique vazamento antes de avaliar."
            )

    return pd.DataFrame(
        [
            {
                "image_id": image_id,
                "image_path": str(image_by_id[image_id]),
                "mask_path": str(mask_by_id[image_id]),
            }
            for image_id in sorted(image_by_id)
        ]
    )


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
    if CSV_PATH.exists() or JSON_PATH.exists():
        raise FileExistsError(
            "Relatório oficial já existe. Confira os arquivos antes de uma nova execução."
        )

    pairs = collect_pairs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if checkpoint.get("quick_test") is True:
        raise ValueError("O checkpoint indicado é de um teste rápido, não do treino final.")
    image_size = int(checkpoint.get("image_size", 256))
    threshold = float(checkpoint.get("threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError(f"Limiar inválido no checkpoint: {threshold}")

    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        ISICSegmentationDataset(pairs, image_size),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    rows = []
    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            predictions = (torch.sigmoid(logits) >= threshold).float()
            masks = masks.to(device)
            intersection = (predictions * masks).sum(dim=(1, 2, 3))
            prediction_area = predictions.sum(dim=(1, 2, 3))
            mask_area = masks.sum(dim=(1, 2, 3))
            dice = (2 * intersection + 1) / (prediction_area + mask_area + 1)
            union = prediction_area + mask_area - intersection
            iou = (intersection + 1) / (union + 1)
            for d, j in zip(dice.tolist(), iou.tolist()):
                pair = pairs.iloc[len(rows)]
                rows.append({
                    "image_id": pair["image_id"],
                    "dice": d,
                    "iou": j,
                })

    summary = {
        "dataset": "ISIC 2016 official Task 1 test",
        "images_evaluated": len(rows),
        "mean_dice": sum(row["dice"] for row in rows) / len(rows),
        "mean_iou": sum(row["iou"] for row in rows) / len(rows),
        "model_checkpoint": str(MODEL_PATH),
        "image_size": image_size,
        "mask_threshold": threshold,
        "metric_aggregation": "arithmetic mean of per-image scores",
        "metric_smoothing": 1.0,
        "device": str(device),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["image_id", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    with JSON_PATH.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Relatórios salvos em {CSV_PATH} e {JSON_PATH}")


if __name__ == "__main__":
    main()
