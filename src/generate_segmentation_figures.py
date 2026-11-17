import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as transform_functional

from train_segmenter import IMAGE_SIZE, PROJECT_ROOT, UNet


REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = PROJECT_ROOT / "models" / "segmenter_unet_isic2016.pth"

HISTORY_PATH = REPORTS_DIR / "segmentation_training_history.csv"
TEST_METRICS_PATH = REPORTS_DIR / "segmentation_test_metrics.json"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "segmentation_test_predictions.csv"


def prepare_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    image_tensor = transform_functional.to_tensor(image)

    image_tensor = transform_functional.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    return image, image_tensor.unsqueeze(0)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(HISTORY_PATH)
    test_predictions = pd.read_csv(TEST_PREDICTIONS_PATH)

    with open(TEST_METRICS_PATH, "r", encoding="utf-8") as file:
        test_metrics = json.load(file)

    # ------------------------------------------------------------
    # 1. Curvas de treinamento
    # ------------------------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(
        history["epoch"],
        history["training_loss"],
        label="Training",
        color="#2563eb",
    )
    axes[0].plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation",
        color="#dc2626",
    )
    axes[0].set_title("Loss during training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        history["epoch"],
        history["training_dice"],
        label="Training",
        color="#2563eb",
    )
    axes[1].plot(
        history["epoch"],
        history["validation_dice"],
        label="Validation",
        color="#dc2626",
    )
    axes[1].set_title("Dice during training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        history["epoch"],
        history["training_iou"],
        label="Training",
        color="#2563eb",
    )
    axes[2].plot(
        history["epoch"],
        history["validation_iou"],
        label="Validation",
        color="#dc2626",
    )
    axes[2].set_title("IoU during training")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "segmentation_training_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    # ------------------------------------------------------------
    # 2. Comparação validação × teste
    # ------------------------------------------------------------
    best_validation_row = history.loc[
        history["validation_dice"].idxmax()
    ]

    metric_names = ["Dice", "IoU"]

    validation_values = [
        best_validation_row["validation_dice"],
        best_validation_row["validation_iou"],
    ]

    test_values = [
        test_metrics["mean_dice"],
        test_metrics["mean_iou"],
    ]

    positions = np.arange(len(metric_names))
    width = 0.35

    figure, axis = plt.subplots(figsize=(8, 5))

    validation_bars = axis.bar(
        positions - width / 2,
        validation_values,
        width,
        label="Validation",
        color="#2563eb",
    )

    test_bars = axis.bar(
        positions + width / 2,
        test_values,
        width,
        label="Test",
        color="#16a34a",
    )

    axis.set_title("U-Net segmentation performance")
    axis.set_ylabel("Metric value")
    axis.set_ylim(0, 1)
    axis.set_xticks(positions)
    axis.set_xticklabels(metric_names)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    for bars in [validation_bars, test_bars]:
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.4f}",
                ha="center",
                va="bottom",
            )

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "segmentation_validation_test_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    # ------------------------------------------------------------
    # 3. Exemplos: melhor, mediano e mais difícil no teste
    # ------------------------------------------------------------
    ordered_predictions = test_predictions.sort_values(
        "dice"
    ).reset_index(drop=True)

    selected_examples = pd.concat(
        [
            ordered_predictions.iloc[[0]],
            ordered_predictions.iloc[[len(ordered_predictions) // 2]],
            ordered_predictions.iloc[[-1]],
        ],
        ignore_index=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    threshold = float(checkpoint.get("threshold", 0.5))

    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    row_names = [
        "Most difficult example",
        "Median example",
        "Best example",
    ]

    figure, axes = plt.subplots(3, 4, figsize=(16, 12))

    with torch.no_grad():
        for row_index, (_, example) in enumerate(
            selected_examples.iterrows()
        ):
            image, image_tensor = prepare_image(
                example["image_path"],
                image_size,
            )

            image_tensor = image_tensor.to(device)

            probabilities = torch.sigmoid(
                model(image_tensor)
            )

            predicted_mask = (
                probabilities >= threshold
            ).float().squeeze().cpu().numpy()

            ground_truth_mask = Image.open(
                example["mask_path"]
            ).convert("L")

            ground_truth_mask = ground_truth_mask.resize(
                (image_size, image_size),
                Image.Resampling.NEAREST,
            )

            ground_truth_mask = (
                np.array(ground_truth_mask) > 0
            ).astype(np.uint8)

            axes[row_index, 0].imshow(image)
            axes[row_index, 0].set_title(
                f"{row_names[row_index]}\nOriginal image"
            )
            axes[row_index, 0].axis("off")

            axes[row_index, 1].imshow(
                ground_truth_mask,
                cmap="gray",
            )
            axes[row_index, 1].set_title("Ground-truth mask")
            axes[row_index, 1].axis("off")

            axes[row_index, 2].imshow(
                predicted_mask,
                cmap="gray",
            )
            axes[row_index, 2].set_title(
                f"Predicted mask\nDice: {example['dice']:.4f}"
            )
            axes[row_index, 2].axis("off")

            axes[row_index, 3].imshow(image)
            axes[row_index, 3].contour(
                predicted_mask,
                levels=[0.5],
                colors="#00e5ff",
                linewidths=2,
            )
            axes[row_index, 3].set_title("Predicted contour")
            axes[row_index, 3].axis("off")

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "segmentation_test_examples.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("Figuras geradas com sucesso em:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
