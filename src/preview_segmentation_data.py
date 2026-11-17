from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "isic2016_segmentation_splits.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "segmentation_data_samples.png"
)


def main():
    dataframe = pd.read_csv(SPLITS_PATH)

    splits = ["train", "validation", "test"]

    figure, axes = plt.subplots(3, 2, figsize=(10, 12))

    for row_index, split_name in enumerate(splits):
        sample = dataframe[dataframe["split"] == split_name].sample(
            n=1,
            random_state=42 + row_index,
        ).iloc[0]

        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")

        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(
            f"Imagem original — {split_name}\n{sample['image_id']}"
        )
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(mask, cmap="gray")
        axes[row_index, 1].set_title(f"Máscara da lesão — {split_name}")
        axes[row_index, 1].axis("off")

    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print("Figura de conferência criada em:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()