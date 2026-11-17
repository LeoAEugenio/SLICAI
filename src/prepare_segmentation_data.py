from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "isic2016_segmentation"
    / "images"
)

MASKS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "isic2016_segmentation"
    / "masks"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "isic2016_segmentation_splits.csv"
)


def main():
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))

    records = []

    for image_path in image_paths:
        image_id = image_path.stem
        mask_path = MASKS_DIR / f"{image_id}_Segmentation.png"

        if not mask_path.exists():
            print(f"Máscara não encontrada para: {image_path.name}")
            continue

        records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
            }
        )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise ValueError("Nenhum par imagem/máscara foi encontrado.")

    train_data, test_data = train_test_split(
        dataframe,
        test_size=0.15,
        random_state=42,
        shuffle=True,
    )

    train_data, validation_data = train_test_split(
        train_data,
        test_size=0.1764705882,
        random_state=42,
        shuffle=True,
    )

    train_data = train_data.copy()
    validation_data = validation_data.copy()
    test_data = test_data.copy()

    train_data["split"] = "train"
    validation_data["split"] = "validation"
    test_data["split"] = "test"

    final_dataframe = pd.concat(
        [train_data, validation_data, test_data],
        ignore_index=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_dataframe.to_csv(OUTPUT_PATH, index=False)

    print("SLICAI — Preparação dos dados de segmentação")
    print("=" * 55)
    print(f"Pares imagem/máscara encontrados: {len(final_dataframe)}")
    print("\nDistribuição das partições:")
    print(final_dataframe["split"].value_counts())
    print(f"\nArquivo criado em:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()