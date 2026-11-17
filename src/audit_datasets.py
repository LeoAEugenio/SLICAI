from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(r"C:\SLICAI\data\raw")

DATASETS = {
    "HAM10000": {
        "images": ROOT / "ham10000" / "images",
        "metadata": ROOT / "ham10000" / "metadata.csv",
    },
    "ISIC 2016 - Images": {
        "images": ROOT / "isic2016_segmentation" / "images",
    },
    "ISIC 2016 - Masks": {
        "images": ROOT / "isic2016_segmentation" / "masks",
    },
    "BCN20000": {
        "images": ROOT / "bcn20000" / "images",
        "metadata": ROOT / "bcn20000" / "metadata.csv",
    },
}


def find_images(folder):
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        file
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    )


def verify_metadata(metadata_path, image_files):
    if not metadata_path.exists():
        print(f"  CSV não encontrado: {metadata_path}")
        return

    dataframe = pd.read_csv(metadata_path, low_memory=False)

    print(f"  Registros no CSV: {len(dataframe)}")
    print(f"  Colunas: {list(dataframe.columns)}")

    possible_id_columns = ["image_id", "isic_id", "image", "name"]
    id_column = next(
        (column for column in possible_id_columns if column in dataframe.columns),
        None,
    )

    if id_column is None:
        print("  Não foi possível identificar automaticamente a coluna da imagem.")
        return

    metadata_ids = (
        dataframe[id_column]
        .dropna()
        .astype(str)
        .map(lambda value: Path(value).stem)
    )

    file_ids = {file.stem for file in image_files}
    missing_images = [image_id for image_id in metadata_ids if image_id not in file_ids]

    print(f"  Coluna identificadora: {id_column}")
    print(f"  Registros sem imagem correspondente: {len(missing_images)}")

    if missing_images:
        print(f"  Exemplos ausentes: {missing_images[:5]}")


def verify_segmentation_pairs():
    image_folder = ROOT / "isic2016_segmentation" / "images"
    mask_folder = ROOT / "isic2016_segmentation" / "masks"

    images = find_images(image_folder)
    masks = find_images(mask_folder)

    image_ids = {image.stem for image in images}
    mask_ids = {
        mask.stem.replace("_Segmentation", "").replace("_segmentation", "")
        for mask in masks
    }

    images_without_mask = image_ids - mask_ids
    masks_without_image = mask_ids - image_ids

    print("\nISIC 2016 - Correspondência entre imagens e máscaras")
    print(f"  Imagens sem máscara: {len(images_without_mask)}")
    print(f"  Máscaras sem imagem: {len(masks_without_image)}")


for dataset_name, paths in DATASETS.items():
    print(f"\n{dataset_name}")
    print("-" * len(dataset_name))

    image_folder = paths["images"]

    if not image_folder.exists():
        print(f"  Pasta não encontrada: {image_folder}")
        continue

    images = find_images(image_folder)
    print(f"  Quantidade de imagens: {len(images)}")

    if images:
        sample = images[0]

        try:
            with Image.open(sample) as image:
                print(f"  Exemplo: {sample.name}")
                print(f"  Dimensões: {image.size}")
                print(f"  Formato: {image.format}")
        except Exception as error:
            print(f"  Não foi possível abrir a imagem: {error}")

    if "metadata" in paths:
        verify_metadata(paths["metadata"], images)

verify_segmentation_pairs()