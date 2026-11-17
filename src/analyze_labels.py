from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\SLICAI\data\raw")

DATASETS = {
    "HAM10000": ROOT / "ham10000" / "metadata.csv",
    "BCN20000": ROOT / "bcn20000" / "metadata.csv",
}

LABEL_COLUMNS = [
    "diagnosis_1",
    "diagnosis_2",
    "diagnosis_3",
    "melanoma",
    "melanocytic",
    "diagnosis_confirm_type",
]


def analyze_dataset(dataset_name, metadata_path):
    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    dataframe = pd.read_csv(metadata_path, low_memory=False)

    print(f"Quantidade de registros: {len(dataframe)}")
    print(f"Quantidade de colunas: {len(dataframe.columns)}")

    if "isic_id" in dataframe.columns:
        duplicated_images = dataframe["isic_id"].duplicated().sum()
        print(f"Identificadores de imagem duplicados: {duplicated_images}")

    if "lesion_id" in dataframe.columns:
        lesion_ids = dataframe["lesion_id"].dropna()

        print(f"Registros com lesion_id: {len(lesion_ids)}")
        print(f"Registros sem lesion_id: {dataframe['lesion_id'].isna().sum()}")
        print(f"Lesões únicas: {lesion_ids.nunique()}")

        repeated_lesions = lesion_ids.duplicated(keep=False).sum()
        print(f"Imagens pertencentes a lesões repetidas: {repeated_lesions}")
    else:
        print("A coluna lesion_id não foi encontrada.")

    for column in LABEL_COLUMNS:
        if column in dataframe.columns:
            print(f"\nDistribuição de {column}:")
            print(
                dataframe[column]
                .fillna("Missing")
                .value_counts(dropna=False)
                .to_string()
            )


for name, path in DATASETS.items():
    if path.exists():
        analyze_dataset(name, path)
    else:
        print(f"CSV não encontrado: {path}")