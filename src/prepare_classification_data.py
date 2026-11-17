from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(r"C:\SLICAI")
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

RANDOM_STATE = 42

PROCESSED.mkdir(parents=True, exist_ok=True)


def prepare_ham10000():
    metadata_path = RAW / "ham10000" / "metadata.csv"
    dataframe = pd.read_csv(metadata_path, low_memory=False)

    dataframe["original_label"] = (
        dataframe["diagnosis_1"]
        .fillna("Missing")
        .astype(str)
        .str.strip()
    )

    valid_labels = ["Benign", "Malignant"]

    eligible = dataframe[
        dataframe["original_label"].isin(valid_labels)
    ].copy()

    excluded = dataframe[
        ~dataframe["original_label"].isin(valid_labels)
    ].copy()

    eligible["target"] = eligible["original_label"].map({
        "Benign": 0,
        "Malignant": 1,
    })

    eligible["target_name"] = eligible["original_label"].str.lower()

    if eligible["lesion_id"].isna().any():
        raise ValueError("Existem registros do HAM10000 sem lesion_id.")

    conflicting_lesions = (
        eligible.groupby("lesion_id")["target"]
        .nunique()
    )

    conflicting_lesions = conflicting_lesions[
        conflicting_lesions > 1
    ]

    if not conflicting_lesions.empty:
        raise ValueError(
            "Existem lesões relacionadas a mais de uma classe."
        )

    lesions = (
        eligible[["lesion_id", "target"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    train_lesions, temporary_lesions = train_test_split(
        lesions,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=lesions["target"],
    )

    validation_lesions, test_lesions = train_test_split(
        temporary_lesions,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary_lesions["target"],
    )

    split_mapping = {}

    split_mapping.update({
        lesion_id: "train"
        for lesion_id in train_lesions["lesion_id"]
    })

    split_mapping.update({
        lesion_id: "validation"
        for lesion_id in validation_lesions["lesion_id"]
    })

    split_mapping.update({
        lesion_id: "internal_test"
        for lesion_id in test_lesions["lesion_id"]
    })

    eligible["split"] = eligible["lesion_id"].map(split_mapping)

    eligible["image_path"] = eligible["isic_id"].apply(
        lambda image_id:
        f"data/raw/ham10000/images/{image_id}.jpg"
    )

    eligible.to_csv(
        PROCESSED / "ham10000_splits.csv",
        index=False,
    )

    excluded["exclusion_reason"] = excluded["original_label"]

    excluded.to_csv(
        PROCESSED / "ham10000_excluded.csv",
        index=False,
    )

    print("\nHAM10000")
    print("-" * 60)

    summary = (
        eligible.groupby(["split", "target_name"])
        .size()
        .unstack(fill_value=0)
    )

    summary["total"] = summary.sum(axis=1)
    print(summary)

    train_ids = set(train_lesions["lesion_id"])
    validation_ids = set(validation_lesions["lesion_id"])
    test_ids = set(test_lesions["lesion_id"])

    overlap = (
        train_ids.intersection(validation_ids)
        | train_ids.intersection(test_ids)
        | validation_ids.intersection(test_ids)
    )

    print(f"\nLesões presentes em mais de uma partição: {len(overlap)}")
    print(f"Registros elegíveis: {len(eligible)}")
    print(f"Registros excluídos: {len(excluded)}")


def prepare_bcn20000():
    metadata_path = RAW / "bcn20000" / "metadata.csv"
    dataframe = pd.read_csv(metadata_path, low_memory=False)

    dataframe["original_label"] = (
        dataframe["diagnosis_1"]
        .fillna("Missing")
        .astype(str)
        .str.strip()
    )

    valid_labels = ["Benign", "Malignant"]

    eligible = dataframe[
        dataframe["original_label"].isin(valid_labels)
    ].copy()

    excluded = dataframe[
        ~dataframe["original_label"].isin(valid_labels)
    ].copy()

    eligible["target"] = eligible["original_label"].map({
        "Benign": 0,
        "Malignant": 1,
    })

    eligible["target_name"] = eligible["original_label"].str.lower()
    eligible["split"] = "external_test"

    eligible["image_path"] = eligible["isic_id"].apply(
        lambda image_id:
        f"data/raw/bcn20000/images/{image_id}.jpg"
    )

    eligible.to_csv(
        PROCESSED / "bcn20000_external_test.csv",
        index=False,
    )

    excluded["exclusion_reason"] = excluded["original_label"]

    excluded.to_csv(
        PROCESSED / "bcn20000_excluded.csv",
        index=False,
    )

    print("\nBCN20000")
    print("-" * 60)
    print(eligible["target_name"].value_counts())
    print(f"\nRegistros elegíveis: {len(eligible)}")
    print(f"Registros excluídos: {len(excluded)}")


prepare_ham10000()
prepare_bcn20000()

print("\nArquivos criados em:")
print(PROCESSED)