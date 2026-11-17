import json

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from train_classifier import (
    PROJECT_ROOT,
    SkinLesionDataset,
    validation_transform,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifier_efficientnet_b0_selected.pth"
)

EXTERNAL_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bcn20000_external_test.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

BATCH_SIZE = 16


def main():
    print("=" * 60)
    print("SLICAI - Avaliação externa no BCN20000")
    print("=" * 60)

    external_dataframe = pd.read_csv(EXTERNAL_CSV_PATH)

    required_columns = {
        "image_path",
        "target",
        "target_name",
    }

    missing_columns = (
        required_columns
        - set(external_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colunas ausentes: "
            + ", ".join(sorted(missing_columns))
        )

    print(
        f"\nImagens na avaliação externa: "
        f"{len(external_dataframe)}"
    )

    print("\nDistribuição das classes:")
    print(
        external_dataframe["target_name"].value_counts()
    )

    dataset = SkinLesionDataset(
        external_dataframe,
        transform=validation_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    threshold = checkpoint["decision_threshold"]

    model = efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    targets = []
    malignant_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(
            loader,
            desc="Avaliação externa",
        ):
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            targets.extend(labels.numpy())

            malignant_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    targets = np.array(targets)

    malignant_probabilities = np.array(
        malignant_probabilities
    )

    predictions = (
        malignant_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0,
    )

    sensitivity = recall_score(
        targets,
        predictions,
        zero_division=0,
    )

    specificity = tn / (tn + fp)

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        targets,
        malignant_probabilities,
    )

    metrics = {
        "dataset": "BCN20000 external test",
        "total_images": int(len(external_dataframe)),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    print(f"\nLimiar utilizado: {threshold:.4f}")

    print("\nResultados da avaliação externa:")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_report = external_dataframe.copy()

    prediction_report["probability_malignant"] = (
        malignant_probabilities
    )

    prediction_report["predicted_target"] = predictions

    prediction_report["predicted_class"] = np.where(
        predictions == 1,
        "malignant",
        "benign",
    )

    prediction_report["classification_result"] = np.select(
        [
            (targets == 0) & (predictions == 0),
            (targets == 0) & (predictions == 1),
            (targets == 1) & (predictions == 0),
            (targets == 1) & (predictions == 1),
        ],
        [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ],
        default="unclassified",
    )

    metrics_path = (
        REPORTS_DIR
        / "external_bcn20000_metrics.json"
    )

    predictions_path = (
        REPORTS_DIR
        / "external_bcn20000_predictions.csv"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    prediction_report.to_csv(
        predictions_path,
        index=False,
    )

    print("\nArquivos gerados:")
    print(metrics_path)
    print(predictions_path)


if __name__ == "__main__":
    main()