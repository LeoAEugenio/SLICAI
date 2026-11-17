import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from train_classifier import (
    PROJECT_ROOT,
    CSV_PATH,
    SkinLesionDataset,
    validation_transform,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifier_efficientnet_b0_finetuned.pth"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifier_efficientnet_b0_selected.pth"
)

BATCH_SIZE = 16

# Sensibilidade mínima desejada na validação
TARGET_SENSITIVITY = 0.80


def main():
    print("=" * 60)
    print("SLICAI - Seleção do limiar de classificação")
    print("=" * 60)

    dataframe = pd.read_csv(CSV_PATH)

    validation_dataframe = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    dataset = SkinLesionDataset(
        validation_dataframe,
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
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            targets.extend(labels.numpy())

            malignant_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    targets = np.array(targets)
    malignant_probabilities = np.array(
        malignant_probabilities
    )

    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            targets,
            malignant_probabilities,
        )
    )

    specificity_values = 1 - false_positive_rate

    valid_indices = np.where(
        true_positive_rate >= TARGET_SENSITIVITY
    )[0]

    if len(valid_indices) == 0:
        raise ValueError(
            "Nenhum limiar alcançou a sensibilidade desejada."
        )

    best_index = valid_indices[
        np.argmax(specificity_values[valid_indices])
    ]

    selected_threshold = float(
        thresholds[best_index]
    )

    predictions = (
        malignant_probabilities >= selected_threshold
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

    auc = roc_auc_score(
        targets,
        malignant_probabilities,
    )

    print(f"\nLimiar selecionado: {selected_threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"True negative: {tn}")
    print(f"False positive: {fp}")
    print(f"False negative: {fn}")
    print(f"True positive: {tp}")

    checkpoint["decision_threshold"] = selected_threshold

    checkpoint["threshold_selection"] = {
        "dataset": "HAM10000 validation",
        "target_sensitivity": TARGET_SENSITIVITY,
        "achieved_sensitivity": sensitivity,
        "specificity": specificity,
    }

    torch.save(
        checkpoint,
        OUTPUT_PATH,
    )

    print(
        f"\nModelo com limiar salvo em:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()