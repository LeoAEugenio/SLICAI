import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

from train_classifier import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

INTERNAL_METRICS_PATH = REPORTS_DIR / "internal_test_metrics.json"
EXTERNAL_METRICS_PATH = REPORTS_DIR / "external_bcn20000_metrics.json"

INTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "internal_test_predictions.csv"
EXTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "external_bcn20000_predictions.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_probability_column(dataframe):
    possible_columns = [
        "predicted_probability",
        "malignant_probability",
        "probability_malignant",
        "probability",
    ]

    for column in possible_columns:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "Não foi encontrada uma coluna de probabilidade no CSV. "
        f"Colunas disponíveis: {list(dataframe.columns)}"
    )


def save_confusion_matrix(dataframe, output_path, title):
    y_true = dataframe["target"].astype(int)
    y_pred = dataframe["predicted_target"].astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Benign", "Malignant"],
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    internal_metrics = load_json(INTERNAL_METRICS_PATH)
    external_metrics = load_json(EXTERNAL_METRICS_PATH)

    # pandas lê corretamente campos de texto que possuem vírgulas,
    # como “Melanoma, NOS”.
    internal_data = pd.read_csv(INTERNAL_PREDICTIONS_PATH)
    external_data = pd.read_csv(EXTERNAL_PREDICTIONS_PATH)

    print("Colunas do teste interno:")
    print(list(internal_data.columns))

    print("\nColunas da avaliação externa:")
    print(list(external_data.columns))

    # ------------------------------------------------------------
    # 1. Comparação de métricas entre teste interno e externo
    # ------------------------------------------------------------
    metric_keys = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
    ]

    metric_labels = [
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1-score",
        "ROC-AUC",
    ]

    internal_values = [internal_metrics[key] for key in metric_keys]
    external_values = [external_metrics[key] for key in metric_keys]

    positions = np.arange(len(metric_labels))
    width = 0.38

    figure, axis = plt.subplots(figsize=(12, 6))

    internal_bars = axis.bar(
        positions - width / 2,
        internal_values,
        width,
        label="HAM10000 — internal test",
        color="#2563eb",
    )

    external_bars = axis.bar(
        positions + width / 2,
        external_values,
        width,
        label="BCN20000 — external test",
        color="#dc2626",
    )

    axis.set_title("Classifier performance comparison")
    axis.set_ylabel("Metric value")
    axis.set_ylim(0, 1)
    axis.set_xticks(positions)
    axis.set_xticklabels(metric_labels)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    for bars in [internal_bars, external_bars]:
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "classification_performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    # ------------------------------------------------------------
    # 2. Matrizes de confusão
    # ------------------------------------------------------------
    save_confusion_matrix(
        internal_data,
        FIGURES_DIR / "internal_test_confusion_matrix.png",
        "Confusion Matrix — HAM10000 (Internal Test)",
    )

    save_confusion_matrix(
        external_data,
        FIGURES_DIR / "external_bcn20000_confusion_matrix.png",
        "Confusion Matrix — BCN20000 (External Test)",
    )

    # ------------------------------------------------------------
    # 3. Curvas ROC
    # ------------------------------------------------------------
    internal_probability_column = get_probability_column(internal_data)
    external_probability_column = get_probability_column(external_data)

    internal_fpr, internal_tpr, _ = roc_curve(
        internal_data["target"].astype(int),
        internal_data[internal_probability_column],
    )

    external_fpr, external_tpr, _ = roc_curve(
        external_data["target"].astype(int),
        external_data[external_probability_column],
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        internal_fpr,
        internal_tpr,
        color="#2563eb",
        linewidth=2,
        label=f"HAM10000 — AUC = {internal_metrics['roc_auc']:.4f}",
    )

    axis.plot(
        external_fpr,
        external_tpr,
        color="#dc2626",
        linewidth=2,
        label=f"BCN20000 — AUC = {external_metrics['roc_auc']:.4f}",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier",
    )

    axis.set_title("Classifier ROC curves")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "roc_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nFiguras geradas com sucesso em:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
