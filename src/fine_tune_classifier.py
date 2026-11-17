import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from train_classifier import (
    PROJECT_ROOT,
    CSV_PATH,
    SkinLesionDataset,
    train_transform,
    validation_transform,
    run_training_epoch,
    run_validation,
)


INITIAL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifier_efficientnet_b0.pth"
)

OUTPUT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifier_efficientnet_b0_finetuned.pth"
)

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.0001


def main():
    print("=" * 60)
    print("SLICAI - Fine-tuning do classificador")
    print("=" * 60)

    if not INITIAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo inicial não encontrado: {INITIAL_MODEL_PATH}"
        )

    dataframe = pd.read_csv(CSV_PATH)

    train_dataframe = dataframe[
        dataframe["split"] == "train"
    ].copy()

    validation_dataframe = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    print(f"\nTreinamento: {len(train_dataframe)} imagens")
    print(f"Validação: {len(validation_dataframe)} imagens")

    train_dataset = SkinLesionDataset(
        train_dataframe,
        transform=train_transform,
    )

    validation_dataset = SkinLesionDataset(
        validation_dataframe,
        transform=validation_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo utilizado: {device}")

    # Criação da mesma arquitetura utilizada anteriormente
    model = efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    checkpoint = torch.load(
        INITIAL_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Primeiro, todas as camadas são congeladas
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Destrava somente os dois últimos blocos da EfficientNet
    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True

    # Mantém o classificador destravado
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model = model.to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Parâmetros treináveis: "
        f"{trainable_parameters:,}"
    )

    class_counts = (
        train_dataframe["target"]
        .value_counts()
        .sort_index()
    )

    class_weights = (
        len(train_dataframe)
        / (2 * class_counts.values)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print(
        "Pesos das classes:",
        class_weights.cpu().numpy(),
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    best_auc = -1.0

    for epoch in range(EPOCHS):
        print(f"\nÉpoca {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = run_training_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_metrics = run_validation(
            model,
            validation_loader,
            criterion,
            device,
        )

        print("\nResultados do treinamento:")
        print(f"Loss: {train_loss:.4f}")
        print(f"Accuracy: {train_accuracy:.4f}")

        print("\nResultados da validação:")

        for metric_name, metric_value in validation_metrics.items():
            if isinstance(metric_value, float):
                print(
                    f"{metric_name}: {metric_value:.4f}"
                )
            else:
                print(
                    f"{metric_name}: {metric_value}"
                )

        current_auc = validation_metrics["roc_auc"]

        if current_auc > best_auc:
            best_auc = current_auc

            torch.save(
                {
                    "architecture": "efficientnet_b0",
                    "model_state_dict": model.state_dict(),
                    "class_names": {
                        0: "benign",
                        1: "malignant",
                    },
                    "image_size": 224,
                    "stage": "fine_tuning",
                    "best_validation_auc": best_auc,
                    "validation_metrics": validation_metrics,
                },
                OUTPUT_MODEL_PATH,
            )

            print(
                "\nNovo melhor modelo salvo."
            )

    print("\nFine-tuning concluído.")
    print(f"Melhor AUC de validação: {best_auc:.4f}")
    print(f"Modelo salvo em:\n{OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()