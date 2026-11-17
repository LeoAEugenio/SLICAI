from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ---------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ham10000_splits.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifier_efficientnet_b0.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

# True: executa um teste rápido com parte dos dados.
# False: utiliza todo o conjunto de treinamento.
SMOKE_TEST = False

SMOKE_TRAIN_SIZE = 600
SMOKE_VALIDATION_SIZE = 200

RANDOM_SEED = 42


# ---------------------------------------------------------
# REPRODUTIBILIDADE
# ---------------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(str(row["image_path"]))

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        target = int(row["target"])

        return image, target


# ---------------------------------------------------------
# TRANSFORMAÇÕES
# ---------------------------------------------------------

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.80, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std,
        ),
    ]
)

validation_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std,
        ),
    ]
)


# ---------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------

def validate_columns(dataframe):
    required_columns = {
        "image_path",
        "target",
        "split",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Colunas ausentes no CSV: "
            + ", ".join(sorted(missing_columns))
        )


def run_training_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    predictions = []
    targets = []

    progress_bar = tqdm(
        data_loader,
        desc="Treinamento",
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predicted_classes = outputs.argmax(dim=1)

        predictions.extend(
            predicted_classes.detach().cpu().numpy()
        )
        targets.extend(labels.detach().cpu().numpy())

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(targets, predictions)

    return average_loss, accuracy


def run_validation(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []
    malignant_probabilities = []

    with torch.no_grad():
        progress_bar = tqdm(
            data_loader,
            desc="Validação",
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)

            predictions.extend(
                predicted_classes.cpu().numpy()
            )
            targets.extend(labels.cpu().numpy())
            malignant_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    average_loss = total_loss / len(data_loader.dataset)

    accuracy = accuracy_score(targets, predictions)

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

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        targets,
        malignant_probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics


# ---------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("SLICAI - Treinamento do classificador")
    print("=" * 60)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {CSV_PATH}"
        )

    dataframe = pd.read_csv(CSV_PATH)

    validate_columns(dataframe)

    train_dataframe = dataframe[
        dataframe["split"] == "train"
    ].copy()

    validation_dataframe = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    if SMOKE_TEST:
        train_dataframe = train_dataframe.sample(
            n=min(
                SMOKE_TRAIN_SIZE,
                len(train_dataframe),
            ),
            random_state=RANDOM_SEED,
        )

        validation_dataframe = validation_dataframe.sample(
            n=min(
                SMOKE_VALIDATION_SIZE,
                len(validation_dataframe),
            ),
            random_state=RANDOM_SEED,
        )

        print(
            "\nMODO DE TESTE RÁPIDO ATIVADO"
        )

    print(
        f"\nImagens de treinamento: "
        f"{len(train_dataframe)}"
    )

    print(
        f"Imagens de validação: "
        f"{len(validation_dataframe)}"
    )

    print("\nDistribuição do treinamento:")
    print(
        train_dataframe["target_name"]
        .value_counts()
    )

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
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDispositivo utilizado: {device}")

    print(
        "\nCarregando EfficientNet-B0 "
        "pré-treinada..."
    )

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )

    # No primeiro teste, somente a última camada será treinada.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        2,
    )

    model = model.to(device)

    class_counts = (
        train_dataframe["target"]
        .value_counts()
        .sort_index()
    )

    if len(class_counts) != 2:
        raise ValueError(
            "O conjunto de treinamento precisa "
            "conter as classes 0 e 1."
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

    for epoch in range(EPOCHS):
        print(
            f"\nÉpoca {epoch + 1}/{EPOCHS}"
        )

        train_loss, train_accuracy = (
            run_training_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        validation_metrics = run_validation(
            model,
            validation_loader,
            criterion,
            device,
        )

        print("\nResultados do treinamento:")
        print(f"Loss: {train_loss:.4f}")
        print(
            f"Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print("\nResultados da validação:")

        for metric_name, metric_value in (
            validation_metrics.items()
        ):
            if isinstance(metric_value, float):
                print(
                    f"{metric_name}: "
                    f"{metric_value:.4f}"
                )
            else:
                print(
                    f"{metric_name}: "
                    f"{metric_value}"
                )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "architecture": "efficientnet_b0",
            "model_state_dict": model.state_dict(),
            "class_names": {
                0: "benign",
                1: "malignant",
            },
            "image_size": IMAGE_SIZE,
            "imagenet_mean": imagenet_mean,
            "imagenet_std": imagenet_std,
            "smoke_test": SMOKE_TEST,
            "validation_metrics": validation_metrics,
        },
        MODEL_PATH,
    )

    print(
        f"\nModelo salvo em:\n{MODEL_PATH}"
    )

    print(
        "\nTeste concluído com sucesso."
    )


if __name__ == "__main__":
    main()