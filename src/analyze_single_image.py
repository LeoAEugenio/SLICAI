import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage
from torchvision import models
from torchvision.transforms import functional as transform_functional
from functools import lru_cache
from train_segmenter import PROJECT_ROOT, UNet


CLASSIFIER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifier_efficientnet_b0_selected.pth"
)

SEGMENTER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "segmenter_unet_isic2016.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "storage" / "analyses"


def get_model_state(checkpoint):
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    return checkpoint

@lru_cache(maxsize=1)
def load_classifier(device):
    checkpoint = torch.load(
        CLASSIFIER_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = models.efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    model.load_state_dict(get_model_state(checkpoint))
    model.to(device)
    model.eval()

    threshold = float(
        checkpoint.get(
            "decision_threshold",
            checkpoint.get("threshold", 0.5),
        )
    )

    return model, threshold

@lru_cache(maxsize=1)
def load_segmenter(device):
    checkpoint = torch.load(
        SEGMENTER_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = UNet()
    model.load_state_dict(get_model_state(checkpoint))
    model.to(device)
    model.eval()

    image_size = int(checkpoint.get("image_size", 256))
    threshold = float(checkpoint.get("threshold", 0.5))

    return model, image_size, threshold


def prepare_classifier_image(image):
    image = transform_functional.resize(image, 256)
    image = transform_functional.center_crop(image, 224)

    tensor = transform_functional.to_tensor(image)

    tensor = transform_functional.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    return tensor.unsqueeze(0)


def prepare_segmenter_image(image, image_size):
    image = image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    tensor = transform_functional.to_tensor(image)

    tensor = transform_functional.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    return image, tensor.unsqueeze(0)

def keep_largest_component(mask):
    labels, number_of_components = ndimage.label(mask)

    if number_of_components == 0:
        return mask

    component_sizes = ndimage.sum(
        mask,
        labels,
        range(1, number_of_components + 1),
    )

    largest_component_label = (
        int(np.argmax(component_sizes)) + 1
    )

    return (labels == largest_component_label).astype(np.uint8)

def main(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    classifier, classifier_threshold = load_classifier(device)

    (
        segmenter,
        segmentation_image_size,
        segmentation_threshold,
    ) = load_segmenter(device)

    original_image = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        classifier_input = prepare_classifier_image(
            original_image
        ).to(device)

        classifier_output = classifier(classifier_input)

        probabilities = torch.softmax(
            classifier_output,
            dim=1,
        )

        malignant_probability = float(
            probabilities[0, 1].item()
        )

        predicted_class = (
            "malignant"
            if malignant_probability >= classifier_threshold
            else "benign"
        )

        segmentation_image, segmentation_input = (
            prepare_segmenter_image(
                original_image,
                segmentation_image_size,
            )
        )

        segmentation_input = segmentation_input.to(device)

        segmentation_output = segmenter(segmentation_input)

        segmentation_probability = torch.sigmoid(
            segmentation_output
        )

        predicted_mask = (
            segmentation_probability >= segmentation_threshold
        ).float()

        predicted_mask = (
            predicted_mask.squeeze().cpu().numpy().astype(np.uint8)
        )

        predicted_mask = keep_largest_component(predicted_mask)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_name = image_path.stem

    mask_output_path = OUTPUT_DIR / f"{image_name}_mask.png"
    figure_output_path = OUTPUT_DIR / f"{image_name}_analysis.png"
    report_output_path = OUTPUT_DIR / f"{image_name}_analysis.json"

    Image.fromarray(predicted_mask * 255).save(mask_output_path)

    if predicted_class == "malignant":
        classification_text = (
            "Preliminary classification: possible malignant pattern"
        )
    else:
        classification_text = (
            "Preliminary classification: possible benign pattern"
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(segmentation_image)
    axes[0].set_title("Uploaded image")
    axes[0].axis("off")

    axes[1].imshow(predicted_mask, cmap="gray")
    axes[1].set_title("Predicted mask")
    axes[1].axis("off")

    axes[2].imshow(segmentation_image)
    axes[2].contour(
        predicted_mask,
        levels=[0.5],
        colors="#00e5ff",
        linewidths=2,
    )
    axes[2].set_title("Predicted contour")
    axes[2].axis("off")

    figure.suptitle(
        f"{classification_text}\n"
        f"Malignancy probability: "
        f"{malignant_probability:.2%} | "
        f"Threshold: {classifier_threshold:.4f}",
        fontsize=12,
    )

    figure.tight_layout()
    figure.savefig(
        figure_output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    report = {
        "input_image": str(image_path.resolve()),
        "classification_result": predicted_class,
        "malignant_probability": malignant_probability,
        "classification_threshold": classifier_threshold,
        "segmentation_threshold": segmentation_threshold,
        "predicted_mask": str(mask_output_path.resolve()),
        "analysis_figure": str(figure_output_path.resolve()),
        "medical_notice": (
            "Experimental decision-support result; "
            "it does not constitute a medical diagnosis."
        ),
    }

    with open(report_output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("=" * 60)
    print("SLICAI — Single-image analysis completed")
    print("=" * 60)
    print(classification_text)
    print(
        "Malignancy probability: "
        f"{malignant_probability:.2%}"
    )
    print(f"\nMask saved to:\n{mask_output_path}")
    print(f"\nFigure saved to:\n{figure_output_path}")
    print(f"\nReport saved to:\n{report_output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python src\\analyze_single_image.py "
            "\"path_to_image.jpg\""
        )
        sys.exit(1)

    main(sys.argv[1])
