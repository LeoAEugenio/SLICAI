import json
import sys
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_single_image import main as run_analysis
from api.repository import list_analyses, save_analysis


UPLOADS_DIR = PROJECT_ROOT / "storage" / "uploads"
ANALYSES_DIR = PROJECT_ROOT / "storage" / "analyses"
WEB_DIR = PROJECT_ROOT / "web"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="SLICAI API",
    description=(
        "Experimental API for preliminary classification and "
        "segmentation of dermoscopic skin lesion images."
    ),
    version="1.1.0",
)


app.mount(
    "/uploads",
    StaticFiles(directory=UPLOADS_DIR),
    name="uploads",
)

app.mount(
    "/analysis-files",
    StaticFiles(directory=ANALYSES_DIR),
    name="analysis-files",
)


def validate_image(content):
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from error


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "database": "PostgreSQL",
    }


@app.get("/api/analyses")
def get_analysis_history(
    limit: int = Query(default=50, ge=1, le=100),
):
    try:
        analyses = list_analyses(limit=limit)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The analysis history could not be retrieved.",
        ) from error

    return {
        "total": len(analyses),
        "analyses": analyses,
    }


@app.post("/api/analyze")
async def analyze_image(image: UploadFile = File(...)):
    original_name = image.filename or "image.jpg"
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, or PNG image.",
        )

    content = await image.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="The image must not exceed 10 MB.",
        )

    validate_image(content)

    analysis_id = uuid.uuid4().hex
    uploaded_image_name = f"{analysis_id}{extension}"
    uploaded_image_path = UPLOADS_DIR / uploaded_image_name

    uploaded_image_path.write_bytes(content)

    try:
        run_analysis(str(uploaded_image_path))

        report_path = (
            ANALYSES_DIR
            / f"{analysis_id}_analysis.json"
        )

        with open(report_path, "r", encoding="utf-8") as file:
            report = json.load(file)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The image could not be analyzed. "
                "Check the server terminal for details."
            ),
        ) from error

    uploaded_image_url = f"/uploads/{uploaded_image_name}"
    mask_url = (
        f"/analysis-files/{analysis_id}_mask.png"
    )
    analysis_figure_url = (
        f"/analysis-files/{analysis_id}_analysis.png"
    )

    try:
        save_analysis(
            analysis_id=analysis_id,
            original_file_name=original_name,
            classification_result=report[
                "classification_result"
            ],
            malignant_probability=report[
                "malignant_probability"
            ],
            classification_threshold=report[
                "classification_threshold"
            ],
            segmentation_threshold=report[
                "segmentation_threshold"
            ],
            uploaded_image_url=uploaded_image_url,
            mask_url=mask_url,
            analysis_figure_url=analysis_figure_url,
            classifier_model=(
                "classifier_efficientnet_b0_selected.pth"
            ),
            segmenter_model=(
                "segmenter_unet_isic2016.pth"
            ),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The analysis was generated, but it could not "
                "be saved to the database."
            ),
        ) from error

    return {
        "analysis_id": analysis_id,
        "original_file_name": original_name,
        "classification_result": report[
            "classification_result"
        ],
        "malignant_probability": report[
            "malignant_probability"
        ],
        "classification_threshold": report[
            "classification_threshold"
        ],
        "segmentation_threshold": report[
            "segmentation_threshold"
        ],
        "uploaded_image_url": uploaded_image_url,
        "mask_url": mask_url,
        "analysis_figure_url": analysis_figure_url,
        "medical_notice": report["medical_notice"],
    }
