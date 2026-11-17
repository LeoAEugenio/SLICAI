from sqlalchemy import select

from api.database import SessionLocal
from api.models import Analysis


def save_analysis(
    analysis_id,
    original_file_name,
    classification_result,
    malignant_probability,
    classification_threshold,
    segmentation_threshold,
    uploaded_image_url,
    mask_url,
    analysis_figure_url,
    classifier_model,
    segmenter_model,
):
    with SessionLocal() as session:
        analysis = Analysis(
            id=analysis_id,
            original_file_name=original_file_name,
            classification_result=classification_result,
            malignant_probability=malignant_probability,
            classification_threshold=classification_threshold,
            segmentation_threshold=segmentation_threshold,
            uploaded_image_url=uploaded_image_url,
            mask_url=mask_url,
            analysis_figure_url=analysis_figure_url,
            classifier_model=classifier_model,
            segmenter_model=segmenter_model,
        )

        session.add(analysis)
        session.commit()


def list_analyses(limit=50):
    with SessionLocal() as session:
        statement = (
            select(Analysis)
            .order_by(Analysis.created_at.desc())
            .limit(limit)
        )

        analyses = session.scalars(statement).all()

        return [
            {
                "analysis_id": analysis.id,
                "original_file_name": analysis.original_file_name,
                "classification_result": (
                    analysis.classification_result
                ),
                "malignant_probability": (
                    analysis.malignant_probability
                ),
                "classification_threshold": (
                    analysis.classification_threshold
                ),
                "segmentation_threshold": (
                    analysis.segmentation_threshold
                ),
                "uploaded_image_url": analysis.uploaded_image_url,
                "mask_url": analysis.mask_url,
                "analysis_figure_url": (
                    analysis.analysis_figure_url
                ),
                "classifier_model": analysis.classifier_model,
                "segmenter_model": analysis.segmenter_model,
                "created_at": analysis.created_at.isoformat(),
            }
            for analysis in analyses
        ]