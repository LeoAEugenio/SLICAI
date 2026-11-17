from datetime import datetime

from sqlalchemy import DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
    )

    original_file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    classification_result: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    malignant_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    classification_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    segmentation_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    uploaded_image_url: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    mask_url: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    analysis_figure_url: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    classifier_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    segmenter_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )