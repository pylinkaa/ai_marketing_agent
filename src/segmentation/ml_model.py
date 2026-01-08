"""ML-based segmentation model (supervised)."""

import logging
from typing import Tuple, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def train_segmentation_model(
    features_df: pd.DataFrame,
    labels: pd.Series,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, LabelEncoder]:
    """Train supervised ML model to predict user segment labels.

    The target is expected to be a categorical label representing the
    user's segment (e.g. true_segment_label or rule-based segment).
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels.astype(str).fillna("UNKNOWN"))

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(features_df, y)

    logger.info(
        "Trained ML segmentation model on %d users with %d classes",
        features_df.shape[0],
        len(encoder.classes_),
    )
    return model, encoder


def predict_segments(
    model: RandomForestClassifier,
    encoder: LabelEncoder,
    features_df: pd.DataFrame,
) -> pd.Series:
    """Predict segment labels for users as strings."""
    y_pred = model.predict(features_df)
    labels = encoder.inverse_transform(y_pred)
    return pd.Series(labels, index=features_df.index, name="segment_label")


