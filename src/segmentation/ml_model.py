"""ML-based segmentation models (supervised and unsupervised)."""

import logging
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def train_segmentation_model(
    features_df: pd.DataFrame,
    labels: pd.Series,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[RandomForestClassifier, LabelEncoder, Dict[str, Any]]:
    """Train supervised ML model with train/test split and evaluation metrics.
    
    Args:
        features_df: Feature matrix
        labels: Target labels
        n_estimators: Number of trees in RandomForest
        max_depth: Max depth of trees
        random_state: Random seed for reproducibility
        test_size: Proportion of data for test set (0.0 to 1.0)
        
    Returns:
        Tuple of (model, label_encoder, metrics_dict)
        metrics_dict contains: accuracy, f1_score, confusion_matrix, train_size, test_size
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels.astype(str).fillna("UNKNOWN"))
    
    # Train/test split
    if test_size > 0.0:
        # Check if we can use stratify (need at least 2 samples per class)
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        min_class_count = class_counts.min() if len(class_counts) > 0 else 0
        can_stratify = len(unique_classes) > 1 and min_class_count >= 2
        
        if can_stratify:
            logger.debug("Using stratified train/test split (min class size: %d)", min_class_count)
            stratify_param = y_encoded
        else:
            logger.debug(
                "Cannot use stratified split (classes: %d, min class size: %d), using random split",
                len(unique_classes),
                min_class_count,
            )
            stratify_param = None
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
        )
        logger.info(
            "Train/test split: %d train, %d test (%.1f%%)",
            len(X_train),
            len(X_test),
            test_size * 100,
        )
    else:
        # Use all data for training (no test set)
        X_train, X_test = features_df, features_df.iloc[:0]
        y_train, y_test = y_encoded, np.array([])
        logger.info("Using full dataset for training (test_size=0)")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    
    logger.info(
        "Trained ML segmentation model on %d users with %d classes",
        len(X_train),
        len(encoder.classes_),
    )
    
    # Calculate metrics
    metrics: Dict[str, Any] = {
        "model": "RandomForestClassifier",
        "n_classes": len(encoder.classes_),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": random_state,
    }
    
    if len(X_test) > 0:
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {
            "matrix": cm.tolist(),
            "labels": encoder.classes_.tolist(),
        }
        
        metrics.update({
            "test_accuracy": float(accuracy),
            "test_f1_macro": float(f1_macro),
            "test_f1_weighted": float(f1_weighted),
            "confusion_matrix": cm_dict,
        })
        
        logger.info(
            "Test metrics - Accuracy: %.3f, F1 (macro): %.3f, F1 (weighted): %.3f",
            accuracy,
            f1_macro,
            f1_weighted,
        )
    else:
        logger.warning("No test set available for evaluation (test_size=0)")
        metrics["test_accuracy"] = None
        metrics["test_f1_macro"] = None
        metrics["test_f1_weighted"] = None
        metrics["confusion_matrix"] = None
    
    return model, encoder, metrics


def predict_segments(
    model: RandomForestClassifier,
    encoder: LabelEncoder,
    features_df: pd.DataFrame,
) -> pd.Series:
    """Predict segment labels for users as strings."""
    y_pred = model.predict(features_df)
    labels = encoder.inverse_transform(y_pred)
    return pd.Series(labels, index=features_df.index, name="segment_label")


def cluster_users_kmeans(
    features_df: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42,
    max_iter: int = 300,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Cluster users using K-Means (unsupervised segmentation).
    
    Args:
        features_df: Feature matrix
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations for K-Means
        
    Returns:
        Tuple of (segment_labels, metrics_dict)
        metrics_dict contains: n_clusters, inertia, n_iter, random_state
    """
    logger.info("Clustering %d users into %d clusters using K-Means", len(features_df), n_clusters)
    
    # Fit K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        n_init=10,
        n_jobs=-1,
    )
    cluster_labels = kmeans.fit_predict(features_df)
    
    # Convert to segment labels (Cluster_0, Cluster_1, ...)
    segment_labels = pd.Series(
        [f"Cluster_{i}" for i in cluster_labels],
        index=features_df.index,
        name="segment_label",
    )
    
    # Calculate metrics
    metrics: Dict[str, Any] = {
        "model": "KMeans",
        "n_clusters": n_clusters,
        "inertia": float(kmeans.inertia_),
        "n_iter": int(kmeans.n_iter_),
        "random_state": random_state,
        "cluster_sizes": {
            label: int((segment_labels == label).sum())
            for label in segment_labels.unique()
        },
    }
    
    logger.info(
        "K-Means clustering completed: inertia=%.2f, iterations=%d",
        kmeans.inertia_,
        kmeans.n_iter_,
    )
    
    return segment_labels, metrics