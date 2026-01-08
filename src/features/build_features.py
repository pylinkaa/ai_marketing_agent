"""Feature engineering for segmentation."""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


TRUE_FIELDS_PREFIX = "true_"


def get_true_fields(df: pd.DataFrame) -> List[str]:
    """Get list of true_* field names."""
    return [col for col in df.columns if col.startswith(TRUE_FIELDS_PREFIX)]


def build_features(
    df: pd.DataFrame,
    exclude_true_fields: bool = True,
    normalize: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build numerical features for segmentation.
    
    Excludes true_* fields from features (they're for validation only).
    
    Args:
        df: Input DataFrame with user data
        exclude_true_fields: Whether to exclude true_* fields
        normalize: Whether to normalize features (for ML)
        
    Returns:
        Tuple of (features_df, original_df)
        features_df: DataFrame with numerical features only
        original_df: Original DataFrame with all columns
    """
    logger.info("Building features for segmentation")
    
    # Identify true_* fields
    true_fields = get_true_fields(df) if exclude_true_fields else []
    if true_fields:
        logger.debug(f"Excluding {len(true_fields)} true_* fields: {true_fields}")
    
    # Select feature columns (exclude true_* and non-numerical categorical)
    feature_cols = []
    
    # Activity features
    activity_cols = [
        "sessions_7d",
        "sessions_30d",
        "days_since_last_activity",
        "days_since_signup",
    ]
    feature_cols.extend([c for c in activity_cols if c in df.columns])
    
    # Purchase features
    purchase_cols = [
        "purchase_count_90d",
        "gmv_90d_rub",
        "aov_rub",
        "first_purchase_done",
        "refund_rate_90d",
        "promo_share_90d",
    ]
    feature_cols.extend([c for c in purchase_cols if c in df.columns])
    
    # Behavioral features
    behavioral_cols = [
        "product_views_7d",
        "product_views_30d",
        "add_to_cart_30d",
        "abandoned_cart_flag",
        "cart_value_rub",
        "wishlist_items",
    ]
    feature_cols.extend([c for c in behavioral_cols if c in df.columns])
    
    # Value features
    value_cols = [
        "ltv_proxy",
        "churn_risk",
    ]
    feature_cols.extend([c for c in value_cols if c in df.columns])
    
    # Loyalty level (encode)
    if "loyalty_level" in df.columns:
        loyalty_encoded = pd.get_dummies(df["loyalty_level"], prefix="loyalty", dummy_na=True)
        df = pd.concat([df, loyalty_encoded], axis=1)
        loyalty_cols = [c for c in loyalty_encoded.columns]
        feature_cols.extend(loyalty_cols)
    
    # Fatigue features
    fatigue_cols = [
        "channel_fatigue_score",
        "days_since_last_campaign",
        "push_sent_30d",
        "email_sent_30d",
        "inapp_shown_30d",
    ]
    feature_cols.extend([c for c in fatigue_cols if c in df.columns])
    
    # Channel features
    channel_cols = [
        "push_opt_in",
        "email_opt_in",
        "sms_opt_in",
        "push_open_rate_90d",
        "email_open_rate_90d",
        "click_rate_90d",
    ]
    feature_cols.extend([c for c in channel_cols if c in df.columns])
    
    # Additional numerical features
    additional_cols = [
        "onboarding_completed",
        "has_subscription",
        "support_tickets_90d",
        "nps_last",
        "crm_conversions_90d",
        "price_sensitivity",
    ]
    feature_cols.extend([c for c in additional_cols if c in df.columns])
    
    # Remove true_* fields and non-existent columns
    feature_cols = [c for c in feature_cols if c not in true_fields and c in df.columns]
    
    # Select only numerical features
    features_df = df[feature_cols].copy()
    
    # Fill NaN with 0 for numerical features
    features_df = features_df.fillna(0)
    
    # Normalize if requested
    if normalize:
        logger.debug("Normalizing features")
        scaler = StandardScaler()
        feature_values = scaler.fit_transform(features_df)
        features_df = pd.DataFrame(
            feature_values,
            columns=features_df.columns,
            index=features_df.index,
        )
    
    logger.info(f"Built {len(feature_cols)} features for {len(df)} users")
    
    return features_df, df

