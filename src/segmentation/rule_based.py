"""Rule-based segmentation for e-commerce."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def segment_users(
    df: pd.DataFrame,
    new_user_days: int = 30,
    dormant_days: int = 60,
    vip_purchase_threshold: int = 3,
    vip_ltv_threshold: float = 3000.0,
    active_days_threshold: int = 30,
) -> pd.Series:
    """
    Segment users using rule-based approach.
    
    Segments:
    - New_Unactivated: new users without purchase
    - Active_NonBuyer: active but no purchases
    - Active_Buyer: active with purchases
    - Dormant: inactive users
    - VIP: high-value customers
    - Price_Sensitive: price-sensitive users
    - Promo_Hunter: users with high promo share
    
    Args:
        df: DataFrame with user data
        new_user_days: Days threshold for new users
        dormant_days: Days threshold for dormant users
        vip_purchase_threshold: Purchase count for VIP
        vip_ltv_threshold: LTV threshold for VIP
        active_days_threshold: Days since last activity for active
        
    Returns:
        Series with segment labels
    """
    logger.info("Performing rule-based segmentation")
    
    segments = pd.Series(index=df.index, dtype=str)
    
    # Helper columns
    days_since_signup = df.get("days_since_signup", 0)
    days_since_activity = df.get("days_since_last_activity", 999)
    purchase_count = df.get("purchase_count_90d", 0)
    first_purchase = df.get("first_purchase_done", 0)
    ltv = df.get("ltv_proxy", 0)
    promo_share = df.get("promo_share_90d", 0)
    price_sensitivity = df.get("price_sensitivity", 0)
    sessions_30d = df.get("sessions_30d", 0)
    
    # New_Unactivated: new users without purchase
    mask_new = (days_since_signup < new_user_days) & (first_purchase == 0)
    segments[mask_new] = "New_Unactivated"
    
    # Dormant: inactive users
    mask_dormant = days_since_activity > dormant_days
    segments[mask_dormant] = "Dormant"
    
    # VIP: high-value customers
    mask_vip = (purchase_count >= vip_purchase_threshold) & (ltv >= vip_ltv_threshold)
    segments[mask_vip] = "VIP"
    
    # Active_Buyer: active with purchases
    mask_active_buyer = (
        (purchase_count > 0)
        & (days_since_activity <= active_days_threshold)
        & ~mask_vip
    )
    segments[mask_active_buyer] = "Active_Buyer"
    
    # Active_NonBuyer: active but no purchases
    mask_active_nonbuyer = (
        (sessions_30d > 0)
        & (purchase_count == 0)
        & (days_since_activity <= active_days_threshold)
        & ~mask_new
    )
    segments[mask_active_nonbuyer] = "Active_NonBuyer"
    
    # Fill remaining with default
    segments[segments == ""] = "Other"
    
    # Add sub-segments for price sensitivity and promo hunting
    # Price_Sensitive sub-segment
    mask_price_sensitive = (price_sensitivity > 0.5) & (segments.isin(["Active_NonBuyer", "Active_Buyer"]))
    segments[mask_price_sensitive] = segments[mask_price_sensitive] + "_Price_Sensitive"
    
    # Promo_Hunter sub-segment
    mask_promo = (promo_share > 0.7) & (segments.isin(["Active_NonBuyer", "Active_Buyer"]))
    segments[mask_promo] = segments[mask_promo].str.replace("_Price_Sensitive", "") + "_Promo_Hunter"
    
    # Count segments
    segment_counts = segments.value_counts()
    logger.info(f"Segmentation complete. Segments: {dict(segment_counts)}")
    
    return segments

