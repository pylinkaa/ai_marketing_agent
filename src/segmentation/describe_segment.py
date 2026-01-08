"""Generate segment profiles from aggregated data."""

import pandas as pd
import logging
from typing import Dict, List
from collections import Counter

from src.core.types import SegmentProfile

logger = logging.getLogger(__name__)


def describe_segment(
    df: pd.DataFrame,
    segment_label: str,
    segment_mask: pd.Series,
) -> SegmentProfile:
    """
    Create segment profile from user data.
    
    Args:
        df: DataFrame with user data
        segment_label: Label of the segment
        segment_mask: Boolean mask for users in segment
        
    Returns:
        SegmentProfile with aggregated information
    """
    segment_df = df[segment_mask].copy()
    size = len(segment_df)
    
    if size == 0:
        logger.warning(f"Empty segment: {segment_label}")
        return SegmentProfile(
            segment_label=segment_label,
            size=0,
            avg_sessions_30d=0.0,
            avg_days_since_last_activity=0.0,
            avg_purchase_count_90d=0.0,
            avg_gmv_90d_rub=0.0,
            avg_ltv_proxy=0.0,
            avg_churn_risk=0.0,
            avg_channel_fatigue=0.0,
            top_categories={},
            device_distribution={},
            city_tier_distribution={},
            loyalty_level_distribution={},
            avg_cart_value=0.0,
            avg_product_views_30d=0.0,
            abandoned_cart_rate=0.0,
            push_opt_in_rate=0.0,
            email_opt_in_rate=0.0,
            avg_push_open_rate=0.0,
            avg_email_open_rate=0.0,
        )
    
    # Numerical aggregations
    avg_sessions_30d = segment_df.get("sessions_30d", 0).mean()
    avg_days_since_last_activity = segment_df.get("days_since_last_activity", 0).mean()
    avg_purchase_count_90d = segment_df.get("purchase_count_90d", 0).mean()
    avg_gmv_90d_rub = segment_df.get("gmv_90d_rub", 0).mean()
    avg_ltv_proxy = segment_df.get("ltv_proxy", 0).mean()
    avg_churn_risk = segment_df.get("churn_risk", 0).mean()
    avg_channel_fatigue = segment_df.get("channel_fatigue_score", 0).mean()
    
    # Categorical distributions
    top_categories = dict(Counter(segment_df.get("category_affinity_top", pd.Series()).dropna()))
    device_distribution = dict(Counter(segment_df.get("device", pd.Series()).dropna()))
    city_tier_distribution = dict(Counter(segment_df.get("city_tier", pd.Series()).dropna()))
    loyalty_level_distribution = dict(Counter(segment_df.get("loyalty_level", pd.Series()).dropna()))
    
    # Behavioral patterns
    avg_cart_value = segment_df.get("cart_value_rub", 0).mean()
    avg_product_views_30d = segment_df.get("product_views_30d", 0).mean()
    abandoned_cart_count = segment_df.get("abandoned_cart_flag", 0).sum()
    abandoned_cart_rate = abandoned_cart_count / size if size > 0 else 0.0
    
    # Channel preferences
    push_opt_in_count = segment_df.get("push_opt_in", 0).sum()
    push_opt_in_rate = push_opt_in_count / size if size > 0 else 0.0
    
    email_opt_in_count = segment_df.get("email_opt_in", 0).sum()
    email_opt_in_rate = email_opt_in_count / size if size > 0 else 0.0
    
    avg_push_open_rate = segment_df.get("push_open_rate_90d", 0).mean()
    avg_email_open_rate = segment_df.get("email_open_rate_90d", 0).mean()
    
    # Generate text description
    description = _generate_description(
        segment_label=segment_label,
        size=size,
        avg_sessions_30d=avg_sessions_30d,
        avg_gmv_90d_rub=avg_gmv_90d_rub,
        avg_ltv_proxy=avg_ltv_proxy,
        avg_purchase_count_90d=avg_purchase_count_90d,
        top_categories=top_categories,
        abandoned_cart_rate=abandoned_cart_rate,
        avg_churn_risk=avg_churn_risk,
        device_distribution=device_distribution,
    )
    
    profile = SegmentProfile(
        segment_label=segment_label,
        size=size,
        avg_sessions_30d=avg_sessions_30d,
        avg_days_since_last_activity=avg_days_since_last_activity,
        avg_purchase_count_90d=avg_purchase_count_90d,
        avg_gmv_90d_rub=avg_gmv_90d_rub,
        avg_ltv_proxy=avg_ltv_proxy,
        avg_churn_risk=avg_churn_risk,
        avg_channel_fatigue=avg_channel_fatigue,
        top_categories=top_categories,
        device_distribution=device_distribution,
        city_tier_distribution=city_tier_distribution,
        loyalty_level_distribution=loyalty_level_distribution,
        avg_cart_value=avg_cart_value,
        avg_product_views_30d=avg_product_views_30d,
        abandoned_cart_rate=abandoned_cart_rate,
        push_opt_in_rate=push_opt_in_rate,
        email_opt_in_rate=email_opt_in_rate,
        avg_push_open_rate=avg_push_open_rate,
        avg_email_open_rate=avg_email_open_rate,
        description=description,
    )
    
    return profile


def _generate_description(
    segment_label: str,
    size: int,
    avg_sessions_30d: float,
    avg_gmv_90d_rub: float,
    avg_ltv_proxy: float,
    avg_purchase_count_90d: float,
    top_categories: Dict[str, int],
    abandoned_cart_rate: float,
    avg_churn_risk: float,
    device_distribution: Dict[str, int],
) -> str:
    """Generate human-readable description of segment."""
    parts = [f"Сегмент '{segment_label}' включает {size} пользователей."]
    
    # Activity
    if avg_sessions_30d > 5:
        parts.append("Высокая активность: более 5 сессий за 30 дней.")
    elif avg_sessions_30d > 0:
        parts.append(f"Средняя активность: {avg_sessions_30d:.1f} сессий за 30 дней.")
    else:
        parts.append("Низкая активность.")
    
    # Value
    if avg_gmv_90d_rub > 0:
        parts.append(f"Средний GMV за 90 дней: {avg_gmv_90d_rub:.0f} руб.")
        parts.append(f"Средний LTV: {avg_ltv_proxy:.0f} руб.")
        parts.append(f"Среднее количество покупок: {avg_purchase_count_90d:.1f}.")
    
    # Categories
    if top_categories:
        top_cat = max(top_categories.items(), key=lambda x: x[1])
        parts.append(f"Наиболее популярная категория: {top_cat[0]} ({top_cat[1]} пользователей).")
    
    # Behavioral insights
    if abandoned_cart_rate > 0.3:
        parts.append("Высокий процент брошенных корзин - требуется стимулирование завершения покупок.")
    
    if avg_churn_risk > 0.7:
        parts.append("Высокий риск оттока - необходимы меры по удержанию.")
    
    # Device
    if device_distribution:
        top_device = max(device_distribution.items(), key=lambda x: x[1])
        parts.append(f"Преобладающее устройство: {top_device[0]} ({top_device[1]} пользователей).")
    
    return " ".join(parts)


def describe_all_segments(
    df: pd.DataFrame,
    segment_labels: pd.Series,
) -> Dict[str, SegmentProfile]:
    """
    Create profiles for all segments.
    
    Args:
        df: DataFrame with user data
        segment_labels: Series with segment labels for each user
        
    Returns:
        Dictionary mapping segment_label -> SegmentProfile
    """
    logger.info("Describing all segments")
    
    profiles = {}
    unique_segments = segment_labels.unique()
    
    for segment_label in unique_segments:
        mask = segment_labels == segment_label
        profile = describe_segment(df, segment_label, mask)
        profiles[segment_label] = profile
    
    logger.info(f"Described {len(profiles)} segments")
    
    return profiles

