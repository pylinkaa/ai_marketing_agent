"""Dataclasses for the marketing agent pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime


@dataclass
class UserRecord:
    """Single user record from CSV."""
    user_id: str
    signup_date: str
    cohort_month: str
    last_activity_date: str
    days_since_signup: int
    days_since_last_activity: int
    sessions_7d: int
    sessions_30d: int
    onboarding_completed: int
    first_purchase_done: int
    purchase_count_90d: int
    gmv_90d_rub: float
    aov_rub: float
    refund_rate_90d: float
    promo_share_90d: float
    price_sensitivity: float
    category_affinity_top: str
    loyalty_level: str
    has_subscription: int
    preferred_channel: str
    push_opt_in: int
    email_opt_in: int
    sms_opt_in: int
    push_open_rate_90d: float
    email_open_rate_90d: float
    click_rate_90d: float
    send_time_pref: str
    device: str
    city_tier: int
    support_tickets_90d: int
    nps_last: int
    churn_risk: float
    ltv_proxy: float
    push_sent_30d: int
    email_sent_30d: int
    inapp_shown_30d: int
    channel_fatigue_score: float
    last_campaign_type: str
    days_since_last_campaign: int
    crm_conversions_90d: int
    # True fields (for validation only, not features)
    true_value_tier: Optional[str] = None
    true_lifecycle_stage: Optional[str] = None
    true_price_segment: Optional[str] = None
    true_segment_label: Optional[str] = None
    true_recommended_channel: Optional[str] = None
    true_next_goal: Optional[str] = None
    true_trigger_event: Optional[str] = None
    true_nba_action: Optional[str] = None
    # Behavioral fields
    product_views_7d: int = 0
    product_views_30d: int = 0
    add_to_cart_30d: int = 0
    wishlist_items: int = 0
    last_view_category: Optional[str] = None
    days_since_last_cart_add: Optional[float] = None
    abandoned_cart_flag: int = 0
    cart_value_rub: float = 0.0


@dataclass
class SegmentProfile:
    """Aggregated segment description without PII."""
    segment_label: str
    size: int
    
    # Aggregated numerical features
    avg_sessions_30d: float
    avg_days_since_last_activity: float
    avg_purchase_count_90d: float
    avg_gmv_90d_rub: float
    avg_ltv_proxy: float
    avg_churn_risk: float
    avg_channel_fatigue: float
    
    # Categorical distributions
    top_categories: Dict[str, int]  # category -> count
    device_distribution: Dict[str, int]  # device -> count
    city_tier_distribution: Dict[int, int]  # tier -> count
    loyalty_level_distribution: Dict[str, int]  # level -> count
    
    # Behavioral patterns
    avg_cart_value: float
    avg_product_views_30d: float
    abandoned_cart_rate: float
    
    # Channel preferences
    push_opt_in_rate: float
    email_opt_in_rate: float
    avg_push_open_rate: float
    avg_email_open_rate: float
    
    # Text description (generated from aggregations)
    description: str = ""
    
    def to_brief(self) -> str:
        """Generate brief text description of the segment."""
        if self.description:
            return self.description
        
        parts = [
            f"Сегмент {self.segment_label} ({self.size} пользователей)",
            f"Средняя активность: {self.avg_sessions_30d:.1f} сессий за 30 дней",
            f"Средний GMV: {self.avg_gmv_90d_rub:.0f} руб",
            f"Средний LTV: {self.avg_ltv_proxy:.0f} руб",
        ]
        
        if self.top_categories:
            top_cat = max(self.top_categories.items(), key=lambda x: x[1])[0]
            parts.append(f"Топ категория: {top_cat}")
        
        if self.abandoned_cart_rate > 0.3:
            parts.append("Высокий процент брошенных корзин")
        
        return ". ".join(parts) + "."


@dataclass
class CampaignRequest:
    """Campaign configuration."""
    goal: str  # активация, реактивация, удержание, upsell, промо, сервис
    channel: str  # push, email, inapp
    style: str = "дружелюбный"  # дружелюбный, формальный, срочный, информативный
    max_length: Optional[int] = None  # Will be set from config based on channel


@dataclass
class GeneratedMessage:
    """Generated message result."""
    user_id: str
    segment_label: str
    segment_profile_brief: str
    goal: str
    channel: str
    message: str  # Best selected message (always present)
    message_v1: Optional[str] = None  # Variant 1 (if n_variants > 1)
    message_v2: Optional[str] = None  # Variant 2 (if n_variants > 1)
    message_v3: Optional[str] = None  # Variant 3 (if n_variants >= 3)
    ranking_score: Optional[float] = None  # Score of selected message
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        result = {
            "user_id": self.user_id,
            "segment_label": self.segment_label,
            "segment_profile_brief": self.segment_profile_brief,
            "goal": self.goal,
            "channel": self.channel,
            "message": self.message,
            "generation_metadata": str(self.generation_metadata) if self.generation_metadata else "",
        }
        # Add variants if present
        if self.message_v1:
            result["message_v1"] = self.message_v1
        if self.message_v2:
            result["message_v2"] = self.message_v2
        if self.message_v3:
            result["message_v3"] = self.message_v3
        if self.ranking_score is not None:
            result["ranking_score"] = self.ranking_score
        return result


@dataclass
class SegmentMetrics:
    """Segmentation quality metrics."""
    segment_sizes: Dict[str, int]
    total_users: int
    
    # Clustering metrics (if ML mode)
    clustering_metrics: Optional[Dict[str, float]] = None
    
    # Validation metrics (comparison with true_* fields)
    validation_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        result = {
            "segment_sizes": self.segment_sizes,
            "total_users": self.total_users,
        }
        if self.clustering_metrics:
            result["clustering_metrics"] = self.clustering_metrics
        if self.validation_metrics:
            result["validation_metrics"] = self.validation_metrics
        return result

