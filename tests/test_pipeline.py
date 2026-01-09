"""Tests for pipeline."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from src.core.types import CampaignRequest
from src.core.pipeline import run_pipeline


@pytest.fixture
def sample_data(tmp_path):
    """Create sample CSV data."""
    data = {
        "user_id": ["u1", "u2", "u3"],
        "signup_date": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "cohort_month": ["2025-01", "2025-02", "2025-03"],
        "last_activity_date": ["2025-12-01", "2025-12-15", "2025-11-01"],
        "days_since_signup": [10, 20, 100],
        "days_since_last_activity": [5, 2, 70],
        "sessions_7d": [2, 3, 0],
        "sessions_30d": [5, 7, 1],
        "onboarding_completed": [1, 1, 0],
        "first_purchase_done": [0, 1, 0],
        "purchase_count_90d": [0, 3, 0],
        "gmv_90d_rub": [0.0, 10000.0, 0.0],
        "aov_rub": [0.0, 3333.0, 0.0],
        "refund_rate_90d": [0.0, 0.05, 0.0],
        "promo_share_90d": [0.0, 0.2, 0.0],
        "price_sensitivity": [0.5, 0.3, 0.6],
        "category_affinity_top": ["Электроника", "Одежда", "Спорт"],
        "loyalty_level": ["Silver", "Gold", "Silver"],
        "has_subscription": [0, 1, 0],
        "preferred_channel": ["push", "email", "inapp"],
        "push_opt_in": [1, 1, 0],
        "email_opt_in": [1, 1, 1],
        "sms_opt_in": [0, 0, 0],
        "push_open_rate_90d": [0.3, 0.4, 0.0],
        "email_open_rate_90d": [0.2, 0.3, 0.0],
        "click_rate_90d": [0.1, 0.15, 0.0],
        "send_time_pref": ["утро", "день", "вечер"],
        "device": ["iOS", "Android", "iOS"],
        "city_tier": [1, 2, 3],
        "support_tickets_90d": [0, 1, 0],
        "nps_last": [8, 9, 7],
        "churn_risk": [0.5, 0.2, 0.8],
        "ltv_proxy": [1000.0, 5000.0, 500.0],
        "push_sent_30d": [5, 3, 0],
        "email_sent_30d": [3, 2, 1],
        "inapp_shown_30d": [2, 1, 0],
        "channel_fatigue_score": [0.3, 0.2, 0.5],
        "last_campaign_type": ["активация", "промо", "реактивация"],
        "days_since_last_campaign": [5, 3, 20],
        "crm_conversions_90d": [0, 2, 0],
        "true_value_tier": ["High", "High", "Low"],
        "true_lifecycle_stage": ["New_NoPurchase", "Active_Buyer", "Dormant"],
        "true_price_segment": ["Not_Price_Sensitive", "Not_Price_Sensitive", "Price_Sensitive"],
        "true_segment_label": ["New_Unactivated", "Active_Buyer", "Dormant"],
        "true_recommended_channel": ["push", "email", "push"],
        "true_next_goal": ["активация", "upsell", "реактивация"],
        "product_views_7d": [5, 10, 0],
        "product_views_30d": [20, 50, 2],
        "add_to_cart_30d": [2, 5, 0],
        "wishlist_items": [1, 3, 0],
        "last_view_category": ["Электроника", "Одежда", "Спорт"],
        "days_since_last_cart_add": [3.0, 1.0, None],
        "abandoned_cart_flag": [1, 0, 0],
        "cart_value_rub": [500.0, 0.0, 0.0],
        "true_trigger_event": ["abandoned_cart", "none", "none"],
        "true_nba_action": ["cart_recovery", "personalized_upsell", "winback"],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_pipeline_basic(sample_data, tmp_path):
    """Test basic pipeline execution with mock LLM."""
    campaign_request = CampaignRequest(
        goal="активация",
        channel="push",
        style="дружелюбный",
        n_variants=3,  # Test with 3 variants for full coverage
    )
    
    messages, metrics = run_pipeline(
        input_path=sample_data,
        campaign_request=campaign_request,
        segmentation_mode="rule",
        llm_mode="mock",
    )
    
    # Check results
    assert len(messages) == 3
    assert metrics.total_users == 3
    assert len(metrics.segment_sizes) > 0
    
    # Check message structure
    for msg in messages:
        assert msg.user_id in ["u1", "u2", "u3"]
        assert msg.goal == "активация"
        assert msg.channel == "push"
        assert len(msg.message) > 0  # Best message always present
        # Variants may be None if n_variants=1
        if msg.message_v1:
            assert len(msg.message_v1) > 0
        if msg.message_v2:
            assert len(msg.message_v2) > 0
        if msg.message_v3:
            assert len(msg.message_v3) > 0
    
    # Check validation metrics if available
    if metrics.validation_metrics:
        assert "segment_label_accuracy" in metrics.validation_metrics


def test_pipeline_different_goal_channel(sample_data):
    """Test pipeline with different goal and channel."""
    campaign_request = CampaignRequest(
        goal="реактивация",
        channel="email",
        style="формальный",
    )
    
    messages, metrics = run_pipeline(
        input_path=sample_data,
        campaign_request=campaign_request,
        segmentation_mode="rule",
        llm_mode="mock",
    )
    
    assert len(messages) == 3
    for msg in messages:
        assert msg.goal == "реактивация"
        assert msg.channel == "email"

