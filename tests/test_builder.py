"""Tests for prompt builder."""

import pytest

from src.prompting.builder import build_prompt
from src.core.types import SegmentProfile, CampaignRequest


@pytest.fixture
def sample_segment_profile():
    """Create sample segment profile."""
    return SegmentProfile(
        segment_label="Test",
        size=10,
        avg_sessions_30d=5.0,
        avg_gmv_90d_rub=1000.0,
        avg_ltv_proxy=5000.0,
        avg_days_since_last_activity=10.0,
        avg_purchase_count_90d=2.0,
        avg_churn_risk=0.3,
        avg_channel_fatigue=0.2,
        top_categories={"Электроника": 5},
        device_distribution={},
        city_tier_distribution={},
        loyalty_level_distribution={},
        avg_cart_value=1000.0,
        avg_product_views_30d=10.0,
        abandoned_cart_rate=0.2,
        push_opt_in_rate=0.8,
        email_opt_in_rate=0.7,
        avg_push_open_rate=0.3,
        avg_email_open_rate=0.2,
    )


def test_build_prompt_without_user_context(sample_segment_profile):
    """Test prompt building without user context."""
    campaign_request = CampaignRequest(goal="активация", channel="push")
    
    prompt = build_prompt(sample_segment_profile, campaign_request)
    
    assert "активация" in prompt.lower()
    assert "push" in prompt.lower()
    assert "СТРОГИЕ ПРАВИЛА" in prompt
    assert "ТОЛЬКО русский язык" in prompt
    assert "ЗАПРЕЩЕНЫ кавычки" in prompt
    assert "Верни ТОЛЬКО текст сообщения" in prompt


def test_build_prompt_with_user_context(sample_segment_profile):
    """Test prompt building with user context."""
    campaign_request = CampaignRequest(goal="активация", channel="push")
    user_context = {
        "category_affinity_top": "Электроника",
        "abandoned_cart_flag": True,
        "days_since_last_activity": 5.0,
        "price_sensitivity": 0.6,
    }
    
    prompt = build_prompt(sample_segment_profile, campaign_request, user_context=user_context)
    
    assert "Контекст пользователя" in prompt
    assert "Электроника" in prompt
    assert "брошенная корзина" in prompt
    assert "Дней с последней активности" in prompt
    assert "чувствительность к цене" in prompt


def test_build_prompt_forbids_cliches(sample_segment_profile):
    """Test that prompt forbids cliches."""
    campaign_request = CampaignRequest(goal="активация", channel="push")
    
    prompt = build_prompt(sample_segment_profile, campaign_request)
    
    assert "у нас много новинок" in prompt
    assert "заканчивай покупку" in prompt
    assert "возвращайся к" in prompt
    assert "мы заметили, что" in prompt
