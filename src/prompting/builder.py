"""Build prompts from segment profiles and campaign requests."""

import logging
from typing import Optional

from src.core.types import SegmentProfile, CampaignRequest
from src.prompting.templates import get_template

logger = logging.getLogger(__name__)


def build_prompt(
    segment_profile: SegmentProfile,
    campaign_request: CampaignRequest,
    max_length: Optional[int] = None,
) -> str:
    """
    Build complete prompt for LLM generation.
    
    Args:
        segment_profile: Segment profile with aggregated data
        campaign_request: Campaign configuration
        max_length: Maximum message length (overrides channel default)
        
    Returns:
        Complete prompt string
    """
    # Get base template
    template = get_template(
        goal=campaign_request.goal,
        channel=campaign_request.channel,
        style=campaign_request.style,
    )
    
    # Add segment description
    segment_context = f"""
Контекст сегмента:
{segment_profile.description or segment_profile.to_brief()}

Ключевые характеристики сегмента:
- Размер сегмента: {segment_profile.size} пользователей
- Средняя активность: {segment_profile.avg_sessions_30d:.1f} сессий за 30 дней
- Средний GMV: {segment_profile.avg_gmv_90d_rub:.0f} руб
- Средний LTV: {segment_profile.avg_ltv_proxy:.0f} руб
"""
    
    if segment_profile.top_categories:
        top_cat = max(segment_profile.top_categories.items(), key=lambda x: x[1])[0]
        segment_context += f"- Популярная категория: {top_cat}\n"
    
    if segment_profile.abandoned_cart_rate > 0.3:
        segment_context += "- Высокий процент брошенных корзин\n"
    
    if segment_profile.avg_churn_risk > 0.7:
        segment_context += "- Высокий риск оттока\n"
    
    # Add length constraint
    length_limit = max_length or campaign_request.max_length
    if length_limit:
        length_instruction = f"\nОграничение длины: сообщение должно быть не более {length_limit} символов."
    else:
        length_instruction = ""
    
    # Add quality instructions
    quality_instructions = """
Требования к качеству сообщения:
- Сообщение должно быть персонализированным и релевантным для данного сегмента
- Используй конкретные детали из описания сегмента для повышения релевантности
- Сообщение должно быть убедительным и мотивирующим
- Избегай общих фраз, используй специфичные для сегмента аргументы
- Учитывай психологию сегмента (активность, покупки, риски)
- Сообщение должно звучать естественно и человечно
"""
    
    # Combine
    full_prompt = f"""{template}

{segment_context}

{quality_instructions}

Задача: Создай ОДНО высококачественное персонализированное маркетинговое сообщение для этого сегмента пользователей.{length_instruction}

Сообщение должно быть:
- Максимально релевантным для характеристик сегмента
- Убедительным и мотивирующим
- Естественным и человечным
- Соответствующим цели кампании и каналу коммуникации

Формат ответа: просто напиши текст сообщения без дополнительных пояснений, меток или нумерации."""
    
    logger.debug(f"Built prompt for segment {segment_profile.segment_label}, goal {campaign_request.goal}")
    
    return full_prompt

