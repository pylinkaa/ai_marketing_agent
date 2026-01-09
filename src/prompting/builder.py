"""Build prompts from segment profiles and campaign requests."""

import logging
from typing import Optional, Dict, Any

from src.core.types import SegmentProfile, CampaignRequest
from src.prompting.templates import get_template

logger = logging.getLogger(__name__)

# Запрещенные клише
FORBIDDEN_CLICHES = [
    "у нас много новинок",
    "заканчивай покупку",
    "возвращайся к",
    "мы заметили, что",
    "у нас широкий ассортимент",
    "большой выбор",
    "много товаров",
]


def build_prompt(
    segment_profile: SegmentProfile,
    campaign_request: CampaignRequest,
    max_length: Optional[int] = None,
    user_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build complete prompt for LLM generation with strict copy rules.
    
    Args:
        segment_profile: Segment profile with aggregated data
        campaign_request: Campaign configuration
        max_length: Maximum message length (overrides channel default)
        user_context: Optional user-level context (without PII)
        
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
    
    # Add user-level context (without PII)
    user_context_str = ""
    if user_context:
        user_parts = []
        
        # Category interest
        category = (
            user_context.get("last_view_category")
            or user_context.get("category_affinity_top")
            or user_context.get("last_category")
        )
        if category:
            user_parts.append(f"- Интерес к категории: {category}")
        
        # Abandoned cart
        if user_context.get("abandoned_cart_flag"):
            user_parts.append("- Есть брошенная корзина")
        
        # Days since last activity
        days_inactive = user_context.get("days_since_last_activity")
        if days_inactive is not None:
            user_parts.append(f"- Дней с последней активности: {int(days_inactive)}")
        
        # Price sensitivity
        price_sens = user_context.get("price_sensitivity")
        if price_sens is not None:
            if price_sens > 0.6:
                user_parts.append("- Высокая чувствительность к цене")
            elif price_sens < 0.4:
                user_parts.append("- Низкая чувствительность к цене")
        
        if user_parts:
            user_context_str = f"""
Контекст пользователя (без PII):
{chr(10).join(user_parts)}
"""
    
    # Add length constraint
    length_limit = max_length or campaign_request.max_length
    if length_limit:
        length_instruction = f"\nОграничение длины: сообщение должно быть не более {length_limit} символов."
    else:
        length_instruction = ""
    
    # Channel-specific rules
    channel = campaign_request.channel
    if channel == "push":
        channel_rules = """
Правила для Push-уведомлений:
- Одна главная мысль
- Один четкий CTA (призыв к действию)
- Максимум 1 эмодзи
- Не более 1 восклицательного знака
- Структура: Хук → Выгода/Причина → CTA
"""
    else:
        channel_rules = ""
    
    # Strict copy rules
    copy_rules = f"""
СТРОГИЕ ПРАВИЛА КОПИРАЙТИНГА (ОБЯЗАТЕЛЬНО):

1. ЯЗЫК:
   - ТОЛЬКО русский язык
   - ЗАПРЕЩЕНА латиница (A-Z, a-z) в тексте сообщения
   - Используй только кириллицу

2. ФОРМАТ:
   - ЗАПРЕЩЕНЫ кавычки вокруг сообщения (", «», "")
   - ЗАПРЕЩЕНО окончание на "..." или "..."
   - Сообщение должно заканчиваться точкой, восклицательным знаком или без знака препинания

3. ЗАПРЕЩЕННЫЕ КЛИШЕ (НЕ ИСПОЛЬЗУЙ):
{chr(10).join(f"   - {cliche}" for cliche in FORBIDDEN_CLICHES)}

4. СТРУКТУРА:
   - Начни с хука (привлечение внимания)
   - Добавь выгоду или причину
   - Заверши четким CTA (призыв к действию)
   - Если известна категория интереса - упомяни её естественно 1 раз

5. КОНКРЕТИКА:
   - Используй конкретные детали из контекста сегмента
   - Избегай общих фраз
   - Будь специфичным и релевантным
{channel_rules}
"""
    
    # Combine
    full_prompt = f"""{template}

{segment_context}
{user_context_str}

{copy_rules}

Задача: Создай ОДНО высококачественное персонализированное маркетинговое сообщение для этого сегмента пользователей.{length_instruction}

ВАЖНО: Верни ТОЛЬКО текст сообщения, без пояснений, без кавычек, без меток, без нумерации. Просто чистый текст сообщения на русском языке."""
    
    logger.debug(f"Built prompt for segment {segment_profile.segment_label}, goal {campaign_request.goal}")
    
    return full_prompt
