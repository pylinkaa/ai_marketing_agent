"""Message ranking and selection based on quality criteria."""

import logging
import re
from typing import List, Tuple, Dict, Any, Optional

from src.core.types import CampaignRequest

logger = logging.getLogger(__name__)

# Запрещенные клише (расширенный список)
FORBIDDEN_CLICHES = [
    r"у нас\s+много\s+новинок",
    r"заканчивай\s+покупку",
    r"возвращайся\s+к",
    r"мы\s+заметили,\s+что",
    r"у нас\s+широкий\s+ассортимент",
    r"большой\s+выбор",
    r"много\s+товаров",
    r"разнообразие\s+товаров",
    r"огромный\s+выбор",
    r"множество\s+вариантов",
]

# CTA глаголы (расширенный список)
CTA_VERBS = [
    r"оформи",
    r"забери",
    r"вернись",
    r"смотри",
    r"выбери",
    r"получи",
    r"купи",
    r"закажи",
    r"воспользуйся",
    r"начни",
    r"успей",
    r"перейди",
    r"открой",
    r"попробуй",
]

# Специфичные фразы (бонусы)
SPECIFIC_PHRASES = [
    r"\d+%",  # Percentage discount
    r"\d+\s*руб",  # Specific price
    r"скидк[аи]",  # Discount
    r"акци[яи]",  # Promotion
    r"только\s+сегодня",  # Urgency
    r"специальн[аяое]",  # Special offer
    r"эксклюзивн[аяое]",  # Exclusive
]

# Латиница
LATIN_PATTERN = re.compile(r'\b[A-Za-z]+\b')


def rank_messages(
    messages: List[str],
    campaign_request: CampaignRequest,
    user_category: Optional[str] = None,
) -> Tuple[str, float, Dict[str, Any]]:
    """Rank messages and select the best one with enhanced scoring.
    
    Scoring criteria:
    - Heavy penalties for Latin, quotes, ellipsis, cliches
    - Bonuses for CTA, category mention, proper length, emojis
    
    Args:
        messages: List of message variants
        campaign_request: Campaign request with constraints
        user_category: Optional user category for bonus scoring
        
    Returns:
        Tuple of (best_message, score, ranking_details)
    """
    if not messages:
        raise ValueError("Empty messages list")
    
    if len(messages) == 1:
        return messages[0], 0.0, {"reason": "single_variant"}
    
    max_length = campaign_request.max_length or 500
    style = campaign_request.style
    channel = campaign_request.channel
    
    scored_messages: List[Tuple[str, float, Dict[str, Any]]] = []
    
    for i, msg in enumerate(messages, 1):
        score = 0.0
        details: Dict[str, Any] = {
            "variant": i,
            "length": len(msg),
            "penalties": [],
            "bonuses": [],
        }
        
        # 1. LATIN PENALTY (very heavy)
        latin_words = LATIN_PATTERN.findall(msg)
        # Filter out common abbreviations
        latin_words = [w for w in latin_words if w.lower() not in ['iphone', 'wifi', 'ios', 'android', 'sms', 'email', 'push']]
        if latin_words:
            penalty = -100
            score += penalty
            details["penalties"].append(f"latin_characters({len(latin_words)}): {penalty:.1f}")
        
        # 2. QUOTES PENALTY
        if (msg.strip().startswith('"') and msg.strip().endswith('"')) or \
           (msg.strip().startswith('«') and msg.strip().endswith('»')) or \
           (msg.strip().startswith("'") and msg.strip().endswith("'")):
            penalty = -20
            score += penalty
            details["penalties"].append(f"quotes: {penalty:.1f}")
        
        # 3. ELLIPSIS PENALTY
        if msg.rstrip().endswith("...") or msg.rstrip().endswith("…"):
            penalty = -15
            score += penalty
            details["penalties"].append(f"trailing_ellipsis: {penalty:.1f}")
        
        # 4. CLICHES PENALTY (heavy)
        cliche_count = sum(1 for cliche in FORBIDDEN_CLICHES if re.search(cliche, msg, re.IGNORECASE))
        if cliche_count > 0:
            penalty = cliche_count * -30
            score += penalty
            details["penalties"].append(f"cliches({cliche_count}): {penalty:.1f}")
        
        # 5. Length penalty (exponential for severe violations)
        length_ratio = len(msg) / max_length if max_length > 0 else 1.0
        if length_ratio > 1.0:
            penalty = (length_ratio - 1.0) ** 2 * 50  # Quadratic penalty
            score -= penalty
            details["penalties"].append(f"length_exceeded: -{penalty:.1f}")
        elif length_ratio < 0.3:
            # Too short might be incomplete
            penalty = (0.3 - length_ratio) * 10
            score -= penalty
            details["penalties"].append(f"too_short: -{penalty:.1f}")
        
        # 6. GENERIC TEXT PENALTY (no category, no CTA)
        has_category = user_category and user_category.lower() in msg.lower()
        has_cta = any(re.search(verb, msg, re.IGNORECASE) for verb in CTA_VERBS)
        
        if not has_category and not has_cta:
            penalty = -20
            score += penalty
            details["penalties"].append(f"too_generic: {penalty:.1f}")
        elif not has_category:
            penalty = -10
            score += penalty
            details["penalties"].append(f"no_category_mention: {penalty:.1f}")
        elif not has_cta:
            penalty = -10
            score += penalty
            details["penalties"].append(f"no_cta: {penalty:.1f}")
        
        # 7. CTA BONUS
        cta_count = sum(1 for verb in CTA_VERBS if re.search(verb, msg, re.IGNORECASE))
        if cta_count > 0:
            bonus = 10
            score += bonus
            details["bonuses"].append(f"cta_verbs({cta_count}): +{bonus:.1f}")
        
        # 8. CATEGORY MENTION BONUS
        if user_category and user_category.lower() in msg.lower():
            bonus = 10
            score += bonus
            details["bonuses"].append(f"category_mention: +{bonus:.1f}")
        
        # 9. SPECIFIC DETAILS BONUS
        specific_count = sum(1 for phrase in SPECIFIC_PHRASES if re.search(phrase, msg, re.IGNORECASE))
        if specific_count > 0:
            bonus = specific_count * 10
            score += bonus
            details["bonuses"].append(f"specific_details({specific_count}): +{bonus:.1f}")
        
        # 10. LENGTH BONUS (fits channel)
        if channel == "push" and len(msg) <= max_length:
            bonus = 10
            score += bonus
            details["bonuses"].append(f"push_length_ok: +{bonus:.1f}")
        elif channel in ["email", "inapp"] and 50 <= len(msg) <= max_length:
            bonus = 10
            score += bonus
            details["bonuses"].append(f"channel_length_ok: +{bonus:.1f}")
        
        # 11. EMOJI BONUS (for friendly style, 0-1 emoji)
        if style == "дружелюбный":
            emoji_count = len(re.findall(r'[🎉🔥⚡🎁💎✨🌟💫]', msg))
            if emoji_count == 0 or emoji_count == 1:
                bonus = 5
                score += bonus
                details["bonuses"].append(f"emoji_count_ok({emoji_count}): +{bonus:.1f}")
            elif emoji_count > 2:
                penalty = -5
                score += penalty
                details["penalties"].append(f"too_many_emojis({emoji_count}): {penalty:.1f}")
        
        # 12. Style compliance bonus
        style_bonus = _check_style_compliance(msg, style, channel)
        if style_bonus > 0:
            score += style_bonus
            details["bonuses"].append(f"style_compliance: +{style_bonus:.1f}")
        
        details["total_score"] = score
        scored_messages.append((msg, score, details))
    
    # Sort by score (descending)
    scored_messages.sort(key=lambda x: x[1], reverse=True)
    
    best_message, best_score, best_details = scored_messages[0]
    
    # Log detailed breakdown
    logger.debug(
        "Selected message variant %d with score %.1f (from %d variants)",
        best_details["variant"],
        best_score,
        len(messages),
    )
    logger.debug(
        "Score breakdown - Penalties: %s, Bonuses: %s",
        best_details["penalties"],
        best_details["bonuses"],
    )
    
    return best_message, best_score, best_details


def _check_style_compliance(message: str, style: str, channel: str) -> float:
    """Check if message complies with requested style.
    
    Returns bonus score for style compliance.
    """
    bonus = 0.0
    message_lower = message.lower()
    
    if style == "формальный":
        # Check for formal language (no emojis, proper punctuation)
        if "🎉" not in message and "🔥" not in message and "⚡" not in message:
            bonus += 5
        if message.endswith(".") or message.endswith("!"):
            bonus += 3
        if "вы" in message_lower or "вас" in message_lower:
            bonus += 2
    
    elif style == "срочный":
        # Check for urgency indicators
        urgency_words = ["только", "сегодня", "сейчас", "успейте", "ограничено"]
        if any(word in message_lower for word in urgency_words):
            bonus += 8
        if "!" in message:
            bonus += 3
    
    elif style == "дружелюбный":
        # Check for friendly tone
        friendly_words = ["добро пожаловать", "спасибо", "рады"]
        if any(word in message_lower for word in friendly_words):
            bonus += 5
    
    elif style == "информативный":
        # Check for informative content
        if len(message) > 50:  # Longer messages are more informative
            bonus += 3
        if any(char.isdigit() for char in message):  # Contains numbers
            bonus += 2
    
    # Channel-specific compliance
    if channel == "push" and len(message) <= 100:
        bonus += 3  # Push should be short
    elif channel == "email" and len(message) > 50:
        bonus += 3  # Email can be longer
    
    return bonus
