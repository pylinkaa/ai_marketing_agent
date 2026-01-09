"""Message ranking and selection based on quality criteria."""

import logging
import re
from typing import List, Tuple, Dict, Any

from src.core.types import CampaignRequest

logger = logging.getLogger(__name__)

# Common generic phrases that should be penalized
GENERIC_PHRASES = [
    r"у нас\s+много",
    r"широкий\s+ассортимент",
    r"большой\s+выбор",
    r"много\s+товаров",
    r"разнообразие\s+товаров",
    r"огромный\s+выбор",
    r"множество\s+вариантов",
]

# Specific phrases that should be rewarded
SPECIFIC_PHRASES = [
    r"\d+%",  # Percentage discount
    r"\d+\s*руб",  # Specific price
    r"скидк[аи]",  # Discount
    r"акци[яи]",  # Promotion
    r"только\s+сегодня",  # Urgency
    r"специальн[аяое]",  # Special offer
    r"эксклюзивн[аяое]",  # Exclusive
]

# CTA phrases
CTA_PHRASES = [
    r"купите",
    r"закажите",
    r"оформите",
    r"получите",
    r"воспользуйтесь",
    r"начните",
    r"успейте",
    r"перейдите",
]


def rank_messages(
    messages: List[str],
    campaign_request: CampaignRequest,
) -> Tuple[str, float, Dict[str, Any]]:
    """Rank messages and select the best one.
    
    Scoring criteria:
    - Penalty for exceeding max_length
    - Penalty for generic phrases
    - Bonus for specific details (discounts, prices, categories)
    - Bonus for CTA phrases
    - Bonus for style compliance
    
    Args:
        messages: List of message variants
        campaign_request: Campaign request with constraints
        
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
        
        # 1. Length penalty (exponential for severe violations)
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
        
        # 2. Generic phrases penalty
        generic_count = sum(1 for phrase in GENERIC_PHRASES if re.search(phrase, msg, re.IGNORECASE))
        if generic_count > 0:
            penalty = generic_count * 5
            score -= penalty
            details["penalties"].append(f"generic_phrases({generic_count}): -{penalty:.1f}")
        
        # 3. Specific details bonus
        specific_count = sum(1 for phrase in SPECIFIC_PHRASES if re.search(phrase, msg, re.IGNORECASE))
        if specific_count > 0:
            bonus = specific_count * 10
            score += bonus
            details["bonuses"].append(f"specific_details({specific_count}): +{bonus:.1f}")
        
        # 4. CTA bonus
        cta_count = sum(1 for phrase in CTA_PHRASES if re.search(phrase, msg, re.IGNORECASE))
        if cta_count > 0:
            bonus = cta_count * 8
            score += bonus
            details["bonuses"].append(f"cta_phrases({cta_count}): +{bonus:.1f}")
        
        # 5. Style compliance bonus
        style_bonus = _check_style_compliance(msg, style, channel)
        if style_bonus > 0:
            score += style_bonus
            details["bonuses"].append(f"style_compliance: +{style_bonus:.1f}")
        
        details["total_score"] = score
        scored_messages.append((msg, score, details))
    
    # Sort by score (descending)
    scored_messages.sort(key=lambda x: x[1], reverse=True)
    
    best_message, best_score, best_details = scored_messages[0]
    
    logger.debug(
        "Selected message variant %d with score %.1f (from %d variants)",
        best_details["variant"],
        best_score,
        len(messages),
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
        if "🎉" in message or "🎁" in message:
            bonus += 2
    
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
