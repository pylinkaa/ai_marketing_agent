"""Post-processing of generated messages."""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Запрещенные клише (расширенный список)
FORBIDDEN_PHRASES = [
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
    r"обязательно",
    r"должен",
    r"нужно",
]

# Латиница (английские слова)
LATIN_PATTERN = re.compile(r'\b[A-Za-z]+\b')


def postprocess_messages(
    messages: List[str],
    max_length: Optional[int] = None,
    style: str = "дружелюбный",
) -> List[str]:
    """
    Post-process generated messages with quality improvements.
    
    Args:
        messages: List of message variants
        max_length: Maximum length in characters
        style: Expected style (for validation)
        
    Returns:
        List of post-processed messages
    """
    processed = []
    
    for msg in messages:
        original = msg
        
        # 1. Remove surrounding quotes
        msg = _remove_quotes(msg)
        
        # 2. Remove trailing ellipsis
        msg = _remove_trailing_ellipsis(msg)
        
        # 3. Remove forbidden phrases
        msg = _remove_forbidden(msg)
        
        # 4. Soft clean Latin characters (only isolated words, not in numbers/URLs)
        msg = _soft_clean_latin(msg)
        
        # 5. Trim to max length
        if max_length:
            msg = _trim_to_length(msg, max_length)
        
        # 6. Clean up whitespace
        msg = _clean_whitespace(msg)
        
        # Safety check: if message became too short, use original
        if len(msg.strip()) < 10:
            logger.warning(f"Post-processed message too short ({len(msg)}), using original")
            msg = original
            # Still apply basic cleanup
            msg = _remove_quotes(msg)
            msg = _remove_trailing_ellipsis(msg)
            msg = _clean_whitespace(msg)
        
        processed.append(msg)
    
    logger.debug(f"Post-processed {len(processed)} messages")
    
    return processed


def _remove_quotes(text: str) -> str:
    """Remove surrounding quotes if they wrap the entire text."""
    text = text.strip()
    
    # Check for various quote types
    quote_pairs = [
        ('"', '"'),
        ('«', '»'),
        ('"', '"'),
        ("'", "'"),
        ("'", "'"),
    ]
    
    for start_quote, end_quote in quote_pairs:
        if text.startswith(start_quote) and text.endswith(end_quote):
            text = text[len(start_quote):-len(end_quote)].strip()
            break
    
    # Also check for single quotes at start and end
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    
    return text


def _remove_trailing_ellipsis(text: str) -> str:
    """Remove trailing ellipsis (... or …)."""
    text = text.rstrip()
    
    # Remove trailing ellipsis
    while text.endswith("...") or text.endswith("…"):
        if text.endswith("..."):
            text = text[:-3].rstrip()
        elif text.endswith("…"):
            text = text[:-1].rstrip()
        else:
            break
    
    return text


def _remove_forbidden(text: str) -> str:
    """Remove forbidden words/phrases."""
    result = text
    for phrase in FORBIDDEN_PHRASES:
        # Case-insensitive replacement
        pattern = re.compile(phrase, re.IGNORECASE)
        result = pattern.sub("", result)
    return result


def _soft_clean_latin(text: str) -> str:
    """
    Softly clean Latin characters - remove isolated English words.
    Preserves numbers, URLs, and common abbreviations.
    """
    # Find all Latin words
    latin_words = LATIN_PATTERN.findall(text)
    
    # Remove common isolated English words that shouldn't be in Russian text
    # But preserve if they're part of a larger context (like "iPhone" or "WiFi")
    for word in latin_words:
        # Skip if it's a common abbreviation or brand name
        if word.lower() in ['iphone', 'wifi', 'ios', 'android', 'sms', 'email', 'push']:
            continue
        
        # Skip if it's part of a number or URL
        if re.search(r'\d', word):
            continue
        
        # Remove isolated short English words (likely typos)
        if len(word) <= 3:
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            text = pattern.sub('', text)
    
    return text


def _trim_to_length(text: str, max_length: int) -> str:
    """
    Trim text to maximum length, preserving word boundaries.
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Trimmed text
    """
    if len(text) <= max_length:
        return text
    
    # Trim to max_length, then find last space to preserve word boundary
    trimmed = text[:max_length]
    last_space = trimmed.rfind(" ")
    
    if last_space > max_length * 0.8:  # If space is reasonably close
        trimmed = trimmed[:last_space]
    else:
        # Just cut at max_length
        trimmed = trimmed[:max_length]
    
    # Don't add ellipsis (it's forbidden)
    trimmed = trimmed.rstrip()
    
    return trimmed


def _clean_whitespace(text: str) -> str:
    """Clean up excessive whitespace."""
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    # Replace multiple newlines with single newline
    text = re.sub(r"\n+", "\n", text)
    # Replace newlines with spaces (for single-line messages)
    text = text.replace("\n", " ")
    # Strip leading/trailing whitespace
    text = text.strip()
    return text
