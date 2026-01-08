"""Post-processing of generated messages."""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


# Forbidden words/phrases (can be extended)
FORBIDDEN_PHRASES = [
    "обязательно",
    "должен",
    "нужно",
    # Add more as needed
]


def postprocess_messages(
    messages: List[str],
    max_length: Optional[int] = None,
    style: str = "дружелюбный",
) -> List[str]:
    """
    Post-process generated messages.
    
    Args:
        messages: List of message variants
        max_length: Maximum length in characters
        style: Expected style (for validation)
        
    Returns:
        List of post-processed messages
    """
    processed = []
    
    for msg in messages:
        # Remove forbidden phrases
        msg = _remove_forbidden(msg)
        
        # Trim to max length
        if max_length:
            msg = _trim_to_length(msg, max_length)
        
        # Clean up whitespace
        msg = _clean_whitespace(msg)
        
        processed.append(msg)
    
    logger.debug(f"Post-processed {len(processed)} messages")
    
    return processed


def _remove_forbidden(text: str) -> str:
    """Remove forbidden words/phrases."""
    result = text
    for phrase in FORBIDDEN_PHRASES:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        result = pattern.sub("", result)
    return result


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
    
    # Add ellipsis if truncated
    if len(text) > max_length:
        trimmed = trimmed.rstrip() + "..."
    
    return trimmed


def _clean_whitespace(text: str) -> str:
    """Clean up excessive whitespace."""
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    # Replace multiple newlines with single newline
    text = re.sub(r"\n+", "\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

