"""Tests for post-processing functions."""

import pytest

from src.llm.postprocess import (
    postprocess_messages,
    _remove_quotes,
    _remove_trailing_ellipsis,
    _soft_clean_latin,
)


def test_remove_quotes():
    """Test quote removal."""
    assert _remove_quotes('"test"') == "test"
    assert _remove_quotes('«test»') == "test"
    assert _remove_quotes("'test'") == "test"
    assert _remove_quotes("test") == "test"
    assert _remove_quotes('"test with spaces"') == "test with spaces"


def test_remove_trailing_ellipsis():
    """Test ellipsis removal."""
    assert _remove_trailing_ellipsis("test...") == "test"
    assert _remove_trailing_ellipsis("test…") == "test"
    assert _remove_trailing_ellipsis("test") == "test"
    assert _remove_trailing_ellipsis("test... ") == "test"


def test_soft_clean_latin():
    """Test Latin character cleaning."""
    # Should remove isolated English words
    assert "lately" not in _soft_clean_latin("test lately message")
    # Should preserve common abbreviations
    assert "iPhone" in _soft_clean_latin("test iPhone message")
    assert "WiFi" in _soft_clean_latin("test WiFi message")


def test_postprocess_messages():
    """Test full post-processing."""
    messages = [
        '"Test message..."',
        '«Another message»',
        "Normal message",
    ]
    
    processed = postprocess_messages(messages, max_length=100)
    
    assert len(processed) == 3
    assert processed[0] == "Test message"
    assert processed[1] == "Another message"
    assert processed[2] == "Normal message"
    
    # Check that quotes and ellipsis are removed
    assert '"' not in processed[0]
    assert "..." not in processed[0]
    assert "«" not in processed[1]
    assert "»" not in processed[1]


def test_postprocess_preserves_short_messages():
    """Test that very short messages are preserved."""
    messages = ['"Hi"']
    processed = postprocess_messages(messages)
    
    # Should not become empty
    assert len(processed[0]) > 0
