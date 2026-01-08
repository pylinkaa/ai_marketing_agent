"""Tests for I/O utilities."""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path

from src.utils.io import load_csv, validate_schema, save_outputs
from src.core.types import GeneratedMessage, SegmentMetrics


def test_load_csv_valid(tmp_path):
    """Test loading valid CSV."""
    # Create test CSV
    data = {
        "user_id": ["u1", "u2"],
        "days_since_signup": [10, 20],
        "days_since_last_activity": [5, 15],
        "sessions_7d": [2, 3],
        "sessions_30d": [5, 7],
        "first_purchase_done": [0, 1],
        "purchase_count_90d": [0, 2],
        "gmv_90d_rub": [0.0, 5000.0],
        "aov_rub": [0.0, 2500.0],
        "churn_risk": [0.5, 0.3],
        "ltv_proxy": [1000.0, 5000.0],
        "channel_fatigue_score": [0.3, 0.2],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    
    # Load
    loaded_df = load_csv(str(csv_path))
    
    assert len(loaded_df) == 2
    assert "user_id" in loaded_df.columns


def test_load_csv_missing_file():
    """Test loading non-existent CSV."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent.csv")


def test_validate_schema_valid():
    """Test schema validation with valid data."""
    data = {
        "user_id": ["u1"],
        "days_since_signup": [10],
        "days_since_last_activity": [5],
        "sessions_7d": [2],
        "sessions_30d": [5],
        "first_purchase_done": [0],
        "purchase_count_90d": [0],
        "gmv_90d_rub": [0.0],
        "aov_rub": [0.0],
        "churn_risk": [0.5],
        "ltv_proxy": [1000.0],
        "channel_fatigue_score": [0.3],
    }
    df = pd.DataFrame(data)
    
    # Should not raise
    validate_schema(df)


def test_validate_schema_missing_column():
    """Test schema validation with missing column."""
    data = {
        "user_id": ["u1"],
        # Missing required columns
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(ValueError):
        validate_schema(df)


def test_save_outputs(tmp_path):
    """Test saving outputs."""
    messages = [
        GeneratedMessage(
            user_id="u1",
            segment_label="Active_Buyer",
            segment_profile_brief="Test segment",
            goal="активация",
            channel="push",
            message_v1="Message 1",
            message_v2="Message 2",
            message_v3="Message 3",
        )
    ]
    
    metrics = SegmentMetrics(
        segment_sizes={"Active_Buyer": 1},
        total_users=1,
    )
    
    saved_files = save_outputs(
        messages,
        metrics,
        output_dir=str(tmp_path),
        save_csv=True,
        save_json=True,
    )
    
    assert "messages_csv" in saved_files
    assert "messages_json" in saved_files
    assert "metrics_json" in saved_files
    
    # Check files exist
    assert Path(saved_files["messages_csv"]).exists()
    assert Path(saved_files["messages_json"]).exists()
    assert Path(saved_files["metrics_json"]).exists()

