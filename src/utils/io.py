"""Input/output utilities for CSV and JSON."""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.core.types import GeneratedMessage, SegmentMetrics

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = [
    "user_id",
    "days_since_signup",
    "days_since_last_activity",
    "sessions_7d",
    "sessions_30d",
    "first_purchase_done",
    "purchase_count_90d",
    "gmv_90d_rub",
    "aov_rub",
    "churn_risk",
    "ltv_proxy",
    "channel_fatigue_score",
]


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load CSV file with validation.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading CSV from {file_path}")
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    validate_schema(df)
    
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If required columns are missing
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.debug("Schema validation passed")


def save_outputs(
    messages: List[GeneratedMessage],
    metrics: SegmentMetrics,
    output_dir: str = "outputs",
    save_csv: bool = True,
    save_json: bool = True,
) -> Dict[str, str]:
    """
    Save generated messages and metrics to CSV and JSON.
    
    Args:
        messages: List of generated messages
        metrics: Segmentation metrics
        output_dir: Output directory
        save_csv: Whether to save CSV
        save_json: Whether to save JSON
        
    Returns:
        Dictionary with paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    
    # Save messages
    if messages:
        messages_data = [msg.to_dict() for msg in messages]
        messages_df = pd.DataFrame(messages_data)
        
        if save_csv:
            csv_path = output_path / f"messages_{timestamp}.csv"
            messages_df.to_csv(csv_path, index=False, encoding="utf-8")
            saved_files["messages_csv"] = str(csv_path)
            logger.info(f"Saved {len(messages)} messages to {csv_path}")
        
        if save_json:
            json_path = output_path / f"messages_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(messages_data, f, ensure_ascii=False, indent=2)
            saved_files["messages_json"] = str(json_path)
            logger.info(f"Saved messages to {json_path}")
    
    # Save metrics
    if save_json:
        metrics_path = output_path / f"metrics_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)
        saved_files["metrics_json"] = str(metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")
    
    return saved_files

