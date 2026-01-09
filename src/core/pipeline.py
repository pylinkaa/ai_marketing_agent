"""End-to-end pipeline for marketing agent."""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

from src.core.types import (
    CampaignRequest,
    GeneratedMessage,
    SegmentMetrics,
    SegmentProfile,
)
from src.utils.io import load_csv, save_outputs
from src.features.build_features import build_features
from src.segmentation.rule_based import segment_users
from src.segmentation.describe_segment import describe_all_segments
from src.segmentation.ml_model import (
    train_segmentation_model,
    predict_segments,
    cluster_users_kmeans,
)
from src.prompting.builder import build_prompt
from src.llm.generation import generate_messages
from src.llm.postprocess import postprocess_messages
from src.llm.ranking import rank_messages

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def calculate_metrics(
    df: pd.DataFrame,
    segment_labels: pd.Series,
    segmentation_mode: str = "rule",
    clustering_metrics: Optional[Dict] = None,
) -> SegmentMetrics:
    """
    Calculate segmentation quality metrics.
    
    Args:
        df: Original DataFrame
        segment_labels: Segment labels for each user
        segmentation_mode: "rule" or "ml"
        clustering_metrics: Clustering metrics (if ML mode)
        
    Returns:
        SegmentMetrics object
    """
    # Segment sizes
    segment_sizes = segment_labels.value_counts().to_dict()
    total_users = len(df)
    
    # Validation metrics (compare with true_* fields if available)
    validation_metrics = None
    if "true_segment_label" in df.columns:
        # Calculate accuracy
        true_labels = df["true_segment_label"].fillna("")
        pred_labels = segment_labels.fillna("")
        
        # Exact match accuracy
        exact_matches = (true_labels == pred_labels).sum()
        accuracy = exact_matches / len(df) if len(df) > 0 else 0.0
        
        validation_metrics = {
            "segment_label_accuracy": accuracy,
            "exact_matches": int(exact_matches),
            "total_users": total_users,
        }
        
        # Compare with true_next_goal if available
        if "true_next_goal" in df.columns:
            # This would be compared against campaign goal in real scenario
            pass
        
        # Compare with true_recommended_channel if available
        if "true_recommended_channel" in df.columns:
            # This would be compared against campaign channel in real scenario
            pass
    
    metrics = SegmentMetrics(
        segment_sizes=segment_sizes,
        total_users=total_users,
        clustering_metrics=clustering_metrics,
        validation_metrics=validation_metrics,
    )
    
    return metrics


def run_pipeline(
    input_path: str,
    campaign_request: CampaignRequest,
    config_path: str = "configs/default.yaml",
    segmentation_mode: str = "rule",
    llm_mode: str = "mock",
) -> Tuple[List[GeneratedMessage], SegmentMetrics]:
    """Run end-to-end pipeline."""
    logger.info("Starting pipeline")
    
    # Load config
    config = load_config(config_path)
    
    # Set max_length from config
    channel_limits = config.get("channel_limits", {})
    campaign_request.max_length = channel_limits.get(campaign_request.channel)
    
    # Step 1: Load data
    logger.info("Step 1: Loading data")
    df = load_csv(input_path)
    
    # Step 2: Build features
    logger.info("Step 2: Building features")
    features_config = config.get("features", {})
    features_df, original_df = build_features(
        df,
        exclude_true_fields=features_config.get("exclude_true_fields", True),
        normalize=features_config.get("normalize_for_ml", False) and segmentation_mode == "ml",
    )
    
    # Step 3: Segment users
    logger.info(f"Step 3: Segmenting users (mode: {segmentation_mode})")
    segmentation_cfg = config.get("segmentation", {})
    if segmentation_mode == "rule":
        seg_config = segmentation_cfg.get("rule_based", {})
        segment_labels = segment_users(
            original_df,
            new_user_days=seg_config.get("new_user_days", 30),
            dormant_days=seg_config.get("dormant_days", 60),
            vip_purchase_threshold=seg_config.get("vip_purchase_threshold", 3),
            vip_ltv_threshold=seg_config.get("vip_ltv_threshold", 3000.0),
            active_days_threshold=seg_config.get("active_days_threshold", 30),
        )
        clustering_metrics = None
    elif segmentation_mode == "ml":
        ml_cfg = segmentation_cfg.get("ml", {})
        algorithm = ml_cfg.get("algorithm", "rf")

        if algorithm == "rf":
            # Supervised: RandomForest
            # Target for supervised segmentation
            target_column = ml_cfg.get("target_column", "true_segment_label")
            if target_column in original_df.columns:
                logger.info("Using %s as target for ML segmentation", target_column)
                y_labels = original_df[target_column]
            else:
                logger.warning(
                    "Target column %s not found, falling back to rule-based segments as target",
                    target_column,
                )
                rule_cfg = segmentation_cfg.get("rule_based", {})
                y_labels = segment_users(
                    original_df,
                    new_user_days=rule_cfg.get("new_user_days", 30),
                    dormant_days=rule_cfg.get("dormant_days", 60),
                    vip_purchase_threshold=rule_cfg.get("vip_purchase_threshold", 3),
                    vip_ltv_threshold=rule_cfg.get("vip_ltv_threshold", 3000.0),
                    active_days_threshold=rule_cfg.get("active_days_threshold", 30),
                )

            # Train model with train/test split
            test_size = ml_cfg.get("test_size", 0.2)
            model, label_encoder, ml_metrics = train_segmentation_model(
                features_df,
                y_labels,
                n_estimators=ml_cfg.get("n_estimators", 200),
                max_depth=ml_cfg.get("max_depth"),
                random_state=ml_cfg.get("random_state", 42),
                test_size=test_size,
            )

            # Predict segments on full dataset (for message generation)
            segment_labels = predict_segments(model, label_encoder, features_df)

            clustering_metrics = ml_metrics

        elif algorithm == "kmeans":
            # Unsupervised: K-Means clustering
            n_clusters = ml_cfg.get("n_clusters", 5)
            segment_labels, clustering_metrics = cluster_users_kmeans(
                features_df,
                n_clusters=n_clusters,
                random_state=ml_cfg.get("random_state", 42),
                max_iter=ml_cfg.get("max_iter", 300),
            )
        else:
            raise ValueError(f"Unknown ML algorithm: {algorithm}")
    else:
        raise ValueError(f"Unknown segmentation mode: {segmentation_mode}")
    
    # Step 4: Describe segments
    logger.info("Step 4: Describing segments")
    segment_profiles = describe_all_segments(original_df, segment_labels)
    
    # Step 5: Generate messages for each user
    logger.info("Step 5: Generating messages")
    generated_messages = []
    
    for user_idx, row in original_df.iterrows():
        user_id = row["user_id"]
        segment_label = segment_labels.iloc[user_idx]
        segment_profile = segment_profiles[segment_label]
        
        # Build user context (without PII)
        user_context = {}
        
        # Category interest (try different column names)
        category = (
            row.get("last_view_category")
            or row.get("category_affinity_top")
            or row.get("last_category")
        )
        if pd.notna(category) and category:
            user_context["category_affinity_top"] = str(category)
            user_context["last_view_category"] = str(category)
        
        # Abandoned cart
        if "abandoned_cart_flag" in row:
            user_context["abandoned_cart_flag"] = bool(row.get("abandoned_cart_flag", 0))
        
        # Days since last activity
        if "days_since_last_activity" in row:
            days = row.get("days_since_last_activity")
            if pd.notna(days):
                user_context["days_since_last_activity"] = float(days)
        
        # Price sensitivity
        if "price_sensitivity" in row:
            sens = row.get("price_sensitivity")
            if pd.notna(sens):
                user_context["price_sensitivity"] = float(sens)
        
        # Build prompt with user context
        prompt = build_prompt(segment_profile, campaign_request, user_context=user_context if user_context else None)
        
        # Generate variants
        llm_config = config.get("llm", {})
        llm_mode_actual = llm_mode or llm_config.get("mode", "mock")
        
        # Prepare configs for different LLM providers
        openai_config = llm_config.get("openai", {})
        hf_config = llm_config.get("hf", {})
        groq_config = llm_config.get("groq", {})
        
        # Generate message variants
        raw_variants = generate_messages(
            prompt,
            campaign_request,
            llm_mode=llm_mode_actual,
            openai_config=openai_config,
            hf_config=hf_config,
            groq_config=groq_config,
        )
        
        # Post-process all variants
        processed_variants = postprocess_messages(
            raw_variants,
            max_length=campaign_request.max_length,
            style=campaign_request.style,
        )
        
        # Extract user category for ranking bonus
        user_category = None
        if user_context:
            user_category = (
                user_context.get("last_view_category")
                or user_context.get("category_affinity_top")
                or user_context.get("last_category")
            )
        
        # Rank and select best message
        if len(processed_variants) > 1:
            best_message, ranking_score, ranking_details = rank_messages(
                processed_variants,
                campaign_request,
                user_category=user_category,
            )
        else:
            best_message = processed_variants[0] if processed_variants else "Сообщение не сгенерировано"
            ranking_score = None
            ranking_details = {}
        
        # Create GeneratedMessage with variants
        message = GeneratedMessage(
            user_id=user_id,
            segment_label=segment_label,
            segment_profile_brief=segment_profile.to_brief(),
            goal=campaign_request.goal,
            channel=campaign_request.channel,
            message=best_message,
            message_v1=processed_variants[0] if len(processed_variants) > 0 else None,
            message_v2=processed_variants[1] if len(processed_variants) > 1 else None,
            message_v3=processed_variants[2] if len(processed_variants) > 2 else None,
            ranking_score=ranking_score,
            generation_metadata={
                "llm_mode": llm_mode_actual,
                "timestamp": pd.Timestamp.now().isoformat(),
                "n_variants": len(processed_variants),
                "ranking_details": ranking_details,
            },
        )
        generated_messages.append(message)
    
    # Step 6: Calculate metrics
    logger.info("Step 6: Calculating metrics")
    metrics = calculate_metrics(
        original_df,
        segment_labels,
        segmentation_mode=segmentation_mode,
        clustering_metrics=clustering_metrics,
    )
    
    logger.info("Pipeline completed successfully")
    
    return generated_messages, metrics

