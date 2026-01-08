"""Command-line interface for marketing agent."""

import argparse
import logging
import sys
from pathlib import Path

from src.core.types import CampaignRequest
from src.core.pipeline import run_pipeline
from src.utils.io import save_outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ИИ-агент для персональных маркетинговых коммуникаций"
    )
    
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Запустить в интерактивном режиме",
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Путь к входному CSV файлу",
    )
    
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        choices=["активация", "реактивация", "удержание", "upsell", "промо", "сервис"],
        help="Цель кампании",
    )
    
    parser.add_argument(
        "--channel",
        type=str,
        required=True,
        choices=["push", "email", "inapp"],
        help="Канал коммуникации",
    )
    
    parser.add_argument(
        "--style",
        type=str,
        default="дружелюбный",
        choices=["дружелюбный", "формальный", "срочный", "информативный"],
        help="Стиль сообщения (по умолчанию: дружелюбный)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Директория для сохранения результатов (по умолчанию: outputs)",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="rule",
        choices=["rule", "ml"],
        help="Режим сегментации: rule (rule-based) или ml (ML clustering) (по умолчанию: rule)",
    )
    
    parser.add_argument(
        "--llm-mode",
        type=str,
        default="mock",
        choices=["mock", "hf", "groq", "openai"],
        help="Режим LLM: mock (без API), hf (Hugging Face, бесплатно), groq (Groq, бесплатно), или openai (платно) (по умолчанию: mock)",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Путь к конфигурационному файлу (по умолчанию: configs/default.yaml)",
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        from src.app.interactive import interactive_mode
        interactive_mode()
        return
    
    # Validate input file for non-interactive mode
    if not args.input:
        logger.error("Требуется указать --input или использовать --interactive")
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Входной файл не найден: {args.input}")
        sys.exit(1)
    
    # Create campaign request
    campaign_request = CampaignRequest(
        goal=args.goal,
        channel=args.channel,
        style=args.style,
    )
    
    # Run pipeline
    try:
        logger.info("Запуск пайплайна...")
        messages, metrics = run_pipeline(
            input_path=str(input_path),
            campaign_request=campaign_request,
            config_path=args.config,
            segmentation_mode=args.mode,
            llm_mode=args.llm_mode,
        )
        
        # Save outputs
        output_config = {
            "save_csv": True,
            "save_json": True,
        }
        saved_files = save_outputs(
            messages,
            metrics,
            output_dir=args.output_dir,
            **output_config,
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ПАЙПЛАЙНА")
        print("=" * 60)
        print(f"\nОбработано пользователей: {len(messages)}")
        print(f"\nСегменты:")
        for segment, size in metrics.segment_sizes.items():
            print(f"  - {segment}: {size} пользователей")
        
        if metrics.validation_metrics:
            print(f"\nМетрики валидации:")
            if "segment_label_accuracy" in metrics.validation_metrics:
                acc = metrics.validation_metrics["segment_label_accuracy"]
                print(f"  - Точность сегментации: {acc:.2%}")
        
        print(f"\nСохраненные файлы:")
        for key, path in saved_files.items():
            print(f"  - {key}: {path}")
        print("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении пайплайна: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

