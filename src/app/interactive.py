"""Интерактивный диалоговый интерфейс для маркетингового агента."""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from src.core.types import CampaignRequest, SegmentProfile, GeneratedMessage
from src.core.pipeline import run_pipeline, load_config, calculate_metrics
from src.utils.io import load_csv, save_outputs
from src.features.build_features import build_features
from src.segmentation.rule_based import segment_users
from src.segmentation.describe_segment import describe_all_segments
from src.prompting.builder import build_prompt
from src.llm.generation import generate_messages
from src.llm.ranking import rank_messages
from src.llm.postprocess import postprocess_messages

logging.basicConfig(
    level=logging.WARNING,  # Уменьшаем логи для интерактивного режима
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def print_separator():
    """Печать разделителя."""
    print("\n" + "=" * 60 + "\n")


def ask_file_path() -> str:
    """Запросить путь к CSV файлу."""
    print_separator()
    print("📊 ЗАГРУЗКА ДАННЫХ")
    print_separator()
    
    while True:
        file_path = input("Введите путь к CSV файлу (или нажмите Enter для data/synthetic_users.csv): ").strip()
        
        if not file_path:
            file_path = "data/synthetic_users.csv"
        
        path = Path(file_path)
        if path.exists():
            print(f"✅ Файл найден: {file_path}")
            return str(path)
        else:
            print(f"❌ Файл не найден: {file_path}")
            retry = input("Попробовать снова? (y/n): ").strip().lower()
            if retry != 'y':
                raise FileNotFoundError(f"Файл не найден: {file_path}")


def show_analytics(df: pd.DataFrame, segment_labels: pd.Series, segment_profiles: Dict[str, SegmentProfile]):
    """Показать аналитику по сегментам."""
    print_separator()
    print("📈 АНАЛИТИКА СЕГМЕНТОВ")
    print_separator()
    
    print(f"Всего пользователей: {len(df)}\n")
    
    # Показать сегменты
    segment_counts = segment_labels.value_counts().sort_values(ascending=False)
    
    print("Сегменты пользователей:")
    for i, (segment, count) in enumerate(segment_counts.items(), 1):
        percentage = (count / len(df)) * 100
        profile = segment_profiles[segment]
        
        print(f"\n{i}. {segment} ({count} пользователей, {percentage:.1f}%)")
        print(f"   📊 Средний GMV: {profile.avg_gmv_90d_rub:.0f} руб")
        print(f"   💰 Средний LTV: {profile.avg_ltv_proxy:.0f} руб")
        print(f"   📱 Активность: {profile.avg_sessions_30d:.1f} сессий за 30 дней")
        
        # Рекомендации по целям
        recommendations = get_recommendations(segment, profile)
        if recommendations:
            print(f"   🎯 Рекомендуемые цели: {', '.join(recommendations)}")
    
    print_separator()


def get_recommendations(segment_label: str, profile: SegmentProfile) -> List[str]:
    """Получить рекомендации по целям для сегмента."""
    recommendations = []
    
    if "New_Unactivated" in segment_label:
        recommendations.append("активация")
    elif "Dormant" in segment_label:
        recommendations.append("реактивация")
    elif "Active_Buyer" in segment_label or "VIP" in segment_label:
        recommendations.append("удержание")
        recommendations.append("upsell")
    elif "Active_NonBuyer" in segment_label:
        recommendations.append("активация")
        if profile.abandoned_cart_rate > 0.3:
            recommendations.append("реактивация")
    
    if profile.avg_churn_risk > 0.7:
        recommendations.append("удержание")
    
    return list(set(recommendations))  # Убрать дубликаты


def select_segment(segment_labels: pd.Series) -> str:
    """Выбрать сегмент для работы."""
    print_separator()
    print("🎯 ВЫБОР СЕГМЕНТА")
    print_separator()
    
    unique_segments = segment_labels.unique()
    segment_counts = segment_labels.value_counts()
    
    print("Доступные сегменты:")
    for i, segment in enumerate(unique_segments, 1):
        count = segment_counts[segment]
        print(f"{i}. {segment} ({count} пользователей)")
    
    while True:
        try:
            choice = input(f"\nВыберите сегмент (1-{len(unique_segments)}) или 'all' для всех: ").strip()
            
            if choice.lower() == 'all':
                return "all"
            
            idx = int(choice) - 1
            if 0 <= idx < len(unique_segments):
                selected = unique_segments[idx]
                print(f"✅ Выбран сегмент: {selected}")
                return selected
            else:
                print(f"❌ Неверный номер. Введите число от 1 до {len(unique_segments)}")
        except ValueError:
            print("❌ Введите число или 'all'")


def select_goal() -> str:
    """Выбрать цель кампании."""
    print_separator()
    print("📝 ВЫБОР ЦЕЛИ КАМПАНИИ")
    print_separator()
    
    goals = {
        "1": "активация",
        "2": "реактивация",
        "3": "удержание",
        "4": "upsell",
        "5": "промо",
        "6": "сервис",
    }
    
    print("Доступные цели:")
    for key, goal in goals.items():
        print(f"{key}. {goal}")
    
    while True:
        choice = input(f"\nВыберите цель (1-6): ").strip()
        if choice in goals:
            selected = goals[choice]
            print(f"✅ Выбрана цель: {selected}")
            return selected
        else:
            print("❌ Неверный выбор. Введите число от 1 до 6")


def select_channel() -> str:
    """Выбрать канал коммуникации."""
    print_separator()
    print("📱 ВЫБОР КАНАЛА")
    print_separator()
    
    channels = {
        "1": ("push", "Push-уведомления (до 100 символов)"),
        "2": ("email", "Email-рассылки (до 500 символов)"),
        "3": ("inapp", "In-app сообщения (до 300 символов)"),
    }
    
    print("Доступные каналы:")
    for key, (channel, desc) in channels.items():
        print(f"{key}. {desc}")
    
    while True:
        choice = input(f"\nВыберите канал (1-3): ").strip()
        if choice in channels:
            selected = channels[choice][0]
            print(f"✅ Выбран канал: {selected}")
            return selected
        else:
            print("❌ Неверный выбор. Введите число от 1 до 3")


def select_style() -> str:
    """Выбрать стиль сообщения."""
    print_separator()
    print("✍️  ВЫБОР СТИЛЯ")
    print_separator()
    
    styles = {
        "1": ("дружелюбный", "Неформальный, теплый стиль"),
        "2": ("формальный", "Официальный, профессиональный стиль"),
        "3": ("срочный", "Создает ощущение дедлайна"),
        "4": ("информативный", "Нейтральный, фактологический стиль"),
    }
    
    print("Доступные стили:")
    for key, (style, desc) in styles.items():
        print(f"{key}. {desc}")
    
    while True:
        choice = input(f"\nВыберите стиль (1-4, Enter для 'дружелюбный'): ").strip()
        if not choice:
            return "дружелюбный"
        if choice in styles:
            selected = styles[choice][0]
            print(f"✅ Выбран стиль: {selected}")
            return selected
        else:
            print("❌ Неверный выбор. Введите число от 1 до 4")


def show_messages(messages: List[GeneratedMessage], segment_label: str):
    """Показать сгенерированные сообщения."""
    print_separator()
    print(f"💬 СООБЩЕНИЯ ДЛЯ СЕГМЕНТА: {segment_label}")
    print_separator()
    
    # Показать примеры для первых 5 пользователей
    sample_size = min(5, len(messages))
    
    print(f"Показаны примеры для первых {sample_size} пользователей из {len(messages)}:\n")
    
    for i, msg in enumerate(messages[:sample_size], 1):
        print(f"\n--- Пользователь {msg.user_id} ---")
        print(f"✅ Выбранное сообщение: {msg.message}")
        if msg.ranking_score is not None:
            print(f"   (Оценка ранжирования: {msg.ranking_score:.1f})")
        if msg.message_v1 or msg.message_v2 or msg.message_v3:
            print("Варианты:")
            if msg.message_v1:
                print(f"  Вариант 1: {msg.message_v1}")
            if msg.message_v2:
                print(f"  Вариант 2: {msg.message_v2}")
            if msg.message_v3:
                print(f"  Вариант 3: {msg.message_v3}")
    
    if len(messages) > sample_size:
        print(f"\n... и еще {len(messages) - sample_size} пользователей")
    
    print_separator()


def ask_save_results() -> bool:
    """Спросить, сохранять ли результаты."""
    while True:
        choice = input("Сохранить результаты в файл? (y/n): ").strip().lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            print("Введите 'y' или 'n'")


def interactive_mode():
    """Основной интерактивный режим."""
    print("\n" + "=" * 60)
    print("🤖 ИИ-АГЕНТ ДЛЯ МАРКЕТИНГОВЫХ КОММУНИКАЦИЙ")
    print("=" * 60)
    
    try:
        # Шаг 1: Загрузка данных
        file_path = ask_file_path()
        print("\n⏳ Загрузка и обработка данных...")
        df = load_csv(file_path)
        print(f"✅ Загружено {len(df)} пользователей")
        
        # Шаг 2: Сегментация
        print("\n⏳ Выполнение сегментации...")
        config = load_config()
        seg_config = config.get("segmentation", {}).get("rule_based", {})
        segment_labels = segment_users(df, **seg_config)
        
        # Шаг 3: Описание сегментов
        print("⏳ Анализ сегментов...")
        segment_profiles = describe_all_segments(df, segment_labels)
        
        # Шаг 4: Показать аналитику
        show_analytics(df, segment_labels, segment_profiles)
        
        # Шаг 5: Выбрать сегмент
        selected_segment = select_segment(segment_labels)
        
        # Шаг 6: Выбрать параметры кампании
        goal = select_goal()
        channel = select_channel()
        style = select_style()
        
        # Шаг 7: Генерация сообщений
        print_separator()
        print("⏳ Генерация сообщений...")
        
        campaign_request = CampaignRequest(
            goal=goal,
            channel=channel,
            style=style,
        )
        
        # Установить max_length из конфига
        channel_limits = config.get("channel_limits", {})
        campaign_request.max_length = channel_limits.get(channel)
        
        # Фильтровать пользователей по сегменту
        if selected_segment == "all":
            target_users = df
            target_labels = segment_labels
        else:
            mask = segment_labels == selected_segment
            target_users = df[mask]
            target_labels = segment_labels[mask]
        
        # Генерация сообщений
        generated_messages = []
        segment_profile = segment_profiles[selected_segment] if selected_segment != "all" else None
        
        for user_idx, row in target_users.iterrows():
            user_id = row["user_id"]
            seg_label = target_labels.iloc[user_idx] if selected_segment == "all" else selected_segment
            
            # Использовать профиль конкретного сегмента
            if selected_segment == "all":
                seg_profile = segment_profiles[seg_label]
            else:
                seg_profile = segment_profile
            
            # Build user context (without PII)
            user_context = {}
            
            # Category interest
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
            
            # Построить промпт с user context
            prompt = build_prompt(seg_profile, campaign_request, user_context=user_context if user_context else None)
            
            # Сгенерировать варианты сообщений
            raw_variants = generate_messages(
                prompt,
                campaign_request,
                llm_mode="mock",
            )
            
            # Постобработка
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
            
            # Ранжирование и выбор лучшего
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
            
            # Создать сообщение
            message = GeneratedMessage(
                user_id=user_id,
                segment_label=seg_label,
                segment_profile_brief=seg_profile.to_brief(),
                goal=goal,
                channel=channel,
                message=best_message,
                message_v1=processed_variants[0] if len(processed_variants) > 0 else None,
                message_v2=processed_variants[1] if len(processed_variants) > 1 else None,
                message_v3=processed_variants[2] if len(processed_variants) > 2 else None,
                ranking_score=ranking_score,
                generation_metadata={
                    "llm_mode": "mock",
                    "n_variants": len(processed_variants),
                    "ranking_details": ranking_details,
                },
            )
            generated_messages.append(message)
        
        print(f"✅ Сгенерировано {len(generated_messages)} сообщений")
        
        # Шаг 8: Показать сообщения
        show_messages(generated_messages, selected_segment)
        
        # Шаг 9: Сохранение
        if ask_save_results():
            metrics = calculate_metrics(df, segment_labels, "rule")
            saved_files = save_outputs(
                generated_messages,
                metrics,
                output_dir="outputs",
            )
            print(f"\n✅ Результаты сохранены:")
            for key, path in saved_files.items():
                print(f"   - {path}")
        
        print_separator()
        print("✨ Работа завершена!")
        print_separator()
        
    except KeyboardInterrupt:
        print("\n\n👋 Выход из программы")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    interactive_mode()

