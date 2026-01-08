"""Streamlit web interface for AI Marketing Agent."""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import streamlit as st
from io import StringIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.types import CampaignRequest, GeneratedMessage, SegmentMetrics, SegmentProfile
from src.core.pipeline import run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Marketing Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .segment-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .message-variant {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = None
    if "segment_profiles" not in st.session_state:
        st.session_state.segment_profiles = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "pipeline_run" not in st.session_state:
        st.session_state.pipeline_run = False


def render_header():
    """Render page header."""
    st.markdown('<h1 class="main-header">🚀 AI Marketing Agent</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Персонализированные маркетинговые коммуникации для e-commerce</p>',
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render sidebar with configuration options."""
    st.sidebar.header("⚙️ Настройки кампании")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Загрузите CSV файл с данными пользователей",
        type=["csv"],
        help="CSV файл должен содержать колонки: user_id, sessions_30d, gmv_90d_rub, и др.",
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Загружено {len(df)} пользователей")
            st.session_state.df = df
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка загрузки файла: {e}")
            st.session_state.df = None
    else:
        # Try to load default file
        default_path = Path("data/synthetic_users.csv")
        if default_path.exists():
            try:
                df = pd.read_csv(default_path)
                st.sidebar.info(f"📂 Используется файл по умолчанию: {default_path}")
                st.session_state.df = df
            except Exception as e:
                st.sidebar.warning(f"Не удалось загрузить файл по умолчанию: {e}")
                st.session_state.df = None
    
    st.sidebar.divider()
    
    # Campaign goal
    goal = st.sidebar.selectbox(
        "🎯 Цель кампании",
        options=["активация", "реактивация", "удержание", "upsell", "промо", "сервис"],
        index=0,
        help="Выберите цель маркетинговой кампании",
    )
    
    # Channel
    channel = st.sidebar.selectbox(
        "📱 Канал коммуникации",
        options=["push", "email", "inapp"],
        index=0,
        help="Выберите канал для отправки сообщений",
    )
    
    # Style
    style = st.sidebar.selectbox(
        "✍️ Стиль сообщения",
        options=["дружелюбный", "формальный", "срочный", "информативный"],
        index=0,
        help="Выберите тон и стиль сообщения",
    )
    
    # Segmentation mode
    segmentation_mode = st.sidebar.radio(
        "🔀 Режим сегментации",
        options=["rule", "ml"],
        index=0,
        help="rule: правило-основанная сегментация\nml: машинное обучение",
    )
    
    # LLM mode
    llm_mode = st.sidebar.radio(
        "🤖 Режим генерации",
        options=["mock", "hf", "groq", "openai"],
        index=0,
        help="mock: мокированные ответы\nhf: бесплатный Hugging Face API\ngroq: бесплатный Groq API (быстрый)\nopenai: OpenAI API (платный)",
    )
    
    if llm_mode == "hf":
        st.sidebar.info("💡 HF: Бесплатный API. Требуется HF_TOKEN или HUGGINGFACE_API_KEY")
    elif llm_mode == "groq":
        st.sidebar.info("💡 Groq: Бесплатный и быстрый API. Требуется GROQ_API_KEY")
    elif llm_mode == "openai":
        st.sidebar.warning("⚠️ OpenAI: Платный API. Требуется OPENAI_API_KEY")
    
    st.sidebar.divider()
    
    return {
        "goal": goal,
        "channel": channel,
        "style": style,
        "segmentation_mode": segmentation_mode,
        "llm_mode": llm_mode,
    }


def run_pipeline_ui(config: dict) -> Tuple[Optional[List[GeneratedMessage]], Optional[SegmentMetrics]]:
    """Run pipeline with progress indicators."""
    if st.session_state.df is None:
        st.error("❌ Пожалуйста, загрузите CSV файл с данными пользователей")
        return None, None
    
    # Save uploaded file temporarily
    input_path = "temp_input.csv"
    st.session_state.df.to_csv(input_path, index=False)
    
    try:
        campaign_request = CampaignRequest(
            goal=config["goal"],
            channel=config["channel"],
            style=config["style"],
        )
        
        with st.spinner("🔄 Запуск пайплайна..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Loading
            status_text.text("Шаг 1/5: Загрузка данных...")
            progress_bar.progress(20)
            
            # Step 2: Features
            status_text.text("Шаг 2/5: Построение признаков...")
            progress_bar.progress(40)
            
            # Step 3: Segmentation
            status_text.text(f"Шаг 3/5: Сегментация пользователей ({config['segmentation_mode']})...")
            progress_bar.progress(60)
            
            # Step 4: Describing segments
            status_text.text("Шаг 4/5: Описание сегментов...")
            progress_bar.progress(80)
            
            # Step 5: Generation
            status_text.text("Шаг 5/5: Генерация сообщений...")
            progress_bar.progress(90)
            
            # Run actual pipeline
            messages, metrics = run_pipeline(
                input_path=input_path,
                campaign_request=campaign_request,
                segmentation_mode=config["segmentation_mode"],
                llm_mode=config["llm_mode"],
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Пайплайн завершен!")
            
            return messages, metrics
            
    except Exception as e:
        st.error(f"❌ Ошибка при выполнении пайплайна: {e}")
        logger.exception("Pipeline error")
        return None, None
    finally:
        # Clean up temp file
        if Path(input_path).exists():
            Path(input_path).unlink()


def render_metrics(metrics: SegmentMetrics):
    """Render segmentation metrics."""
    st.header("📊 Метрики сегментации")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Всего пользователей", metrics.total_users)
    
    with col2:
        st.metric("Количество сегментов", len(metrics.segment_sizes))
    
    with col3:
        if metrics.clustering_metrics:
            model_type = metrics.clustering_metrics.get("model", "Unknown")
            st.metric("ML модель", model_type)
        else:
            st.metric("Режим", "Rule-based")
    
    st.divider()
    
    # Segment sizes
    st.subheader("📈 Распределение по сегментам")
    
    segment_df = pd.DataFrame(
        list(metrics.segment_sizes.items()),
        columns=["Сегмент", "Количество пользователей"],
    )
    segment_df["Доля, %"] = (segment_df["Количество пользователей"] / metrics.total_users * 100).round(1)
    segment_df = segment_df.sort_values("Количество пользователей", ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart
        st.bar_chart(segment_df.set_index("Сегмент")["Количество пользователей"], use_container_width=True)
    
    with col2:
        # Table
        st.dataframe(segment_df, use_container_width=True, hide_index=True)
    
    # Pie chart alternative (using bar chart)
    st.subheader("🥧 Доля сегментов (%)")
    st.bar_chart(segment_df.set_index("Сегмент")["Доля, %"])


def render_segments(messages: List[GeneratedMessage], metrics: SegmentMetrics):
    """Render segment information and messages."""
    st.header("👥 Сегменты и сообщения")
    
    # Get unique segments
    segments = list(metrics.segment_sizes.keys())
    
    # Create tabs for each segment
    tabs = st.tabs([f"{seg} ({metrics.segment_sizes[seg]})" for seg in segments])
    
    for tab_idx, selected_segment in enumerate(segments):
        with tabs[tab_idx]:
            # Filter messages for selected segment
            segment_messages = [m for m in messages if m.segment_label == selected_segment]
            
            if not segment_messages:
                st.warning(f"Нет сообщений для сегмента '{selected_segment}'")
                continue
            
            # Segment info
            segment_size = metrics.segment_sizes[selected_segment]
            percentage = segment_size / metrics.total_users * 100
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"### 📌 Сегмент: **{selected_segment}**")
            with col2:
                st.metric("Пользователей", f"{segment_size} ({percentage:.1f}%)")
            
            # Show segment profile from first message
            if segment_messages:
                profile_brief = segment_messages[0].segment_profile_brief
                with st.expander("📋 Описание сегмента", expanded=True):
                    st.markdown(profile_brief)
            
            st.divider()
            
            # Messages
            st.subheader(f"💬 Сгенерированные сообщения")
            
            # Filter options
            col1, col2 = st.columns([3, 1])
            with col2:
                show_all = st.checkbox("Показать все", value=False, key=f"show_all_{selected_segment}")
            
            # Show messages
            display_messages = segment_messages if show_all else segment_messages[:10]
            if not show_all and len(segment_messages) > 10:
                st.caption(f"Показано 10 из {len(segment_messages)} сообщений. Отметьте 'Показать все' для просмотра всех.")
            
            for idx, msg in enumerate(display_messages):
                with st.container():
                    st.markdown(f"**👤 Пользователь:** `{msg.user_id}`")
                    st.markdown(f'<div class="message-variant">{msg.message}</div>', unsafe_allow_html=True)
                    
                    if idx < len(display_messages) - 1:
                        st.divider()


def render_download_section(messages: List[GeneratedMessage], metrics: SegmentMetrics):
    """Render download section for results."""
    st.header("💾 Скачать результаты")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Convert messages to DataFrame
        messages_data = []
        for msg in messages:
            messages_data.append({
                "user_id": msg.user_id,
                "segment_label": msg.segment_label,
                "goal": msg.goal,
                "channel": msg.channel,
                "message": msg.message,
            })
        
        messages_df = pd.DataFrame(messages_data)
        csv_messages = messages_df.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            label="📥 Скачать сообщения (CSV)",
            data=csv_messages,
            file_name="generated_messages.csv",
            mime="text/csv",
        )
    
    with col2:
        # Metrics as JSON
        import json
        metrics_dict = {
            "segment_sizes": metrics.segment_sizes,
            "total_users": metrics.total_users,
            "validation_metrics": metrics.validation_metrics,
        }
        json_metrics = json.dumps(metrics_dict, ensure_ascii=False, indent=2).encode("utf-8")
        
        st.download_button(
            label="📥 Скачать метрики (JSON)",
            data=json_metrics,
            file_name="segmentation_metrics.json",
            mime="application/json",
        )


def main():
    """Main Streamlit app."""
    init_session_state()
    render_header()
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Main content
    st.divider()
    
    # Run pipeline button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button(
            "🚀 Запустить пайплайн",
            type="primary",
            use_container_width=True,
        )
    
    if run_button:
        if st.session_state.df is None:
            st.error("❌ Пожалуйста, загрузите CSV файл с данными пользователей")
        else:
            messages, metrics = run_pipeline_ui(config)
            if messages and metrics:
                st.session_state.messages = messages
                st.session_state.metrics = metrics
                st.session_state.pipeline_run = True
                st.success("✅ Пайплайн успешно выполнен!")
                st.balloons()
    
    # Display results if pipeline was run
    if st.session_state.pipeline_run and st.session_state.messages and st.session_state.metrics:
        st.divider()
        
        # Metrics
        render_metrics(st.session_state.metrics)
        
        st.divider()
        
        # Segments and messages
        render_segments(st.session_state.messages, st.session_state.metrics)
        
        st.divider()
        
        # Download section
        render_download_section(st.session_state.messages, st.session_state.metrics)
    
    # Footer
    st.divider()
    st.caption("AI Marketing Agent v1.0 | Персонализированные маркетинговые коммуникации")


if __name__ == "__main__":
    main()

