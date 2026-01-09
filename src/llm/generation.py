"""Message generation using LLM (mock, OpenAI, Hugging Face, or Groq)."""

import logging
from typing import Dict, Any, List

from src.core.types import CampaignRequest
from src.llm.openai_client import OpenAIClient
from src.llm.hf_client import HuggingFaceClient
from src.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)


def generate_messages(
    prompt: str,
    campaign_request: CampaignRequest,
    llm_mode: str = "mock",
    **kwargs,
) -> List[str]:
    """Generate N message variants (1-3).
    
    Returns:
        List of message variants (length = campaign_request.n_variants)
    """
    n_variants = max(1, min(3, campaign_request.n_variants))  # Clamp to 1-3
    
    if n_variants == 1:
        # Single message - use existing logic
        if llm_mode == "openai":
            try:
                return [_generate_openai_message(prompt, campaign_request, **kwargs)]
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAI generation failed (%s), falling back to mock", exc)
                return [_generate_mock_message(prompt, campaign_request)]
        
        if llm_mode == "hf" or llm_mode == "huggingface":
            try:
                return [_generate_hf_message(prompt, campaign_request, **kwargs)]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hugging Face generation failed (%s), falling back to mock", exc)
                return [_generate_mock_message(prompt, campaign_request)]
        
        if llm_mode == "groq":
            try:
                return [_generate_groq_message(prompt, campaign_request, **kwargs)]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Groq generation failed (%s), falling back to mock", exc)
                return [_generate_mock_message(prompt, campaign_request)]
        
        # Default / fallback
        if llm_mode != "mock":
            logger.warning("Unknown LLM mode: %s, using mock", llm_mode)
        return [_generate_mock_message(prompt, campaign_request)]
    
    # Multiple variants - generate N times
    variants = []
    for i in range(n_variants):
        if llm_mode == "openai":
            try:
                variants.append(_generate_openai_message(prompt, campaign_request, n=i+1, **kwargs))
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAI generation failed (%s), falling back to mock", exc)
                variants.append(_generate_mock_message(prompt, campaign_request, variant=i+1))
        
        elif llm_mode == "hf" or llm_mode == "huggingface":
            try:
                variants.append(_generate_hf_message(prompt, campaign_request, **kwargs))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hugging Face generation failed (%s), falling back to mock", exc)
                variants.append(_generate_mock_message(prompt, campaign_request, variant=i+1))
        
        elif llm_mode == "groq":
            try:
                variants.append(_generate_groq_message(prompt, campaign_request, **kwargs))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Groq generation failed (%s), falling back to mock", exc)
                variants.append(_generate_mock_message(prompt, campaign_request, variant=i+1))
        
        else:
            # Mock mode
            variants.append(_generate_mock_message(prompt, campaign_request, variant=i+1))
    
    return variants


# Backward compatibility
def generate_message(
    prompt: str,
    campaign_request: CampaignRequest,
    llm_mode: str = "mock",
    **kwargs,
) -> str:
    """Generate a single message (backward compatibility).
    
    For new code, use generate_messages() instead.
    """
    messages = generate_messages(prompt, campaign_request, llm_mode, **kwargs)
    return messages[0] if messages else ""


def _generate_mock_message(
    prompt: str,
    campaign_request: CampaignRequest,
    variant: int = 1,
) -> str:
    """Generate a single high-quality mock message without API call.
    
    Args:
        prompt: Generation prompt
        campaign_request: Campaign request
        variant: Variant number (1-3) for generating different versions
    """
    logger.debug("Generating mock message variant %d", variant)
    
    goal = campaign_request.goal
    channel = campaign_request.channel
    style = campaign_request.style
    
    # Best quality messages for each goal/channel combination
    base_messages = {
        "активация": {
            "push": "🎉 Добро пожаловать! Совершите первую покупку и получите скидку 10% на первый заказ!",
            "email": "Добро пожаловать в наш сервис! Мы подготовили для вас специальное предложение: скидка 10% на первый заказ. Начните делать покупки уже сегодня и откройте для себя широкий ассортимент качественных товаров.",
            "inapp": "Добро пожаловать! Совершите первую покупку и получите скидку 10%. Начните прямо сейчас!",
        },
        "реактивация": {
            "push": "Мы скучаем! Вернитесь и получите персональную скидку 20% на ваш следующий заказ.",
            "email": "Мы заметили, что вы давно не заходили к нам. Чтобы вернуть вас, мы подготовили специальное предложение: скидка 20% на ваш следующий заказ. Загляните к нам снова и откройте для себя обновленный ассортимент.",
            "inapp": "Вернитесь к нам! Специальная скидка 20% ждет вас при следующей покупке.",
        },
        "удержание": {
            "push": "Спасибо за лояльность! Эксклюзивная скидка 15% только для вас.",
            "email": "Спасибо, что остаетесь с нами! Как наш лояльный клиент, вы получаете эксклюзивную скидку 15% на следующий заказ. Продолжайте делать покупки с выгодой!",
            "inapp": "Спасибо за лояльность! Эксклюзивная скидка 15% только для вас.",
        },
        "upsell": {
            "push": "Откройте для себя премиум-варианты! Специальное предложение на апгрейд со скидкой 20%.",
            "email": "Откройте для себя премиум-варианты наших услуг! Мы предлагаем вам специальную скидку 20% на апгрейд. Получите больше возможностей и преимуществ уже сегодня.",
            "inapp": "Откройте премиум-варианты! Специальное предложение на апгрейд со скидкой 20%.",
        },
        "промо": {
            "push": "🔥 Акция! Скидка 30% только сегодня. Успейте купить!",
            "email": "🔥 Большая акция! Скидка 30% на весь ассортимент действует только сегодня. Не упустите возможность сэкономить на любимых товарах. Успейте сделать заказ!",
            "inapp": "🔥 Акция! Скидка 30% только сегодня. Успейте купить!",
        },
        "сервис": {
            "push": "Мы здесь, чтобы помочь! Есть вопросы? Напишите нам.",
            "email": "Здравствуйте! Мы заметили, что у вас могут быть вопросы. Наша служба поддержки готова помочь вам с любыми вопросами или проблемами. Свяжитесь с нами, и мы обязательно решим ваш вопрос.",
            "inapp": "Мы здесь, чтобы помочь! Есть вопросы? Напишите нам в поддержку.",
        },
    }
    
    # Variant messages (different versions for ranking)
    variant_messages = {
        1: {
            "активация": {
                "push": "🎉 Добро пожаловать! Совершите первую покупку и получите скидку 10% на первый заказ!",
                "email": "Добро пожаловать в наш сервис! Мы подготовили для вас специальное предложение: скидка 10% на первый заказ. Начните делать покупки уже сегодня и откройте для себя широкий ассортимент качественных товаров.",
                "inapp": "Добро пожаловать! Совершите первую покупку и получите скидку 10%. Начните прямо сейчас!",
            },
            "реактивация": {
                "push": "Мы скучаем! Вернитесь и получите персональную скидку 20% на ваш следующий заказ.",
                "email": "Мы заметили, что вы давно не заходили к нам. Чтобы вернуть вас, мы подготовили специальное предложение: скидка 20% на ваш следующий заказ. Загляните к нам снова и откройте для себя обновленный ассортимент.",
                "inapp": "Вернитесь к нам! Специальная скидка 20% ждет вас при следующей покупке.",
            },
        },
        2: {
            "активация": {
                "push": "Привет! Получите 10% скидку на первую покупку. У нас много товаров на любой вкус!",
                "email": "Здравствуйте! Предлагаем вам скидку 10% на первый заказ. У нас широкий ассортимент товаров для всей семьи.",
                "inapp": "Получите 10% скидку на первую покупку!",
            },
            "реактивация": {
                "push": "Вернитесь к нам! Скидка 20% на следующий заказ.",
                "email": "Мы рады видеть вас снова! Предлагаем скидку 20% на ваш следующий заказ. У нас большой выбор товаров.",
                "inapp": "Скидка 20% при возврате!",
            },
        },
        3: {
            "активация": {
                "push": "🎁 Специальное предложение для новых: скидка 10% на первый заказ! Купите сейчас!",
                "email": "Добро пожаловать! Как новый клиент, вы получаете эксклюзивную скидку 10% на первый заказ. Оформите заказ сегодня и получите выгоду!",
                "inapp": "Новым клиентам: скидка 10%! Оформите заказ сейчас!",
            },
            "реактивация": {
                "push": "Мы ждем вас! Персональная скидка 20% только для вас. Вернитесь сегодня!",
                "email": "Скучаем по вам! Вернитесь к нам и получите персональную скидку 20% на следующий заказ. Специальное предложение действует ограниченное время.",
                "inapp": "Вернитесь сегодня! Скидка 20% только для вас!",
            },
        },
    }
    
    # Get message for variant, goal and channel
    if variant in variant_messages and goal in variant_messages[variant] and channel in variant_messages[variant][goal]:
        message = variant_messages[variant][goal][channel]
    elif goal in base_messages and channel in base_messages[goal]:
        message = base_messages[goal][channel]
    else:
        message = f"Персонализированное сообщение для {goal} через {channel}. Специальное предложение для вас!"
    
    # Adjust style if needed
    if style == "формальный":
        message = message.replace("!", ".").replace("🎉", "").replace("🔥", "").replace("⚡", "").replace("🎁", "")
    elif style == "срочный":
        if "только сегодня" not in message.lower():
            message = message.replace(".", "! Только сегодня!")
    
    logger.debug(f"Generated mock message variant {variant} for {goal}/{channel}")
    return message


def _generate_openai_message(
    prompt: str,
    campaign_request: CampaignRequest,
    **kwargs,
) -> str:
    """Generate a single high-quality message using OpenAI Chat Completions API."""
    openai_config: Dict[str, Any] = kwargs.get("openai_config") or {}
    model = openai_config.get("model", "gpt-4o-mini")
    api_url = openai_config.get("api_url", "https://api.openai.com/v1/chat/completions")
    temperature = openai_config.get("temperature", 0.7)
    max_tokens = openai_config.get("max_tokens", 200)
    max_retries = openai_config.get("max_retries", 3)
    timeout = openai_config.get("timeout", 30)

    client = OpenAIClient(
        model=model,
        api_url=api_url,
        timeout=timeout,
        max_retries=max_retries,
    )

    system_prompt = (
        "Ты опытный маркетинговый копирайтер для e-commerce. "
        "Твоя задача - создавать высококачественные, персонализированные маркетинговые сообщения на русском языке. "
        "Учитывай цель кампании, канал коммуникации и характеристики сегмента пользователей. "
        "Сообщение должно быть убедительным, естественным и максимально релевантным."
    )

    logger.debug(
        "Calling OpenAI API for goal=%s, channel=%s",
        campaign_request.goal,
        campaign_request.channel,
    )
    
    raw_variants = client.generate(
        system_prompt=system_prompt,
        user_prompt=prompt,
        n=1,  # Generate only 1 message
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if raw_variants:
        return raw_variants[0].strip()
    
    return "Сообщение не сгенерировано"


def _generate_hf_message(
    prompt: str,
    campaign_request: CampaignRequest,
    **kwargs,
) -> str:
    """Generate a single high-quality message using Hugging Face Inference API (free)."""
    hf_config: Dict[str, Any] = kwargs.get("hf_config") or {}
    model = hf_config.get("model", "mistralai/Mistral-7B-Instruct-v0.2")
    api_url = hf_config.get("api_url")
    temperature = hf_config.get("temperature", 0.7)
    max_tokens = hf_config.get("max_tokens", 200)
    max_retries = hf_config.get("max_retries", 3)
    timeout = hf_config.get("timeout", 60)
    
    client = HuggingFaceClient(
        model=model,
        api_url=api_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    
    # Combine system and user prompt for HF
    full_prompt = (
        f"Ты опытный маркетинговый копирайтер для e-commerce. "
        f"Создай высококачественное персонализированное маркетинговое сообщение на русском языке.\n\n"
        f"{prompt}\n\n"
        f"Напиши ОДНО сообщение без дополнительных пояснений."
    )
    
    logger.debug(
        "Calling Hugging Face API for goal=%s, channel=%s",
        campaign_request.goal,
        campaign_request.channel,
    )
    
    raw_variants = client.generate(
        prompt=full_prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        num_return_sequences=1,  # Generate only 1 message
    )
    
    if raw_variants:
        message = raw_variants[0]
        # Remove prompt if present
        if prompt in message:
            message = message.split(prompt, 1)[-1].strip()
        # Take first non-empty line
        lines = [line.strip() for line in message.split("\n") if line.strip()]
        if lines:
            return lines[0]
        return message.strip()
    
    return "Сообщение не сгенерировано"


def _generate_groq_message(
    prompt: str,
    campaign_request: CampaignRequest,
    **kwargs,
) -> str:
    """Generate a single high-quality message using Groq API (free, fast)."""
    groq_config: Dict[str, Any] = kwargs.get("groq_config") or {}
    model = groq_config.get("model", "llama-3.1-8b-instant")
    api_url = groq_config.get("api_url", "https://api.groq.com/openai/v1/chat/completions")
    temperature = groq_config.get("temperature", 0.7)
    max_tokens = groq_config.get("max_tokens", 200)
    max_retries = groq_config.get("max_retries", 3)
    timeout = groq_config.get("timeout", 30)
    
    client = GroqClient(
        model=model,
        api_url=api_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    
    system_prompt = (
        "Ты опытный маркетинговый копирайтер для e-commerce. "
        "Твоя задача - создавать высококачественные, персонализированные маркетинговые сообщения на русском языке. "
        "Учитывай цель кампании, канал коммуникации и характеристики сегмента пользователей. "
        "Сообщение должно быть убедительным, естественным и максимально релевантным."
    )
    
    logger.debug(
        "Calling Groq API for goal=%s, channel=%s",
        campaign_request.goal,
        campaign_request.channel,
    )
    
    raw_variants = client.generate(
        system_prompt=system_prompt,
        user_prompt=prompt,
        n=1,  # Generate only 1 message
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    if raw_variants:
        return raw_variants[0].strip()
    
    return "Сообщение не сгенерировано"
