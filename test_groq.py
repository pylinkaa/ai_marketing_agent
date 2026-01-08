#!/usr/bin/env python3
"""Простой тест подключения к Groq API."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to load .env file if it exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    print(f"📂 Загрузка .env файла: {env_file}")
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                print(f"  ✅ Загружен: {key}")

from src.llm.groq_client import GroqClient

def main():
    """Test Groq API connection."""
    print("🔍 Тестирование подключения к Groq API...")
    print()
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Ошибка: GROQ_API_KEY не установлен!")
        print()
        print("Установите ключ одним из способов:")
        print("1. source EXPORT_GROQ_KEY.sh")
        print("2. export GROQ_API_KEY='YOUR_GROQ_API_KEY_HERE'")
        print("3. ./setup_groq_simple.sh")
        print()
        print("⚠️  ВАЖНО: Запустите команду в ТОМ ЖЕ терминале, где будете запускать тест!")
        return 1
    
    print(f"✅ API ключ найден: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    # Test connection
    try:
        print("📡 Подключение к Groq API...")
        client = GroqClient(
            model="llama-3.1-8b-instant",
            timeout=30,
        )
        
        print("✅ Клиент создан успешно")
        print()
        
        print("🤖 Генерация тестового сообщения...")
        system_prompt = "Ты помощник, который пишет короткие сообщения на русском языке."
        user_prompt = "Напиши приветствие для нового пользователя (1 предложение)."
        
        result = client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            n=1,
            max_tokens=50,
            temperature=0.7,
        )
        
        print("✅ Генерация успешна!")
        print()
        print("📝 Результат:")
        print("-" * 50)
        for i, text in enumerate(result, 1):
            print(f"{i}. {text}")
        print("-" * 50)
        print()
        print("🎉 Всё работает! Можно использовать Groq в пайплайне.")
        return 0
        
    except ValueError as e:
        print(f"❌ Ошибка валидации: {e}")
        return 1
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print()
        print("Возможные причины:")
        print("1. Неверный API ключ")
        print("2. Проблемы с интернет-соединением")
        print("3. Groq API временно недоступен")
        print("4. Превышен лимит запросов")
        return 1

if __name__ == "__main__":
    sys.exit(main())

