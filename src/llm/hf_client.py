"""Hugging Face Inference API client for free LLM generation."""

import logging
import os
import time
import random
from typing import List, Optional
import requests

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for Hugging Face Inference API (free tier)."""
    
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> None:
        self.model = model
        self.api_url = api_url or f"https://router.huggingface.co/models/{model}"
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        if not self.token:
            logger.warning("HF_TOKEN or HUGGINGFACE_API_KEY is not set; requests may fail")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}" if self.token else "",
            "Content-Type": "application/json",
        }
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        num_return_sequences: int = 3,
    ) -> List[str]:
        """
        Generate text using Hugging Face Inference API.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "num_return_sequences": num_return_sequences,
                "return_full_text": False,
            },
        }
        
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                
                # Handle rate limiting
                if resp.status_code == 503:
                    # Model is loading, wait and retry
                    wait_time = self.base_delay * (2 ** (attempt - 1))
                    logger.info(
                        "Model is loading, waiting %.1f seconds before retry %d/%d",
                        wait_time,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(wait_time)
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                # Parse response
                if isinstance(data, list):
                    texts = []
                    for item in data:
                        if isinstance(item, dict):
                            generated_text = item.get("generated_text", "")
                            if generated_text:
                                texts.append(generated_text.strip())
                        elif isinstance(item, str):
                            texts.append(item.strip())
                    
                    if texts:
                        return texts[:num_return_sequences]
                
                # Fallback: try to extract text from response
                if isinstance(data, dict) and "generated_text" in data:
                    return [data["generated_text"].strip()]
                
                logger.warning("Unexpected response format: %s", data)
                return ["Сообщение не сгенерировано"] * num_return_sequences
                
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logger.warning(
                        "HF API request failed (attempt %d/%d): %s. Waiting %.2f seconds",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "HF API request failed (final attempt %d/%d): %s",
                        attempt,
                        self.max_retries,
                        exc,
                    )
        
        raise RuntimeError(
            f"Hugging Face API failed after {self.max_retries} attempts"
        ) from last_error

