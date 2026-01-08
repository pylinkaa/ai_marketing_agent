"""Groq API client for free fast LLM generation."""

import logging
import os
import time
import random
from typing import List, Optional
import requests

logger = logging.getLogger(__name__)


class GroqClient:
    """Client for Groq API (free tier, very fast)."""
    
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_url: str = "https://api.groq.com/openai/v1/chat/completions",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.api_url = api_url
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        if not self.api_key:
            logger.error("GROQ_API_KEY is not set! Please set it as environment variable or pass as parameter.")
            raise ValueError("GROQ_API_KEY is required but not set")
        
        # Validate API key format (Groq keys start with 'gsk_')
        if not self.api_key.startswith("gsk_"):
            logger.warning("Groq API key format looks incorrect (should start with 'gsk_')")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        logger.debug("Groq client initialized with model: %s", self.model)
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int = 3,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Generate text using Groq API (OpenAI-compatible).
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            n: Number of completions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated texts
        """
        # Groq API doesn't reliably support 'n' parameter, so make multiple requests
        all_texts: List[str] = []
        
        for request_num in range(n):
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            last_error: Optional[Exception] = None
            success = False
            
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.debug(
                        "Groq API request %d/%d (attempt %d/%d)",
                        request_num + 1,
                        n,
                        attempt,
                        self.max_retries,
                    )
                    
                    resp = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    
                    # Handle rate limiting
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            wait_time = float(retry_after)
                        else:
                            wait_time = self.base_delay * (2 ** (attempt - 1))
                        
                        if attempt < self.max_retries:
                            logger.warning(
                                "Rate limited (429): waiting %.1f seconds before retry %d/%d",
                                wait_time,
                                attempt + 1,
                                self.max_retries,
                            )
                            time.sleep(wait_time)
                            continue
                    
                    # Check for errors before parsing
                    if resp.status_code != 200:
                        error_text = resp.text
                        logger.error(
                            "Groq API returned status %d: %s",
                            resp.status_code,
                            error_text[:200],
                        )
                        # Try to parse error message
                        try:
                            error_data = resp.json()
                            error_msg = error_data.get("error", {}).get("message", error_text)
                            logger.error("Groq API error: %s", error_msg)
                        except Exception:
                            pass
                        
                        if attempt < self.max_retries:
                            delay = self.base_delay * (2 ** (attempt - 1))
                            logger.warning(
                                "Retrying after %.2f seconds (attempt %d/%d)",
                                delay,
                                attempt + 1,
                                self.max_retries,
                            )
                            time.sleep(delay)
                            continue
                        else:
                            raise requests.exceptions.HTTPError(
                                f"Groq API error {resp.status_code}: {error_text[:200]}"
                            )
                    
                    resp.raise_for_status()
                    data = resp.json()
                    
                    # Parse OpenAI-compatible response
                    choices = data.get("choices", [])
                    if choices:
                        # Get first choice content
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        if content:
                            all_texts.append(content.strip())
                            success = True
                            logger.debug("Successfully generated variant %d/%d", request_num + 1, n)
                            break  # Success, exit retry loop
                    
                    logger.warning("No text generated in response: %s", data)
                    if attempt == self.max_retries:
                        all_texts.append("Сообщение не сгенерировано")
                        break
                    
                except requests.exceptions.RequestException as exc:
                    last_error = exc
                    error_details = str(exc)
                    
                    # Log more details about the error
                    if hasattr(exc, 'response') and exc.response is not None:
                        try:
                            error_data = exc.response.json()
                            error_msg = error_data.get("error", {}).get("message", str(exc))
                            logger.error("Groq API error details: %s", error_msg)
                            error_details = error_msg
                        except Exception:
                            pass
                    
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                        logger.warning(
                            "Groq API request failed (attempt %d/%d): %s. Waiting %.2f seconds",
                            attempt,
                            self.max_retries,
                            error_details,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "Groq API request failed (final attempt %d/%d): %s",
                            attempt,
                            self.max_retries,
                            error_details,
                        )
                        all_texts.append("Сообщение не сгенерировано")
                        break
                
                except Exception as exc:  # noqa: BLE001
                    logger.error("Unexpected error in Groq API call: %s", exc)
                    all_texts.append("Сообщение не сгенерировано")
                    break
            
            # Small delay between requests to avoid rate limits
            if request_num < n - 1:
                time.sleep(0.5)
        
        # Return results
        if all_texts:
            logger.info("Successfully generated %d variant(s) from Groq", len(all_texts))
            # Ensure we have exactly n variants
            while len(all_texts) < n:
                all_texts.append(all_texts[-1] if all_texts else "Сообщение не сгенерировано")
            return all_texts[:n]
        
        logger.error("Failed to generate any text from Groq API")
        return ["Сообщение не сгенерировано"] * n
