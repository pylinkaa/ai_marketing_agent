"""OpenAI Chat Completions API client."""

import logging
import os
import random
import time
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# Global rate limit state: last 429 timestamp and retry-after value
_last_429_time: Optional[float] = None
_last_retry_after: Optional[float] = None


class OpenAIClient:
    """Simple client for OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        self.model = model
        self.api_url = api_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set; OpenAI requests will fail")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int = 3,
        max_tokens: int = 200,
        temperature: float = 0.2,
    ) -> List[str]:
        """Call OpenAI chat completions API and return list of message texts."""
        import json as _json  # local import to avoid unused in tools

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            # Check global rate limit state: if we recently got 429, wait
            global _last_429_time, _last_retry_after
            if _last_429_time is not None and _last_retry_after is not None:
                elapsed = time.time() - _last_429_time
                if elapsed < _last_retry_after:
                    wait_time = _last_retry_after - elapsed
                    logger.info(
                        "Rate limit active: waiting %.1f seconds before retry",
                        wait_time,
                    )
                    time.sleep(wait_time)
                    # Reset after waiting
                    _last_429_time = None
                    _last_retry_after = None

            try:
                resp = requests.post(
                    self.api_url,
                    headers=self.headers,
                    data=_json.dumps(payload),
                    timeout=self.timeout,
                )

                # Handle 429 specifically: respect Retry-After header
                if resp.status_code == 429:
                    retry_after = self._parse_retry_after(resp, attempt)
                    _last_429_time = time.time()
                    _last_retry_after = retry_after

                    if attempt < self.max_retries:
                        wait_time = retry_after
                        retry_after_header = resp.headers.get("Retry-After", "not provided")
                        logger.warning(
                            "Rate limited (429): waiting %.1f seconds (Retry-After header: %s) before retry %d/%d",
                            wait_time,
                            retry_after_header,
                            attempt + 1,
                            self.max_retries,
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        last_error = requests.HTTPError(
                            f"429 Too Many Requests (waiting {retry_after}s)"
                        )
                        break

                resp.raise_for_status()
                data = resp.json()

                # Success: reset rate limit state
                _last_429_time = None
                _last_retry_after = None

                choices = data.get("choices", [])
                texts: List[str] = []
                for ch in choices:
                    msg = ch.get("message", {}) or {}
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        texts.append(content.strip())
                if texts:
                    return texts

                # Fallback: single string from response
                return [str(data)]

            except requests.HTTPError as exc:
                # Already handled 429 above, but catch other HTTP errors
                if exc.response is not None and exc.response.status_code == 429:
                    # Shouldn't reach here, but just in case
                    continue
                last_error = exc
                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        "OpenAI request failed (attempt %d/%d): %s. Waiting %.2f seconds",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "OpenAI request failed (final attempt %d/%d): %s",
                        attempt,
                        self.max_retries,
                        exc,
                    )

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        "OpenAI request failed (attempt %d/%d): %s. Waiting %.2f seconds",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "OpenAI request failed (final attempt %d/%d): %s",
                        attempt,
                        self.max_retries,
                        exc,
                    )

        raise RuntimeError(
            f"OpenAI API failed after {self.max_retries} attempts"
        ) from last_error

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with optional jitter.
        
        Formula: base_delay * (2 ** (attempt - 1)) + jitter
        """
        delay = self.base_delay * (2 ** (attempt - 1))
        
        # Cap at max_delay
        delay = min(delay, self.max_delay)
        
        # Add jitter (±20% random variation)
        if self.jitter:
            jitter_amount = delay * 0.2 * (2 * random.random() - 1)  # ±20%
            delay = delay + jitter_amount
        
        return max(0.0, delay)

    def _parse_retry_after(self, resp: requests.Response, attempt: int) -> float:
        """
        Parse Retry-After header from response.
        Returns delay in seconds.
        
        If Retry-After header is missing, uses aggressive exponential backoff
        for rate limits (minimum 60 seconds for 429).
        """
        retry_after_str = resp.headers.get("Retry-After")
        if retry_after_str:
            try:
                # Retry-After can be seconds (int) or HTTP date
                retry_after = float(retry_after_str)
                # Use header value, but ensure minimum reasonable delay
                return max(retry_after, 10.0)
            except ValueError:
                # Try parsing as HTTP date (RFC 7231)
                try:
                    from email.utils import parsedate_to_datetime
                    retry_date = parsedate_to_datetime(retry_after_str)
                    if retry_date:
                        delay = (retry_date.timestamp() - time.time())
                        return max(delay, 10.0)
                except Exception:  # noqa: BLE001
                    pass
        
        # Fallback: aggressive exponential backoff for 429
        # Start with longer delays: 60s, 120s, 240s...
        base_429_delay = 60.0
        delay = base_429_delay * (2 ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        # Add jitter
        if self.jitter:
            jitter_amount = delay * 0.1 * (2 * random.random() - 1)  # ±10%
            delay = delay + jitter_amount
        
        return max(60.0, delay)  # Minimum 60 seconds for 429 without Retry-After


