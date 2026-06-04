"""Google Gemini provider with structured output."""

from __future__ import annotations

import json
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMProvider, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, default_model: str = "gemini-2.5-flash"):
        self._client = genai.Client(api_key=api_key)
        self._default_model = default_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        reraise=True,
    )
    async def extract_structured(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: type[BaseModel],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        cache_system_prompt: bool = True,
    ) -> LLMResponse:
        model = model or self._default_model

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        usage = LLMUsage(
            input_tokens=response.usage_metadata.prompt_token_count or 0,
            output_tokens=response.usage_metadata.candidates_token_count or 0,
            calls=1,
        )

        text = response.text or ""
        try:
            data = json.loads(text)
            parsed = response_schema.model_validate(data)
            return LLMResponse(content=text, parsed=parsed, usage=usage)
        except Exception as e:
            logger.warning(f"Gemini structured output parse error: {e}")
            return LLMResponse(content=text, parsed=None, usage=usage)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        reraise=True,
    )
    async def extract_text(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        model = model or self._default_model

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        usage = LLMUsage(
            input_tokens=response.usage_metadata.prompt_token_count or 0,
            output_tokens=response.usage_metadata.candidates_token_count or 0,
            calls=1,
        )

        return LLMResponse(content=response.text or "", usage=usage)
