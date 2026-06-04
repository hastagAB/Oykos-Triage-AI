"""OpenAI provider with structured output via response_format."""

from __future__ import annotations

import json
import logging

import openai
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMProvider, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, default_model: str = "gpt-4o"):
        self._client = openai.AsyncOpenAI(api_key=api_key)
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

        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=response_schema,
        )
        if temperature > 0:
            kwargs["temperature"] = temperature
        response = await self._client.beta.chat.completions.parse(**kwargs)

        usage = LLMUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            calls=1,
        )

        message = response.choices[0].message
        if message.parsed:
            return LLMResponse(
                content=message.content or "",
                parsed=message.parsed,
                usage=usage,
            )

        if message.refusal:
            logger.warning(f"OpenAI refused: {message.refusal}")

        return LLMResponse(content=message.content or "", parsed=None, usage=usage)

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

        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        if temperature > 0:
            kwargs["temperature"] = temperature
        response = await self._client.chat.completions.create(**kwargs)

        usage = LLMUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            calls=1,
        )

        text = response.choices[0].message.content or ""
        return LLMResponse(content=text, usage=usage)
