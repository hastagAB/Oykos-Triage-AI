"""Anthropic Claude provider with tool_use structured output and prompt caching."""

from __future__ import annotations

import json
import logging

import anthropic
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMProvider, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)

# Models that do not accept the temperature parameter
_NO_TEMPERATURE_MODELS = {"claude-opus-4-8", "claude-opus-4-5"}


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, default_model: str = "claude-sonnet-4-20250514"):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
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
        schema = response_schema.model_json_schema()
        tool_name = "extract_symptoms"

        tools = [
            {
                "name": tool_name,
                "description": "Extract symptoms from a parent message and return structured JSON.",
                "input_schema": schema,
            }
        ]

        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
            }
        ]
        if cache_system_prompt:
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}

        kwargs = dict(
            model=model,
            max_tokens=4096,
            system=system_blocks,
            tools=tools,
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user_message}],
        )
        if model not in _NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = temperature

        response = await self._client.messages.create(**kwargs)

        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            calls=1,
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                parsed = response_schema.model_validate(block.input)
                return LLMResponse(
                    content=json.dumps(block.input, ensure_ascii=False),
                    parsed=parsed,
                    usage=usage,
                )

        logger.warning("No tool_use block found in response, returning empty result")
        return LLMResponse(content="", parsed=None, usage=usage)

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
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        if model not in _NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = temperature

        response = await self._client.messages.create(**kwargs)

        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            calls=1,
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return LLMResponse(content=text, usage=usage)
