"""Unified LLM provider interface for multi-model support.

Supports four LLM backends:
    - Gemini (Google) — via google-generativeai SDK
    - Claude (Anthropic) — via anthropic SDK
    - OpenAI (GPT-4o, etc.) — via openai SDK
    - Llama (Meta) — via ollama local server or together.ai API

Each provider implements the same interface: send a system prompt + user prompt,
receive a text response. This allows the RAG report generator and oracle to
work with any supported LLM.

Usage:
    provider = get_provider("claude", api_key="sk-ant-...")
    response = provider.generate("System prompt", "User prompt")

    # Or with image (multimodal):
    response = provider.generate_with_image("System prompt", "User prompt", image)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from PIL import Image


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI = "openai"
    LLAMA = "llama"


# Default model names per provider
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GEMINI: "gemini-1.5-pro",
    LLMProvider.CLAUDE: "claude-sonnet-4-20250514",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.LLAMA: "llama3.2-vision",
}


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: LLMProvider
    model_name: str | None = None  # None = use default for provider
    api_key: str | None = None
    base_url: str | None = None  # For Llama/Ollama or custom endpoints
    temperature: float = 0.3
    max_tokens: int = 2048

    @property
    def resolved_model(self) -> str:
        return self.model_name or DEFAULT_MODELS[self.provider]


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text from system + user prompts.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User message / query.

        Returns:
            Generated text response.
        """

    @abstractmethod
    def generate_with_image(
        self, system_prompt: str, user_prompt: str, image: Image.Image
    ) -> str:
        """Generate text from prompts + an image (multimodal).

        Args:
            system_prompt: System-level instructions.
            user_prompt: User message / query.
            image: PIL Image to analyze.

        Returns:
            Generated text response.
        """


class GeminiLLM(BaseLLM):
    """Google Gemini provider via google-generativeai SDK."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._model = None

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai

            if self.config.api_key:
                genai.configure(api_key=self.config.api_key)
            self._model = genai.GenerativeModel(
                self.config.resolved_model,
                system_instruction=None,  # Passed per-call
            )
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        model = self._get_model()
        # Gemini uses system_instruction at model level or prepends to prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
                "response_mime_type": "application/json",
            },
        )
        return response.text

    def generate_with_image(
        self, system_prompt: str, user_prompt: str, image: Image.Image
    ) -> str:
        model = self._get_model()
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = model.generate_content(
            [full_prompt, image],
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
                "response_mime_type": "application/json",
            },
        )
        return response.text


class ClaudeLLM(BaseLLM):
    """Anthropic Claude provider via anthropic SDK."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.config.api_key)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.config.resolved_model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.config.temperature,
        )
        return response.content[0].text

    def generate_with_image(
        self, system_prompt: str, user_prompt: str, image: Image.Image
    ) -> str:
        import base64
        import io

        client = self._get_client()

        # Encode image to base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = client.messages.create(
            model=self.config.resolved_model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }],
            temperature=self.config.temperature,
        )
        return response.content[0].text


class OpenAILLM(BaseLLM):
    """OpenAI GPT provider via openai SDK."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            kwargs: dict = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.config.resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def generate_with_image(
        self, system_prompt: str, user_prompt: str, image: Image.Image
    ) -> str:
        import base64
        import io

        client = self._get_client()

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model=self.config.resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content


class LlamaLLM(BaseLLM):
    """Meta Llama provider via Ollama local server or OpenAI-compatible API.

    Supports:
        - Ollama (default): local inference at http://localhost:11434
        - Together.ai: cloud inference via OpenAI-compatible endpoint
        - Any OpenAI-compatible server hosting Llama models
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._base_url = config.base_url or "http://localhost:11434/v1"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                api_key=self.config.api_key or "ollama",
                base_url=self._base_url,
            )
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.config.resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    def generate_with_image(
        self, system_prompt: str, user_prompt: str, image: Image.Image
    ) -> str:
        import base64
        import io

        client = self._get_client()

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model=self.config.resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content


# ─── Factory ───────────────────────────────────────────────────────────────

_PROVIDER_MAP: dict[LLMProvider, type[BaseLLM]] = {
    LLMProvider.GEMINI: GeminiLLM,
    LLMProvider.CLAUDE: ClaudeLLM,
    LLMProvider.OPENAI: OpenAILLM,
    LLMProvider.LLAMA: LlamaLLM,
}


def get_provider(
    provider: str | LLMProvider,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> BaseLLM:
    """Factory function to create an LLM provider instance.

    Args:
        provider: Provider name ("gemini", "claude", "openai", "llama") or enum.
        model_name: Optional model override (uses provider default if None).
        api_key: API key for the provider.
        base_url: Custom endpoint URL (for Llama/Ollama or proxies).
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.

    Returns:
        Configured LLM provider instance.

    Raises:
        ValueError: If provider is not recognized.
    """
    if isinstance(provider, str):
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            valid = ", ".join(p.value for p in LLMProvider)
            raise ValueError(
                f"Unknown LLM provider '{provider}'. Valid: {valid}"
            ) from None
    else:
        provider_enum = provider

    config = LLMConfig(
        provider=provider_enum,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    cls = _PROVIDER_MAP[provider_enum]
    return cls(config)
