"""Tests for the LLM provider abstraction layer.

Tests provider factory, config resolution, and the unified interface.
Actual API calls are not made — we test config, factory, and structure.
"""

from __future__ import annotations

import pytest

from phytoveda.llm.providers import (
    BaseLLM,
    ClaudeLLM,
    DEFAULT_MODELS,
    GeminiLLM,
    LLMConfig,
    LLMProvider,
    LlamaLLM,
    OpenAILLM,
    get_provider,
)


# ─── LLMProvider Enum ──────────────────────────────────────────────────────


class TestLLMProvider:
    def test_all_providers(self) -> None:
        assert LLMProvider.GEMINI.value == "gemini"
        assert LLMProvider.CLAUDE.value == "claude"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.LLAMA.value == "llama"

    def test_provider_count(self) -> None:
        assert len(LLMProvider) == 4


# ─── LLMConfig ─────────────────────────────────────────────────────────────


class TestLLMConfig:
    def test_default_model_resolution(self) -> None:
        config = LLMConfig(provider=LLMProvider.GEMINI)
        assert config.resolved_model == DEFAULT_MODELS[LLMProvider.GEMINI]

    def test_custom_model_override(self) -> None:
        config = LLMConfig(provider=LLMProvider.CLAUDE, model_name="claude-opus-4-20250514")
        assert config.resolved_model == "claude-opus-4-20250514"

    def test_defaults(self) -> None:
        config = LLMConfig(provider=LLMProvider.OPENAI)
        assert config.temperature == 0.3
        assert config.max_tokens == 2048
        assert config.api_key is None
        assert config.base_url is None

    def test_all_defaults_exist(self) -> None:
        for provider in LLMProvider:
            assert provider in DEFAULT_MODELS


# ─── Factory: get_provider ─────────────────────────────────────────────────


class TestGetProvider:
    def test_create_gemini(self) -> None:
        llm = get_provider("gemini")
        assert isinstance(llm, GeminiLLM)
        assert llm.config.provider == LLMProvider.GEMINI

    def test_create_claude(self) -> None:
        llm = get_provider("claude", api_key="test-key")
        assert isinstance(llm, ClaudeLLM)
        assert llm.config.api_key == "test-key"

    def test_create_openai(self) -> None:
        llm = get_provider("openai", model_name="gpt-4o-mini")
        assert isinstance(llm, OpenAILLM)
        assert llm.config.resolved_model == "gpt-4o-mini"

    def test_create_llama(self) -> None:
        llm = get_provider("llama")
        assert isinstance(llm, LlamaLLM)

    def test_create_llama_custom_url(self) -> None:
        llm = get_provider("llama", base_url="http://myserver:11434/v1")
        assert isinstance(llm, LlamaLLM)
        assert llm._base_url == "http://myserver:11434/v1"

    def test_create_with_enum(self) -> None:
        llm = get_provider(LLMProvider.CLAUDE)
        assert isinstance(llm, ClaudeLLM)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("cohere")

    def test_case_insensitive(self) -> None:
        llm = get_provider("Claude")
        assert isinstance(llm, ClaudeLLM)

    def test_custom_temperature(self) -> None:
        llm = get_provider("gemini", temperature=0.9)
        assert llm.config.temperature == 0.9

    def test_custom_max_tokens(self) -> None:
        llm = get_provider("openai", max_tokens=4096)
        assert llm.config.max_tokens == 4096


# ─── Provider Classes Structure ────────────────────────────────────────────


class TestProviderStructure:
    def test_all_providers_are_base_llm(self) -> None:
        for provider_str in ["gemini", "claude", "openai", "llama"]:
            llm = get_provider(provider_str)
            assert isinstance(llm, BaseLLM)

    def test_all_providers_have_generate(self) -> None:
        for provider_str in ["gemini", "claude", "openai", "llama"]:
            llm = get_provider(provider_str)
            assert hasattr(llm, "generate")
            assert callable(llm.generate)

    def test_all_providers_have_generate_with_image(self) -> None:
        for provider_str in ["gemini", "claude", "openai", "llama"]:
            llm = get_provider(provider_str)
            assert hasattr(llm, "generate_with_image")
            assert callable(llm.generate_with_image)

    def test_llama_default_base_url(self) -> None:
        llm = get_provider("llama")
        assert isinstance(llm, LlamaLLM)
        assert "localhost" in llm._base_url
        assert "11434" in llm._base_url


# ─── Integration with ReportGenerator ─────────────────────────────────────


class TestReportGeneratorMultiLLM:
    def test_report_generator_accepts_provider(self) -> None:
        from phytoveda.rag.report_generator import ReportGenerator
        gen = ReportGenerator(provider="claude", api_key="test")
        assert gen._provider_name == "claude"

    def test_report_generator_default_gemini(self) -> None:
        from phytoveda.rag.report_generator import ReportGenerator
        gen = ReportGenerator()
        assert gen._provider_name == "gemini"

    def test_report_generator_offline_unchanged(self) -> None:
        """Offline report generation doesn't need an LLM provider."""
        from phytoveda.rag.report_generator import ReportGenerator
        from phytoveda.vrikshayurveda.mapper import Dosha, DoshaAssessment
        gen = ReportGenerator(provider="claude")
        dosha = DoshaAssessment(
            dosha=Dosha.PITTA, confidence=0.9,
            cv_features=[], classical_symptoms=[], treatments=["Treat"], contraindications=[],
        )
        report = gen.generate_offline("Neem", "Bacterial Spot", dosha)
        assert report.species_name == "Neem"

    def test_report_generator_injectable_llm(self) -> None:
        """Can inject a custom BaseLLM instance."""
        from phytoveda.rag.report_generator import ReportGenerator
        llm = get_provider("gemini")
        gen = ReportGenerator(llm=llm)
        assert gen._llm is llm
