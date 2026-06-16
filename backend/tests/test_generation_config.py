from unittest.mock import patch

import pytest

from rag import generation


def test_create_llm_uses_ollama_when_enabled():
    with (
        patch.object(generation, "USE_OLLAMA", True),
        patch.object(generation, "OLLAMA_MODEL", "llama3"),
        patch.object(generation, "OLLAMA_HOST", "http://localhost:11434"),
        patch.object(generation, "Ollama") as ollama,
    ):
        result = generation.create_llm()

    assert result is ollama.return_value
    ollama.assert_called_once_with(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=generation.LLM_TEMPERATURE,
    )


def test_create_llm_uses_official_anthropic_api_by_default():
    with (
        patch.object(generation, "USE_OLLAMA", False),
        patch.object(generation, "CLAUDE_API_KEY", "test-key"),
        patch.object(generation, "CLAUDE_MODEL", "claude-haiku-4-5"),
        patch.object(generation, "CLAUDE_URL", ""),
        patch.object(generation, "ChatAnthropic") as anthropic,
    ):
        generation.create_llm()

    kwargs = anthropic.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["api_key"] == "test-key"
    assert "base_url" not in kwargs


def test_create_llm_supports_custom_anthropic_proxy():
    with (
        patch.object(generation, "USE_OLLAMA", False),
        patch.object(generation, "CLAUDE_API_KEY", "test-key"),
        patch.object(generation, "CLAUDE_URL", "http://localhost:6655/anthropic/"),
        patch.object(generation, "ChatAnthropic") as anthropic,
    ):
        generation.create_llm()

    assert anthropic.call_args.kwargs["base_url"] == "http://localhost:6655/anthropic"


def test_create_llm_requires_anthropic_key():
    with (
        patch.object(generation, "USE_OLLAMA", False),
        patch.object(generation, "CLAUDE_API_KEY", ""),
    ):
        with pytest.raises(ValueError, match="CLAUDE_API_KEY"):
            generation.create_llm()
