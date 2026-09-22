"""
Unit tests for Multilingual Translation constraints.
"""

import pytest
from services.translation import MultilingualTranslator


def test_strictly_four_languages_supported():
    translator = MultilingualTranslator()
    assert set(translator.SUPPORTED_LANGUAGES) == {"English", "Hindi", "French", "Spanish"}


def test_unsupported_language_raises_error():
    translator = MultilingualTranslator()
    with pytest.raises(ValueError):
        translator.translate_text("Test query", target_language="German")


def test_english_translation_identity():
    translator = MultilingualTranslator()
    sample = "ResearchLens AI analysis"
    res = translator.translate_text(sample, target_language="English", output_mode="Selected Language")
    assert res == sample


def test_french_translation_mocked():
    from unittest.mock import MagicMock
    from services.foundry_client import MicrosoftFoundryClient

    mock_client = MagicMock(spec=MicrosoftFoundryClient)
    mock_client.is_configured = True
    mock_client.research_agent_name = "researchmate-gpt4-1-mini"
    mock_client.run_agent.return_value = {
        "success": True,
        "content": "L'attention est tout ce dont vous avez besoin [Paper 1, Page 2].",
    }

    translator = MultilingualTranslator(client=mock_client)
    sample = "Attention is all you need [Paper 1, Page 2]."
    res = translator.translate_text(sample, target_language="French")

    assert "L'attention est tout ce dont vous avez besoin" in res
    assert "[Paper 1, Page 2]" in res
    mock_client.run_agent.assert_called_once()


def test_hindi_translation_bilingual_mode():
    from unittest.mock import MagicMock
    from services.foundry_client import MicrosoftFoundryClient

    mock_client = MagicMock(spec=MicrosoftFoundryClient)
    mock_client.is_configured = True
    mock_client.research_agent_name = "researchmate-gpt4-1-mini"
    mock_client.run_agent.return_value = {
        "success": True,
        "content": "Hindi translated text [Evidence 1].",
    }

    translator = MultilingualTranslator(client=mock_client)
    sample = "English research findings [Evidence 1]."
    res = translator.translate_text(sample, target_language="Hindi", output_mode="Bilingual")

    assert "[English Original]" in res
    assert sample in res
    assert "[Hindi Translation]" in res
    assert "Hindi translated text" in res


def test_unconfigured_foundry_graceful_translation():
    from unittest.mock import MagicMock
    from services.foundry_client import MicrosoftFoundryClient

    mock_client = MagicMock(spec=MicrosoftFoundryClient)
    mock_client.is_configured = False

    translator = MultilingualTranslator(client=mock_client)
    sample = "Test analysis"
    res = translator.translate_text(sample, target_language="Spanish")

    assert "Translation unavailable" in res
    assert sample in res
