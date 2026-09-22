"""
Comprehensive unit and integration test suite for Azure AI Speech (Text-to-Speech / Read Aloud).
Tests:
1. Missing Speech configuration
2. Endpoint configuration
3. Voice mapping (canonical voices)
4. English voice resolution
5. Hindi voice resolution
6. French voice resolution
7. Spanish voice resolution
8. Citation cleaning (formal citations, page ranges, section names, simple paper IDs, internal candidate ID removal)
9. Markdown cleaning (headers, bold, italics, code blocks, HTML, bullets, emojis)
10. Speech failure and cancellation handling
11. Long-answer splitting & multi-chunk handling
"""

import pytest
from unittest.mock import MagicMock, patch

from services.speech_service import (
    AzureSpeechService,
    SpeechConfigurationError,
    SpeechSynthesisError,
    LANGUAGE_VOICE_MAP,
)


class TestSpeechConfiguration:
    """Test suite for Speech credentials and endpoint configuration."""

    def test_missing_speech_configuration(self):
        """Verifies that missing credentials mark service as unconfigured."""
        service = AzureSpeechService(endpoint="", api_key="")
        assert not service.is_configured
        with pytest.raises(SpeechConfigurationError) as exc_info:
            service.synthesize_speech("Hello research world.")
        assert "Read Aloud is not configured" in str(exc_info.value)

    def test_missing_api_key_only(self):
        """Verifies that missing API key raises SpeechConfigurationError."""
        service = AzureSpeechService(endpoint="https://valid-endpoint.azure.com/", api_key="")
        assert not service.is_configured
        with pytest.raises(SpeechConfigurationError):
            service.synthesize_speech("Test text")

    def test_endpoint_configuration(self):
        """Verifies custom and default endpoint storage."""
        custom_ep = "https://custom-speech.cognitiveservices.azure.com/"
        service = AzureSpeechService(endpoint=custom_ep, api_key="test_key_123")
        assert service.is_configured
        assert service.endpoint == custom_ep
        assert service.api_key == "test_key_123"


class TestVoiceMapping:
    """Test suite for language to neural voice resolution."""

    def test_voice_mapping_complete(self):
        """Ensures all 4 primary languages are present in LANGUAGE_VOICE_MAP."""
        assert "English" in LANGUAGE_VOICE_MAP
        assert "Hindi" in LANGUAGE_VOICE_MAP
        assert "French" in LANGUAGE_VOICE_MAP
        assert "Spanish" in LANGUAGE_VOICE_MAP

    def test_english_voice(self):
        """Verifies English resolves to the verified DragonHD neural voice."""
        service = AzureSpeechService(api_key="key")
        assert service.get_voice_for_language("English") == "en-US-Ava:DragonHDLatestNeural"
        assert service.get_voice_for_language("english") == "en-US-Ava:DragonHDLatestNeural"

    def test_hindi_voice(self):
        """Verifies Hindi resolves to the canonical neural voice."""
        service = AzureSpeechService(api_key="key")
        assert service.get_voice_for_language("Hindi") == "hi-IN-SwaraNeural"
        assert service.get_voice_for_language("hindi") == "hi-IN-SwaraNeural"

    def test_french_voice(self):
        """Verifies French resolves to the canonical neural voice."""
        service = AzureSpeechService(api_key="key")
        assert service.get_voice_for_language("French") == "fr-FR-DeniseNeural"
        assert service.get_voice_for_language("french") == "fr-FR-DeniseNeural"

    def test_spanish_voice(self):
        """Verifies Spanish resolves to the canonical neural voice."""
        service = AzureSpeechService(api_key="key")
        assert service.get_voice_for_language("Spanish") == "es-ES-ElviraNeural"
        assert service.get_voice_for_language("spanish") == "es-ES-ElviraNeural"

    def test_unknown_language_fallback(self):
        """Verifies unlisted language falls back safely to English voice."""
        service = AzureSpeechService(api_key="key")
        assert service.get_voice_for_language("German") == "en-US-Ava:DragonHDLatestNeural"


class TestCitationCleaning:
    """Test suite for converting academic citations to natural spoken text."""

    def setup_method(self):
        self.service = AzureSpeechService(api_key="key")

    def test_formal_citation_with_page(self):
        """Converts [Paper 1, Page 4] to natural speech."""
        text = "The model achieves strong results [Paper 1, Page 4]."
        cleaned = self.service.clean_text_for_speech(text)
        assert "Paper 1, page 4." in cleaned
        assert "[" not in cleaned
        assert "]" not in cleaned

    def test_formal_citation_with_page_range(self):
        """Converts [Paper 1, Pages 10-12] into natural speech."""
        text = "Detailed hyperparameter analysis is provided [Paper 1, Pages 10-12]."
        cleaned = self.service.clean_text_for_speech(text)
        assert "Paper 1, pages 10 to 12." in cleaned

    def test_formal_citation_with_section(self):
        """Converts [paper_001, Page 2, §Abstract] into natural speech."""
        text = "The authors propose the Transformer architecture [paper_001, Page 2, §Abstract]."
        cleaned = self.service.clean_text_for_speech(text)
        assert "Paper 1, page 2, section Abstract." in cleaned

    def test_simple_paper_id_citation(self):
        """Converts [Paper 1] or [paper_002] into natural speech."""
        text = "As established by [Paper 1] and later confirmed by [paper_002]."
        cleaned = self.service.clean_text_for_speech(text)
        assert "Paper 1." in cleaned
        assert "Paper 2." in cleaned

    def test_removes_internal_candidate_ids(self):
        """Removes internal database IDs like [rq_001] or [gap_002]."""
        text = "This addresses the primary research gap [gap_001] and raises question [rq_002]."
        cleaned = self.service.clean_text_for_speech(text)
        assert "[gap_001]" not in cleaned
        assert "[rq_002]" not in cleaned


class TestMarkdownCleaning:
    """Test suite for stripping presentation formatting while preserving content."""

    def setup_method(self):
        self.service = AzureSpeechService(api_key="key")

    def test_headers_removed(self):
        """Removes #, ##, ### header markup."""
        text = "### 1. Introduction\n#### Methodological Details\nContent here."
        cleaned = self.service.clean_text_for_speech(text)
        assert not cleaned.startswith("#")
        assert "1. Introduction" in cleaned
        assert "Methodological Details" in cleaned

    def test_bold_and_italics(self):
        """Preserves inner text while removing bold and italic asterisks."""
        text = "This **crucial finding** was *consistently demonstrated* across datasets."
        cleaned = self.service.clean_text_for_speech(text)
        assert "crucial finding" in cleaned
        assert "consistently demonstrated" in cleaned
        assert "**" not in cleaned
        assert "*" not in cleaned

    def test_html_tags_removed(self):
        """Strips HTML presentation tags."""
        text = "<div style='color: red;'>Warning: <span>High latency</span></div>"
        cleaned = self.service.clean_text_for_speech(text)
        assert "<div" not in cleaned
        assert "</span>" not in cleaned
        assert "Warning: High latency" in cleaned

    def test_code_blocks_removed(self):
        """Removes code blocks which sound unreadable in TTS."""
        text = "Here is the implementation:\n```python\ndef attention():\n    return 42\n```\nExplanation follows."
        cleaned = self.service.clean_text_for_speech(text)
        assert "def attention" not in cleaned
        assert "Explanation follows." in cleaned

    def test_emojis_and_bullets_removed(self):
        """Removes decorative emojis and list bullet characters."""
        text = "• 🔬 First finding\n- 🚀 Second finding\n* 💡 Third insight"
        cleaned = self.service.clean_text_for_speech(text)
        assert "🔬" not in cleaned
        assert "🚀" not in cleaned
        assert "💡" not in cleaned
        assert "First finding" in cleaned
        assert "Second finding" in cleaned
        assert "Third insight" in cleaned


class TestLongAnswerHandling:
    """Test suite for splitting long academic responses into sentence-bounded chunks."""

    def setup_method(self):
        self.service = AzureSpeechService(api_key="key")

    def test_short_answer_not_split(self):
        """Answers under max_chunk_chars remain as a single chunk."""
        text = "A concise academic answer of 100 characters for testing purposes."
        chunks = self.service.split_text_into_chunks(text, max_chunk_chars=3000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_answer_paragraph_split(self):
        """Splits multi-paragraph long answers at paragraph boundaries."""
        p1 = "Paragraph 1 with evidence and citations. " * 30  # ~1200 chars
        p2 = "Paragraph 2 discussing subsequent experiments. " * 30  # ~1400 chars
        p3 = "Paragraph 3 summarizing conclusions and future work. " * 30  # ~1500 chars
        full_text = f"{p1}\n\n{p2}\n\n{p3}"
        chunks = self.service.split_text_into_chunks(full_text, max_chunk_chars=2000)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 2500  # Within tolerance


class TestSpeechFailureHandling:
    """Test suite for handling errors and cancellation cleanly."""

    def test_empty_cleaned_text_raises_error(self):
        """Verifies empty or pure markup text raises SpeechSynthesisError."""
        service = AzureSpeechService(endpoint="https://dummy.cognitiveservices.azure.com/", api_key="dummy_key")
        with pytest.raises(SpeechSynthesisError) as exc_info:
            service.synthesize_speech("```python\n# only code\n```")
        assert "empty after removing presentation markup" in str(exc_info.value)

    @patch("azure.cognitiveservices.speech.SpeechSynthesizer")
    def test_speech_canceled_handling(self, mock_synth_cls):
        """Verifies that Azure Speech cancellation details are extracted into SpeechSynthesisError."""
        service = AzureSpeechService(endpoint="https://dummy.cognitiveservices.azure.com/", api_key="dummy_key")

        mock_result = MagicMock()
        import azure.cognitiveservices.speech as speechsdk
        mock_result.reason = speechsdk.ResultReason.Canceled
        mock_result.cancellation_details.error_details = "Authentication failed (401)"

        mock_synth = MagicMock()
        mock_synth.speak_text_async.return_value.get.return_value = mock_result
        mock_synth_cls.return_value = mock_synth

        with pytest.raises(SpeechSynthesisError) as exc_info:
            service.synthesize_speech("Valid academic text.")
        assert "Authentication failed (401)" in str(exc_info.value)
