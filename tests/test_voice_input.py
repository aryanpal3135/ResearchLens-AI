"""
Unit and integration tests for Azure AI Speech Speech-to-Text (Voice Input) in ResearchLens AI.
Tests configuration, language resolution, recognition handling, error handling,
maximum utterance limits, and chat input pipeline integration.
"""

import unittest
from unittest.mock import MagicMock, patch
import azure.cognitiveservices.speech as speechsdk

from services.speech_service import (
    AzureSpeechService,
    SpeechConfigurationError,
    SpeechRecognitionError,
    SpeechNoMatchError,
    SpeechTooLongError,
    SpeechRecognitionResult,
    RECOGNITION_LANGUAGE_MAP,
)


class TestAzureSpeechToText(unittest.TestCase):
    """Test suite for AzureSpeechService Speech-to-Text capabilities."""

    def setUp(self):
        self.endpoint = "https://researchmate-resource.cognitiveservices.azure.com/"
        self.api_key = "test_speech_key_12345"
        self.region = "southeastasia"
        self.service = AzureSpeechService(
            endpoint=self.endpoint,
            api_key=self.api_key,
            region=self.region,
        )

    # 1. Missing Speech configuration
    def test_missing_configuration_raises_error(self):
        unconfigured_svc = AzureSpeechService(endpoint="", api_key="", region="")
        self.assertFalse(unconfigured_svc.is_configured)
        with self.assertRaises(SpeechConfigurationError) as ctx:
            unconfigured_svc.recognize_speech_from_audio(b"fake_audio_bytes")
        self.assertIn("Voice Input is not configured", str(ctx.exception))

    def test_missing_configuration_microphone_raises_error(self):
        unconfigured_svc = AzureSpeechService(endpoint="", api_key="", region="")
        with self.assertRaises(SpeechConfigurationError):
            unconfigured_svc.recognize_speech_from_microphone()

    # 2. Speech configuration loading
    def test_speech_configuration_loading(self):
        self.assertTrue(self.service.is_configured)
        self.assertEqual(self.service.endpoint, self.endpoint)
        self.assertEqual(self.service.api_key, self.api_key)
        self.assertEqual(self.service.region, self.region)

    # 3. Language mapping
    def test_language_mapping_values(self):
        self.assertEqual(self.service.get_recognition_language("English"), "en-US")
        self.assertEqual(self.service.get_recognition_language("Hindi"), "hi-IN")
        self.assertEqual(self.service.get_recognition_language("French"), "fr-FR")
        self.assertEqual(self.service.get_recognition_language("Spanish"), "es-ES")

    def test_language_mapping_locales_direct(self):
        self.assertEqual(self.service.get_recognition_language("en-US"), "en-US")
        self.assertEqual(self.service.get_recognition_language("hi-IN"), "hi-IN")
        self.assertEqual(self.service.get_recognition_language("fr-FR"), "fr-FR")
        self.assertEqual(self.service.get_recognition_language("es-ES"), "es-ES")

    def test_language_mapping_default_fallback(self):
        self.assertEqual(self.service.get_recognition_language("German"), "en-US")
        self.assertEqual(self.service.get_recognition_language(""), "en-US")
        self.assertEqual(self.service.get_recognition_language(None), "en-US")

    # 4. Recognized speech handling
    @patch("azure.cognitiveservices.speech.SpeechRecognizer")
    @patch("azure.cognitiveservices.speech.audio.AudioConfig")
    def test_recognized_speech_success(self, mock_audio_config, mock_recognizer_cls):
        mock_recognizer = MagicMock()
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.RecognizedSpeech
        mock_result.text = "What is the main contribution of this paper?"

        mock_async_op = MagicMock()
        mock_async_op.get.return_value = mock_result
        mock_recognizer.recognize_once_async.return_value = mock_async_op
        mock_recognizer_cls.return_value = mock_recognizer

        # Valid audio bytes (>100 bytes)
        fake_audio = b"RIFF" + b"\x00" * 200
        result = self.service.recognize_speech_from_audio(fake_audio, language="English")

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "RecognizedSpeech")
        self.assertEqual(result.text, "What is the main contribution of this paper?")
        self.assertIsNone(result.error_message)
        self.assertEqual(result.language, "en-US")

    # 5. NoMatch handling
    @patch("azure.cognitiveservices.speech.SpeechRecognizer")
    @patch("azure.cognitiveservices.speech.audio.AudioConfig")
    def test_nomatch_speech_handling(self, mock_audio_config, mock_recognizer_cls):
        mock_recognizer = MagicMock()
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.NoMatch
        mock_result.text = ""

        mock_async_op = MagicMock()
        mock_async_op.get.return_value = mock_result
        mock_recognizer.recognize_once_async.return_value = mock_async_op
        mock_recognizer_cls.return_value = mock_recognizer

        fake_audio = b"RIFF" + b"\x00" * 200
        result = self.service.recognize_speech_from_audio(fake_audio, language="English")

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "NoMatch")
        self.assertIn("Could not understand the speech. Please try again.", result.error_message)

    # 6. Canceled / error handling
    @patch("azure.cognitiveservices.speech.SpeechRecognizer")
    @patch("azure.cognitiveservices.speech.audio.AudioConfig")
    def test_canceled_error_handling(self, mock_audio_config, mock_recognizer_cls):
        mock_recognizer = MagicMock()
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.Canceled
        mock_cancellation = MagicMock()
        mock_cancellation.reason = speechsdk.CancellationReason.Error
        mock_cancellation.error_details = "Connection failed: DNS resolution failure."
        mock_result.cancellation_details = mock_cancellation

        mock_async_op = MagicMock()
        mock_async_op.get.return_value = mock_result
        mock_recognizer.recognize_once_async.return_value = mock_async_op
        mock_recognizer_cls.return_value = mock_recognizer

        fake_audio = b"RIFF" + b"\x00" * 200
        result = self.service.recognize_speech_from_audio(fake_audio, language="English")

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "Canceled")
        self.assertIn("Connection failed", result.error_message)

    # 7. Empty recognition result handling
    def test_empty_audio_data_returns_nomatch(self):
        result = self.service.recognize_speech_from_audio(b"", language="English")
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "NoMatch")
        self.assertIn("Could not understand the speech", result.error_message)

    @patch("azure.cognitiveservices.speech.SpeechRecognizer")
    @patch("azure.cognitiveservices.speech.audio.AudioConfig")
    def test_empty_text_in_recognized_speech_returns_nomatch(self, mock_audio_config, mock_recognizer_cls):
        mock_recognizer = MagicMock()
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.RecognizedSpeech
        mock_result.text = "   "  # whitespace only

        mock_async_op = MagicMock()
        mock_async_op.get.return_value = mock_result
        mock_recognizer.recognize_once_async.return_value = mock_async_op
        mock_recognizer_cls.return_value = mock_recognizer

        fake_audio = b"RIFF" + b"\x00" * 200
        result = self.service.recognize_speech_from_audio(fake_audio, language="English")

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "NoMatch")
        self.assertEqual(result.text, "")

    # 8-11. Multilingual recognition targets
    @patch("azure.cognitiveservices.speech.SpeechRecognizer")
    @patch("azure.cognitiveservices.speech.audio.AudioConfig")
    def test_hindi_speech_recognition(self, mock_audio_config, mock_recognizer_cls):
        mock_recognizer = MagicMock()
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.RecognizedSpeech
        mock_result.text = "इस पेपर का मुख्य योगदान क्या है?"

        mock_async_op = MagicMock()
        mock_async_op.get.return_value = mock_result
        mock_recognizer.recognize_once_async.return_value = mock_async_op
        mock_recognizer_cls.return_value = mock_recognizer

        fake_audio = b"RIFF" + b"\x00" * 200
        result = self.service.recognize_speech_from_audio(fake_audio, language="Hindi")

        self.assertTrue(result.success)
        self.assertEqual(result.language, "hi-IN")
        self.assertEqual(result.text, "इस पेपर का मुख्य योगदान क्या है?")

    @patch("azure.cognitiveservices.speech.SpeechRecognizer")
    @patch("azure.cognitiveservices.speech.audio.AudioConfig")
    def test_french_speech_recognition(self, mock_audio_config, mock_recognizer_cls):
        mock_recognizer = MagicMock()
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.RecognizedSpeech
        mock_result.text = "Quelle est la principale contribution de cet article ?"

        mock_async_op = MagicMock()
        mock_async_op.get.return_value = mock_result
        mock_recognizer.recognize_once_async.return_value = mock_async_op
        mock_recognizer_cls.return_value = mock_recognizer

        fake_audio = b"RIFF" + b"\x00" * 200
        result = self.service.recognize_speech_from_audio(fake_audio, language="French")

        self.assertTrue(result.success)
        self.assertEqual(result.language, "fr-FR")
        self.assertIn("contribution", result.text)

    @patch("azure.cognitiveservices.speech.SpeechRecognizer")
    @patch("azure.cognitiveservices.speech.audio.AudioConfig")
    def test_spanish_speech_recognition(self, mock_audio_config, mock_recognizer_cls):
        mock_recognizer = MagicMock()
        mock_result = MagicMock()
        mock_result.reason = speechsdk.ResultReason.RecognizedSpeech
        mock_result.text = "¿Cuál es la principal contribución de este artículo?"

        mock_async_op = MagicMock()
        mock_async_op.get.return_value = mock_result
        mock_recognizer.recognize_once_async.return_value = mock_async_op
        mock_recognizer_cls.return_value = mock_recognizer

        fake_audio = b"RIFF" + b"\x00" * 200
        result = self.service.recognize_speech_from_audio(fake_audio, language="Spanish")

        self.assertTrue(result.success)
        self.assertEqual(result.language, "es-ES")
        self.assertIn("contribución", result.text)

    # 12. Maximum utterance handling
    def test_maximum_utterance_too_long_handling(self):
        # Create oversized payload exceeding max duration (e.g. > 3MB)
        oversized_audio = b"\x00" * (46 * 65000)
        result = self.service.recognize_speech_from_audio(oversized_audio, language="English", max_duration_sec=45.0)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "TooLong")
        self.assertIn("Your voice prompt was too long. Please try a shorter prompt.", result.error_message)

    # 13. Microphone hardware failure handling
    @patch("azure.cognitiveservices.speech.audio.AudioConfig")
    def test_microphone_hardware_exception_handling(self, mock_audio_config):
        mock_audio_config.side_effect = RuntimeError("No audio capture device found.")
        result = self.service.recognize_speech_from_microphone(language="English")

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "MicrophoneError")
        self.assertIn("Microphone access error", result.error_message)


if __name__ == "__main__":
    unittest.main()
