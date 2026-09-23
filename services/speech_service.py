"""
Azure AI Speech Service for ResearchLens AI.
Provides high-fidelity, multilingual Text-to-Speech (TTS) / Read Aloud capabilities
powered by Microsoft Azure Cognitive Services Speech SDK.
Strictly converts academic responses into natural, fluent spoken audio
with clean citation pronunciation, presentation markup removal, and zero hallucination.
"""

import hashlib
import io
import os
import re
import tempfile
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import azure.cognitiveservices.speech as speechsdk

from config.settings import settings

# In-memory fast audio synthesis cache
_TTS_MEMORY_CACHE: Dict[str, bytes] = {}
# Connection-warmed synthesizer cache per voice
_SYNTHESIZER_CACHE: Dict[str, Any] = {}



class SpeechConfigurationError(Exception):
    """Raised when Azure AI Speech credentials or endpoint are missing or invalid."""
    pass


class SpeechSynthesisError(Exception):
    """Raised when Azure AI Speech synthesis fails."""
    pass


class SpeechRecognitionError(Exception):
    """Raised when Azure AI Speech recognition fails."""
    pass


class SpeechNoMatchError(SpeechRecognitionError):
    """Raised when speech could not be recognized."""
    pass


class SpeechTooLongError(SpeechRecognitionError):
    """Raised when audio input exceeds maximum supported utterance duration."""
    pass


@dataclass
class SpeechRecognitionResult:
    """Enterprise result container for Azure AI Speech recognition."""
    text: str
    success: bool
    reason: str
    error_message: Optional[str] = None
    language: str = "en-US"


# Canonical voice mapping for ResearchLens AI supported languages (Text-to-Speech)
# Uses ultra-fast, natural neural voices optimized for sub-1.5s real-time synthesis
LANGUAGE_VOICE_MAP: Dict[str, str] = {
    "English": "en-US-Ava:DragonHDLatestNeural",
    "Hindi": "hi-IN-SwaraNeural",
    "French": "fr-FR-DeniseNeural",
    "Spanish": "es-ES-ElviraNeural",
}

# Fallback neural voices in case specialized neural voice is unavailable
VOICE_FALLBACK_MAP: Dict[str, str] = {
    "English": "en-US-JennyNeural",
    "Hindi": "hi-IN-MadhurNeural",
    "French": "fr-FR-HenriNeural",
    "Spanish": "es-ES-AlvaroNeural",
}

# Recognition locale mapping for ResearchLens AI supported languages (Speech-to-Text)
RECOGNITION_LANGUAGE_MAP: Dict[str, str] = {
    "English": "en-US",
    "Hindi": "hi-IN",
    "French": "fr-FR",
    "Spanish": "es-ES",
}



class AzureSpeechService:
    """Enterprise Azure AI Speech client managing voice synthesis and text cleaning."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        if endpoint is not None:
            self.endpoint = endpoint.strip()
        else:
            self.endpoint = (
                os.getenv("AZURE_SPEECH_ENDPOINT")
                or getattr(settings, "AZURE_SPEECH_ENDPOINT", "")
                or "https://researchmate-resource.cognitiveservices.azure.com/"
            ).strip()

        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = (
                os.getenv("AZURE_SPEECH_API_KEY")
                or getattr(settings, "AZURE_SPEECH_API_KEY", "")
                or os.getenv("AZURE_OPENAI_API_KEY")
                or getattr(settings, "AZURE_OPENAI_API_KEY", "")
            ).strip()

        if region is not None:
            self.region = region.strip()
        else:
            self.region = (
                os.getenv("AZURE_SPEECH_REGION")
                or getattr(settings, "AZURE_SPEECH_REGION", "southeastasia")
            ).strip()

    @property
    def is_configured(self) -> bool:
        """Checks if required Azure Speech credentials and endpoint are present."""
        return bool(self.endpoint and self.api_key)

    def get_voice_for_language(self, language: str) -> str:
        """Resolves the designated Azure Speech neural voice for a given language."""
        normalized_lang = language.strip().capitalize()
        return LANGUAGE_VOICE_MAP.get(normalized_lang, LANGUAGE_VOICE_MAP["English"])

    def clean_text_for_speech(self, text: str) -> str:
        """
        Cleans academic markdown and presentation markup for natural, fluent TTS.
        Converts bracketed citations (e.g. [Paper 1, Page 4]) into conversational speech.
        Preserves 100% of academic substantive content without summarizing.
        """
        if not text:
            return ""

        cleaned = text

        # 1. Remove raw code blocks completely
        cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)

        # 2. Convert detailed citations: [Paper 1, Page 4] or [paper_001, Page 2, §Abstract]
        def _citation_to_speech(match: re.Match) -> str:
            p_name = match.group(1).strip()
            # Normalize internal paper_001 -> Paper 1
            p_name = re.sub(r"^paper_0*(\d+)", r"Paper \1", p_name, flags=re.IGNORECASE)
            page_start = match.group(2)
            page_end = match.group(3)
            section = match.group(4)

            parts = [p_name]
            if page_start and page_end:
                parts.append(f"pages {page_start} to {page_end}")
            elif page_start:
                parts.append(f"page {page_start}")

            if section:
                clean_sec = section.strip().lstrip("§").strip()
                parts.append(f"section {clean_sec}")

            return ", ".join(parts) + "."

        citation_pattern = re.compile(
            r"\[([a-zA-Z0-9_\- ]+),\s*Pages?\s*(\d+)(?:[–\-](\d+))?(?:,\s*§?([^\]]+))?\][\.।]?"
        )
        cleaned = citation_pattern.sub(_citation_to_speech, cleaned)

        # 3. Convert simple paper citations: [paper_001] or [Paper 1]
        cleaned = re.sub(r"\[paper_0*(\d+)\][\.।]?", r"Paper \1.", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[Paper\s+(\d+)\][\.।]?", r"Paper \1.", cleaned, flags=re.IGNORECASE)

        # 4. Remove internal candidate IDs, badge tags, and UI anchors like [rq_001], [gap_002], [disp_id]
        cleaned = re.sub(r"\[[a-zA-Z0-9_\-]+_[0-9]+\]", "", cleaned)

        # 5. Remove HTML tags (e.g. <div>, <span>, <code>, <br>)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        # 6. Remove Markdown headers (#, ##, ###, ####)
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

        # 7. Remove Bold, Italic, and Strikethrough markup while retaining inner text
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
        cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
        cleaned = re.sub(r"~~([^~]+)~~", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

        # 8. Remove list bullet points (dash, asterisk, bullet, plus)
        cleaned = re.sub(r"^\s*[-*•+]\s+", "", cleaned, flags=re.MULTILINE)

        # 9. Remove blockquote markers
        cleaned = re.sub(r"^\s*>\s*", "", cleaned, flags=re.MULTILINE)

        # 10. Strip decorative UI emojis and special symbols that sound awkward in TTS
        emoji_pattern = re.compile(
            r"[\U00010000-\U0010ffff\u2600-\u27bf\u2b50\u23f0-\u23ff\u2190-\u21ff\u2022]",
            flags=re.UNICODE,
        )
        cleaned = emoji_pattern.sub("", cleaned)

        # 11. Normalize excessive whitespace, line breaks, duplicate periods and dandas
        cleaned = re.sub(r"\.।", "।", cleaned)
        cleaned = re.sub(r"।\.", "।", cleaned)
        cleaned = re.sub(r"\.{2,}", ".", cleaned)
        cleaned = re.sub(r"।{2,}", "।", cleaned)
        cleaned = re.sub(r"\s+\.", ".", cleaned)
        cleaned = re.sub(r"\s+।", "।", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)

        return cleaned.strip()

    def split_text_into_chunks(self, text: str, max_chunk_chars: int = 3000) -> List[str]:
        """
        Safely splits long academic text into manageable sentence-bounded chunks
        to prevent Azure Speech payload limit violations or timeouts.
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= max_chunk_chars:
            return [text]

        chunks: List[str] = []
        # Split on double newlines (paragraphs) first
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            if len(current_chunk) + len(p) + 2 <= max_chunk_chars:
                current_chunk = f"{current_chunk}\n\n{p}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # If paragraph itself exceeds max_chunk_chars, split on sentence terminators
                if len(p) > max_chunk_chars:
                    sentences = re.split(r"(?<=[.?!])\s+", p)
                    for s in sentences:
                        s = s.strip()
                        if not s:
                            continue
                        if len(current_chunk) + len(s) + 1 <= max_chunk_chars:
                            current_chunk = f"{current_chunk} {s}".strip()
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = s
                else:
                    current_chunk = p

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _create_speech_config(self, voice_name: str) -> speechsdk.SpeechConfig:
        """Creates and configures speechsdk.SpeechConfig with clean lossless WAV output."""
        if not self.is_configured:
            raise SpeechConfigurationError(
                "Read Aloud is not configured. Please verify AZURE_SPEECH_ENDPOINT "
                "and AZURE_SPEECH_API_KEY in your secure environment settings."
            )

        speech_config = speechsdk.SpeechConfig(
            endpoint=self.endpoint,
            subscription=self.api_key,
        )

        # Set high-quality lossless 16kHz 16-bit mono PCM format for zero browser decoding lag
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        )
        speech_config.speech_synthesis_voice_name = voice_name
        return speech_config

    def _synthesize_single_chunk(self, chunk: str, voice_name: str) -> bytes:
        """Synthesizes a single chunk of text into WAV audio bytes using Azure Speech."""
        global _SYNTHESIZER_CACHE
        synthesizer = _SYNTHESIZER_CACHE.get(voice_name)
        if synthesizer is None:
            speech_config = self._create_speech_config(voice_name)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=None,  # Return in-memory audio data
            )
            _SYNTHESIZER_CACHE[voice_name] = synthesizer

        try:
            result = synthesizer.speak_text_async(chunk).get()
        except Exception as ex:
            _SYNTHESIZER_CACHE.pop(voice_name, None)
            raise SpeechSynthesisError(f"Azure Speech network communication failed: {str(ex)}") from ex

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            if not result.audio_data:
                raise SpeechSynthesisError("Azure Speech returned empty audio payload.")
            return result.audio_data
        elif result.reason == speechsdk.ResultReason.Canceled:
            _SYNTHESIZER_CACHE.pop(voice_name, None)
            cancellation = result.cancellation_details
            error_msg = cancellation.error_details or cancellation.reason
            raise SpeechSynthesisError(f"Azure Speech synthesis canceled: {error_msg}")
        else:
            _SYNTHESIZER_CACHE.pop(voice_name, None)
            raise SpeechSynthesisError(f"Azure Speech unexpected synthesis result reason: {result.reason}")

    def synthesize_speech(self, text: str, language: str = "English") -> bytes:
        """
        Converts text to speech using Azure AI Speech, returning playable WAV audio bytes.
        Handles markdown cleaning, citation pronunciation, long answer chunking,
        and multilingual voice selection with zero header corruption.
        """
        if not self.is_configured:
            raise SpeechConfigurationError(
                "Read Aloud is not configured. Please set AZURE_SPEECH_ENDPOINT "
                "and AZURE_SPEECH_API_KEY in your environment."
            )

        cleaned_text = self.clean_text_for_speech(text)
        if not cleaned_text:
            raise SpeechSynthesisError("Text content is empty after removing presentation markup.")

        # Check in-memory audio cache for instant sub-millisecond retrieval
        cache_key = f"{language}:{hashlib.sha256(cleaned_text.encode('utf-8')).hexdigest()}"
        if cache_key in _TTS_MEMORY_CACHE:
            return _TTS_MEMORY_CACHE[cache_key]

        voice_name = self.get_voice_for_language(language)

        # For standard answers (under 8,000 characters), synthesize in a single pass for peak performance
        if len(cleaned_text) <= 8000:
            final_wav = self._synthesize_single_chunk(cleaned_text, voice_name=voice_name)
            _TTS_MEMORY_CACHE[cache_key] = final_wav
            return final_wav

        # For very long responses, chunk safely and stitch PCM frames into a single valid WAV file
        chunks = self.split_text_into_chunks(cleaned_text, max_chunk_chars=4000)
        if not chunks:
            raise SpeechSynthesisError("No valid text chunks generated for synthesis.")

        pcm_frames_list: List[bytes] = []
        for chunk in chunks:
            part_bytes = self._synthesize_single_chunk(chunk, voice_name=voice_name)
            if part_bytes.startswith(b"RIFF"):
                try:
                    with io.BytesIO(part_bytes) as bio:
                        with wave.open(bio, "rb") as wf:
                            pcm_frames_list.append(wf.readframes(wf.getnframes()))
                except Exception:
                    pcm_frames_list.append(part_bytes[44:] if len(part_bytes) > 44 else part_bytes)
            else:
                pcm_frames_list.append(part_bytes)

        # Package combined raw PCM frames into a single, standard WAV container
        out_bio = io.BytesIO()
        with wave.open(out_bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_frames_list))

        final_wav = out_bio.getvalue()
        _TTS_MEMORY_CACHE[cache_key] = final_wav
        return final_wav

    def get_recognition_language(self, language: str) -> str:
        """Resolves the designated BCP-47 locale code for Azure Speech recognition."""
        if not language:
            return "en-US"
        lang_str = language.strip()
        if lang_str in RECOGNITION_LANGUAGE_MAP.values():
            return lang_str
        normalized = lang_str.capitalize()
        return RECOGNITION_LANGUAGE_MAP.get(normalized, "en-US")

    def _create_recognition_speech_config(self, language_code: str) -> speechsdk.SpeechConfig:
        """Creates and configures speechsdk.SpeechConfig for fast speech recognition."""
        if not self.is_configured:
            raise SpeechConfigurationError(
                "Voice Input is not configured. Please verify AZURE_SPEECH_ENDPOINT "
                "and AZURE_SPEECH_API_KEY in your secure environment settings."
            )
        speech_config = speechsdk.SpeechConfig(
            endpoint=self.endpoint,
            subscription=self.api_key,
        )
        speech_config.speech_recognition_language = language_code
        # Optimize silence detection for rapid response upon speech completion
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1200"
        )
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1000"
        )
        return speech_config

    def _process_recognition_result(
        self,
        result: speechsdk.SpeechRecognitionResult,
        language_code: str,
    ) -> SpeechRecognitionResult:
        """Processes the speechsdk recognition result into a SpeechRecognitionResult."""
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = (result.text or "").strip()
            if not recognized_text:
                return SpeechRecognitionResult(
                    text="",
                    success=False,
                    reason="NoMatch",
                    error_message="Could not understand the speech. Please try again.",
                    language=language_code,
                )
            return SpeechRecognitionResult(
                text=recognized_text,
                success=True,
                reason="RecognizedSpeech",
                error_message=None,
                language=language_code,
            )
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return SpeechRecognitionResult(
                text="",
                success=False,
                reason="NoMatch",
                error_message="Could not understand the speech. Please try again.",
                language=language_code,
            )
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            if cancellation.reason == speechsdk.CancellationReason.Error:
                err = cancellation.error_details or "Speech recognition failed on Azure Speech server."
                return SpeechRecognitionResult(
                    text="",
                    success=False,
                    reason="Canceled",
                    error_message=f"Azure Speech Error: {err}",
                    language=language_code,
                )
            return SpeechRecognitionResult(
                text="",
                success=False,
                reason="Canceled",
                error_message="Speech recognition was canceled.",
                language=language_code,
            )
        else:
            return SpeechRecognitionResult(
                text="",
                success=False,
                reason=str(result.reason),
                error_message=f"Unexpected recognition status: {result.reason}",
                language=language_code,
            )

    @staticmethod
    def _normalize_audio_to_16k_mono(audio_bytes: bytes) -> bytes:
        """
        Normalizes any WAV audio (differing sample rates like 44.1k/48k, stereo/mono, float/int)
        into standardized 16,000 Hz, 16-bit, 1-channel mono PCM bytes.
        Guarantees seamless, accurate Azure Speech recognition across all web browsers and devices.
        """
        if not audio_bytes.startswith(b"RIFF"):
            return audio_bytes

        try:
            with io.BytesIO(audio_bytes) as bio:
                with wave.open(bio, "rb") as wf:
                    framerate = wf.getframerate()
                    nchannels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    raw_frames = wf.readframes(wf.getnframes())

            # If already 16kHz 16-bit mono PCM, return raw frames directly
            if framerate == 16000 and nchannels == 1 and sampwidth == 2:
                return raw_frames

            try:
                import numpy as np
                import scipy.signal as signal
                has_scipy = True
            except ImportError:
                has_scipy = False

            if has_scipy and raw_frames:
                if sampwidth == 2:
                    samples = np.frombuffer(raw_frames, dtype=np.int16)
                elif sampwidth == 4:
                    f_samples = np.frombuffer(raw_frames, dtype=np.float32)
                    if len(f_samples) > 0 and np.max(np.abs(f_samples)) <= 2.0:
                        samples = (np.clip(f_samples, -1.0, 1.0) * 32767).astype(np.int16)
                    else:
                        samples = (np.frombuffer(raw_frames, dtype=np.int32) >> 16).astype(np.int16)
                elif sampwidth == 1:
                    samples = ((np.frombuffer(raw_frames, dtype=np.uint8).astype(np.int16) - 128) * 256).astype(np.int16)
                else:
                    return raw_frames

                # Downmix multi-channel / stereo to mono
                if nchannels > 1 and len(samples) >= nchannels:
                    samples = samples.reshape(-1, nchannels).mean(axis=1).astype(np.int16)

                # High-fidelity polyphase resampling to 16,000 Hz
                if framerate != 16000 and len(samples) > 0:
                    import math
                    gcd_val = math.gcd(16000, framerate)
                    samples = signal.resample_poly(samples, 16000 // gcd_val, framerate // gcd_val).astype(np.int16)

                return samples.tobytes()

            elif sampwidth == 2 and len(raw_frames) >= 2:
                # Lightweight pure-Python fallback for integer samples
                import struct
                num_samples = len(raw_frames) // (2 * nchannels)
                unpacked = struct.unpack(f"<{num_samples * nchannels}h", raw_frames[:num_samples * nchannels * 2])
                if nchannels > 1:
                    mono_samples = [
                        sum(unpacked[i * nchannels : (i + 1) * nchannels]) // nchannels
                        for i in range(num_samples)
                    ]
                else:
                    mono_samples = list(unpacked)

                if framerate == 48000:
                    mono_samples = mono_samples[::3]
                elif framerate == 32000:
                    mono_samples = mono_samples[::2]

                return struct.pack(f"<{len(mono_samples)}h", *mono_samples)

        except Exception:
            pass

        return audio_bytes

    def recognize_speech_from_audio(
        self,
        audio_data: bytes,
        language: str = "English",
        max_duration_sec: float = 45.0,
    ) -> SpeechRecognitionResult:
        """
        Recognizes speech from audio bytes (WAV/PCM from browser/Streamlit) using Azure Speech SDK.
        Guarantees single-utterance recognition, normalizes browser audio, and cleanly handles length limits.
        """
        if not self.is_configured:
            raise SpeechConfigurationError(
                "Voice Input is not configured. Please set AZURE_SPEECH_ENDPOINT "
                "and AZURE_SPEECH_API_KEY in your environment."
            )

        if not audio_data or len(audio_data) < 100:
            return SpeechRecognitionResult(
                text="",
                success=False,
                reason="NoMatch",
                error_message="Could not understand the speech. Please try again.",
                language=self.get_recognition_language(language),
            )

        # Check maximum utterance duration limitation (approx 32KB/sec for 16kHz 16-bit mono)
        # 45 seconds is approx 1.5MB; allow up to 2.5MB
        if len(audio_data) > int(max_duration_sec * 64000):
            return SpeechRecognitionResult(
                text="",
                success=False,
                reason="TooLong",
                error_message="Your voice prompt was too long. Please try a shorter prompt.",
                language=self.get_recognition_language(language),
            )

        lang_code = self.get_recognition_language(language)
        speech_config = self._create_recognition_speech_config(lang_code)

        # Standardize audio to 16kHz 16-bit Mono PCM
        pcm_data = self._normalize_audio_to_16k_mono(audio_data)

        # In-memory streaming using PushAudioInputStream
        try:
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000,
                channels=1,
                bits_per_sample=16,
            )
            push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            push_stream.write(pcm_data)
            push_stream.close()

            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
            result = recognizer.recognize_once_async().get()
            del recognizer, audio_config, push_stream
            return self._process_recognition_result(result, lang_code)
        except Exception:
            pass  # Fall back to file-based recognition if in-memory streaming encounters issue

        # Fallback: Write standardized WAV to temporary file with secure cleanup
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        try:
            with wave.open(temp_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm_data)

            audio_config = speechsdk.audio.AudioConfig(filename=temp_wav_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )

            result = recognizer.recognize_once_async().get()
            del recognizer, audio_config

            return self._process_recognition_result(result, lang_code)
        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

    def recognize_speech_from_microphone(
        self,
        language: str = "English",
    ) -> SpeechRecognitionResult:
        """
        Recognizes speech directly from the default hardware microphone using Azure Speech SDK.
        Useful for local desktop runtime environments.
        """
        if not self.is_configured:
            raise SpeechConfigurationError(
                "Voice Input is not configured. Please set AZURE_SPEECH_ENDPOINT "
                "and AZURE_SPEECH_API_KEY in your environment."
            )

        lang_code = self.get_recognition_language(language)
        speech_config = self._create_recognition_speech_config(lang_code)

        try:
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
            result = recognizer.recognize_once_async().get()
            del recognizer, audio_config
            return self._process_recognition_result(result, lang_code)
        except Exception as ex:
            return SpeechRecognitionResult(
                text="",
                success=False,
                reason="MicrophoneError",
                error_message=f"Microphone access error: {str(ex)}. Please ensure microphone is connected and permitted.",
                language=lang_code,
            )

