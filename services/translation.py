"""
Multilingual Translation Service for ResearchLens AI.
Strictly supports exactly four languages: English, Hindi, French, Spanish.
Supports Selected Language and Bilingual output modes without mutating original citations.
Designed for Phase 11 implementation.
"""

from typing import Dict, Any
from config.settings import settings


from typing import Dict, Any, Optional
from config.settings import settings
from services.foundry_client import MicrosoftFoundryClient


class MultilingualTranslator:
    """
    Academic translation service connecting to Microsoft Foundry (researchmate-gpt4-1-mini).
    Strictly supports exactly four languages: English, Hindi, French, Spanish.
    Preserves exact citations, page references, mathematical notation, and technical terms.
    """

    SUPPORTED_LANGUAGES = ["English", "Hindi", "French", "Spanish"]

    def __init__(self, client: Optional[MicrosoftFoundryClient] = None):
        self.supported = self.SUPPORTED_LANGUAGES
        self.client = client or MicrosoftFoundryClient()

    def translate_text(
        self,
        text: str,
        target_language: str,
        output_mode: str = "Selected Language",
    ) -> str:
        """
        Translates AI-generated explanations and outputs into the target language.
        Preserves original citation markers, page numbers, and technical terms intact.
        """
        if target_language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{target_language}' is not supported. Supported: {self.SUPPORTED_LANGUAGES}")

        clean_text = text.strip()
        if not clean_text:
            return ""

        if target_language == "English":
            return clean_text

        # If Foundry is not configured, report clear diagnostic message
        if not getattr(self.client, "is_configured", False):
            if output_mode == "Bilingual":
                return f"[English Original]\n{clean_text}\n\n[{target_language}]\n⚠️ Translation unavailable: Microsoft Foundry is not configured."
            return f"⚠️ Translation unavailable: Microsoft Foundry is not configured.\n\n{clean_text}"

        prompt = (
            f"You are an academic research translation assistant.\n\n"
            f"Translate the following academic research analysis into {target_language}.\n\n"
            f"CRITICAL PRESERVATION RULES:\n"
            f"1. Preserve ALL citation tags, chunk tags, and page references exactly as they appear "
            f"(e.g., [Evidence 1: ...], [Paper 1, Page 3], [Chunk ...], §Section, pp. X-Y).\n"
            f"2. Preserve mathematical equations, model names (Transformer, BERT, LoRA), metrics (BLEU, MRR), "
            f"and dataset names in English.\n"
            f"3. Maintain formal academic scholarly tone in {target_language}.\n"
            f"4. Output ONLY the translated text without preface or conversational filler.\n\n"
            f"Text to translate:\n{clean_text}"
        )

        res = self.client.run_agent(
            agent_name=self.client.research_agent_name,
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=3000,
        )

        if not res.get("success"):
            err_msg = res.get("error", "Unknown Foundry error")
            if output_mode == "Bilingual":
                return f"[English Original]\n{clean_text}\n\n[{target_language}]\n⚠️ Translation error: {err_msg}"
            return f"⚠️ Translation error: {err_msg}\n\n{clean_text}"

        translated = res.get("content", "").strip()

        if output_mode == "Bilingual":
            return f"[English Original]\n{clean_text}\n\n[{target_language} Translation]\n{translated}"
        return translated
