"""
Chat & Research Hub for ResearchLens AI.
Flagship LLM-First conversational interface integrating:
- Direct PDF upload & ingestion
- Strict paper scoping & cross-paper querying
- Real Microsoft Foundry Prompt Agents: Paper-Chat-Agent (v2) & researchmate-gpt4-1-mini (v1)
- Verifiable evidence provenance with section/page/chunk citations
- Direct quick actions: Deep Analysis, Compare Papers, Research Gaps, Research Questions, Future Directions, Literature Review
- Multilingual translation (English, Hindi, French, Spanish) via Microsoft Foundry
- Zero hallucination and strict zero-fallback policy
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st

from config.settings import settings
from frontend.components import render_header
from frontend.retrieval_ui import get_or_create_retrieval_engine, sync_papers_to_retrieval_engine
from models.analysis import ResearchAnswer
from models.paper import PaperDocument
from services.chunking import DocumentChunker
from services.foundry_agent import ResearchLensAgent
from services.pdf_processor import PDFProcessor
from services.translation import MultilingualTranslator
from services.i18n import t
from services.speech_service import (
    AzureSpeechService,
    SpeechConfigurationError,
    SpeechSynthesisError,
    SpeechRecognitionError,
    SpeechNoMatchError,
    SpeechTooLongError,
    SpeechRecognitionResult,
)


def handle_pdf_upload(uploaded_files: List[Any], processor: PDFProcessor, chunker: DocumentChunker):
    """Processes, chunks, and indexes uploaded research PDFs into the active session."""
    if not uploaded_files:
        return

    progress_bar = st.progress(0, text="Processing uploaded PDFs...")
    engine = get_or_create_retrieval_engine()
    papers = st.session_state.get("uploaded_papers", [])

    for idx, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        file_bytes = uploaded_file.getvalue()

        # 1. Validation
        is_valid, msg = processor.validate_pdf(file_bytes, filename)
        if not is_valid:
            st.error(f"❌ Failed to ingest '{filename}': {msg}")
            continue

        # Prevent duplicate in session
        if any(p.filename == filename for p in papers):
            continue

        # 2. Save
        saved_path = processor.save_uploaded_file(file_bytes, filename)

        # 3. Generate sequential ID
        paper_id = processor.generate_paper_id(len(papers))

        # 4. Deep Extraction
        paper_doc = processor.extract_document(saved_path, paper_id)

        # 5. Chunking & Indexing
        chunker.chunk_document(paper_doc)
        engine.index_paper(paper_doc)

        papers.append(paper_doc)
        progress_bar.progress((idx + 1) / len(uploaded_files), text=f"Indexed [{paper_id}] {filename}")

    st.session_state["uploaded_papers"] = papers
    st.success(f"✅ Successfully cataloged and indexed {len(uploaded_files)} paper(s) with verifiable chunks.")
    st.rerun()


def render_chat_page():
    """Renders the comprehensive LLM-First Research Assistant Hub."""
    papers: List[PaperDocument] = st.session_state.get("uploaded_papers", [])
    processor = PDFProcessor(upload_dir=settings.UPLOAD_DIR)
    chunker = DocumentChunker()
    translator = MultilingualTranslator()

    # Initialize Foundry Agent
    agent = ResearchLensAgent()
    status = agent.get_connection_status()

    # User Auth & Session Status
    user_info = getattr(st, "user", None)
    is_user_logged_in = getattr(user_info, "is_logged_in", False)
    user_label = getattr(user_info, "name", None) or getattr(user_info, "email", None) if is_user_logged_in else "Session User"

    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(
            f"""
            <div style="margin-bottom: 8px;">
                <h1 style="margin: 0; color: #38BDF8; font-size: 2rem;">{t('chat_hub_title', '🔬 ResearchLens AI')}</h1>
                <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                    {t('chat_hub_subtitle', 'Academic Research Assistant powered by Microsoft Foundry & Hybrid RAG.')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_h2:
        if is_user_logged_in:
            st.markdown(f"<div style='text-align: right; padding-top: 10px; color: #34D399; font-size: 0.85rem;'>👤 {user_label}</div>", unsafe_allow_html=True)
            if st.button("Sign Out", key="top_logout_btn"):
                st.logout()
        else:
            st.markdown(
                f"<div style='text-align: right; padding-top: 8px;'>"
                f"<span style='background-color: #064E3B; color: #34D399; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;'>"
                f"🔒 Private Session Memory</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Foundry Platform Status Banner
    st.markdown(
        f"""
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 10px 14px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-size: 0.9rem; font-weight: 700; color: #38BDF8;">🧠 Microsoft Foundry:</span>
                    <span style="color: #F8FAFC; font-size: 0.85rem; margin-left: 6px;">
                        Chat Agent: <code>{status.get('chat_agent', 'Paper-Chat-Agent')} (v2)</code> &bull;
                        Research Agent: <code>{status.get('research_agent', 'researchmate-gpt4-1-mini')} (v1)</code>
                    </span>
                </div>
                <div>
                    {'<span style="background-color: #064E3B; color: #34D399; padding: 3px 8px; border-radius: 10px; font-size: 0.78rem; font-weight: 600;">🟢 Cloud Agents Active</span>' if status.get('connected') else '<span style="background-color: #7F1D1D; color: #F87171; padding: 3px 8px; border-radius: 10px; font-size: 0.78rem; font-weight: 600;">🔴 Foundry Disconnected</span>'}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 1. PDF Upload Section (Prompt: [ Upload Research Paper(s) ])
    # --------------------------------------------------------------------------
    if not papers:
        st.markdown("### 📥 Upload Research Paper(s)")
        st.caption("Upload one or more academic research PDFs. Text, structure, and tables are immediately chunked and indexed into the hybrid retrieval engine.")
        uploaded_files = st.file_uploader(
            label="Upload research PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="chat_initial_uploader",
            label_visibility="collapsed",
        )
        if uploaded_files:
            handle_pdf_upload(uploaded_files, processor, chunker)
        st.info("💡 To begin, drag and drop one or more academic PDFs above (e.g. *Attention Is All You Need*, *BERT*).")
        return

    # If papers are already loaded, provide a compact manage bar
    with st.expander(f"📚 Uploaded Papers ({len(papers)} active) — Click to view or add more", expanded=False):
        for p in papers:
            st.markdown(f"- **`[{p.id}]`** {p.metadata.title or p.filename} — `{p.page_count} pages`, `{len(p.chunks)} chunks`")

        col_up, col_clr = st.columns([3, 1])
        with col_up:
            more_files = st.file_uploader("Add more PDFs:", type=["pdf"], accept_multiple_files=True, key="chat_more_uploader")
            if more_files:
                handle_pdf_upload(more_files, processor, chunker)
        with col_clr:
            st.write("")
            st.write("")
            if st.button("🗑️ Clear Session Papers", use_container_width=True, type="secondary"):
                st.session_state["uploaded_papers"] = []
                st.session_state["chat_messages_v4"] = []
                if "hybrid_retrieval_engine" in st.session_state:
                    del st.session_state["hybrid_retrieval_engine"]
                st.rerun()

    # Ensure papers are indexed
    sync_papers_to_retrieval_engine()
    engine = get_or_create_retrieval_engine()

    # --------------------------------------------------------------------------
    # 2. Target Scope Selector
    # --------------------------------------------------------------------------
    col_sc1, col_sc2 = st.columns([2, 3])
    with col_sc1:
        scope_mode = st.radio(
            "Query Scope:",
            options=["All Uploaded Papers", "Specific Paper Only"],
            horizontal=True,
            key="chat_scope_mode",
        )
    with col_sc2:
        selected_paper_id: Optional[str] = None
        if scope_mode == "Specific Paper Only":
            paper_options = {p.id: f"[{p.id}] {p.metadata.title or p.filename}" for p in papers}
            selected_paper_id = st.selectbox(
                "Target Paper:",
                options=list(paper_options.keys()),
                format_func=lambda pid: paper_options[pid],
                key="chat_target_paper",
            )
        else:
            st.caption(f"🌐 Cross-paper search active across {len(papers)} papers.")

    # --------------------------------------------------------------------------
    # 3. Quick Action Buttons (Verified Foundry Capabilities)
    # --------------------------------------------------------------------------
    st.markdown("##### ⚡ Quick Research Actions:")
    btn_cols = st.columns(6)

    trigger_action = None
    with btn_cols[0]:
        if st.button(t("action_analysis", "📄 Deep Analysis"), use_container_width=True, help="7-section structured academic analysis"):
            trigger_action = "analysis"
    with btn_cols[1]:
        if st.button(t("action_comparison", "⚖️ Compare Papers"), use_container_width=True, disabled=(len(papers) < 2), help="Comparative cross-paper synthesis"):
            trigger_action = "comparison"
    with btn_cols[2]:
        if st.button(t("action_gaps", "🔍 Research Gaps"), use_container_width=True, help="Evidence-grounded research gaps"):
            trigger_action = "gaps"
    with btn_cols[3]:
        if st.button(t("action_questions", "❓ Research Questions"), use_container_width=True, help="Novel, gap-derived research questions"):
            trigger_action = "questions"
    with btn_cols[4]:
        if st.button(t("action_future", "🔮 Future Directions"), use_container_width=True, help="Author-stated & literature-derived future work"):
            trigger_action = "future_directions"
    with btn_cols[5]:
        if st.button(t("action_review", "📚 Literature Review"), use_container_width=True, help="Comprehensive academic literature review"):
            trigger_action = "literature_review"

    # Initialize chat message history
    if "chat_messages_v4" not in st.session_state:
        st.session_state["chat_messages_v4"] = [
            {
                "role": "assistant",
                "content": (
                    f"Hello! I am **ResearchLens AI**, connected to Microsoft Foundry cloud agents.\n\n"
                    f"I have indexed **{len(papers)} research paper(s)** with strict hybrid RAG retrieval. "
                    f"You can ask any research question, or use the quick actions above to analyze, compare, "
                    f"find research gaps, generate questions, or synthesize a literature review."
                ),
                "confidence_status": "grounded",
                "evidence_items": [],
                "citation_labels": [],
                "model": status.get("chat_agent", "Paper-Chat-Agent"),
                "latency_ms": 0.0,
            }
        ]

    # Render previous messages
    for msg_idx, msg in enumerate(st.session_state["chat_messages_v4"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Evidence provenance
            if msg.get("evidence_items"):
                conf = msg.get("confidence_status", "grounded")
                conf_badge = "🟢 Grounded in Document Evidence" if conf == "grounded" else "🟡 Insufficient Direct Evidence"
                with st.expander(f"📍 Verifiable Evidence & Provenance ({len(msg['evidence_items'])} chunks) — {conf_badge}", expanded=False):
                    for idx, ev in enumerate(msg["evidence_items"], start=1):
                        st.markdown(
                            f"**[{idx}] Paper:** `{ev.get('paper_id')}` | "
                            f"**Pages:** {ev.get('page_start')}–{ev.get('page_end')} | "
                            f"**Section:** `{ev.get('normalized_section')}` | "
                            f"**Score:** `{ev.get('score', 0.0):.4f}`"
                        )
                        st.markdown(
                            f"<div style='background-color: #0F172A; border-left: 3px solid #38BDF8; padding: 8px 12px; margin-bottom: 8px; font-size: 0.85rem; max-height: 200px; overflow-y: auto;'>"
                            f"<code>{ev.get('chunk_id')}</code><br/>"
                            f"{ev.get('text', '')}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Read Aloud & Translation controls for assistant answers (all assistant messages)
            if msg["role"] == "assistant":
                audio_cache_key = f"audio_cache_{msg_idx}"
                audio_tr_cache_key = f"audio_tr_cache_{msg_idx}"

                t_col_audio, t_col1, t_col2, t_col3 = st.columns([1.6, 1, 1, 1])
                with t_col_audio:
                    if st.button(t("read_aloud", "🔊 Read aloud"), key=f"btn_read_aloud_{msg_idx}", help="Listen to this answer via Azure AI Speech"):
                        with st.spinner("Generating audio via Azure AI Speech..."):
                            speech_svc = AzureSpeechService()
                            if not speech_svc.is_configured:
                                st.warning("⚠️ Read Aloud is not configured. Please set AZURE_SPEECH_ENDPOINT and AZURE_SPEECH_API_KEY.")
                            else:
                                try:
                                    current_lang = st.session_state.get("app_language", "English")
                                    audio_bytes = speech_svc.synthesize_speech(
                                        text=msg["content"],
                                        language=current_lang,
                                    )
                                    st.session_state[audio_cache_key] = audio_bytes
                                    st.rerun()
                                except SpeechConfigurationError as e:
                                    st.warning(f"⚠️ Read Aloud is not configured: {str(e)}")
                                except SpeechSynthesisError as e:
                                    st.error(f"❌ Azure Speech Error: {str(e)}")
                                except Exception as e:
                                    st.error(f"❌ Azure Speech Error: {str(e)}")

                with t_col1:
                    if st.button("🌐 हिन्दी", key=f"tr_hi_{msg_idx}", help="Translate to Hindi via Foundry"):
                        with st.spinner("Translating via Foundry researchmate-gpt4-1-mini..."):
                            tr_res = translator.translate_text(msg["content"], "Hindi", output_mode="Selected Language")
                            st.session_state[f"translated_{msg_idx}"] = tr_res
                            st.session_state[f"translated_lang_{msg_idx}"] = "Hindi"
                            if audio_tr_cache_key in st.session_state:
                                del st.session_state[audio_tr_cache_key]
                            st.rerun()
                with t_col2:
                    if st.button("🌐 Français", key=f"tr_fr_{msg_idx}", help="Translate to French via Foundry"):
                        with st.spinner("Translating via Foundry researchmate-gpt4-1-mini..."):
                            tr_res = translator.translate_text(msg["content"], "French", output_mode="Selected Language")
                            st.session_state[f"translated_{msg_idx}"] = tr_res
                            st.session_state[f"translated_lang_{msg_idx}"] = "French"
                            if audio_tr_cache_key in st.session_state:
                                del st.session_state[audio_tr_cache_key]
                            st.rerun()
                with t_col3:
                    if st.button("🌐 Español", key=f"tr_es_{msg_idx}", help="Translate to Spanish via Foundry"):
                        with st.spinner("Translating via Foundry researchmate-gpt4-1-mini..."):
                            tr_res = translator.translate_text(msg["content"], "Spanish", output_mode="Selected Language")
                            st.session_state[f"translated_{msg_idx}"] = tr_res
                            st.session_state[f"translated_lang_{msg_idx}"] = "Spanish"
                            if audio_tr_cache_key in st.session_state:
                                del st.session_state[audio_tr_cache_key]
                            st.rerun()

                # Render audio player for main answer with automatic playback
                if audio_cache_key in st.session_state:
                    st.audio(st.session_state[audio_cache_key], format="audio/wav", autoplay=True)

                # Render translated output container with its own Read Aloud
                if f"translated_{msg_idx}" in st.session_state:
                    tr_text = st.session_state[f"translated_{msg_idx}"]
                    tr_lang = st.session_state.get(f"translated_lang_{msg_idx}", "Hindi")
                    with st.expander(f"🌍 Translated Academic Output ({tr_lang})", expanded=True):
                        st.markdown(tr_text)
                        if st.button(f"🔊 Read aloud ({tr_lang})", key=f"btn_read_aloud_tr_{msg_idx}", help=f"Listen to {tr_lang} translation via Azure Speech"):
                            with st.spinner(f"Generating {tr_lang} audio via Azure AI Speech..."):
                                speech_svc = AzureSpeechService()
                                if not speech_svc.is_configured:
                                    st.warning("⚠️ Read Aloud is not configured. Please set AZURE_SPEECH_ENDPOINT and AZURE_SPEECH_API_KEY.")
                                else:
                                    try:
                                        audio_tr_bytes = speech_svc.synthesize_speech(
                                            text=tr_text,
                                            language=tr_lang,
                                        )
                                        st.session_state[audio_tr_cache_key] = audio_tr_bytes
                                        st.rerun()
                                    except SpeechConfigurationError as e:
                                        st.warning(f"⚠️ Read Aloud is not configured: {str(e)}")
                                    except SpeechSynthesisError as e:
                                        st.error(f"❌ Azure Speech Error: {str(e)}")
                                    except Exception as e:
                                        st.error(f"❌ Azure Speech Error: {str(e)}")

                        if audio_tr_cache_key in st.session_state:
                            st.audio(st.session_state[audio_tr_cache_key], format="audio/wav", autoplay=True)

    # --------------------------------------------------------------------------
    # 4. Handle Voice Input & Chat Input (Unified Pipeline)
    # --------------------------------------------------------------------------
    speech_svc = AzureSpeechService()

    # Voice Input Session State
    if "voice_mode_open" not in st.session_state:
        st.session_state["voice_mode_open"] = False
    if "recognized_voice_query" not in st.session_state:
        st.session_state["recognized_voice_query"] = ""
    if "voice_widget_version" not in st.session_state:
        st.session_state["voice_widget_version"] = 0
    if "voice_selected_lang" not in st.session_state:
        cur_lang = st.session_state.get("app_language", "English")
        st.session_state["voice_selected_lang"] = cur_lang if cur_lang in settings.SUPPORTED_LANGUAGES else "English"

    voice_submitted_query = None

    # Voice Input Controls Bar
    v_col1, v_col2, v_col3 = st.columns([3, 2, 5])
    with v_col1:
        voice_label = t("voice_prompt", "🎤 Voice Prompt (Azure Speech)")
        if st.button(
            voice_label,
            key="btn_toggle_voice_prompt",
            help="Click to open microphone and dictate your question via Azure AI Speech",
        ):
            st.session_state["voice_mode_open"] = not st.session_state["voice_mode_open"]
            st.rerun()

    with v_col2:
        lang_options = ["English", "Hindi", "French", "Spanish"]
        curr_idx = lang_options.index(st.session_state["voice_selected_lang"]) if st.session_state["voice_selected_lang"] in lang_options else 0
        selected_voice_lang = st.selectbox(
            t("voice_language", "Spoken Language:"),
            options=lang_options,
            index=curr_idx,
            key="voice_lang_selector",
            label_visibility="collapsed",
        )
        st.session_state["voice_selected_lang"] = selected_voice_lang

    with v_col3:
        pass

    # Voice Recording & Review Container
    if st.session_state.get("voice_mode_open"):
        with st.container():
            rec_locale = speech_svc.get_recognition_language(selected_voice_lang)
            st.markdown(
                f"""
                <div style="background-color: #0F172A; border: 1px solid #0284C7; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
                        <span style="color: #38BDF8; font-weight: 700; font-size: 0.92rem;">
                            🎙️ Azure Speech-to-Text — <code>{selected_voice_lang} ({rec_locale})</code>
                        </span>
                        <span style="color: #F43F5E; font-size: 0.85rem; font-weight: 600;">
                            🔴 Listening... Speak your research question.
                        </span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # Auto-send toggle (ChatGPT style)
            auto_send_col, _ = st.columns([3, 7])
            with auto_send_col:
                auto_send_enabled = st.checkbox(
                    "⚡ Auto-Send after speaking (ChatGPT style)",
                    value=True,
                    key="voice_auto_send_toggle",
                    help="When enabled, your spoken question is immediately transcribed by Azure Speech and submitted directly to ResearchLens.",
                )

            # Dynamic widget key to ensure the audio recorder completely resets after each prompt
            widget_key = f"voice_audio_input_widget_{st.session_state.get('voice_widget_version', 0)}"
            audio_recording = st.audio_input(
                label=t("listening_prompt", "🔴 Listening... Speak your research question."),
                key=widget_key,
                help="Speak clearly into your microphone. Azure AI Speech will transcribe your question.",
            )

            if audio_recording is not None:
                audio_bytes = audio_recording.getvalue() if hasattr(audio_recording, "getvalue") else audio_recording.read()
                audio_hash = f"audio_{len(audio_bytes)}_{audio_bytes[:24]}"
                if st.session_state.get("last_processed_audio_hash") != audio_hash:
                    with st.spinner(f"Transcribing speech via Azure AI Speech ({selected_voice_lang})..."):
                        if not speech_svc.is_configured:
                            st.warning("⚠️ Voice Input is not configured. Please set AZURE_SPEECH_ENDPOINT and AZURE_SPEECH_API_KEY.")
                        else:
                            rec_result = speech_svc.recognize_speech_from_audio(
                                audio_data=audio_bytes,
                                language=selected_voice_lang,
                            )
                            st.session_state["last_processed_audio_hash"] = audio_hash
                            if rec_result.success and rec_result.text:
                                if auto_send_enabled:
                                    # ChatGPT style: immediately submit the recognized query!
                                    voice_submitted_query = rec_result.text
                                    st.session_state["recognized_voice_query"] = ""
                                    # DO NOT rerun here; let it flow to user_query below so it executes!
                                else:
                                    st.session_state["recognized_voice_query"] = rec_result.text
                                    st.rerun()
                            elif rec_result.reason == "NoMatch":
                                st.warning("⚠️ Could not understand the speech. Please speak clearly or try again.")
                            elif rec_result.reason == "TooLong":
                                st.warning("⚠️ Your voice prompt was too long. Please try a shorter prompt.")
                            else:
                                err = rec_result.error_message or "Speech recognition failed."
                                if "denied" in err.lower() or "microphone" in err.lower():
                                    st.error("❌ Microphone access was denied. Please allow microphone access in your browser/system settings.")
                                else:
                                    st.error(f"❌ {err}")

            # Editable Review & Send box (when auto-send is disabled or reviewing)
            if st.session_state.get("recognized_voice_query") and not voice_submitted_query:
                st.markdown(
                    f"<p style='color: #38BDF8; font-size: 0.88rem; font-weight: 600; margin: 10px 0 4px 0;'>"
                    f"{t('review_voice_prompt', 'Review & edit recognized question before sending:')}</p>",
                    unsafe_allow_html=True,
                )
                edited_query = st.text_area(
                    label="Editable Question",
                    value=st.session_state["recognized_voice_query"],
                    key="voice_text_editor",
                    height=75,
                    label_visibility="collapsed",
                )
                col_send, col_clear, _ = st.columns([2, 2, 6])
                with col_send:
                    if st.button(t("send_voice_prompt", "🚀 Send to ResearchLens"), key="btn_send_voice_query", type="primary"):
                        if edited_query.strip():
                            voice_submitted_query = edited_query.strip()
                            st.session_state["recognized_voice_query"] = ""
                            st.session_state["voice_mode_open"] = False
                            # DO NOT rerun here; let it flow to user_query below so it executes!
                with col_clear:
                    if st.button(t("clear_voice_prompt", "🔄 Clear Recording"), key="btn_clear_voice_query"):
                        st.session_state["recognized_voice_query"] = ""
                        st.session_state["last_processed_audio_hash"] = None
                        st.session_state["voice_widget_version"] = st.session_state.get("voice_widget_version", 0) + 1
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    # Standard Chat Input (with embedded microphone directly inside chat_input, exactly like ChatGPT)
    raw_chat_input = st.chat_input(
        placeholder=t("chat_placeholder", "Ask ResearchLens anything about your papers..."),
        key="main_chat_input",
        accept_audio=True,
        audio_sample_rate=16000,
    )

    # Extract query from raw_chat_input (handles string or ChatInputValue from embedded mic)
    typed_or_chat_query = None
    if raw_chat_input is not None:
        if isinstance(raw_chat_input, str):
            typed_or_chat_query = raw_chat_input.strip()
        else:
            # ChatInputValue (user clicked the mic icon inside chat_input, ChatGPT style)
            text_val = getattr(raw_chat_input, "text", "") or ""
            audio_val = getattr(raw_chat_input, "audio", None)
            if audio_val is not None and not text_val:
                with st.spinner("🎙️ Transcribing voice via Azure AI Speech..."):
                    audio_bytes = audio_val.getvalue() if hasattr(audio_val, "getvalue") else audio_val.read()
                    if speech_svc.is_configured:
                        rec_res = speech_svc.recognize_speech_from_audio(
                            audio_data=audio_bytes,
                            language=st.session_state.get("voice_selected_lang", "English"),
                        )
                        if rec_res.success and rec_res.text:
                            # Immediate ChatGPT-style execution: submit recognized voice directly!
                            typed_or_chat_query = rec_res.text
                        elif rec_res.reason == "NoMatch":
                            st.warning("⚠️ Could not understand the speech. Please try again.")
                        elif rec_res.reason == "TooLong":
                            st.warning("⚠️ Your voice prompt was too long. Please try a shorter prompt.")
                        else:
                            st.error(f"❌ {rec_res.error_message or 'Recognition failed.'}")
                    else:
                        st.warning("⚠️ Voice Input is not configured. Please set AZURE_SPEECH_ENDPOINT and AZURE_SPEECH_API_KEY.")
            elif text_val:
                typed_or_chat_query = text_val.strip()

    user_query = voice_submitted_query or typed_or_chat_query

    # Action routing
    if trigger_action:
        target_pid = selected_paper_id or papers[0].id

        if trigger_action == "analysis":
            target_paper = next((p for p in papers if p.id == target_pid), papers[0])
            with st.chat_message("user"):
                st.write(f"Generate deep academic analysis for paper `{target_paper.id}` ({target_paper.metadata.title or target_paper.filename}).")
            st.session_state["chat_messages_v4"].append({
                "role": "user",
                "content": f"Generate deep academic analysis for paper `{target_paper.id}` ({target_paper.metadata.title or target_paper.filename}).",
            })

            with st.chat_message("assistant"):
                with st.spinner(f"Synthesizing 7-section deep analysis via Foundry Agent (`researchmate-gpt4-1-mini`)..."):
                    res = agent.analyze_paper_sections(paper=target_paper, engine=engine)
                    content_lines = [f"### 📄 Deep Academic Analysis: {target_paper.metadata.title or target_paper.filename}\n"]
                    all_ev = []
                    for sec in res.sections.values():
                        content_lines.append(f"#### § {sec.section_name}")
                        content_lines.append(f"{sec.content}\n")
                        for ev in sec.evidence_items:
                            all_ev.append(ev.model_dump())

                    full_text = "\n".join(content_lines)
                    st.markdown(full_text)

                    st.session_state["chat_messages_v4"].append({
                        "role": "assistant",
                        "content": full_text,
                        "confidence_status": "grounded",
                        "evidence_items": all_ev,
                        "model": "researchmate-gpt4-1-mini",
                    })
                    st.rerun()

        elif trigger_action == "comparison":
            target_papers = papers[:3]
            target_ids = [p.id for p in target_papers]
            with st.chat_message("user"):
                st.write(f"Compare research papers: {', '.join(target_ids)}")
            st.session_state["chat_messages_v4"].append({
                "role": "user",
                "content": f"Compare research papers: {', '.join(target_ids)}",
            })

            with st.chat_message("assistant"):
                with st.spinner(f"Synthesizing cross-paper comparison via Foundry Agent (`researchmate-gpt4-1-mini`)..."):
                    res = agent.compare_papers(papers=target_papers, engine=engine)
                    paper_a_title = target_papers[0].metadata.title or target_papers[0].filename
                    paper_b_title = target_papers[1].metadata.title or target_papers[1].filename if len(target_papers) > 1 else "Paper B"

                    comp_lines = [
                        f"### ⚖️ Cross-Paper Comparison: {paper_a_title} vs {paper_b_title}\n"
                    ]
                    if res.custom_query_answer:
                        comp_lines.append(f"**Executive Synthesis:**\n{res.custom_query_answer}\n")

                    if res.dimensions:
                        comp_lines.append("#### 📊 Key Academic Dimensions")
                        dim_items = list(res.dimensions.values())[:6]
                        for dim in dim_items:
                            comp_lines.append(f"**{dim.dimension_name}:**")
                            sum_a = dim.paper_summaries.get(target_papers[0].id, "")
                            sum_b = dim.paper_summaries.get(target_papers[1].id, "") if len(target_papers) > 1 else ""
                            if sum_a and "Not clearly identified" not in sum_a:
                                comp_lines.append(f"- *{paper_a_title[:40]}:* {sum_a}")
                            if sum_b and "Not clearly identified" not in sum_b:
                                comp_lines.append(f"- *{paper_b_title[:40]}:* {sum_b}")
                            if dim.synthesis:
                                comp_lines.append(f"- *Comparative Contrast:* {dim.synthesis}")
                            comp_lines.append("")

                    if res.similarities:
                        comp_lines.append("#### 🔄 Grounded Similarities")
                        for s in res.similarities[:3]:
                            comp_lines.append(f"• **{s.topic}:** {s.description}")
                        comp_lines.append("")

                    if res.differences:
                        comp_lines.append("#### ⚡ Core Differences")
                        for d in res.differences[:3]:
                            comp_lines.append(f"• **{d.topic}:** {d.description}")

                    comp_text = "\n".join(comp_lines)
                    st.markdown(comp_text)
                    all_ev = [ev.model_dump() for ev in res.all_evidence_items]

                    st.session_state["chat_messages_v4"].append({
                        "role": "assistant",
                        "content": comp_text,
                        "confidence_status": "grounded",
                        "evidence_items": all_ev,
                        "model": "researchmate-gpt4-1-mini",
                    })
                    st.rerun()

        elif trigger_action == "gaps":
            target_papers = [p for p in papers if p.id == selected_paper_id] if selected_paper_id else papers
            with st.chat_message("user"):
                st.write("Find genuine evidence-grounded research gaps.")
            st.session_state["chat_messages_v4"].append({
                "role": "user",
                "content": "Find genuine evidence-grounded research gaps.",
            })

            with st.chat_message("assistant"):
                with st.spinner("Detecting research gaps via Foundry Agent (`researchmate-gpt4-1-mini`)..."):
                    res = agent.detect_research_gaps(papers=target_papers, engine=engine)
                    gap_text = f"### 🔍 Detected Research Gaps ({len(res.gaps)} Found)\n\n"
                    for g in res.gaps:
                        gap_text += f"- **[{g.category}]** {g.title}: {g.description}\n"

                    st.markdown(gap_text)
                    all_ev = [ev.model_dump() for ev in res.all_evidence_items]

                    st.session_state["chat_messages_v4"].append({
                        "role": "assistant",
                        "content": gap_text,
                        "confidence_status": "grounded",
                        "evidence_items": all_ev,
                        "model": "researchmate-gpt4-1-mini",
                    })
                    st.rerun()

        elif trigger_action == "questions":
            target_papers = [p for p in papers if p.id == selected_paper_id] if selected_paper_id else papers
            with st.chat_message("user"):
                st.write("Generate novel, evidence-derived research questions.")
            st.session_state["chat_messages_v4"].append({
                "role": "user",
                "content": "Generate novel, evidence-derived research questions.",
            })

            with st.chat_message("assistant"):
                with st.spinner("Generating research questions via Foundry Agent (`researchmate-gpt4-1-mini`)..."):
                    res = agent.generate_research_questions(papers=target_papers, engine=engine)
                    q_text = f"### ❓ Grounded Research Questions ({len(res.questions)} Generated)\n\n"
                    for q in res.questions:
                        q_text += f"- **`[{q.display_id}]`** {q.question}\n  *Type:* `{q.question_type}` | *Papers:* `{', '.join(q.supporting_paper_ids)}`\n"

                    st.markdown(q_text)
                    all_ev = [ev.model_dump() for ev in res.all_evidence_items]

                    st.session_state["chat_messages_v4"].append({
                        "role": "assistant",
                        "content": q_text,
                        "confidence_status": "grounded",
                        "evidence_items": all_ev,
                        "model": "researchmate-gpt4-1-mini",
                    })
                    st.rerun()

        elif trigger_action == "future_directions":
            target_papers = [p for p in papers if p.id == selected_paper_id] if selected_paper_id else papers
            with st.chat_message("user"):
                st.write("Extract future research directions.")
            st.session_state["chat_messages_v4"].append({
                "role": "user",
                "content": "Extract future research directions.",
            })

            with st.chat_message("assistant"):
                with st.spinner("Extracting future directions via Foundry Agent (`researchmate-gpt4-1-mini`)..."):
                    res = agent.generate_research_questions(papers=target_papers, engine=engine)
                    fd_text = f"### 🔮 Future Research Directions ({len(res.future_directions)} Directions)\n\n"
                    for fd in res.future_directions:
                        fd_text += f"- **[{fd.direction_type.upper()}]** {fd.statement}\n  *Rationale:* {fd.rationale}\n"

                    st.markdown(fd_text)
                    all_ev = [ev.model_dump() for ev in res.all_evidence_items]

                    st.session_state["chat_messages_v4"].append({
                        "role": "assistant",
                        "content": fd_text,
                        "confidence_status": "grounded",
                        "evidence_items": all_ev,
                        "model": "researchmate-gpt4-1-mini",
                    })
                    st.rerun()

        elif trigger_action == "literature_review":
            target_papers = papers
            with st.chat_message("user"):
                st.write("Synthesize comprehensive academic literature review.")
            st.session_state["chat_messages_v4"].append({
                "role": "user",
                "content": "Synthesize comprehensive academic literature review.",
            })

            with st.chat_message("assistant"):
                with st.spinner("Synthesizing literature review via Foundry Agent (`researchmate-gpt4-1-mini`)..."):
                    res = agent.generate_literature_review(papers=target_papers, engine=engine)
                    lit_text = f"### 📚 {res.title}\n\n"
                    lit_text += f"**Introduction:**\n{res.introduction}\n\n"
                    lit_text += f"**Methodological Synthesis:**\n{res.methodological_synthesis}\n\n"
                    lit_text += f"**Empirical Findings:**\n{res.findings_and_evidence}\n\n"
                    lit_text += f"**Conclusion:**\n{res.conclusion}\n"

                    st.markdown(lit_text)
                    all_ev = [ev.model_dump() for ev in res.all_evidence_items]

                    st.session_state["chat_messages_v4"].append({
                        "role": "assistant",
                        "content": lit_text,
                        "confidence_status": "grounded",
                        "evidence_items": all_ev,
                        "model": "researchmate-gpt4-1-mini",
                    })
                    st.rerun()

    elif user_query:
        # Standard Conversational QA -> Paper-Chat-Agent (v2)
        st.session_state["chat_messages_v4"].append({
            "role": "user",
            "content": user_query,
            "evidence_items": [],
        })
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing grounded answer via Foundry Paper-Chat-Agent (v2)..."):
                ans: ResearchAnswer = agent.answer_question(
                    query=user_query,
                    paper_id=selected_paper_id,
                    engine=engine,
                    top_k=5,
                )

                st.markdown(ans.answer)

                if ans.evidence_items:
                    conf_badge = "🟢 Grounded in Document Evidence" if ans.confidence_status == "grounded" else "🟡 Insufficient Direct Evidence"
                    with st.expander(f"📍 Verifiable Evidence & Provenance ({len(ans.evidence_items)} chunks) — {conf_badge}", expanded=True):
                        for idx, ev in enumerate(ans.evidence_items, start=1):
                            st.markdown(
                                f"**[{idx}] Paper:** `{ev.paper_id}` | "
                                f"**Pages:** {ev.page_start}–{ev.page_end} | "
                                f"**Section:** `{ev.normalized_section}` | "
                                f"**Score:** `{ev.score:.4f}`"
                            )
                            st.markdown(
                                f"<div style='background-color: #0F172A; border-left: 3px solid #38BDF8; padding: 8px 12px; margin-bottom: 8px; font-size: 0.85rem; max-height: 200px; overflow-y: auto;'>"
                                f"<code>{ev.chunk_id}</code><br/>"
                                f"{ev.text}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                st.caption(f"Agent: `{ans.model_used or 'Paper-Chat-Agent'}` | Latency: `{ans.execution_latency_ms:.1f} ms` | Status: `{ans.confidence_status}`")

        st.session_state["chat_messages_v4"].append({
            "role": "assistant",
            "content": ans.answer,
            "confidence_status": ans.confidence_status,
            "evidence_items": [ev.model_dump() for ev in ans.evidence_items],
            "citation_labels": ans.citation_labels,
            "model": ans.model_used,
            "latency_ms": ans.execution_latency_ms,
        })

        # Clean up any lingering voice states for the next prompt
        st.session_state["recognized_voice_query"] = ""
        st.session_state["last_processed_audio_hash"] = None
        st.session_state["voice_widget_version"] = st.session_state.get("voice_widget_version", 0) + 1
        st.rerun()
