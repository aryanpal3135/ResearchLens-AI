"""
Translation Page for ResearchLens AI.
Strictly supports exactly four languages: English, Hindi, French, Spanish.
Provides Selected Language and Bilingual output modes while preserving source citations.
"""

import streamlit as st
from config.settings import settings
from frontend.components import render_header, render_info_card


def render_translation_page():
    """Renders the multilingual translation settings and interactive testing suite."""
    render_header(
        title="🌐 Multilingual Research Translation",
        subtitle="Translate paper analyses, gaps, literature reviews, and questions into 4 supported languages."
    )

    render_info_card(
        title="Academic Translation Standards",
        description=(
            "• **Supported Languages (Strictly 4):** English, Hindi (हिन्दी), French (Français), Spanish (Español).\n"
            "• **Output Modes:**\n"
            "   1. *Selected Language:* Outputs synthesized research directly in the chosen tongue.\n"
            "   2. *Bilingual:* Displays English alongside the translated text for cross-verification.\n"
            "• **Citation Integrity:** Original paper filenames, page numbers, and quote markers are **never mutated** "
            "to ensure academic traceability remains 100% auditable."
        )
    )

    # Translation Controls
    col1, col2 = st.columns(2)

    with col1:
        selected_language = st.selectbox(
            "Select Target Language:",
            options=settings.SUPPORTED_LANGUAGES,
            index=settings.SUPPORTED_LANGUAGES.index(st.session_state.get("app_language", "English"))
        )
        st.session_state["app_language"] = selected_language

    with col2:
        output_mode = st.radio(
            "Output Mode:",
            options=["Selected Language", "Bilingual"],
            horizontal=True,
            index=0 if st.session_state.get("output_mode", "Selected Language") == "Selected Language" else 1
        )
        st.session_state["output_mode"] = output_mode

    st.divider()
    st.subheader(f"🔍 Translation Preview ({selected_language} | Mode: {output_mode})")

    sample_academic_snippets = {
        "English": {
            "title": "Literature Review Synthesis Snippet",
            "text": "Existing studies primarily rely on historical datasets, which leaves real-time data drift largely unaddressed.",
            "citation": "[Paper: paper1.pdf, Page: 5, Section: Methodology]"
        },
        "Hindi": {
            "title": "साहित्य समीक्षा संश्लेषण सारांश",
            "text": "मौजूदा अध्ययन मुख्य रूप से ऐतिहासिक डेटासेट पर निर्भर करते हैं, जिससे वास्तविक समय (रियल-टाइम) डेटा विचलन काफी हद तक अनसुलझा रह जाता है।",
            "citation": "[Paper: paper1.pdf, Page: 5, Section: Methodology]"
        },
        "French": {
            "title": "Extrait de synthèse de la revue de la littérature",
            "text": "Les études existantes s'appuient principalement sur des ensembles de données historiques, ce qui laisse la dérive des données en temps réel largement non traitée.",
            "citation": "[Paper: paper1.pdf, Page: 5, Section: Methodology]"
        },
        "Spanish": {
            "title": "Fragmento de síntesis de revisión de literatura",
            "text": "Los estudios existentes se basan principalmente en conjuntos de datos históricos, lo que deja la deriva de datos en tiempo real en gran medida sin abordar.",
            "citation": "[Paper: paper1.pdf, Page: 5, Section: Methodology]"
        }
    }

    english_data = sample_academic_snippets["English"]
    target_data = sample_academic_snippets[selected_language]

    if output_mode == "Selected Language":
        st.markdown(f"#### 📖 {target_data['title']}")
        st.write(target_data["text"])
        st.caption(f"**Source Evidence:** `{target_data['citation']}` *(Preserved unchanged)*")
    else:  # Bilingual
        st.markdown("#### 📖 Bilingual Comparative Display")
        col_eng, col_trans = st.columns(2)
        with col_eng:
            st.markdown("**English (Original Synthesis):**")
            st.info(english_data["text"])
            st.caption(f"Citation: `{english_data['citation']}`")
        with col_trans:
            st.markdown(f"**{selected_language} (Translated Synthesis):**")
            st.success(target_data["text"])
            st.caption(f"Citation: `{target_data['citation']}` *(Intact)*")

    # Interactive Translator Playground
    st.divider()
    st.subheader("✍️ Quick Translation Playground")
    user_text = st.text_area(
        "Enter text to test academic translation:",
        value="The methodology combines dense vector embeddings with sparse BM25 retrieval."
    )

    if st.button("Translate Text"):
        if selected_language == "English":
            st.write(user_text)
        elif selected_language == "Hindi":
            st.write("कार्यप्रणाली घने वेक्टर एम्बेडिंग को विरल BM25 पुनर्प्राप्ति के साथ जोड़ती है।")
        elif selected_language == "French":
            st.write("La méthodologie combine des plongements vectoriels denses avec une recherche clairsemée BM25.")
        elif selected_language == "Spanish":
            st.write("La metodología combina incrustaciones vectoriales densas con recuperación dispersa BM25.")
        st.caption("*(Full model-driven translation using Microsoft Foundry will execute in Phase 11)*")
