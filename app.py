"""
ResearchLens AI - Main Streamlit Application Entry Point.
Chat-First Academic Research Assistant powered by Microsoft Foundry and Hybrid RAG.
Features public landing page, user registration/login gating, and authenticated research workspace.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure Azure CLI standard path is present in PATH on Windows
az_standard_path = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
if os.path.exists(az_standard_path) and az_standard_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = az_standard_path + os.pathsep + os.environ.get("PATH", "")

# Add project root to sys.path for clean modular imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from services.i18n import t
from frontend.landing_page import render_landing_page
from frontend.chat import render_chat_page
from frontend.analysis import render_analysis_page
from frontend.comparison import render_comparison_page
from frontend.gaps import render_gaps_page
from frontend.questions import render_questions_page
from frontend.literature_review import render_literature_review_page
from services.foundry_client import MicrosoftFoundryClient


def initialize_session_state():
    """Ensures consistent state initialization across page navigation."""
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = None
    if "uploaded_papers" not in st.session_state:
        st.session_state["uploaded_papers"] = []
    if "paper_analyses" not in st.session_state:
        st.session_state["paper_analyses"] = {}
    if "research_gaps" not in st.session_state:
        st.session_state["research_gaps"] = []
    if "research_questions" not in st.session_state:
        st.session_state["research_questions"] = []
    if "future_directions" not in st.session_state:
        st.session_state["future_directions"] = []
    if "app_language" not in st.session_state:
        st.session_state["app_language"] = settings.DEFAULT_LANGUAGE
    if "foundry_status_cached" not in st.session_state:
        is_conf = bool(settings.AZURE_OPENAI_ENDPOINT and settings.AZURE_OPENAI_API_KEY)
        res_agent = settings.FOUNDRY_RESEARCH_AGENT_NAME
        chat_agent = settings.FOUNDRY_CHAT_AGENT_NAME
        st.session_state["foundry_status_cached"] = {
            "connected": is_conf,
            "configured": is_conf,
            "label": f"🧠 Microsoft Foundry Agents ({res_agent} | {chat_agent})",
            "badge": "Cloud Agents Active" if is_conf else "Foundry Disconnected",
            "research_agent": res_agent,
            "research_agent_version": settings.FOUNDRY_RESEARCH_AGENT_VERSION,
            "chat_agent": chat_agent,
            "chat_agent_version": settings.FOUNDRY_CHAT_AGENT_VERSION,
            "deployment": settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            "agent_name": res_agent,
            "endpoint": settings.FOUNDRY_PROJECT_ENDPOINT,
            "model_attribution": f"Microsoft Foundry ({res_agent}, {settings.AZURE_OPENAI_CHAT_DEPLOYMENT})",
            "message": "Microsoft Foundry Agents connected and active.",
        }



def render_sidebar(current_user: dict):
    """Renders the streamlined sidebar navigation and authenticated user profile with dynamic i18n."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 6px 0 14px 0;">
                <h1 style="margin: 0; font-size: 1.5rem; color: #38BDF8;">{t("app_title")}</h1>
                <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94A3B8;">
                    {t("app_subtitle")}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. User Profile Card
        st.markdown(f"### {t('profile_heading')}")
        initial = current_user.get("name", "U")[0].upper()
        name = current_user.get("name", "Researcher")
        email = current_user.get("email", "user@researchlens.ai")
        institution = current_user.get("institution", "Academic Institution")
        created = current_user.get("created_at", "Active")[:10]

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 14px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <div style="background: #0284C7; color: white; border-radius: 50%; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1rem; flex-shrink: 0;">
                        {initial}
                    </div>
                    <div style="overflow: hidden;">
                        <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">{name}</div>
                        <div style="font-size: 0.78rem; color: #38BDF8; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">{institution}</div>
                    </div>
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; word-break: break-all; margin-bottom: 6px;">
                    📧 {email}
                </div>
                <div style="font-size: 0.72rem; color: #64748B;">
                    {t('member_since')}: <code>{created}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(t("sign_out"), key="sidebar_logout_btn", use_container_width=True, type="secondary"):
            st.session_state["current_user"] = None
            st.rerun()

        st.divider()

        # 2. Global Language Settings (Placed prominently so language switch updates whole view)
        st.markdown(f"### {t('academic_translation')}")
        current_lang = st.session_state.get("app_language", "English")
        new_lang = st.selectbox(
            t("target_language"),
            options=settings.SUPPORTED_LANGUAGES,
            index=settings.SUPPORTED_LANGUAGES.index(current_lang) if current_lang in settings.SUPPORTED_LANGUAGES else 0,
            help="Supported: English, Hindi, French, Spanish. Selecting will instantly localize the entire UI.",
            key="sidebar_lang_selector",
        )
        if new_lang != current_lang:
            st.session_state["app_language"] = new_lang
            st.rerun()

        st.divider()

        # 3. Primary Navigation (Chat-First + Deep Views)
        st.markdown(f"### {t('workspace_views')}")
        nav_items = [
            ("nav_chat", "💬 ResearchLens AI (Chat & Hub)"),
            ("nav_analysis", "📄 Paper Deep Analysis View"),
            ("nav_comparison", "⚖️ Paper Comparison View"),
            ("nav_gaps", "🔍 Research Gaps View"),
            ("nav_questions", "❓ Research Questions View"),
            ("nav_lit_review", "📚 Literature Review View"),
        ]

        nav_display_names = [t(k) for k, _ in nav_items]
        current_selected_idx = st.session_state.get("sidebar_nav_index", 0)

        selected_idx = st.radio(
            label="Go to View:",
            options=range(len(nav_items)),
            index=current_selected_idx if current_selected_idx < len(nav_items) else 0,
            format_func=lambda i: nav_display_names[i],
            label_visibility="collapsed"
        )
        st.session_state["sidebar_nav_index"] = selected_idx
        selected_page = nav_items[selected_idx][1]

        st.divider()

        # 4. Session Papers & Controls
        paper_count = len(st.session_state.get("uploaded_papers", []))
        st.markdown(f"**{t('active_papers')}** `{paper_count}`")
        if paper_count > 0:
            if st.button(t("reset_session"), use_container_width=True, type="secondary"):
                st.session_state["uploaded_papers"] = []
                st.session_state["chat_messages_v4"] = []
                st.session_state["paper_analyses"] = {}
                st.session_state["research_gaps"] = []
                st.session_state["research_questions"] = []
                st.session_state["future_directions"] = []
                if "hybrid_retrieval_engine" in st.session_state:
                    del st.session_state["hybrid_retrieval_engine"]
                st.rerun()

        # 5. Microsoft Foundry Platform Status
        f_status = st.session_state.get("foundry_status_cached", {})
        if f_status.get("configured"):
            st.markdown(
                f"<div style='margin-top: 12px; font-size: 0.8rem; color: #34D399;'>"
                f"🟢 <strong>{t('foundry_status')}</strong> <code>researchmate</code><br/>"
                f"<span style='color: #94A3B8;'>Agents: <code>{f_status.get('chat_agent', 'Paper-Chat')}</code> & <code>{f_status.get('research_agent', 'gpt4-1-mini')}</code></span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("🔴 **Foundry:** `Disconnected`")

        st.caption("ResearchLens AI v1.0.0 | Enterprise Academic")

    return selected_page


def inject_global_css():
    """Injects universal CSS to fix dropdown text truncation, wrapping, and text areas."""
    st.markdown(
        """
        <style>
        /* 1. BaseWeb Selectbox Container: Allow multi-line height and text wrapping */
        div[data-baseweb="select"] {
            min-height: 46px !important;
            height: auto !important;
        }
        div[data-baseweb="select"] > div {
            min-height: 46px !important;
            height: auto !important;
            white-space: normal !important;
            padding-top: 4px !important;
            padding-bottom: 4px !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div[aria-selected="true"],
        div[data-baseweb="select"] div {
            white-space: normal !important;
            word-break: break-word !important;
            overflow: visible !important;
            text-overflow: unset !important;
            line-height: 1.45 !important;
        }

        /* 2. Popover Dropdown Menu List: Wide popover, full multi-line text wrapping */
        div[data-baseweb="popover"] {
            min-width: 100% !important;
            max-width: 95vw !important;
            width: auto !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6) !important;
        }
        ul[role="listbox"] {
            max-height: 480px !important;
            width: 100% !important;
            padding: 6px !important;
        }
        ul[role="listbox"] li,
        ul[role="listbox"] li[role="option"] {
            white-space: normal !important;
            word-break: break-word !important;
            overflow: visible !important;
            text-overflow: unset !important;
            height: auto !important;
            min-height: 44px !important;
            line-height: 1.45 !important;
            padding: 10px 14px !important;
            margin-bottom: 2px !important;
            border-bottom: 1px solid rgba(51, 65, 85, 0.4) !important;
        }
        ul[role="listbox"] li > div,
        ul[role="listbox"] li[role="option"] > div,
        ul[role="listbox"] li span {
            white-space: normal !important;
            word-break: break-word !important;
            overflow: visible !important;
            text-overflow: unset !important;
            line-height: 1.45 !important;
        }

        /* 3. Streamlit TextArea full text display */
        .stTextArea textarea {
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
            white-space: normal !important;
            word-break: break-word !important;
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Main application loop with landing page gating and centralized error handling."""
    st.set_page_config(
        page_title="ResearchLens AI — Academic Research Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_global_css()
    initialize_session_state()

    current_user = st.session_state.get("current_user")

    # If user is NOT logged in: show the public landing page with Sign In / Sign Up
    if not current_user:
        with st.sidebar:
            st.markdown(
                """
                <div style="text-align: center; padding: 12px 0 20px 0;">
                    <h2 style="margin: 0; font-size: 1.4rem; color: #38BDF8;">🔬 ResearchLens AI</h2>
                    <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94A3B8;">
                        Microsoft Foundry Academic Assistant
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info("🔒 **Access Gated**\n\nPlease Sign In or Create an Account on the main page to unlock all research tools.")
            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.6;">
                    <strong>Features available after login:</strong><br/>
                    • 📄 Interactive Paper Chat<br/>
                    • 🔬 7-Dimension Deep Analysis<br/>
                    • ⚖️ Multi-Paper Comparative Matrix<br/>
                    • 🔍 Grounded Research Gap Detection<br/>
                    • ❓ Novel Research Questions<br/>
                    • 📚 Publication-Grade Literature Review<br/>
                    • 🌐 Academic Multilingual Translation
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.divider()
            st.caption("ResearchLens AI • Microsoft Foundry Connected")

        render_landing_page()
        return

    # User IS logged in: Render the authenticated research workspace
    try:
        page = render_sidebar(current_user)

        # Top welcome banner
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(51, 65, 85, 0.6); border-radius: 8px; padding: 8px 16px; margin-bottom: 16px;">
                <span style="font-size: 0.88rem; color: #94A3B8;">
                    👤 {t('researcher')}: <strong style="color: #F8FAFC;">{current_user['name']}</strong> &bull; <span style="color: #38BDF8;">{current_user.get('institution', 'Academic Researcher')}</span>
                </span>
                <span style="font-size: 0.8rem; background: rgba(52, 211, 153, 0.15); color: #34D399; padding: 2px 10px; border-radius: 9999px; font-weight: 600;">
                    {t('workspace_active')}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if page == "💬 ResearchLens AI (Chat & Hub)":
            render_chat_page()
        elif page == "📄 Paper Deep Analysis View":
            render_analysis_page()
        elif page == "⚖️ Paper Comparison View":
            render_comparison_page()
        elif page == "🔍 Research Gaps View":
            render_gaps_page()
        elif page == "❓ Research Questions View":
            render_questions_page()
        elif page == "📚 Literature Review View":
            render_literature_review_page()

    except Exception as e:
        st.error(
            f"⚠️ An unexpected error occurred: **{str(e)}**\n\n"
            "Please check your inputs or configuration."
        )
        if settings.DEBUG:
            st.exception(e)


if __name__ == "__main__":
    main()
