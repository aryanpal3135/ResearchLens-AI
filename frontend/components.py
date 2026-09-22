"""
Reusable UI components and styles for Streamlit frontend.
"""

import streamlit as st


def render_header(title: str, subtitle: str, badge: str = "Phase 1: Architecture & UI"):
    """Renders a standard, professional page header."""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(title)
        st.caption(subtitle)
    with col2:
        st.markdown(
            f"""
            <div style="background-color: #1E293B; color: #38BDF8; padding: 6px 12px;
                        border-radius: 8px; font-size: 0.82rem; font-weight: 600;
                        text-align: center; border: 1px solid #334155; margin-top: 15px;">
                {badge}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()


def render_info_card(title: str, description: str, icon: str = "ℹ️"):
    """Displays a stylized informative callout."""
    st.info(f"{icon} **{title}**\n\n{description}")


def render_evidence_badge(is_explicit: bool):
    """Renders an Explicit vs Inferred badge."""
    if is_explicit:
        return '<span style="background-color:#064E3B;color:#34D399;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;">Explicitly Stated</span>'
    return '<span style="background-color:#431407;color:#FB923C;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;">AI Inferred</span>'
