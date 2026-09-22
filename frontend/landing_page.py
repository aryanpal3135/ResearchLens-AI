"""
ResearchLens AI - Public Landing Page & Authentication UI.
Presents a high-end academic SaaS landing page with feature showcase,
interactive Sign In / Sign Up forms, and 1-click Researcher Demo Login.
"""

from typing import Callable, Optional
import streamlit as st
from services.auth_service import AuthService


def render_landing_page(on_login_success: Optional[Callable] = None) -> None:
    """Renders the public landing page with authentication forms."""
    auth_service = AuthService()

    # Custom styling for the landing page
    st.markdown(
        """
        <style>
        .hero-container {
            text-align: center;
            padding: 30px 10px 20px 10px;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: #38BDF8;
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 16px;
            letter-spacing: 0.5px;
        }
        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #F8FAFC 0%, #38BDF8 50%, #818CF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 1.15rem;
            color: #94A3B8;
            max-width: 780px;
            margin: 0 auto 24px auto;
            line-height: 1.6;
        }
        .feature-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(51, 65, 85, 0.8);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            transition: all 0.2s ease-in-out;
            height: 100%;
        }
        .feature-card:hover {
            border-color: #38BDF8;
            box-shadow: 0 8px 24px -8px rgba(56, 189, 248, 0.25);
            transform: translateY(-2px);
        }
        .feature-icon {
            font-size: 1.8rem;
            margin-bottom: 8px;
        }
        .feature-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #F1F5F9;
            margin-bottom: 6px;
        }
        .feature-desc {
            font-size: 0.88rem;
            color: #94A3B8;
            line-height: 1.5;
        }
        .auth-container {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
        }
        .stats-badge {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(51, 65, 85, 0.6);
            border-radius: 10px;
            padding: 14px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 1. Top Navigation Bar
    nav_col1, nav_col2 = st.columns([7, 3])
    with nav_col1:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; padding: 8px 0;">
                <span style="font-size: 1.8rem;">🔬</span>
                <div>
                    <span style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC;">ResearchLens AI</span>
                    <span style="margin-left: 8px; font-size: 0.75rem; background: #0369A1; color: white; padding: 2px 8px; border-radius: 6px; font-weight: 600;">FOUNDRY V2</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav_col2:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 12px;">
                <span style="color: #34D399; font-size: 0.85rem; font-weight: 600;">🟢 Cloud Agents Live</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin: 8px 0 24px 0; border-color: rgba(51, 65, 85, 0.5);'>", unsafe_allow_html=True)

    # 2. Hero Section
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">⚡ POWERED BY MICROSOFT FOUNDRY & HYBRID RAG</div>
            <div class="hero-title">Evidence-Grounded AI Research Intelligence</div>
            <div class="hero-subtitle">
                Synthesize complex academic literature with mathematical rigor. Chat with papers using exact page-level citations, extract research gaps, conduct deep 7-dimension analyses, and generate publication-grade reviews.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Two-Column Layout: Left = Value Proposition & Stats, Right = Sign In / Sign Up Form
    left_col, right_col = st.columns([6, 5], gap="large")

    with left_col:
        st.markdown("### 🌟 Why Researchers Choose ResearchLens AI")
        st.markdown(
            """
            - **Cloud-Managed Agents:** Driven by Microsoft Foundry agents (`Paper-Chat-Agent` and `researchmate-gpt4-1-mini`).
            - **Zero Hallucination Tolerance:** Every assertion is bound to verified chunk provenance with page numbers and paragraph offsets.
            - **Hybrid RAG Engine:** Dense vector similarity (`text-embedding-3-large`, 3072 dims) fused with Sparse BM25 and Reciprocal Rank Fusion.
            - **Academic Multilingual Synthesis:** English, Hindi, French, and Spanish translation with 100% preservation of citation anchors.
            """
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        # Stats Badges
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(
                """
                <div class="stats-badge">
                    <div style="font-size: 1.5rem; font-weight: 800; color: #38BDF8;">3072-D</div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">Dense Embeddings</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                """
                <div class="stats-badge">
                    <div style="font-size: 1.5rem; font-weight: 800; color: #818CF8;">10 Dims</div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">Paper Comparison</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                """
                <div class="stats-badge">
                    <div style="font-size: 1.5rem; font-weight: 800; color: #34D399;">4 Langs</div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">Academic Provenance</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown(
            """
            <div style="margin-bottom: 12px;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC;">🔐 Researcher Portal</span>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">
                    Sign in to access your private research workspace, papers, and AI agents.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        auth_tab_login, auth_tab_signup = st.tabs(["🔑 Sign In", "📝 Create Account"])

        # Tab 1: Log In
        with auth_tab_login:
            with st.form("login_form", clear_on_submit=False):
                email_input = st.text_input("Academic or Work Email:", placeholder="researcher@university.edu", key="login_email")
                password_input = st.text_input("Password:", type="password", placeholder="••••••••", key="login_password")
                login_submitted = st.form_submit_button("Sign In to Workspace 🚀", use_container_width=True, type="primary")

                if login_submitted:
                    success, msg, user = auth_service.authenticate_user(email_input, password_input)
                    if success and user:
                        st.session_state["current_user"] = user
                        st.success(f"Welcome back, {user['name']}!")
                        if on_login_success:
                            on_login_success()
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("<div style='text-align: center; margin: 12px 0 8px 0; color: #64748B; font-size: 0.82rem;'>OR QUICK PREVIEW</div>", unsafe_allow_html=True)
            if st.button("⚡ 1-Click Researcher Demo Login", key="quick_demo_btn", use_container_width=True):
                demo_user = auth_service.get_or_create_demo_user()
                st.session_state["current_user"] = demo_user
                st.success(f"Signed in as Demo Researcher ({demo_user['name']})")
                if on_login_success:
                    on_login_success()
                st.rerun()

        # Tab 2: Create Account
        with auth_tab_signup:
            with st.form("signup_form", clear_on_submit=False):
                new_name = st.text_input("Full Name:", placeholder="Dr. Jane Doe", key="signup_name")
                new_email = st.text_input("Email Address:", placeholder="jane.doe@lab.org", key="signup_email")
                new_institution = st.text_input("Institution / Organization:", placeholder="MIT CSAIL / DeepMind", key="signup_inst")
                new_password = st.text_input("Password (min 6 characters):", type="password", placeholder="••••••••", key="signup_password")
                signup_submitted = st.form_submit_button("Create Researcher Account ✨", use_container_width=True, type="primary")

                if signup_submitted:
                    success, msg, user = auth_service.register_user(
                        name=new_name,
                        email=new_email,
                        password=new_password,
                        institution=new_institution,
                    )
                    if success and user:
                        st.session_state["current_user"] = user
                        st.success("Account created successfully! Redirecting...")
                        if on_login_success:
                            on_login_success()
                        st.rerun()
                    else:
                        st.error(msg)

    st.markdown("<br/><hr style='border-color: rgba(51, 65, 85, 0.4); margin: 32px 0;'><br/>", unsafe_allow_html=True)

    # 4. Feature Showcase Grid
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 28px;">
            <h2 style="font-size: 1.8rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
                Comprehensive Academic Intelligence Suite
            </h2>
            <p style="color: #94A3B8; font-size: 0.95rem;">
                Built specifically for graduate researchers, principal investigators, and enterprise R&D teams.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">Paper Chat & Q&A</div>
                <div class="feature-desc">
                    Conversational agent backed by <code>Paper-Chat-Agent</code> v2. Answers questions strictly using retrieved chunks and provides inline clickable citations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">7-Dimension Analysis</div>
                <div class="feature-desc">
                    Automatically extracts Problem Statement, Methodology, Theoretical Basis, Empirical Findings, Limitations, Datasets, and Real-world Applications.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⚖️</div>
                <div class="feature-title">Multi-Paper Comparison</div>
                <div class="feature-desc">
                    Cross-examines 2+ research papers across 10 academic dimensions with strict context isolation to prevent hallucinated overlap.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Research Gap Detection</div>
                <div class="feature-desc">
                    Identifies unaddressed limitations, theoretical bounds, and empirical blind spots directly documented in the source literature.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📚</div>
                <div class="feature-title">Literature Review Synthesis</div>
                <div class="feature-desc">
                    Generates formal publication-grade literature reviews with thematic synthesis, methodological critique, and verified provenance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <div class="feature-title">Academic Translation</div>
                <div class="feature-desc">
                    One-click translation into English, Hindi, French, and Spanish with strict identity preservation of in-text citation tags.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/><br/>", unsafe_allow_html=True)

    # 5. Footer
    st.markdown(
        """
        <div style="text-align: center; color: #64748B; font-size: 0.8rem; padding: 20px 0;">
            ResearchLens AI • Powered by Microsoft Foundry Project <code>researchmate</code> • Southeast Asia
        </div>
        """,
        unsafe_allow_html=True,
    )
