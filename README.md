# 🔬 ResearchLens AI

### Advanced Multilingual Academic Research Intelligence & Literature Synthesis Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-FF4B4B.svg)](https://streamlit.io/)
[![Microsoft Foundry](https://img.shields.io/badge/Azure%20AI-Foundry%20%2F%20OpenAI-0078D4.svg)](https://azure.microsoft.com/)
[![Azure Speech](https://img.shields.io/badge/Azure%20AI-Speech%20Services-0089D6.svg)](https://azure.microsoft.com/services/cognitive-services/speech-services/)
[![Tests](https://img.shields.io/badge/Tests-215%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 1. Overview

**ResearchLens AI** is an enterprise-grade academic research intelligence and literature synthesis platform. Unlike simple document QA wrappers or generic PDF summarizers, ResearchLens AI is built to automate rigorous academic literature workflows: deep document ingestion, section-aware chunking, hybrid dense/sparse vector retrieval, structured 18-parameter paper analysis, 6-dimensional research gap detection, hypothesis generation, scholarly literature review synthesis, real-time voice interaction, and multi-language translation.

The platform is designed around **Microsoft Foundry**, **Azure OpenAI Service**, and **Azure AI Speech Services**, delivering low latency, verifiable evidence provenance, and strict cross-paper isolation.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             ResearchLens AI Workflow                            │
└──────────────────────────────────────────────────────────────────────────────────┘
   PDF Upload ➔ Deep Text & Table Extraction ➔ Section-Aware Sliding Chunking
        ➔ 3072d Dense Vector + BM25 Sparse Index ➔ Reciprocal Rank Fusion (RRF)
             ➔ 18-Parameter Academic Extraction (Explicit vs. Inferred)
                  ➔ Multi-Paper Comparison Matrix & Contradiction Mapping
                       ➔ 6-Dimensional Systematic Gap Detection
                            ➔ AI Research Questions & Future Directions Roadmap
                                 ➔ 11-Section Scholarly Literature Review
                                      ➔ Multi-Language Translation (EN/HI/FR/ES)
                                           ➔ Voice Input (STT) & Read Aloud (TTS)
```

---

## ✨ 2. Key Features

The platform provides 18 fully implemented, end-to-end working features:

1. **Academic PDF Ingestion & Robust Extraction**
   - High-fidelity extraction powered by PyMuPDF (`fitz`) and PDFMiner.
   - Extracts structured sections, abstracts, full text, tables (converted to Markdown), and formal bibliographic references without truncation.

2. **Metadata Extraction & Verification**
   - Automatically parses paper title, authors, publication year, domain, and abstract.
   - Graceful fallback heuristics for non-standard academic paper formats.

3. **Section-Aware Semantic Chunking**
   - Retains subsection hierarchy and boundaries with a 50–100 token sliding paragraph overlap.
   - Generates dedicated chunks for tables and references to prevent context bleeding across mathematical formulations and data matrices.

4. **Dense Neural Embeddings & Sparse Lexical Indexing**
   - Dense representations generated via Azure OpenAI `text-embedding-3-large` (3,072 dimensions) with intelligent in-memory batching (128 chunks/batch).
   - In-memory BM25Okapi sparse lexical index ($k_1=1.5, b=0.75$) with sub-linear term frequency.

5. **Reciprocal Rank Fusion (RRF) Hybrid Retrieval**
   - Combines dense cosine similarity with sparse BM25 scoring:
     $$\text{RRF Score}(d) = \sum_{c \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_c(d)}$$
   - Enforces strict cross-paper isolation during targeted single-paper queries.

6. **Structured 18-Parameter Single-Paper Extraction**
   - Standardized academic extraction across 18 parameters: Title, Authors, Year, Domain, Problem, Objective, Research Questions, Methodology, Dataset, Dataset Size, Models/Algorithms, Features/Variables, Evaluation Metrics, Main Results, Key Findings, Limitations, Future Work, Keywords.
   - Visually tags each parameter as **Explicitly Stated** (with exact page citations) or **AI-Inferred**.

7. **Multi-Paper Comparative Matrix**
   - Side-by-side synthesis across datasets, models, metrics, and trade-offs.
   - Detects commonalities, complementary methodologies, and empirical divergences across uploaded papers.

8. **6-Dimensional Systematic Research Gap Discovery**
   - Categorizes open academic challenges into 6 dimensions:
     - *Repeated Limitations:* Recurring constraints noted across independent studies.
     - *Underexplored Areas:* Critical problem spaces or modalities neglected in current work.
     - *Methodological Gaps:* Model architectures or training paradigms lacking empirical validation.
     - *Dataset Gaps:* Shortcomings in dataset scale, diversity, or real-world coverage.
     - *Evaluation Gaps:* Missing statistical significance tests or narrow evaluation metrics.
     - *Contradictory Findings:* Empirical disagreements between published studies presented neutrally.

9. **AI Research Question (RQ) Formulation**
   - Derives actionable, high-impact research questions (RQ1, RQ2...) directly mapped to identified literature gaps.
   - Clearly labeled with academic AI-generation disclaimers.

10. **Future Directions Roadmap**
    - Suggests concrete thesis roadmaps including proposed methodologies, potential benchmarks, and expected scientific contributions.

11. **11-Section Scholarly Literature Review Synthesizer**
    - Generates comprehensive academic reviews following formal scholarly structures with verified inline citation anchors (`[Paper Title, p. X]`).

12. **Grounded RAG Conversational Assistant**
    - Context-grounded Q&A over the entire paper corpus or individual papers.
    - Every claim is tied to chunk-level provenance with clickable citations.

13. **Multilingual Translation & Full UI Localization**
    - High-fidelity academic translation across 4 supported languages:
      - 🇬🇧 **English**
      - 🇮🇳 **Hindi (हिन्दी)**
      - 🇫🇷 **French (Français)**
      - 🇪🇸 **Spanish (Español)**
    - Supports dynamic whole-app UI localization and bilingual comparative reading modes without mutating citations or author names.

14. **Real-Time Azure Speech-to-Text (STT / Voice Input)**
    - Natural voice prompting powered by Azure AI Speech Services.
    - In-memory audio stream processing (`PushAudioInputStream`) with ChatGPT-style auto-submit and dynamic widget lifecycle management.

15. **Real-Time Azure Text-to-Speech (TTS / Read Aloud)**
    - Converts assistant answers into high-clarity neural speech using Azure AI Speech Services.
    - Integrated in-memory WAV caching for immediate playback via the Streamlit audio player.

16. **Scientific Ground-Truth Benchmark Evaluation Suite**
    - Built-in evaluation dashboard benchmarking retrieval accuracy (MRR, Precision@5, Recall@5, Hit@5) and latency across standardized ground-truth queries.

17. **Multi-User Authentication & Session Management**
    - Secure SQLite-backed user authentication with salted PBKDF2 password hashing.
    - Fast session caching (<50ms login response) and independent per-user workspaces.

18. **Interactive Streamlit Dashboard**
    - Real-time corpus statistics, upload management, paper inspector, token usage monitoring, and system health status.

---

## 🏗️ 3. Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                               Streamlit Frontend                              │
│   Dashboard │ Upload │ Analysis │ Comparison │ Gaps │ Review │ Chat │ Voice   │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           Application & Service Layer                         │
│  ┌──────────────────────┬────────────────────────┬─────────────────────────┐  │
│  │   PDFProcessor       │    DocumentChunker     │    RetrievalEngine      │  │
│  │   (PyMuPDF/PDFMiner) │  (Section-Aware+Slide) │    (RRF Dense + BM25)   │  │
│  ├──────────────────────┼────────────────────────┼─────────────────────────┤  │
│  │   PaperAnalyzer      │   GapDetector          │   SpeechService         │  │
│  │   (18 Parameters)    │   (6-Dimensional)      │   (Azure STT + TTS)     │  │
│  └──────────────────────┴────────────────────────┴─────────────────────────┘  │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
          ┌────────────────────────────┴───────────────────────────┐
          ▼                                                        ▼
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│   Azure OpenAI / Foundry         │     │   Azure AI Speech Services       │
│   • GPT-4o (Chat / Synthesis)    │     │   • Speech-to-Text (Streaming)   │
│   • text-embedding-3-large (3072)│     │   • Neural Text-to-Speech (TTS)  │
└──────────────────────────────────┘     └──────────────────────────────────┘
```

---

## ☁️ 4. Microsoft Foundry & Azure AI Integration

ResearchLens AI integrates directly with Microsoft Azure AI services:

| Component | Azure Service | Model / Configuration | Function |
|---|---|---|---|
| **Language Model** | Azure OpenAI Service | `gpt-4o` | Structured extraction, synthesis, chat, translation |
| **Embeddings** | Azure OpenAI Service | `text-embedding-3-large` (3,072d) | High-dimensional semantic representation |
| **Speech-to-Text** | Azure AI Speech | Streaming Audio Recognizer | Real-time speech input & voice prompting |
| **Text-to-Speech** | Azure AI Speech | Neural Speech Synthesizer | Natural voice read-aloud playback |
| **Agent Layer** | Microsoft Foundry | Azure AI Agent SDK | Autonomous research analysis orchestration |

---

## 🚀 5. Getting Started

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- Active Azure OpenAI & Azure AI Speech resources (or use offline/local mode for tests)

### Step 1: Clone the Repository
```bash
git clone https://github.com/aryanpal3135/ResearchLens-AI.git
cd ResearchLens-AI
```

### Step 2: Create a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy the template file to `.env`:
```bash
cp .env.example .env
```
Fill in your Azure credentials in `.env`:
```env
# Azure OpenAI / Foundry
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Azure AI Speech
AZURE_SPEECH_ENDPOINT=https://<your-speech-resource>.cognitiveservices.azure.com/
AZURE_SPEECH_API_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=eastus
```

### Step 5: Launch the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Default local login credentials can be registered instantly from the login screen.

---

## 🧪 6. Verification & Test Suite

The repository contains an exhaustive automated test suite covering all functional pipelines:

```bash
python -m pytest -q
```

### Test Results
```
215 passed in 79.88s (100% pass rate)
```

Test coverage includes:
- **`tests/test_pdf.py`**: PDF ingestion, metadata parsing, table preservation, and error handling.
- **`tests/test_retrieval.py`**: Chunker windowing, BM25 indexing, cosine similarity, RRF scoring, and cross-paper isolation.
- **`tests/test_analysis.py`**: 18-parameter Pydantic validation, explicit vs inferred badge attribution.
- **`tests/test_gaps.py`**: 6-dimensional gap categorization and contradiction detection logic.
- **`tests/test_translation.py`**: 4-language constraint enforcement, zero citation mutation, and bilingual mappings.
- **`tests/test_speech.py`**: Audio stream handling, WAV formatting, and TTS caching.

---

## 📁 7. Project Structure

```
ResearchLens-AI/
├── app.py                     # Streamlit application entry point & page router
├── requirements.txt           # Python package dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Exclusion rules (secrets, caches, PDFs, database)
├── README.md                  # Comprehensive project documentation
│
├── config/
│   ├── __init__.py
│   └── settings.py            # Centralized settings and Azure configuration
│
├── frontend/                  # Streamlit UI Pages & Views
│   ├── __init__.py
│   ├── auth.py                # User login, registration & session auth
│   ├── components.py          # Shared UI widgets, badges, and headers
│   ├── dashboard.py           # Corpus overview, stats & charts
│   ├── upload.py              # PDF upload, extraction, and validation view
│   ├── analysis.py            # Structured 18-parameter single-paper view
│   ├── comparison.py          # Side-by-side comparative matrix view
│   ├── gaps.py                # 6-dimensional research gap detection view
│   ├── questions.py           # AI research questions & future pathways
│   ├── literature_review.py   # 11-section scholarly literature review view
│   ├── chat.py                # Grounded RAG conversational interface & voice input
│   ├── translation.py         # Multilingual translation & bilingual view
│   └── evaluation.py          # Scientific benchmark evaluation suite
│
├── services/                  # Business Logic & Core Engines
│   ├── __init__.py
│   ├── auth_service.py        # SQLite + PBKDF2 authentication service
│   ├── pdf_processor.py       # PyMuPDF text & table extraction engine
│   ├── chunking.py            # Section-aware semantic chunking
│   ├── embeddings.py          # Azure OpenAI dense embeddings service
│   ├── retrieval.py           # Hybrid RAG engine (Dense + BM25 + RRF)
│   ├── foundry_client.py      # Microsoft Foundry & Azure OpenAI client
│   ├── paper_analyzer.py      # 18-parameter structured extraction engine
│   ├── comparison_engine.py   # Multi-paper comparative synthesizer
│   ├── gap_detector.py        # 6-dimensional gap discovery engine
│   ├── question_generator.py  # Research question & thesis pathway generator
│   ├── literature_review.py   # Scholarly literature review synthesizer
│   ├── translation.py         # Academic 4-language translation service
│   ├── speech_service.py      # Azure AI Speech (STT + TTS) streaming service
│   └── citation_service.py    # Evidence verification & citation formatter
│
├── models/                    # Pydantic Schemas & Data Models
│   ├── __init__.py
│   ├── paper.py               # PaperDocument, Chunk & Metadata schemas
│   ├── analysis.py            # 18-Parameter analysis & Citation schemas
│   └── research_gap.py        # Gap, Question, & FutureDirection schemas
│
├── prompts/                   # Structured Academic Prompts
│   ├── paper_analysis.txt     # 18-parameter extraction prompt
│   ├── comparison.txt         # Multi-paper comparative prompt
│   ├── gap_detection.txt      # 6-dimensional gap discovery prompt
│   ├── question_generation.txt# Research question formulation prompt
│   ├── literature_review.txt  # 11-section scholarly review prompt
│   └── translation.txt        # Academic translation prompt
│
└── tests/                     # Automated Test Suite (215 Tests)
    ├── test_pdf.py
    ├── test_retrieval.py
    ├── test_analysis.py
    ├── test_gaps.py
    ├── test_questions.py
    ├── test_literature_review.py
    ├── test_translation.py
    ├── test_speech.py
    └── test_auth.py
```

---

## 🔒 8. Security & Privacy

- **Zero Hardcoded Credentials:** All API keys, endpoints, and credentials are read strictly from environment variables or Streamlit secrets.
- **Repository Safety:** `.env`, user databases (`users.db`), processed artifacts, local vector stores, and research PDFs are strictly excluded via `.gitignore`.
- **Credential Storage:** User passwords are encrypted using salted PBKDF2 key derivation with SHA-256 before storage.
- **Academic Citation Integrity:** Translation and synthesis pipelines preserve verbatim bibliographic references, DOI markers, and author names.

---

## ⚠️ 9. Academic Boundaries & Limitations

- **Evidence Boundary:** The RAG assistant and synthesis engines are constrained to retrieve only from uploaded and indexed papers. When information is absent from the papers, the system reports it explicitly rather than hallucinating.
- **AI-Inferred Distinction:** All single-paper parameters and research questions distinguish explicitly between facts directly stated in the text and AI inferences.
- **Academic Support:** ResearchLens AI is built as an academic research accelerator to assist scholars and evaluators; it does not replace peer review or critical scientific analysis.

---

## 📄 10. License

This project is licensed under the [MIT License](LICENSE).
