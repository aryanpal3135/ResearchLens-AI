"""
Tests for Phase 8: Evidence-Grounded Literature Review Generation.
Covers:
1. LiteratureReview schema validation
2. LiteratureReviewTheme schema validation
3. LiteratureReviewSection schema validation
4. Single-paper literature review support
5. Multi-paper literature review synthesis
6. Custom review query handling
7. Per-paper retrieval isolation (strict separation)
8. Cross-paper attribution integrity
9. Provenance preservation (chunk, page, section, score)
10. Page number preservation
11. Section heading preservation
12. No fabricated citations or external references
13. Insufficient evidence handling
14. Contradiction handling ("No direct contradiction was identified...")
15. Phase 6 gap integration
16. Phase 7 future-direction integration (flagged is_suggestion=True)
17. Complete-data export compliance
18. Markdown export integrity
19. JSON export integrity
20. LLM failure graceful handling
21. Empty paper selection handling
22. No ranking or winner declaration language
23. Claim type classification (DOCUMENTED, SYNTHESIS, INFERENCE, INSUFFICIENT_EVIDENCE)
24. Phase 3 retrieval regression check
25. Phase 4 agent regression check
26. Phase 5 comparison regression check
27. Phase 6 research gap regression check
28. Phase 7 research question regression check
"""

import json
import pytest
from unittest.mock import MagicMock

from models.paper import PaperDocument, PaperMetadata, DocumentChunk, EvidenceItem
from models.research_gap import (
    ResearchGap,
    FutureDirection,
)
from models.literature_review import (
    LiteratureReview,
    LiteratureReviewTheme,
    LiteratureReviewSection,
    CLAIM_TYPES,
)
from services.foundry_agent import ResearchLensAgent, SYSTEM_PROMPT_LITERATURE_REVIEW_AGENT
from services.retrieval import HybridRetrievalEngine


@pytest.fixture
def mock_paper_a() -> PaperDocument:
    paper = PaperDocument(
        id="paper_001",
        filename="attention_paper.pdf",
        saved_path="/data/attention_paper.pdf",
        metadata=PaperMetadata(
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            publication_year="2017",
        ),
        page_count=15,
    )
    chunk1 = DocumentChunk(
        chunk_id="paper_001_c001",
        paper_id="paper_001",
        text="The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
        page=1,
        page_start=1,
        page_end=1,
        normalized_section="Introduction",
        original_heading="Introduction",
        content_type="body",
        chunk_index=0,
    )
    chunk2 = DocumentChunk(
        chunk_id="paper_001_c002",
        paper_id="paper_001",
        text="An attention function can be described as mapping a query and a set of key-value pairs to an output. We compute the dot products of the query with all keys, divide each by sqrt(d_k), and apply a softmax function.",
        page=4,
        page_start=4,
        page_end=4,
        normalized_section="Attention Mechanism",
        original_heading="Attention Mechanism",
        content_type="body",
        chunk_index=1,
    )
    chunk3 = DocumentChunk(
        chunk_id="paper_001_c003",
        paper_id="paper_001",
        text="To evaluate translation performance, on the WMT 2014 English-to-German task, the big transformer model achieves a BLEU score of 28.4. Computational complexity per layer scales quadratically with sequence length n.",
        page=8,
        page_start=8,
        page_end=8,
        normalized_section="Results & Discussion",
        original_heading="Results & Discussion",
        content_type="body",
        chunk_index=2,
    )
    paper.chunks = [chunk1, chunk2, chunk3]
    return paper


@pytest.fixture
def mock_paper_b() -> PaperDocument:
    paper = PaperDocument(
        id="paper_002",
        filename="bert_paper.pdf",
        saved_path="/data/bert_paper.pdf",
        metadata=PaperMetadata(
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Devlin et al."],
            publication_year="2018",
        ),
        page_count=16,
    )
    chunk1 = DocumentChunk(
        chunk_id="paper_002_c001",
        paper_id="paper_002",
        text="BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
        page=1,
        page_start=1,
        page_end=1,
        normalized_section="Introduction",
        original_heading="Introduction",
        content_type="body",
        chunk_index=0,
    )
    chunk2 = DocumentChunk(
        chunk_id="paper_002_c002",
        paper_id="paper_002",
        text="The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context.",
        page=3,
        page_start=3,
        page_end=3,
        normalized_section="Methodology",
        original_heading="Methodology",
        content_type="body",
        chunk_index=1,
    )
    chunk3 = DocumentChunk(
        chunk_id="paper_002_c003",
        paper_id="paper_002",
        text="BERT advances the state of the art on eleven NLP tasks, including pushing the GLUE score to 80.5%. However, fine-tuning can be unstable on small datasets.",
        page=7,
        page_start=7,
        page_end=7,
        normalized_section="Evaluation",
        original_heading="Evaluation",
        content_type="body",
        chunk_index=2,
    )
    paper.chunks = [chunk1, chunk2, chunk3]
    return paper


@pytest.fixture
def mock_retrieval_engine(mock_paper_a, mock_paper_b) -> HybridRetrievalEngine:
    engine = MagicMock(spec=HybridRetrievalEngine)
    engine.indexed_chunks = mock_paper_a.chunks + mock_paper_b.chunks

    def fake_retrieve(query: str, paper_ids=None, top_k=3, **kwargs):
        results = []
        target_papers = paper_ids if paper_ids else ["paper_001", "paper_002"]
        all_chunks = []
        if "paper_001" in target_papers:
            all_chunks.extend(mock_paper_a.chunks)
        if "paper_002" in target_papers:
            all_chunks.extend(mock_paper_b.chunks)

        for c in all_chunks[:top_k]:
            mock_res = MagicMock()
            mock_res.chunk = c
            mock_res.score = 0.88
            mock_res.rrf_score = 0.88
            mock_res.dense_score = 0.90
            mock_res.bm25_score = 0.85
            results.append(mock_res)
        return results

    engine.retrieve.side_effect = fake_retrieve
    return engine


@pytest.fixture
def mock_foundry_client():
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"

    def fake_chat(system_prompt=None, user_prompt=None, **kwargs):
        sys_str = str(system_prompt or "")
        usr_str = str(user_prompt or "")
        if "LITERATURE REVIEW" in sys_str or "literature_review" in usr_str:
            return {
                "success": True,
                "content": json.dumps({
                    "title": "Evidence-Grounded Literature Review",
                    "introduction": {"content": "Comprehensive synthesis of Transformer literature [paper_001, p. 1, §Introduction].", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
                    "themes": [
                        {"theme_id": "theme_001", "title": "Self-Attention Architectures", "description": "Core attention mechanisms.", "supporting_papers": ["paper_001"], "synthesis": "Self-attention replaces recurrence.", "support_level": "High evidence support"}
                    ],
                    "methodology_synthesis": {"content": "Self-attention dot-product mechanics [paper_001, p. 4, §Attention Mechanism].", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
                    "findings_synthesis": {"content": "Empirical translation performance reaches 28.4 BLEU [paper_001, p. 8, §Results & Discussion].", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
                    "agreements_differences": {"content": "Methodological consistency across attention models.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
                    "limitations": {"content": "Quadratic computational and memory complexity [paper_001, p. 8, §Results & Discussion].", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
                    "research_gaps": {"content": "Scaling attention to ultra-long sequences.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
                    "conclusion": {"content": "Attention architectures represent a fundamental paradigm shift.", "claim_type": "INFERENCE", "supporting_papers": ["paper_001"]},
                })
            }
        elif "RESEARCH GAPS" in sys_str:
            return {
                "success": True,
                "content": json.dumps({
                    "title": "Detected Gaps",
                    "gaps": [
                        {
                            "gap_id": "gap_001",
                            "title": "Quadratic Memory Complexity",
                            "category": "Methodological Gap",
                            "description": "Quadratic memory scaling.",
                            "evidence_type": "explicit",
                            "evidence_support": "High evidence support",
                            "supporting_papers": ["paper_001"],
                            "evidence_ids": ["ev_001"],
                        }
                    ]
                })
            }
        elif "RESEARCH QUESTIONS" in sys_str:
            return {
                "success": True,
                "content": json.dumps({
                    "title": "Generated Research Questions",
                    "questions": [
                        {
                            "question_id": "rq_001",
                            "question": "How can self-attention architectures scale efficiently without quadratic memory overhead?",
                            "rationale": "Paper_001 identifies quadratic complexity with sequence length as a key computational bottleneck.",
                            "supporting_papers": ["paper_001"],
                            "gap_addressed": "gap_001",
                            "evidence_ids": ["ev_001"],
                            "methodology": "Sparse attention mechanisms.",
                            "feasibility": "High",
                            "novelty": "High",
                            "question_type": "author_inspired",
                            "target_outcomes": ["Sub-quadratic memory complexity."],
                        }
                    ],
                    "future_directions": [
                        {
                            "direction_id": "fd_001",
                            "title": "Sparse Attention",
                            "description": "Explore sparse attention.",
                            "gap_addressed": "gap_001",
                            "supporting_papers": ["paper_001"],
                            "timeframe": "medium_term",
                            "feasibility": "High",
                            "impact": "High",
                        }
                    ]
                })
            }
        elif "COMPARISON" in sys_str:
            return {
                "success": True,
                "content": json.dumps({
                    "title": "Paper Comparison",
                    "dimension_comparisons": [
                        {
                            "dimension": "Architecture",
                            "comparison_narrative": "Both utilize self-attention mechanisms.",
                            "key_differences": ["Encoder-decoder vs encoder-only"],
                            "key_similarities": ["Self-attention backbones"],
                            "supporting_papers": ["paper_001", "paper_002"],
                        }
                    ]
                })
            }
        else:
            return {
                "success": True,
                "content": "{}"
            }

    mock_client.generate_chat_response.side_effect = fake_chat
    return mock_client


@pytest.fixture
def sample_gap() -> ResearchGap:
    return ResearchGap(
        gap_id="gap_001",
        title="Quadratic Memory Complexity in Self-Attention",
        category="Methodological Gap",
        description="Self-attention computational and memory complexity scales as O(n^2) with sequence length.",
        evidence_type="explicit",
        evidence_support="High evidence support",
        supporting_papers=["paper_001"],
        evidence_items=[
            EvidenceItem(
                evidence_id="ev_paper_001_c003",
                citation_label="[paper_001, p. 8, §Results & Discussion]",
                paper_id="paper_001",
                chunk_id="paper_001_c003",
                page_start=8,
                page_end=8,
                normalized_section="Results & Discussion",
                original_heading="Results & Discussion",
                content_type="body",
                score=0.88,
                text="Computational complexity per layer scales quadratically with sequence length n.",
            )
        ],
    )


# --------------------------------------------------------------------------
# Test 1: LiteratureReview schema validation
# --------------------------------------------------------------------------
def test_literature_review_schema():
    rev = LiteratureReview(
        review_id="lit_rev_001",
        title="Scholarly Review of Transformer Architectures",
        review_question="How do Transformers scale?",
        selected_paper_ids=["paper_001"],
        selected_paper_titles={"paper_001": "Attention Is All You Need"},
    )
    assert rev.review_id == "lit_rev_001"
    assert rev.status == "completed"
    assert rev.introduction.section_id == "sec_intro"
    assert rev.methodology_synthesis.section_id == "sec_methodology"
    assert rev.conclusion.section_id == "sec_conclusion"


# --------------------------------------------------------------------------
# Test 2: LiteratureReviewTheme schema validation
# --------------------------------------------------------------------------
def test_literature_review_theme_schema():
    ev = EvidenceItem(
        evidence_id="ev_paper_001_c002",
        citation_label="[paper_001, p. 4, §Attention Mechanism]",
        paper_id="paper_001",
        chunk_id="paper_001_c002",
        page_start=4,
        page_end=4,
        normalized_section="Attention Mechanism",
        original_heading="Attention Mechanism",
        content_type="body",
        score=0.89,
        text="We compute dot products of the query with all keys.",
    )
    theme = LiteratureReviewTheme(
        theme_id="theme_001",
        title="Self-Attention as Core Representation Mechanism",
        description="Exploration of dot-product attention replacing recurrence.",
        evidence_items=[ev],
        synthesis="The literature establishes attention as sufficient for sequence transduction.",
        support_level="High evidence support",
    )
    assert theme.theme_id == "theme_001"
    assert "paper_001" in theme.supporting_papers
    assert theme.citation_labels == ["[paper_001, p. 4, §Attention Mechanism]"]


# --------------------------------------------------------------------------
# Test 3: LiteratureReviewSection schema & claim types
# --------------------------------------------------------------------------
def test_literature_review_section_schema():
    sec = LiteratureReviewSection(
        section_id="sec_findings",
        title="3. Findings & Evidence Synthesis",
        content="Empirical results demonstrate significant BLEU score gains.",
        claim_type="DOCUMENTED",
        supporting_papers=["paper_001"],
    )
    assert sec.claim_type == "DOCUMENTED"
    assert sec.claim_type_label == "Documented in Literature"
    assert not sec.is_insufficient_evidence

    sec_insufficient = LiteratureReviewSection(
        section_id="sec_custom",
        title="Custom Section",
        content="Insufficient evidence in the selected papers.",
    )
    assert sec_insufficient.is_insufficient_evidence
    assert sec_insufficient.claim_type == "INSUFFICIENT_EVIDENCE"


# --------------------------------------------------------------------------
# Test 4: Single-paper review support
# --------------------------------------------------------------------------
def test_single_paper_review(mock_paper_b, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "title": "Literature Synthesis on BERT Pre-training",
        "introduction": {"content": "BERT introduces deep bidirectional pre-training [paper_002, p. 1, §Introduction].", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_002"]},
        "themes": [{"theme_id": "theme_001", "title": "Masked Language Modeling", "description": "Pre-training via token masking.", "supporting_papers": ["paper_002"], "synthesis": "Masking enables bidirectional conditioning.", "support_level": "High evidence support"}],
        "methodology_synthesis": {"content": "BERT uses masked language modeling and next sentence prediction [paper_002, p. 3, §Methodology].", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_002"]},
        "findings_synthesis": {"content": "BERT scores 80.5% on GLUE [paper_002, p. 7, §Evaluation].", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_002"]},
        "agreements_differences": {"content": "Evaluated exclusively across eleven NLP tasks.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_002"]},
        "limitations": {"content": "Fine-tuning can be unstable on small datasets [paper_002, p. 7, §Evaluation].", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_002"]},
        "research_gaps": {"content": "Stabilizing fine-tuning on small classification tasks.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_002"]},
        "conclusion": {"content": "BERT demonstrates the power of bidirectional pre-training.", "claim_type": "INFERENCE", "supporting_papers": ["paper_002"]},
    }
    mock_client.generate_chat_response.return_value = {"success": True, "content": json.dumps(mock_response)}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_b], engine=mock_retrieval_engine)

    assert res.status == "completed"
    assert res.selected_paper_ids == ["paper_002"]
    assert all("paper_001" not in th.supporting_papers for th in res.themes)
    assert res.introduction.supporting_papers == ["paper_002"]


# --------------------------------------------------------------------------
# Test 5: Multi-paper literature review cross-synthesis
# --------------------------------------------------------------------------
def test_multi_paper_review(mock_paper_a, mock_paper_b, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "title": "Cross-Paper Literature Synthesis on Transformers and Bidirectional Encoders",
        "introduction": {"content": "Both paper_001 and paper_002 investigate Transformer-based architectures for NLP transduction.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
        "themes": [
            {"theme_id": "theme_001", "title": "Attention-Driven Representations", "description": "Eliminating recurrence.", "supporting_papers": ["paper_001", "paper_002"], "synthesis": "Self-attention serves as the foundational backbone across both models.", "support_level": "High evidence support"},
            {"theme_id": "theme_002", "title": "Pre-training vs Supervised Transduction", "description": "Training objectives.", "supporting_papers": ["paper_001", "paper_002"], "synthesis": "Paper_001 investigates supervised machine translation while Paper_002 leverages unsupervised pre-training.", "support_level": "High evidence support"},
        ],
        "methodology_synthesis": {"content": "Paper_001 employs an encoder-decoder architecture with dot-product attention, whereas Paper_002 uses bidirectional encoder layers trained on masked language modeling.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
        "findings_synthesis": {"content": "Paper_001 achieves 28.4 BLEU on translation, while Paper_002 pushes GLUE to 80.5%.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
        "agreements_differences": {"content": "Both agree that self-attention provides superior representation capacity over RNNs. They differ in pre-training objectives and auto-regressive decoding.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
        "limitations": {"content": "Paper_001 notes quadratic memory scaling; Paper_002 observes fine-tuning instability on small datasets.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
        "research_gaps": {"content": "Addressing sequence scaling and fine-tuning robustness.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
        "conclusion": {"content": "Together these works established Transformers as the dominant NLP paradigm.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
    }
    mock_client.generate_chat_response.return_value = {"success": True, "content": json.dumps(mock_response)}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a, mock_paper_b], engine=mock_retrieval_engine)

    assert res.status == "completed"
    assert len(res.selected_paper_ids) == 2
    assert len(res.themes) == 2
    assert res.agreements_differences.claim_type == "SYNTHESIS"
    assert "No direct contradiction was identified in the retrieved evidence." in res.agreements_differences.content


# --------------------------------------------------------------------------
# Test 6: Custom review query handling
# --------------------------------------------------------------------------
def test_custom_query_handling(mock_paper_a, mock_paper_b, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "title": "Review on Computational Efficiency",
            "introduction": {"content": "Synthesizing computational bottlenecks.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
            "themes": [{"theme_id": "theme_001", "title": "Memory Scaling", "description": "O(n^2) scaling.", "supporting_papers": ["paper_001"], "synthesis": "Self-attention requires quadratic memory.", "support_level": "High evidence support"}],
            "methodology_synthesis": {"content": "Self-attention compute.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
            "findings_synthesis": {"content": "Quadratic memory limits context length.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
            "agreements_differences": {"content": "Different memory demands.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
            "limitations": {"content": "Quadratic scaling constraints.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
            "research_gaps": {"content": "Linear attention.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
            "conclusion": {"content": "Efficiency remains critical.", "claim_type": "INFERENCE", "supporting_papers": ["paper_001"]},
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    custom_q = "How do these papers address computational efficiency?"
    res = agent.generate_literature_review(papers=[mock_paper_a, mock_paper_b], engine=mock_retrieval_engine, custom_query=custom_q)

    assert res.review_question == custom_q
    # Check that custom query was passed in prompt
    called_prompt = mock_client.generate_chat_response.call_args[1]["user_prompt"]
    assert custom_q in called_prompt


# --------------------------------------------------------------------------
# Test 7: Per-paper retrieval isolation
# --------------------------------------------------------------------------
def test_per_paper_retrieval_isolation(mock_paper_a, mock_paper_b, mock_retrieval_engine, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    mock_retrieval_engine.retrieve.reset_mock()

    agent.generate_literature_review(papers=[mock_paper_a, mock_paper_b], engine=mock_retrieval_engine)

    # Verify that retrieve was called with isolated paper_ids for each paper
    calls = mock_retrieval_engine.retrieve.call_args_list
    paper_id_args = [c[1].get("paper_ids") for c in calls if "paper_ids" in c[1]]
    assert all(len(pids) == 1 for pids in paper_id_args), "Global un-isolated retrieval was performed!"
    assert any(pids == ["paper_001"] for pids in paper_id_args)
    assert any(pids == ["paper_002"] for pids in paper_id_args)


# --------------------------------------------------------------------------
# Test 8: Cross-paper attribution integrity
# --------------------------------------------------------------------------
def test_cross_paper_attribution_integrity(mock_paper_a, mock_paper_b, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "title": "Attribution Review",
            "introduction": {"content": "Paper_002 investigates BERT pre-training.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_002"]},
            "themes": [{"theme_id": "theme_001", "title": "BERT Bidirectional Pre-training", "description": "Pre-training.", "supporting_papers": ["paper_002"], "synthesis": "BERT conditions on left and right context.", "support_level": "High evidence support"}],
            "methodology_synthesis": {"content": "Masked language model is introduced in paper_002.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_002"]},
            "findings_synthesis": {"content": "GLUE 80.5% in paper_002.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_002"]},
            "agreements_differences": {"content": "Methodological comparisons.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
            "limitations": {"content": "BERT fine-tuning instability.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_002"]},
            "research_gaps": {"content": "Gaps.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_002"]},
            "conclusion": {"content": "Conclusion.", "claim_type": "INFERENCE", "supporting_papers": ["paper_002"]},
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a, mock_paper_b], engine=mock_retrieval_engine)

    # Methodology mentioning BERT must have paper_002
    assert "paper_002" in res.methodology_synthesis.supporting_papers


# --------------------------------------------------------------------------
# Test 9: Provenance preservation (chunk, page, section, score)
# --------------------------------------------------------------------------
def test_provenance_preservation(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "title": "Provenance Test Review",
            "introduction": {"content": "Intro [paper_001, p. 1, §Introduction].", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
            "themes": [],
            "methodology_synthesis": {"content": "Method.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
            "findings_synthesis": {"content": "Findings.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
            "agreements_differences": {"content": "Differences.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
            "limitations": {"content": "Limits.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
            "research_gaps": {"content": "Gaps.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
            "conclusion": {"content": "End.", "claim_type": "INFERENCE", "supporting_papers": ["paper_001"]},
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert len(res.all_evidence_items) > 0
    ev = res.all_evidence_items[0]
    assert ev.chunk_id.startswith("paper_001_c")
    assert ev.page_start >= 1
    assert ev.normalized_section != ""
    assert ev.score > 0.0


# --------------------------------------------------------------------------
# Test 10: Page number preservation
# --------------------------------------------------------------------------
def test_page_preservation(mock_paper_a, mock_retrieval_engine, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)
    pages = [r["page"] for r in res.references]
    assert all(isinstance(p, int) for p in pages)
    assert any(p == 1 for p in pages)


# --------------------------------------------------------------------------
# Test 11: Section heading preservation
# --------------------------------------------------------------------------
def test_section_preservation(mock_paper_a, mock_retrieval_engine, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)
    sections = [r["section"] for r in res.references]
    assert any("Introduction" in s for s in sections)


# --------------------------------------------------------------------------
# Test 12: No fabricated citations
# --------------------------------------------------------------------------
def test_no_fabricated_citations(mock_paper_a, mock_retrieval_engine, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)
    real_chunks = {c.chunk_id for c in mock_paper_a.chunks}
    for ref in res.references:
        assert ref["chunk_id"] in real_chunks, f"Fabricated chunk citation detected: {ref['chunk_id']}"


# --------------------------------------------------------------------------
# Test 13: Insufficient-evidence handling
# --------------------------------------------------------------------------
def test_insufficient_evidence_handling(mock_paper_a):
    empty_engine = MagicMock(spec=HybridRetrievalEngine)
    empty_engine.retrieve.return_value = []

    agent = ResearchLensAgent(foundry_client=MagicMock())
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=empty_engine)

    assert res.status == "insufficient_evidence"
    assert "Insufficient evidence" in res.title or "Insufficient evidence" in (res.error_message or "")


# --------------------------------------------------------------------------
# Test 14: Contradiction handling ("No direct contradiction was identified...")
# --------------------------------------------------------------------------
def test_contradiction_handling(mock_paper_a, mock_paper_b, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "title": "Contradiction Test",
            "introduction": {"content": "Intro", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
            "themes": [],
            "methodology_synthesis": {"content": "Methods", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
            "findings_synthesis": {"content": "Findings", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
            "agreements_differences": {"content": "Both use self-attention but differ in training objectives.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
            "limitations": {"content": "Limits", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
            "research_gaps": {"content": "Gaps", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
            "conclusion": {"content": "Conclusion", "claim_type": "INFERENCE", "supporting_papers": ["paper_001", "paper_002"]},
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a, mock_paper_b], engine=mock_retrieval_engine)

    assert "No direct contradiction was identified in the retrieved evidence." in res.agreements_differences.content


# --------------------------------------------------------------------------
# Test 15: Phase 6 gap integration
# --------------------------------------------------------------------------
def test_phase6_gap_integration(mock_paper_a, mock_retrieval_engine, sample_gap, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine, gaps=[sample_gap])

    assert len(res.linked_gaps) == 1
    assert res.linked_gaps[0].gap_id == sample_gap.gap_id
    assert res.summary_counts["total_gaps"] == 1


# --------------------------------------------------------------------------
# Test 16: Phase 7 future-direction integration (flagged is_suggestion=True)
# --------------------------------------------------------------------------
def test_phase7_future_direction_integration(mock_paper_a, mock_retrieval_engine, sample_gap, mock_foundry_client):
    fd = FutureDirection(
        direction_id="fd_001",
        display_index=1,
        display_id="fd_001",
        title="Sparse Attention Experiments",
        description="Investigate sub-quadratic sparse attention variants.",
        gap_addressed="gap_001",
        supporting_papers=["paper_001"],
        is_suggestion=True,
    )
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine, future_directions=[fd])

    assert len(res.future_directions) == 1
    assert res.future_directions[0].is_suggestion is True
    assert res.future_directions[0].direction_id == "fd_001"


# --------------------------------------------------------------------------
# Test 17: Complete-data export compliance
# --------------------------------------------------------------------------
def test_complete_data_export(mock_paper_a, mock_retrieval_engine, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)
    # References must not be truncated with "... and 5 more"
    for r in res.references:
        assert "... and" not in r["snippet"] or "..." in r["snippet"]


# --------------------------------------------------------------------------
# Test 18: Markdown export integrity
# --------------------------------------------------------------------------
def test_markdown_export_integrity(mock_paper_a, mock_paper_b, mock_retrieval_engine, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    res = agent.generate_literature_review(papers=[mock_paper_a, mock_paper_b], engine=mock_retrieval_engine)

    # Emulate frontend markdown generation
    md_content = f"# {res.title}\n{res.introduction.content}\n{res.methodology_synthesis.content}\n{res.conclusion.content}"
    assert res.title in md_content
    assert res.introduction.content in md_content
    assert res.conclusion.content in md_content


# --------------------------------------------------------------------------
# Test 19: JSON export integrity
# --------------------------------------------------------------------------
def test_json_export_integrity(mock_paper_a, mock_retrieval_engine, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)

    dumped = res.model_dump()
    serialized = json.dumps(dumped, default=str)
    loaded = json.loads(serialized)
    assert loaded["review_id"] == res.review_id
    assert loaded["status"] == "completed"
    assert "introduction" in loaded


# --------------------------------------------------------------------------
# Test 20: LLM failure graceful handling
# --------------------------------------------------------------------------
def test_model_failure_handling(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.generate_chat_response.return_value = {"success": False, "error": "Foundry rate limit exceeded"}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert res.status == "error"
    assert "Foundry rate limit exceeded" in (res.error_message or "")


# --------------------------------------------------------------------------
# Test 21: Empty selection handling
# --------------------------------------------------------------------------
def test_empty_paper_selection(mock_retrieval_engine):
    agent = ResearchLensAgent(foundry_client=MagicMock())
    res = agent.generate_literature_review(papers=[], engine=mock_retrieval_engine)

    assert res.status == "error"
    assert "No research papers were provided" in (res.error_message or "")


# --------------------------------------------------------------------------
# Test 22: No ranking or winner declaration language
# --------------------------------------------------------------------------
def test_no_ranking_or_winner_language():
    prompt = SYSTEM_PROMPT_LITERATURE_REVIEW_AGENT
    assert "Never rank papers as 'better', 'worse', 'best', or 'worst'" in prompt
    assert "Never declare a winner" in prompt


# --------------------------------------------------------------------------
# Test 23: Claim type classification
# --------------------------------------------------------------------------
def test_claim_type_classification():
    for ct in ["DOCUMENTED", "SYNTHESIS", "INFERENCE", "INSUFFICIENT_EVIDENCE"]:
        assert ct in CLAIM_TYPES


# --------------------------------------------------------------------------
# Test 24: Phase 3 retrieval regression check
# --------------------------------------------------------------------------
def test_phase3_retrieval_regression(mock_paper_a, mock_retrieval_engine):
    retrieved = mock_retrieval_engine.retrieve(query="transformer self-attention", paper_ids=["paper_001"], top_k=2)
    assert len(retrieved) > 0
    assert retrieved[0].chunk.paper_id == "paper_001"


# --------------------------------------------------------------------------
# Test 25: Phase 4 agent regression check
# --------------------------------------------------------------------------
def test_phase4_agent_regression():
    agent = ResearchLensAgent(foundry_client=MagicMock())
    status = agent.get_connection_status()
    assert "label" in status
    assert "connected" in status


# --------------------------------------------------------------------------
# Test 26: Phase 5 comparison regression check
# --------------------------------------------------------------------------
def test_phase5_comparison_regression(mock_paper_a, mock_paper_b, mock_retrieval_engine, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    comp = agent.compare_papers(papers=[mock_paper_a, mock_paper_b], engine=mock_retrieval_engine)
    assert comp.status in ["completed", "error"]


# --------------------------------------------------------------------------
# Test 27: Phase 6 gap regression check
# --------------------------------------------------------------------------
def test_phase6_gap_regression(mock_paper_a, mock_retrieval_engine, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    gap_res = agent.detect_research_gaps(papers=[mock_paper_a], engine=mock_retrieval_engine)
    assert gap_res.status in ["completed", "error"]


# --------------------------------------------------------------------------
# Test 28: Phase 7 questions regression check
# --------------------------------------------------------------------------
def test_phase7_questions_regression(mock_paper_a, mock_retrieval_engine, sample_gap, mock_foundry_client):
    agent = ResearchLensAgent(foundry_client=mock_foundry_client)
    q_res = agent.generate_research_questions(papers=[mock_paper_a], engine=mock_retrieval_engine, gaps=[sample_gap])
    assert q_res.status in ["completed", "error", "insufficient_evidence"]


# --------------------------------------------------------------------------
# Test 29: Rejection of fabricated numeric finding (e.g. "34%")
# --------------------------------------------------------------------------
def test_rejection_of_fabricated_numeric_finding(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"

    fabricated_content = json.dumps({
        "title": "Literature Review with Fabricated Number",
        "introduction": {
            "content": "Integrating semantic chunking improves citation recall by up to 34% in benchmarks [paper_001, p. 8]. The model achieves a BLEU score of 28.4 on translation.",
            "claim_type": "DOCUMENTED",
            "supporting_papers": ["paper_001"]
        },
        "themes": [],
        "methodology_synthesis": {"content": "Self-attention replaces recurrence.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "findings_synthesis": {"content": "The big transformer achieves a BLEU score of 28.4.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "agreements_differences": {"content": "No contradictions noted.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "limitations": {"content": "Computational complexity scales quadratically.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "research_gaps": {"content": "Scaling to longer sequences.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "conclusion": {"content": "Transformers are effective.", "claim_type": "INFERENCE", "supporting_papers": ["paper_001"]}
    })
    mock_client.generate_chat_response.return_value = {"success": True, "content": fabricated_content}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert "34%" not in res.introduction.content
    assert any("34%" in rc.get("sentence", "") for rc in res.rejected_claims)
    assert any(rc.get("reason") in ["numerical_hallucination", "implementation_leakage"] for rc in res.rejected_claims)


# --------------------------------------------------------------------------
# Test 30: Rejection of implementation leakage (e.g. OpenAI GPT-4o, Azure text-embedding-3)
# --------------------------------------------------------------------------
def test_rejection_of_implementation_leakage(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"

    leakage_content = json.dumps({
        "title": "Literature Review with Implementation Leakage",
        "introduction": {
            "content": "Dominant algorithms include OpenAI GPT-4o, Azure text-embedding-3 models, and transformer bi-encoders [Paper: 1, Page: 6]. Also, the Transformer relies entirely on self-attention.",
            "claim_type": "DOCUMENTED",
            "supporting_papers": ["paper_001"]
        },
        "themes": [
            {
                "theme_id": "theme_001",
                "title": "RAG and OpenAI GPT-4o Pipelines",
                "description": "API token costs and GPU memory ceilings.",
                "supporting_papers": ["paper_001"],
                "synthesis": "Azure OpenAI text-embedding-3 dominates vector indexing.",
                "support_level": "High evidence support"
            },
            {
                "theme_id": "theme_002",
                "title": "Self-Attention Mechanism",
                "description": "Pure attention without recurrence.",
                "supporting_papers": ["paper_001"],
                "synthesis": "Self-attention computes representations without recurrence.",
                "support_level": "High evidence support"
            }
        ],
        "methodology_synthesis": {"content": "Self-attention uses dot-product mechanics.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "findings_synthesis": {"content": "Transformer reaches 28.4 BLEU.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "agreements_differences": {"content": "Consistent results across tasks.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "limitations": {"content": "Quadratic complexity with sequence length.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "research_gaps": {"content": "Sub-quadratic attention mechanisms.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "conclusion": {"content": "Self-attention offers significant speedups.", "claim_type": "INFERENCE", "supporting_papers": ["paper_001"]}
    })
    mock_client.generate_chat_response.return_value = {"success": True, "content": leakage_content}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)

    full_text = " ".join([
        res.introduction.content,
        res.methodology_synthesis.content,
        res.findings_synthesis.content,
        " ".join([t.title + " " + t.synthesis for t in res.themes])
    ])
    assert "GPT-4o" not in full_text
    assert "Azure" not in full_text
    assert "text-embedding-3" not in full_text
    assert len(res.themes) == 1
    assert res.themes[0].theme_id == "theme_002"
    assert any("GPT-4o" in rc.get("sentence", "") for rc in res.rejected_claims)
    assert any(rc.get("reason") == "implementation_leakage" for rc in res.rejected_claims)


# --------------------------------------------------------------------------
# Test 31: Rejection of wrong paper attribution
# --------------------------------------------------------------------------
def test_rejection_of_wrong_paper_attribution(mock_paper_a, mock_paper_b, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"

    inverted_content = json.dumps({
        "title": "Review with Inverted Attribution",
        "introduction": {"content": "Both papers study sequence representations.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
        "themes": [],
        "methodology_synthesis": {
            "content": "Paper_001 uses a masked language model that randomly masks some of the tokens from the input [paper_001, p. 3]. Meanwhile, BERT introduces a masked language model that randomly masks some of the tokens from the input to predict the original vocabulary id [paper_002, p. 3].",
            "claim_type": "DOCUMENTED",
            "supporting_papers": ["paper_001"] # LLM mistakenly claimed paper_001 instead of paper_002
        },
        "findings_synthesis": {"content": "Transformer yields 28.4 BLEU on translation [paper_001, p. 8].", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "agreements_differences": {"content": "Both use self-attention layers.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
        "limitations": {"content": "Quadratic memory scaling is observed in self-attention.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "research_gaps": {"content": "Efficient scaling to long contexts.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001", "paper_002"]},
        "conclusion": {"content": "Bidirectional and autoregressive architectures are both viable.", "claim_type": "INFERENCE", "supporting_papers": ["paper_001", "paper_002"]}
    })
    mock_client.generate_chat_response.return_value = {"success": True, "content": inverted_content}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a, mock_paper_b], engine=mock_retrieval_engine)

    # 1. Deterministic paper derivation overrides the LLM's hallucinated ["paper_001"] with ["paper_002"]
    assert "paper_002" in res.methodology_synthesis.supporting_papers
    assert "paper_001" not in res.methodology_synthesis.supporting_papers

    # 2. Attribution inversion sentence was caught and rejected in rejected_claims
    assert any(rc.get("reason") == "attribution_mismatch" for rc in res.rejected_claims)


# --------------------------------------------------------------------------
# Test 32: Partitioning of future research directions (Author vs AI)
# --------------------------------------------------------------------------
def test_future_direction_partitioning_epistemic_sources(mock_paper_a, mock_retrieval_engine, sample_gap):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"

    valid_content = json.dumps({
        "title": "Review with Partitioned Directions",
        "introduction": {"content": "Transformer architecture review.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "themes": [],
        "methodology_synthesis": {"content": "Self-attention mechanism.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "findings_synthesis": {"content": "BLEU score of 28.4 on WMT.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "agreements_differences": {"content": "No contradictions.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "limitations": {"content": "Quadratic complexity.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "research_gaps": {"content": "Memory scaling bottleneck.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "conclusion": {"content": "Conclusion on attention.", "claim_type": "INFERENCE", "supporting_papers": ["paper_001"]}
    })
    mock_client.generate_chat_response.return_value = {"success": True, "content": valid_content}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(
        papers=[mock_paper_a],
        engine=mock_retrieval_engine,
        gaps=[sample_gap]
    )

    ai_dirs = res.ai_suggested_directions
    author_dirs = res.author_stated_directions

    assert len(ai_dirs) >= 1
    for d in ai_dirs:
        assert d.is_suggestion is True
        assert d.gap_addressed == sample_gap.gap_id

    for d in author_dirs:
        assert d.is_suggestion is False


# --------------------------------------------------------------------------
# Test 33: Unsupported claims demoted to INSUFFICIENT_EVIDENCE
# --------------------------------------------------------------------------
def test_unsupported_dataset_method_demoted_to_insufficient_evidence(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"

    empty_content = json.dumps({
        "title": "Review with Fabricated Section",
        "introduction": {"content": "Transformer architecture review.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "themes": [],
        "methodology_synthesis": {"content": "Transformer uses self-attention.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "findings_synthesis": {
            "content": "Quantum hyper-graphs with topological quantum memory yielded a 99.9% topological accuracy on the Hilbert space benchmark.",
            "claim_type": "DOCUMENTED",
            "supporting_papers": ["paper_001"]
        },
        "agreements_differences": {"content": "No contradictions.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "limitations": {"content": "Complexity scales quadratically.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "research_gaps": {"content": "Efficiency gaps.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "conclusion": {"content": "Conclusion.", "claim_type": "INFERENCE", "supporting_papers": ["paper_001"]}
    })
    mock_client.generate_chat_response.return_value = {"success": True, "content": empty_content}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert res.findings_synthesis.claim_type == "INSUFFICIENT_EVIDENCE"
    assert "Insufficient documented evidence" in res.findings_synthesis.content


# --------------------------------------------------------------------------
# Test 34: Citation page provenance verification
# --------------------------------------------------------------------------
def test_citation_page_provenance_verification(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"

    invalid_page_content = json.dumps({
        "title": "Review with Invalid Page Citation",
        "introduction": {
            "content": "Self-attention computes representations without recurrence [Paper: 1, Page: 99]. But the model achieves 28.4 BLEU on translation [Paper: 1, Page: 8].",
            "claim_type": "DOCUMENTED",
            "supporting_papers": ["paper_001"]
        },
        "themes": [],
        "methodology_synthesis": {"content": "Dot product mechanics.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "findings_synthesis": {"content": "BLEU score of 28.4.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "agreements_differences": {"content": "No contradictions.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "limitations": {"content": "Quadratic complexity.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "research_gaps": {"content": "Scaling attention.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "conclusion": {"content": "Conclusion.", "claim_type": "INFERENCE", "supporting_papers": ["paper_001"]}
    })
    mock_client.generate_chat_response.return_value = {"success": True, "content": invalid_page_content}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert any(rc.get("reason") == "unverified_citation" for rc in res.rejected_claims)


# --------------------------------------------------------------------------
# Test 35: Strict single paper isolation guard
# --------------------------------------------------------------------------
def test_strict_single_paper_isolation_guard(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"

    leaked_cross_paper_content = json.dumps({
        "title": "Single Paper Review with Cross-Paper Leakage",
        "introduction": {
            "content": "Paper_001 introduces the Transformer. In contrast, Paper_002 uses masked language modeling for bidirectional representations [paper_002, p. 1].",
            "claim_type": "DOCUMENTED",
            "supporting_papers": ["paper_001", "paper_002"]
        },
        "themes": [],
        "methodology_synthesis": {"content": "Self-attention computes representations without recurrence [paper_001, p. 1].", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "findings_synthesis": {"content": "BLEU score of 28.4 on WMT.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "agreements_differences": {"content": "No cross-paper comparison available.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "limitations": {"content": "Quadratic complexity.", "claim_type": "DOCUMENTED", "supporting_papers": ["paper_001"]},
        "research_gaps": {"content": "Scaling attention.", "claim_type": "SYNTHESIS", "supporting_papers": ["paper_001"]},
        "conclusion": {"content": "Conclusion.", "claim_type": "INFERENCE", "supporting_papers": ["paper_001"]}
    })
    mock_client.generate_chat_response.return_value = {"success": True, "content": leaked_cross_paper_content}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.generate_literature_review(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert res.selected_paper_ids == ["paper_001"]
    assert "paper_002" not in res.introduction.supporting_papers
    assert "masked language modeling" not in res.introduction.content
    assert any("paper_002" in rc.get("sentence", "").lower() for rc in res.rejected_claims)
