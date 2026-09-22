"""
Unit tests for 18-parameter Analysis and Citation models.
"""

from models.analysis import PaperAnalysisResult, AnalyzedParameter, EvidenceCitation


def test_18_parameter_model_structure():
    """Verify that all 18 required parameters exist in PaperAnalysisResult."""
    analysis = PaperAnalysisResult(
        paper_id="paper_001",
        paper_filename="deep_learning_survey.pdf"
    )

    required_parameters = [
        "title", "authors", "publication_year", "research_domain",
        "research_problem", "research_objective", "research_questions",
        "methodology", "dataset", "dataset_size", "algorithms",
        "features", "evaluation_metrics", "main_results",
        "key_findings", "limitations", "future_work", "keywords"
    ]

    for param in required_parameters:
        assert hasattr(analysis, param), f"Missing required parameter: {param}"
        param_obj = getattr(analysis, param)
        assert isinstance(param_obj, AnalyzedParameter)


def test_evidence_citation_attribution():
    """Verify explicit vs inferred attribution tracking."""
    citation = EvidenceCitation(
        claim="XGBoost achieved 92.4% accuracy",
        paper_id="p1",
        paper_filename="test.pdf",
        page=5,
        section="Results",
        is_explicit=True,
        confidence="High"
    )
    assert citation.is_explicit is True
    assert citation.confidence == "High"
