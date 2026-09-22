"""
Multi-Paper Comparison Engine for ResearchLens AI.
Builds comparison matrices and identifies common patterns, repeated limitations, and contradictions.
Designed for Phase 7 implementation.
"""

from typing import List, Dict, Any
from models.analysis import PaperAnalysisResult


class ComparisonEngine:
    """Service to compare multiple analyzed papers without ranking."""

    def build_comparison_matrix(self, analyses: List[PaperAnalysisResult]) -> List[Dict[str, Any]]:
        """
        Builds comparison rows across key parameters:
        Dataset, Dataset Size, Methodology, Algorithm, Evaluation Metric, Main Result, Limitation, Future Work.
        """
        parameters = [
            "Dataset",
            "Dataset Size",
            "Methodology",
            "Algorithms / Models",
            "Evaluation Metrics",
            "Main Results",
            "Limitations",
            "Future Work",
        ]

        matrix = []
        for param in parameters:
            row = {"Parameter": param}
            for analysis in analyses:
                # Find matching parameter value
                val = "Pending Analysis"
                if param == "Dataset":
                    val = analysis.dataset.value or "N/A"
                elif param == "Dataset Size":
                    val = analysis.dataset_size.value or "N/A"
                elif param == "Methodology":
                    val = analysis.methodology.value or "N/A"
                elif param == "Algorithms / Models":
                    val = analysis.algorithms.value or "N/A"
                elif param == "Evaluation Metrics":
                    val = analysis.evaluation_metrics.value or "N/A"
                elif param == "Main Results":
                    val = analysis.main_results.value or "N/A"
                elif param == "Limitations":
                    val = analysis.limitations.value or "N/A"
                elif param == "Future Work":
                    val = analysis.future_work.value or "N/A"
                row[analysis.paper_filename] = val
            matrix.append(row)

        return matrix

    def detect_cross_paper_patterns(self, analyses: List[PaperAnalysisResult]) -> Dict[str, List[str]]:
        """Identifies common patterns, repeated limitations, and contradictions."""
        return {
            "common_datasets": [],
            "common_algorithms": [],
            "common_methodologies": [],
            "repeated_limitations": [],
            "contradictory_findings": [],
            "underexplored_areas": [],
        }
