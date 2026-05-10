import statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from detect_hallucination.hallucinate_detection import LayerResult


# ============================================================
# Metrics aggregation
# ============================================================

def metrics_summary(layer_results: List[LayerResult]) -> Dict[str, Any]:
    if not layer_results:
        return {
            "support_score": None,
            "hallucination_score": None,
            "risk_level": "baseline_only",
        }

    hallucination_scores = [r.hallucination_score for r in layer_results]
    support_scores = [r.support_score for r in layer_results]
    flagged_layers = [r.layer_name for r in layer_results if r.status == "flag"]

    avg_hallucination = statistics.mean(hallucination_scores)
    max_hallucination = max(hallucination_scores)
    avg_support = statistics.mean(support_scores)

    if max_hallucination >= 0.75:
        risk = "high"
    elif max_hallucination >= 0.5:
        risk = "medium"
    elif max_hallucination >= 0.25:
        risk = "low"
    else:
        risk = "very_low"

    return {
        "support_score": round(avg_support, 3),
        "hallucination_score_avg": round(avg_hallucination, 3),
        "hallucination_score_max": round(max_hallucination, 3),
        "risk_level": risk,
        "flagged_layers": flagged_layers,
        "num_layers": len(layer_results),
    }