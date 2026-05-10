import json
from typing import List, Dict, Any, Optional, Callable
from generate.baseline import baseline_generate
from detect_hallucination.hallucinate_detection import (
    ExperimentResult,
    lexical_support_layer,
    llm_groundedness_layer,
    selfcheck_layer,
    metarag_layer,
    semantic_entropy_layer,
)
from metrics import metrics_summary
from dataclasses import dataclass, asdict



# ============================================================
# Experiment runner
# ============================================================

def run_rag_experiment(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    generation_model: str = "mistral",
    enabled_layers: Optional[List[Callable]] = None,
) -> ExperimentResult:

    baseline_answer = baseline_generate(
        query=query,
        retrieved_docs=retrieved_docs,
        model=generation_model,
    )

    if enabled_layers is None:
        enabled_layers = [
            lexical_support_layer,
            llm_groundedness_layer,
            selfcheck_layer,
            metarag_layer,
            semantic_entropy_layer,
        ]

    layer_results = []

    for layer in enabled_layers:
        result = layer(
            query=query,
            answer=baseline_answer,
            retrieved_docs=retrieved_docs,
            model=generation_model,
            judge_model=generation_model,
        )
        layer_results.append(result)

    final_metrics = metrics_summary(layer_results)

    return ExperimentResult(
        query=query,
        model=generation_model,
        baseline_answer=baseline_answer,
        retrieved_docs=retrieved_docs,
        layer_results=layer_results,
        final_metrics=final_metrics,
    )


def print_experiment_result(result: ExperimentResult):
    print("\n=== BASELINE ANSWER ===")
    print(result.baseline_answer)

    print("\n=== LAYER RESULTS ===")
    for layer in result.layer_results:
        print(f"\n[{layer.layer_name}]")
        print(f"status: {layer.status}")
        print(f"support_score: {layer.support_score}")
        print(f"hallucination_score: {layer.hallucination_score}")

    print("\n=== FINAL METRICS ===")
    print(json.dumps(result.final_metrics, indent=2))