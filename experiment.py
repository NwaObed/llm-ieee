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
from generate.query import docs_retriver, db_connector
from augment.embed import embed



# ============================================================
# Experiment runner
# ============================================================

def run_rag_experiment(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    generation_model: str = "llama3",
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
        print(f"Running layer: {layer.__name__}")
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

query="When can HMRC issue a discovery assessment?"
query_embedding = embed(query)
TOP_K = 5
results = db_connector(query_embedding, TOP_K)
retrieved_docs = docs_retriver(results)



if __name__ == "__main__":
    result = run_rag_experiment(
    query=query,
    retrieved_docs=retrieved_docs,
    generation_model="llama3",
    enabled_layers=None,  # Enable all layers
    )

    print(f"\n\n=== EXPERIMENT METRICS RESULT ===\n\n {result.final_metrics}")
    # print(result.baseline_answer)

    # experiments = {
    # "baseline": [],
    # "lexical_only": [lexical_support_layer],
    # "llm_judge_only": [llm_groundedness_layer],
    # "selfcheck_only": [selfcheck_layer],
    # "metarag_only": [metarag_layer],
    # "semantic_entropy_only": [semantic_entropy_layer],
    # "all_layers": [
    #     lexical_support_layer,
    #     llm_groundedness_layer,
    #     selfcheck_layer,
    #     metarag_layer,
    #     semantic_entropy_layer,
    # ],
    # }

    # for name, layers in experiments.items():
        # result = run_rag_experiment(
        #     query="When can HMRC issue a discovery assessment?",
        #     retrieved_docs=retrieved_docs,
        #     generation_model="llama3",
        #     enabled_layers=layers,
        # )

        # print("\n\n====", name, "====")
        # print(json.dumps(result.final_metrics, indent=2))

        # print(layers)