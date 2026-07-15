result = run_rag_experiment(
    query="When can HMRC issue a discovery assessment?",
    retrieved_docs=retrieved_docs,
    generation_model="mistral",
    enabled_layers=[],
)

print(result.baseline_answer)