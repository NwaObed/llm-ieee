from typing import List, Dict, Any, Optional

from torchgen import model
from generate.rag_model import call_ollama, docs_to_context


# ============================================================
# 0. Baseline: plain RAG response
# ============================================================

def baseline_generate(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    model: str = "mistral",
) -> str:
    context = docs_to_context(retrieved_docs)

    prompt = f"""
You are an HMRC Self Assessment assistant.

Use the retrieved manual context to answer the user question.

Context:
{context}

Question:
{query}

Answer:
"""

    return call_ollama(prompt, model=model, temperature=0.0)