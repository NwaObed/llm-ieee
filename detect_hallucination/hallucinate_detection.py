import json
import re
import math
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable
from generate.rag_model import split_claims, tokenize, docs_to_context, call_ollama, safe_json_parse


# ============================================================
# Data structures
# ============================================================

@dataclass
class LayerResult:
    layer_name: str
    support_score: float
    hallucination_score: float
    status: str
    details: Dict[str, Any]


@dataclass
class ExperimentResult:
    query: str
    model: str
    baseline_answer: str
    retrieved_docs: List[Dict[str, Any]]
    layer_results: List[LayerResult]
    final_metrics: Dict[str, Any]


# ============================================================
# 1. Lexical support layer
# ============================================================

def lexical_support_layer(
    query: str,
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
    **kwargs,
) -> LayerResult:
    claims = split_claims(answer)

    claim_results = []
    support_scores = []

    for claim in claims:
        claim_tokens = set(tokenize(claim))

        if not claim_tokens:
            score = 0.0
            best_doc = None
        else:
            scored = []

            for doc in retrieved_docs:
                doc_tokens = set(tokenize(doc.get("chunk_text", "")))
                overlap = claim_tokens.intersection(doc_tokens)
                score = len(overlap) / len(claim_tokens)
                scored.append((score, list(overlap), doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            score, overlap, best_doc = scored[0]

        support_scores.append(score)

        claim_results.append({
            "claim": claim,
            "support_score": round(score, 3),
            "best_section": best_doc.get("section_code") if best_doc else None,
            "best_title": best_doc.get("title") if best_doc else None,
            "matched_terms": overlap[:10] if best_doc else [],
        })

    avg_support = statistics.mean(support_scores) if support_scores else 0.0
    hallucination_score = 1.0 - avg_support

    return LayerResult(
        layer_name="lexical_support",
        support_score=round(avg_support, 3),
        hallucination_score=round(hallucination_score, 3),
        status="pass" if avg_support >= 0.45 else "flag",
        details={
            "claims": claim_results,
            "claim_count": len(claims),
        },
    )


# ============================================================
# 2. LLM groundedness judge layer
# ============================================================

def llm_groundedness_layer(
    query: str,
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
    judge_model: str = "mistral",
    **kwargs,
) -> LayerResult:
    context = docs_to_context(retrieved_docs)

    prompt = f"""
You are a strict hallucination detector for an HMRC RAG system.

Your task:
Check whether the answer is fully supported by the retrieved context.

Return ONLY valid JSON with this schema:
{{
  "status": "supported" | "partially_supported" | "unsupported",
  "support_score": 0.0,
  "hallucination_score": 0.0,
  "unsupported_claims": ["..."],
  "reason": "..."
}}

Rules:
- Use only the retrieved context.
- If a claim is not explicitly supported, mark it unsupported.
- Do not use outside knowledge.

Question:
{query}

Retrieved context:
{context}

Answer:
{answer}
"""

    raw = call_ollama(prompt, model=judge_model, temperature=0.0)
    data = safe_json_parse(raw) or {}

    support_score = float(data.get("support_score", 0.0))
    hallucination_score = float(data.get("hallucination_score", 1.0))

    return LayerResult(
        layer_name="llm_groundedness_judge",
        support_score=round(support_score, 3),
        hallucination_score=round(hallucination_score, 3),
        status=data.get("status", "unknown"),
        details=data,
    )


# ============================================================
# 3. SelfCheckGPT-style consistency layer
# ============================================================

def selfcheck_layer(
    query: str,
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
    model: str = "mistral",
    judge_model: str = "mistral",
    n_samples: int = 3,
    **kwargs,
) -> LayerResult:
    context = docs_to_context(retrieved_docs)

    samples = []
    for _ in range(n_samples):
        prompt = f"""
Answer the question using the retrieved context.

Context:
{context}

Question:
{query}

Answer:
"""
        sample = call_ollama(prompt, model=model, temperature=0.8)
        samples.append(sample)

    claims = split_claims(answer)
    claim_scores = []

    for claim in claims:
        support_votes = []

        for sample in samples:
            judge_prompt = f"""
Does the sampled answer support the claim?

Return only JSON:
{{"supported": true | false, "reason": "..."}}

Claim:
{claim}

Sampled answer:
{sample}
"""
            raw = call_ollama(judge_prompt, model=judge_model, temperature=0.0)
            data = safe_json_parse(raw) or {}
            support_votes.append(bool(data.get("supported", False)))

        support_ratio = sum(support_votes) / len(support_votes) if support_votes else 0.0
        claim_scores.append({
            "claim": claim,
            "support_ratio": support_ratio,
            "hallucination_score": 1.0 - support_ratio,
        })

    avg_hallucination = statistics.mean(
        item["hallucination_score"] for item in claim_scores
    ) if claim_scores else 0.0

    return LayerResult(
        layer_name="selfcheck_consistency",
        support_score=round(1.0 - avg_hallucination, 3),
        hallucination_score=round(avg_hallucination, 3),
        status="pass" if avg_hallucination < 0.4 else "flag",
        details={
            "samples": samples,
            "claim_scores": claim_scores,
        },
    )


# ============================================================
# 4. MetaRAG-style mutation layer
# ============================================================

def generate_factoid_mutations(
    claim: str,
    model: str = "mistral",
    n: int = 2,
) -> Dict[str, List[str]]:
    prompt = f"""
Create controlled mutations of this factual claim.

Return only JSON:
{{
  "synonyms": ["..."],
  "antonyms": ["..."]
}}

Rules:
- Synonyms must preserve the meaning.
- Antonyms must contradict the meaning.
- Keep each mutation atomic.
- Generate exactly {n} synonyms and {n} antonyms.

Claim:
{claim}
"""

    raw = call_ollama(prompt, model=model, temperature=0.3)
    data = safe_json_parse(raw) or {"synonyms": [], "antonyms": []}

    return {
        "synonyms": data.get("synonyms", [])[:n],
        "antonyms": data.get("antonyms", [])[:n],
    }


def verify_claim_against_context(
    claim: str,
    context: str,
    model: str = "mistral",
) -> str:
    prompt = f"""
Check whether the claim is supported by the context.

Return only JSON:
{{"decision": "yes" | "no" | "not_sure", "reason": "..."}}

Context:
{context}

Claim:
{claim}
"""

    raw = call_ollama(prompt, model=model, temperature=0.0)
    data = safe_json_parse(raw) or {}
    return data.get("decision", "not_sure")


def metarag_layer(
    query: str,
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
    model: str = "mistral",
    judge_model: str = "mistral",
    n_mutations: int = 2,
    **kwargs,
) -> LayerResult:
    context = docs_to_context(retrieved_docs)
    claims = split_claims(answer)

    factoid_scores = []

    for claim in claims:
        mutations = generate_factoid_mutations(claim, model=model, n=n_mutations)

        penalties = []

        for syn in mutations["synonyms"]:
            decision = verify_claim_against_context(syn, context, model=judge_model)
            if decision == "yes":
                penalty = 0.0
            elif decision == "not_sure":
                penalty = 0.5
            else:
                penalty = 1.0
            penalties.append(penalty)

        for ant in mutations["antonyms"]:
            decision = verify_claim_against_context(ant, context, model=judge_model)
            if decision == "no":
                penalty = 0.0
            elif decision == "not_sure":
                penalty = 0.5
            else:
                penalty = 1.0
            penalties.append(penalty)

        score = statistics.mean(penalties) if penalties else 1.0

        factoid_scores.append({
            "claim": claim,
            "mutations": mutations,
            "hallucination_score": round(score, 3),
        })

    response_hallucination_score = max(
        item["hallucination_score"] for item in factoid_scores
    ) if factoid_scores else 0.0

    return LayerResult(
        layer_name="metarag_mutation",
        support_score=round(1.0 - response_hallucination_score, 3),
        hallucination_score=round(response_hallucination_score, 3),
        status="pass" if response_hallucination_score < 0.5 else "flag",
        details={
            "factoid_scores": factoid_scores,
        },
    )


# ============================================================
# 5. Semantic entropy approximation layer
# ============================================================

def jaccard_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))

    if not ta or not tb:
        return 0.0

    return len(ta.intersection(tb)) / len(ta.union(tb))


def cluster_by_similarity(texts: List[str], threshold: float = 0.55) -> List[List[str]]:
    clusters = []

    for text in texts:
        placed = False

        for cluster in clusters:
            if any(jaccard_similarity(text, existing) >= threshold for existing in cluster):
                cluster.append(text)
                placed = True
                break

        if not placed:
            clusters.append([text])

    return clusters


def entropy_from_clusters(clusters: List[List[str]]) -> float:
    total = sum(len(c) for c in clusters)
    if total == 0:
        return 0.0

    entropy = 0.0
    for cluster in clusters:
        p = len(cluster) / total
        entropy -= p * math.log(p + 1e-12)

    max_entropy = math.log(total) if total > 1 else 1.0
    return entropy / max_entropy


def semantic_entropy_layer(
    query: str,
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
    model: str = "mistral",
    n_samples: int = 5,
    **kwargs,
) -> LayerResult:
    context = docs_to_context(retrieved_docs)

    samples = [answer]

    for _ in range(n_samples):
        prompt = f"""
Answer the question using only the retrieved context.

Context:
{context}

Question:
{query}

Answer:
"""
        samples.append(call_ollama(prompt, model=model, temperature=0.9))

    clusters = cluster_by_similarity(samples)
    semantic_entropy = entropy_from_clusters(clusters)

    return LayerResult(
        layer_name="semantic_entropy",
        support_score=round(1.0 - semantic_entropy, 3),
        hallucination_score=round(semantic_entropy, 3),
        status="pass" if semantic_entropy < 0.45 else "flag",
        details={
            "samples": samples,
            "clusters": clusters,
            "semantic_entropy": semantic_entropy,
        },
    )






