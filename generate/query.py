import os
import sys

from augment.embed import embed
import psycopg
import requests
from config import db_cred
import os
from dotenv import load_dotenv

load_dotenv()


OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = "llama3"
TOP_K = 5

# query = "what is VAT?"

# query_embedding = embed(query)  # or API call

# with psycopg.connect(db_cred) as conn:
#     with conn.cursor() as cur:
#         cur.execute("""
#         SELECT chunk_text, section_code, title, url
#         FROM manual_chunks
#         ORDER BY embedding <-> %s::vector
#         LIMIT %s;
#         """, (query_embedding, TOP_K))

#         results = cur.fetchall()

def db_connector(query_embedding, TOP_K):
    with psycopg.connect(db_cred) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT chunk_text, section_code, title, url
            FROM manual_chunks
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
            """, (query_embedding, TOP_K))

            results = cur.fetchall()
    
    return results

def docs_retriver(results):
    retrieved_docs = []

    for row in results:
        retrieved_docs.append({
            "chunk_text": row[0],
            "section_code": row[1],
            "title": row[2],
            "url": row[3]
        })
    
    return retrieved_docs

# retrieved_docs = []

# for row in results:
#     retrieved_docs.append({
#         "chunk_text": row[0],
#         "section_code": row[1],
#         "title": row[2],
#         "url": row[3]
#     })

# results = db_connector(query_embedding, TOP_K)
# query_docs = docs_retriver(results)

# context = "\n\n".join([
#     doc["chunk_text"]
#     for doc in query_docs
# ])

# baseline_prompt = f"""
# You are a UK tax assistant.

# Use the following HMRC manual context to answer:

# {context}

# Question: {query}
# """

# prompt = f"""
# You are a UK tax assistant.

# Answer only from the following retrieved HMRC manual context.
# If the context does not contain enough evidence, say you cannot determine the answer from the retrieved manual sections.

# {context}

# Question: {query}
# """



# response = requests.post(
#     OLLAMA_URL, # local machine
#     json={
#         "model": OLLAMA_MODEL,
#         "prompt": prompt,
#         "stream": False
#     }
# )

# answer = response.json()["response"]

# citations = []

# for doc in query_docs:
#     citations.append(
#         f"{doc['section_code']} - {doc['title']} - {doc['url']}"
#     )

# citations = list(set(citations))


# baseline_response = answer + "\n\nSources:\n"

# for c in citations:
#     baseline_response += f"- {c}\n"

# print(f"Baseline response:\n{baseline_response}")

# report = detect_hallucinations(
#     query=query,
#     answer=answer,
#     retrieved_docs=retrieved_docs,
#     model=OLLAMA_MODEL,
# )

# final_response += "\n" + format_hallucination_report(report)

# final_response = answer + "\n\nSources:\n"

# for c in citations:
#     final_response += f"- {c}\n"
# print(f"Formatted response: {final_response}")
