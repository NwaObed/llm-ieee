import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en")

def embed(text):
    return model.encode(text).tolist()

def build_embedded_chunks():
    from chunks import chunks

    all_chunks = []
    for c in chunks:
        c["embedding"] = embed(c["chunk_text"])
        all_chunks.append(c)

    return all_chunks

if __name__ == "__main__":
    build_embedded_chunks()
