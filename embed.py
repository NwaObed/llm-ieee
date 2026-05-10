import json
from sentence_transformers import SentenceTransformer
from chunks import build_chunks

model = SentenceTransformer("BAAI/bge-base-en")

def embed(text):
    return model.encode(text).tolist()

def build_embedded_chunks():
    
    all_chunks = []
    for c in build_chunks():
        c["embedding"] = embed(c["chunk_text"])
        all_chunks.append(c)

    return all_chunks

if __name__ == "__main__":
    build_embedded_chunks()
