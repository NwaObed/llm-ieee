from embed import build_embedded_chunks

import psycopg
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg.connect(
    "host=" + os.getenv("HOST") + " " \
    "port=" + os.getenv("PORT") + " " \
    "dbname=" + os.getenv("DBNAME") + " " \
    "user=" + os.getenv("DB_USER") + " " \
    "password=" + os.getenv("DB_PASSWORD")
)

cur = conn.cursor()

cur.execute("SELECT current_database();")
print(cur.fetchone())

all_chunks = build_embedded_chunks()

for chunk in all_chunks:
    print(chunk["chunk_text"][:10])  # print first 100 characters of the chunk text
    cur.execute("""
    INSERT INTO manual_chunks (
        chunk_text, embedding, section_code, title, url, chunk_index
    ) VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        chunk["chunk_text"],
        chunk["embedding"],
        chunk["section_code"],
        chunk["title"],
        chunk["url"],
        chunk["chunk_index"]
    ))

conn.commit()

