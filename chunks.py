from extraction import parse_page, get_links
import json

source_link = "https://www.gov.uk/hmrc-internal-manuals/self-assessment-manual"

def chunk_page(page_data, max_paragraphs_per_chunk=3):
    """
    Turns a parsed HMRC manual page into structured chunks.
    """

    content_blocks = page_data["content_blocks"]
    
    chunks = []
    current_chunk = []
    chunk_index = 0

    for block in content_blocks:
        current_chunk.append(block)

        # When chunk is big enough → flush it
        if len(current_chunk) >= max_paragraphs_per_chunk:
            
            chunk_text = "\n".join(current_chunk)

            chunks.append({
                "chunk_text": chunk_text,
                "section_code": page_data.get("section_code"),
                "title": page_data.get("title"),
                "url": page_data.get("url"),
                "chunk_index": chunk_index
            })

            chunk_index += 1
            current_chunk = []

    # flush remaining text
    if current_chunk:
        chunk_text = "\n".join(current_chunk)

        chunks.append({
            "chunk_text": chunk_text,
            "section_code": page_data.get("section_code"),
            "title": page_data.get("title"),
            "url": page_data.get("url"),
            "chunk_index": chunk_index
        })

    return chunks

def build_chunks(source_url=source_link, max_paragraphs_per_chunk=3):
    links = get_links(source_url)
    manual_data = parse_page(links[0])
    
    return chunk_page(manual_data, max_paragraphs_per_chunk)

if __name__ == "__main__":
    build_chunks()
