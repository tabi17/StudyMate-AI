from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


def split_document_pages(pages):
    chunk_records = []

    for page in pages:
        page_number = page["page"]
        page_text = page["text"]
        source_type = page.get("source_type", "unknown")

        chunks = split_text(page_text)

        for chunk_index, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "text": chunk,
                    "page": page_number,
                    "source_type": source_type,
                    "chunk_index": chunk_index,
                }
            )

    return chunk_records