from sentence_transformers import SentenceTransformer

from src import chunking
from src import config
from src import db

_MODEL = None


def get_model():
    """Loads the embedder once per process: it is ~90MB and 2s to load, per call would be absurd."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _MODEL


def vector_literal(values):
    """psycopg2 adapts a Python list to a Postgres ARRAY, which will not cast to vector.

    pgvector's text input format is '[0.1,0.2,...]', so we hand it a string and let
    the %s::vector cast in the SQL do the conversion.
    """
    parts = []
    for value in values:
        parts.append(str(float(value)))
    return "[" + ",".join(parts) + "]"


def embed_texts(texts):
    """Batches of 32 keep CPU memory flat; encode() returns numpy, SQL needs plain floats."""
    model = get_model()
    vectors = model.encode(texts, batch_size=32)
    return vectors.tolist()


def delete_document(conn, filename):
    """Makes re-ingesting safe to run repeatedly instead of silently duplicating every chunk.

    Full delete-and-reload, not a diff. Re-embedding both PDFs costs ~2 minutes of CPU,
    which is cheaper than the code that would work out what actually changed.
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM documents WHERE filename = %s", (filename,))
    cur.close()


def insert_document(conn, title, filename):
    """Returns the new doc_id because every chunk row needs it as a foreign key."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (title, filename) VALUES (%s, %s) RETURNING id",
        (title, filename),
    )
    doc_id = cur.fetchone()[0]
    cur.close()
    return doc_id


def insert_chunks(conn, doc_id, chunks, vectors):
    """One row per chunk; content_tsv fills itself because the column is GENERATED."""
    cur = conn.cursor()
    for i in range(len(chunks)):
        cur.execute(
            "INSERT INTO chunks (doc_id, section_ref, content, embedding)"
            " VALUES (%s, %s, %s, %s::vector)",
            (doc_id, chunks[i]["section_ref"], chunks[i]["content"], vector_literal(vectors[i])),
        )
    cur.close()


def ingest_pdf(filepath, title):
    """PDF to searchable rows in one call, so ingestion is a single command not a notebook."""
    filename = filepath.name

    pages = chunking.read_pdf_pages(filepath)
    sections = chunking.split_by_sections(pages)
    chunks = chunking.enforce_max_size(sections)

    texts = []
    for chunk in chunks:
        texts.append(chunk["content"])
    vectors = embed_texts(texts)

    conn = db.get_connection()
    try:
        delete_document(conn, filename)
        doc_id = insert_document(conn, title, filename)
        insert_chunks(conn, doc_id, chunks, vectors)
        # One commit for the whole document: a half-ingested document is worse than none.
        conn.commit()
    finally:
        conn.close()

    return len(chunks)


if __name__ == "__main__":
    raw_dir = config.PROJECT_ROOT / "data" / "raw"

    count = ingest_pdf(raw_dir / "kyc_md.pdf", "KYC MD")
    print("KYC MD chunks:", count)

    count = ingest_pdf(raw_dir / "ppi_md.pdf", "PPI MD")
    print("PPI MD chunks:", count)