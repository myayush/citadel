import psycopg2

from src import config

# The vector(384) width must match config.EMBEDDING_DIMS. Postgres cannot read
# a Python constant, so changing the embedding model means editing both places
# and rebuilding the table.
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id serial PRIMARY KEY,
    title text NOT NULL,
    filename text NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id serial PRIMARY KEY,
    doc_id integer NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_ref text NOT NULL,
    content text NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING GIN (content_tsv);
"""


def get_connection():
    """One place that knows how to reach Postgres, so no other file builds a DSN."""
    if config.DATABASE_URL:
        return psycopg2.connect(config.DATABASE_URL)
    return psycopg2.connect(**config.DB_PARAMS)


def init_db():
    """Creates extension, tables and index so a fresh clone can ingest immediately."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(SCHEMA_SQL)
        conn.commit()
        cur.close()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("schema ready")