import re

from src import config
from src import db
from src.ingest import get_model
from src.ingest import vector_literal


def _rows_to_results(rows):
    """Both searches return the same shape so the RRF fuser does not care which ran."""
    results = []
    for row in rows:
        results.append(
            {
                "id": row[0],
                "section_ref": row[1],
                "content": row[2],
                "title": row[3],
                "score": row[4],
            }
        )
    return results


def dense_search(query, top_k=20):
    """Semantic half: finds clauses that mean the question without sharing its words."""
    model = get_model()
    vector = model.encode([query])[0].tolist()

    # <=> is cosine DISTANCE, so smaller is better: ORDER BY ASC, and flip it to a
    # similarity below. Getting this backwards returns the worst chunks in the corpus.
    sql = (
        "SELECT c.id, c.section_ref, c.content, d.title,"
        " c.embedding <=> %s::vector AS distance"
        " FROM chunks c JOIN documents d ON d.id = c.doc_id"
        " ORDER BY distance ASC LIMIT %s"
    )

    conn = db.get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, (vector_literal(vector), top_k))
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    results = _rows_to_results(rows)
    for result in results:
        result["score"] = 1 - result["score"]
    return results

def _or_terms(query):
    """websearch_to_tsquery ANDs its terms, so one absent word returns zero rows.

    A natural-language question always carries words the target clause does not use
    ("Section", "requirements for"). We OR the terms instead and let ts_rank do the
    discriminating: a chunk matching four terms outranks one matching one.
    """
    terms = re.findall(r"\w+", query)
    return " OR ".join(terms)

def lexical_search(query, top_k=20):
    """Keyword half: exact hits like 'Section 12' that embeddings blur away."""
    sql = (
        "SELECT c.id, c.section_ref, c.content, d.title,"
        " ts_rank(c.content_tsv, websearch_to_tsquery('english', %s), 1) AS rank"
        " FROM chunks c JOIN documents d ON d.id = c.doc_id"
        " WHERE c.content_tsv @@ websearch_to_tsquery('english', %s)"
        " ORDER BY rank DESC LIMIT %s"
    )
    tsquery_input = _or_terms(query)

    conn = db.get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, (tsquery_input, tsquery_input, top_k))
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    return _rows_to_results(rows)


def _fuse(ranked_list, label, scores, chunks, sources):
    """One list's contribution to the RRF totals; called once per retrieval method."""
    for position in range(len(ranked_list)):
        result = ranked_list[position]
        chunk_id = result["id"]
        # RRF ranks are 1-based, enumerate positions are 0-based. Off by one here
        # silently changes every score.
        rank = position + 1
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (config.RRF_K + rank)
        chunks[chunk_id] = result
        if chunk_id not in sources:
            sources[chunk_id] = []
        sources[chunk_id].append(label)


def hybrid_search(query, top_k=5):
    """Fuses both methods by rank, not score: dense distances and ts_rank are not comparable."""
    dense = dense_search(query, top_k=20)
    lexical = lexical_search(query, top_k=20)

    scores = {}
    chunks = {}
    sources = {}
    _fuse(dense, "dense", scores, chunks, sources)
    _fuse(lexical, "lexical", scores, chunks, sources)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    results = []
    for chunk_id, rrf_score in ordered[:top_k]:
        result = dict(chunks[chunk_id])
        result["score"] = rrf_score
        result["sources"] = sources[chunk_id]
        results.append(result)
    return results


def _print_top3(label, results):
    for result in results[:3]:
        print(label, result["id"], result["title"], result["section_ref"], round(result["score"], 4))


if __name__ == "__main__":
    queries = [
        "requirements for video based customer identification",
        "Section 12 customer due diligence",
    ]
    for query in queries:
        print("QUERY:", query)
        _print_top3("dense   ", dense_search(query, top_k=3))
        _print_top3("lexical ", lexical_search(query, top_k=3))
        _print_top3("hybrid  ", hybrid_search(query, top_k=3))
        print()