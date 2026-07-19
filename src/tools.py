from src import db
from src import retrieval


def search_regulations(query, top_k=5):
    """The agent's eyes: wraps hybrid_search into compact strings it can read cheaply."""
    results = retrieval.hybrid_search(query, top_k=top_k)
    lines = []
    for r in results:
        line = "[" + str(r["id"]) + "] (" + r["title"] + ", section " + r["section_ref"] + "): " + r["content"][:400]
        lines.append(line)
    if not lines:
        return "No results found for this query."
    return "\n\n".join(lines)


def get_section(doc_title, section_ref):
    """Lets the agent read a full section after search showed it only the first 400 chars."""
    sql = (
        "SELECT c.id, c.section_ref, c.content"
        " FROM chunks c JOIN documents d ON d.id = c.doc_id"
        " WHERE d.title = %s AND (c.section_ref = %s OR c.section_ref LIKE %s)"
        " ORDER BY c.id"
    )
    conn = db.get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, (doc_title, section_ref, section_ref + "-part%"))
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    if not rows:
        return "No section " + section_ref + " found in " + doc_title + "."
    lines = []
    for row in rows:
        lines.append("[" + str(row[0]) + "] section " + row[1] + ":\n" + row[2])
    return "\n\n".join(lines)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_regulations",
            "description": "Search both RBI documents for passages relevant to a query. Returns chunk ids, document, section and a preview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": "Fetch the full text of one section of one document. Use after search_regulations shows a promising section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_title": {"type": "string", "description": "Exactly 'KYC MD' or 'PPI MD'"},
                    "section_ref": {"type": "string", "description": "Section number like '42' or '9.2'"},
                },
                "required": ["doc_title", "section_ref"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "search_regulations": search_regulations,
    "get_section": get_section,
}