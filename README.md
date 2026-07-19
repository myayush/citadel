# Citadel

Ask questions about two RBI rulebooks — the KYC Directions (2025) and the
Prepaid Payment Instruments Master Directions (2021) — and get answers where
every claim points to the exact paragraph it came from. If the answer is not
in the documents, it says `NOT_IN_CORPUS` instead of guessing.

Built with FastAPI, Postgres + pgvector, a local embedding model
(all-MiniLM-L6-v2 on CPU), and the Groq API. No RAG frameworks — the search
fusion, agent loop, and evaluation are all hand-written.

## How it works

```
POST /ask
   |
   +-- simple: hybrid_search -> one LLM call with labeled excerpts
   |             |
   |             +-- dense: pgvector cosine search over MiniLM embeddings
   |             +-- lexical: Postgres tsvector full-text search
   |             +-- fused with Reciprocal Rank Fusion (in Python)
   |
   +-- agent: while-loop, max 8 iterations, for multi-document questions
                 LLM <-> search_regulations / get_section (manual dispatch)
                 every chunk id seen in tool results is recorded
                 answer -> verify_citations: cited ids must be a subset
                 of seen ids; anything else is flagged as fabricated
```

The documents are split into 536 chunks, stored in Postgres. Each question is
searched two ways:

- **dense** — vector similarity, finds text that *means* the same thing
- **lexical** — Postgres full-text search, finds exact words like "Section 12"

The two result lists are merged with Reciprocal Rank Fusion (plain Python dict
arithmetic). The top chunks go to the LLM with their ids, and the answer must
cite those ids.

**Simple mode**: search once, answer once. Fast, handles most questions.

**Agent mode**: for questions that need both documents. The LLM runs in a
hand-written loop — it asks for searches, my code runs them and feeds back the
results, up to 8 rounds. Every chunk id the agent is shown gets recorded.
After it answers, a small Python check compares the ids it *cited* against the
ids it was *shown*. Any cited id it was never shown is flagged as a fabricated
citation. No LLM involved in that check — just set membership.

## Results

Measured on 16 handwritten questions. Small set, so treat each cell as rough
(one question moves it by 20 points).

| search method | semantic questions | keyword questions |
|---|---|---|
| dense only | 40% | 100% |
| lexical only | 80% | 100% |
| hybrid (fused) | 60% | 100% |

Simple mode: 88% of answers fully supported by their sources, ~$0.0007 per
question. Agent mode: zero fabricated citations across all runs, ~$0.0018 per
question. Raw result files are in `eval/`.

What the numbers showed:

- Keyword search beat vector search here. Legal text repeats its own key
  terms, which suits keywords, and the small embedding model only reads the
  first ~1000 characters of each chunk.
- Merging two rankings gives you their average, not their best — so hybrid
  landed between the two, not above them.
- The question that needed both documents always failed in simple mode.
  Agent mode was built for exactly that case, and it handles it.
- The LLM judge that grades answers made mistakes of its own, so I checked
  its verdicts by hand across three runs.

## Run it locally

Needs: Docker, Python 3.11, a free Groq API key, and the two RBI PDFs saved
as `data/raw/kyc_md.pdf` and `data/raw/ppi_md.pdf`.

```
cp .env.example .env        # fill in GROQ_API_KEY and the two model ids;
                            # leave DATABASE_URL empty to use local Docker Postgres
docker compose up -d
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m src.db
python -m src.ingest
uvicorn src.api:app --port 8000
```

Open http://localhost:8000 in a browser, or:

```
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What is the cash loading limit for PPIs?","mode":"simple"}'
```

Run the evaluation: `python -m eval.run_eval`

For a hosted Postgres (how the live deployment works), set `DATABASE_URL` in
`.env` — it overrides the local settings.