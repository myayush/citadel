# Citadel

Question answering over two RBI regulatory documents: the Commercial Banks KYC
Directions, 2025 and the Master Directions on Prepaid Payment Instruments, 2021.
Every claim in an answer cites the exact source chunk. If the answer is not in
the documents, it refuses instead of guessing.

Stack: FastAPI, Postgres + pgvector, sentence-transformers (all-MiniLM-L6-v2,
local CPU), Groq API. No LangChain or LlamaIndex - retrieval fusion, the agent
loop, and the eval harness are hand-rolled on purpose.

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

## Results

16 handwritten questions (small set - one question moves a cell by 20 points):

| method | semantic hit@5 | lexical hit@5 |
|---|---|---|
| dense | 40% | 100% |
| lexical | 80% | 100% |
| hybrid | 60% | 100% |

Simple mode: 88% faithfulness, $0.0007/question. Agent mode: 100% citation
validity (zero fabricated citations), $0.0018/question. Raw results in `eval/`.

Main findings: Postgres full-text search beat the dense retriever on this
corpus (formulaic regulatory prose favors keywords; MiniLM's 256-token window
is weak on statutory text). RRF averages its inputs rather than picking the
better one, so hybrid landed between them. The cross-document question failed
in simple mode every run and is the reason agent mode exists. The LLM judge
was the least reliable component and its verdicts were audited by hand.

## Run it

Needs: Docker, Python 3.11, a free Groq API key, and the two RBI PDFs saved as
`data/raw/kyc_md.pdf` and `data/raw/ppi_md.pdf`.

```
cp .env.example .env        # fill in GROQ_API_KEY and the two model ids;
                            # leave DATABASE_URL empty to use the local Docker Postgres
docker compose up -d
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m src.db
python -m src.ingest
uvicorn src.api:app --port 8000
```

Open http://localhost:8000, or:

```
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What is the cash loading limit for PPIs?","mode":"simple"}'
```

Run the eval: `python -m eval.run_eval`

To run against a hosted Postgres instead (this is how the live deployment
works), set DATABASE_URL in .env - it takes precedence over the local values.