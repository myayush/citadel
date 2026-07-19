from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from src import db
from src import generate

from src import agent
from src import verify  
from fastapi.responses import FileResponse
from src import demo_guard

app = FastAPI(title="Citadel")


class AskRequest(BaseModel):
    question: str
    mode: str = "simple"


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    answer: str
    citations: list[int]
    usage: Usage
    mode: str
    citation_validity: float | None = None
    fabricated_ids: list[int] | None = None
    tool_calls: int | None = None
    iterations: int | None = None
    
@app.get("/")
def index():
    """Serves the demo page; everything else on this app is JSON."""
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    """Checks the DB, not just the process: a live app with a dead Postgres answers nothing."""
    conn = db.get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM chunks")
        count = cur.fetchone()[0]
        cur.close()
    finally:
        conn.close()
    return {"status": "ok", "chunks": count}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """One endpoint, two modes: simple answers fast, agent searches iteratively and is verified."""
    if request.mode not in ("simple", "agent"):
        raise HTTPException(status_code=400, detail="mode must be 'simple' or 'agent'")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    
    if not demo_guard.check_and_count():
        raise HTTPException(
            status_code=429,
            detail="Daily demo limit reached. See the repo and demo video: github.com/<your-username>/citadel",
        )
        
    if request.mode == "simple":
        result = generate.answer_simple(request.question)
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "usage": result["usage"],
            "mode": "simple",
        }

    result = agent.run_agent(request.question)
    check = verify.verify_citations(result["answer"], result["seen_chunk_ids"])
    return {
        "answer": result["answer"],
        "citations": check["cited_ids"],
        "usage": result["usage"],
        "mode": "agent",
        "citation_validity": check["citation_validity"],
        "fabricated_ids": check["fabricated_ids"],
        "tool_calls": result["tool_calls"],
        "iterations": result["iterations"],
    }