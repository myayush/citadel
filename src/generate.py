import re

from groq import Groq

from src import config
from src import retrieval

_CLIENT = None

SYSTEM_PROMPT = """You are a compliance analyst answering questions about RBI regulations.

Rules:
- Answer ONLY from the excerpts provided. Never use outside knowledge.
- After every claim, cite the excerpt id in square brackets, like [1042].
- If the excerpts do not contain the answer, reply with exactly NOT_IN_CORPUS and nothing else.
- Be direct. No preamble."""


def get_client():
    """Client creation is cheap but the key read is not: one place to look when auth breaks."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


def build_prompt(question, results):
    """Labels excerpts with their real chunk id so the model's citations are checkable ids, not prose."""
    parts = []
    for result in results:
        header = "[" + str(result["id"]) + "] " + result["title"] + ", section " + result["section_ref"]
        parts.append(header + "\n" + result["content"])
    excerpts = "\n\n".join(parts)
    return "Excerpts:\n\n" + excerpts + "\n\nQuestion: " + question


def extract_citations(text):
    """Pulls the ids the model actually cited, which is what the API returns to the caller."""
    ids = []
    for match in re.findall(r"\[(\d+)\]", text):
        chunk_id = int(match)
        if chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def answer_simple(question):
    """One retrieve, one LLM call: the baseline the agent has to beat in the eval."""
    results = retrieval.hybrid_search(question, top_k=config.TOP_K)

    client = get_client()
    resp = client.chat.completions.create(
        model=config.STRONG_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, results)},
        ],
        max_tokens=2048,
    )

    text = resp.choices[0].message.content or ""
    usage = resp.usage

    return {
        "answer": text.strip(),
        "citations": extract_citations(text),
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        },
    }


if __name__ == "__main__":
    result = answer_simple("What is the video based customer identification process?")
    print(result["answer"])
    print("citations:", result["citations"])
    print("usage:", result["usage"])

    result = answer_simple("What is the interest rate on a 5 year fixed deposit at SBI?")
    print(result["answer"])