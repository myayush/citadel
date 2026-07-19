import json
import re
import time

from src import config
from src import generate
from src import tools
from groq import BadRequestError

SYSTEM_PROMPT = """You are a compliance analyst. You answer questions about exactly two RBI documents:
1. "KYC MD" - Reserve Bank of India (Commercial Banks - Know Your Customer) Directions, 2025
2. "PPI MD" - Master Directions on Prepaid Payment Instruments, 2021

Rules:
- Always search before answering. Never answer from memory.
- Answer ONLY from tool results. Every claim must cite a chunk id in square brackets, like [1042].
- Questions comparing the two documents need searches covering both.
- If the documents do not contain the answer, reply with exactly NOT_IN_CORPUS.
- When you have enough to answer, give the final answer as plain text with citations."""


def extract_chunk_ids(text):
    """The verifier needs to know which ids the agent actually saw, not which it claims."""
    ids = set()
    for match in re.findall(r"\[(\d+)\]", text):
        ids.add(int(match))
    return ids


def run_tool(name, args):
    """Errors go back to the model as text so it can recover instead of crashing the loop."""
    fn = tools.TOOL_REGISTRY.get(name)
    if fn is None:
        return "Unknown tool: " + name
    try:
        return fn(**args)
    except Exception as exc:
        return "Tool error: " + str(exc)


def run_agent(question, max_iterations=8):
    """The hand-written loop: model proposes tool calls, we dispatch, repeat until it answers."""
    client = generate.get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    seen_chunk_ids = set()
    tool_calls_count = 0
    usage = {"input_tokens": 0, "output_tokens": 0}
    last_call_sig = None
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        # llama-3.3-70b occasionally emits malformed tool-call syntax that Groq's
        # parser rejects with 400 tool_use_failed. Retrying the same request
        # usually succeeds; two failures in a row means give up on tools this turn.
        resp = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=config.STRONG_MODEL,
                    messages=messages,
                    tools=tools.TOOL_SCHEMAS,
                    max_tokens=2048,
                )
                break
            except BadRequestError as exc:
                if "tool_use_failed" not in str(exc) or attempt == 2:
                    raise
                time.sleep(1.0)
        usage["input_tokens"] += resp.usage.prompt_tokens
        usage["output_tokens"] += resp.usage.completion_tokens
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            return {
                "answer": answer.strip(),
                "seen_chunk_ids": seen_chunk_ids,
                "tool_calls": tool_calls_count,
                "iterations": iterations,
                "usage": usage,
                "transcript": messages,
                "hit_iteration_limit": False,
            }

        # The assistant turn must be appended with its tool_calls intact, or the
        # tool results that follow have nothing to attach to and the API rejects them.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            tool_calls_count += 1
            # arguments arrives as a JSON string, not a dict
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = run_tool(tc.function.name, args)
            seen_chunk_ids |= extract_chunk_ids(result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # Stall detection: identical tool call twice in a row means it is looping,
        # so we tell it to stop searching and answer with what it has.
        sig = json.dumps([[tc.function.name, tc.function.arguments] for tc in msg.tool_calls])
        if sig == last_call_sig:
            messages.append({
                "role": "user",
                "content": "You repeated the same tool call. Stop searching and answer now from what you have, or reply NOT_IN_CORPUS.",
            })
        last_call_sig = sig

        # free tier: keep sequential calls gentle
        time.sleep(1.0)

    return {
        "answer": "Could not produce an answer within the iteration limit.",
        "seen_chunk_ids": seen_chunk_ids,
        "tool_calls": tool_calls_count,
        "iterations": iterations,
        "usage": usage,
        "transcript": messages,
        "hit_iteration_limit": True,
    }


if __name__ == "__main__":
    result = run_agent(
        "How does the identity information needed for a Small PPI compare with what a bank must collect for an account opened without meeting the customer?"
    )
    print(result["answer"])
    print()
    print("tool_calls:", result["tool_calls"], "iterations:", result["iterations"])
    print("seen_chunk_ids:", sorted(result["seen_chunk_ids"]))
    print("usage:", result["usage"])