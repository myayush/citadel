import json
import time
from datetime import date

from groq import RateLimitError

from eval import judge_prompts
from src import config
from src import generate
from src import retrieval

from src import agent
from src import verify

SLEEP_BETWEEN_CALLS = 2.0
MAX_RETRIES = 5

STRONG_INPUT_PER_MTOK = 0.59
STRONG_OUTPUT_PER_MTOK = 0.79
CHEAP_INPUT_PER_MTOK = 0.05
CHEAP_OUTPUT_PER_MTOK = 0.08


def load_golden_set(path):
    """One JSON object per line, so a typo breaks one question instead of the file."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def call_with_backoff(fn, *args):
    """Free tier 429s on RPM, TPM or TPD; this run makes ~90 calls so it will hit one."""
    delay = 5.0
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = delay * 2


def section_matches(ref, source_section):
    """Plain startswith would let source_section '1' match section '15'. Anchor the boundary."""
    if ref == source_section:
        return True
    if ref.startswith(source_section + "."):
        return True
    if ref.startswith(source_section + "-part"):
        return True
    return False


def hit_at_5(results, row):
    """A hit means the right clause of the right document made the top 5. No partial credit."""
    for result in results[:5]:
        if result["title"] != row["source_doc"]:
            continue
        if section_matches(result["section_ref"], row["source_section"]):
            return 1
    return 0


def run_retrieval_eval(rows):
    """The ablation: same questions, three retrievers, split by query_type. No LLM calls."""
    methods = {
        "dense": retrieval.dense_search,
        "lexical": retrieval.lexical_search,
        "hybrid": retrieval.hybrid_search,
    }
    table = {"dense": {}, "lexical": {}, "hybrid": {}}

    for row in rows:
        if row["query_type"] in ("multi_doc", "refusal"):
            continue
        for name in methods:
            results = methods[name](row["question"], top_k=5)
            bucket = table[name].setdefault(row["query_type"], [])
            bucket.append(hit_at_5(results, row))
    return table


def parse_judge(text):
    """Models fence their JSON despite being told not to; strip first, let the caller retry."""
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "")
    return json.loads(cleaned.strip())


def judge_once(excerpts, answer):
    """Returns (parsed_or_None, usage) so the caller can retry without losing the token count."""
    client = generate.get_client()
    prompt = judge_prompts.FAITHFULNESS.format(excerpts=excerpts, answer=answer)
    resp = client.chat.completions.create(
        model=config.CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    try:
        return parse_judge(resp.choices[0].message.content or ""), usage
    except json.JSONDecodeError:
        return None, usage

def judge_with_retry(excerpts, answer_text):
    """One retry on unparseable judge output; both eval paths need identical handling."""
    verdict, judge_usage = call_with_backoff(judge_once, excerpts, answer_text)
    time.sleep(SLEEP_BETWEEN_CALLS)
    if verdict is None:
        verdict, judge_usage = call_with_backoff(judge_once, excerpts, answer_text)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return verdict, judge_usage


def run_answer_eval(rows):
    """Answers every question and grades it, one call at a time to stay inside the free tier."""
    per_question = []
    for row in rows:
        answer = call_with_backoff(generate.answer_simple, row["question"])
        time.sleep(SLEEP_BETWEEN_CALLS)

        # Rebuilt rather than returned by answer_simple: retrieval is deterministic, so the
        # judge grades against the exact text the answerer saw. Costs one DB call, no tokens.
        results = retrieval.hybrid_search(row["question"], top_k=config.TOP_K)
        excerpts = generate.build_prompt(row["question"], results)

        verdict, judge_usage = judge_with_retry(excerpts, answer["answer"])

        per_question.append({
            "id": row["id"],
            "question": row["question"],
            "query_type": row["query_type"],
            "cohort": row["cohort"],
            "answer": answer["answer"],
            "citations": answer["citations"],
            "verdict": verdict,
            "answer_usage": answer["usage"],
            "judge_usage": judge_usage,
        })
        print("done", row["id"], "verdict", verdict)
    return per_question

def run_agent_eval(rows):
    """Agent mode on the multi_doc questions: the cases single-shot retrieval failed."""
    per_question = []
    for row in rows:
        if row["query_type"] != "multi_doc":
            continue
        result = call_with_backoff(agent.run_agent, row["question"])
        time.sleep(SLEEP_BETWEEN_CALLS)
        check = verify.verify_citations(result["answer"], result["seen_chunk_ids"])

      
   
        tool_texts = []
        for m in result["transcript"]:
            if isinstance(m, dict) and m.get("role") == "tool":
                tool_texts.append(m["content"])
        excerpts = "\n\n".join(tool_texts)

        verdict, judge_usage = judge_with_retry(excerpts, result["answer"])

        per_question.append({
            "id": row["id"],
            "question": row["question"],
            "answer": result["answer"],
            "verdict": verdict,
            "citation_validity": check["citation_validity"],
            "fabricated_ids": check["fabricated_ids"],
            "tool_calls": result["tool_calls"],
            "iterations": result["iterations"],
            "hit_iteration_limit": result["hit_iteration_limit"],
            "answer_usage": result["usage"],
            "judge_usage": judge_usage,
        })
        print("agent done", row["id"], "tools", result["tool_calls"], "validity", check["citation_validity"])
    return per_question


def print_agent_report(per_question):
    if not per_question:
        return
    graded = 0
    faithful = 0
    total_cost = 0.0
    validity_sum = 0.0
    tools_sum = 0
    iters_sum = 0
    for row in per_question:
        total_cost += usd(row["answer_usage"], STRONG_INPUT_PER_MTOK, STRONG_OUTPUT_PER_MTOK)
        total_cost += usd(row["judge_usage"], CHEAP_INPUT_PER_MTOK, CHEAP_OUTPUT_PER_MTOK)
        validity_sum += row["citation_validity"]
        tools_sum += row["tool_calls"]
        iters_sum += row["iterations"]
        if row["verdict"] is not None:
            graded += 1
            if row["verdict"]["verdict"] == "YES":
                faithful += 1

    n = len(per_question)
    print()
    print("agent mode: n=" + str(n))
    if graded:
        print("faithfulness:", round(100.0 * faithful / graded), "%")
    print("citation validity:", round(100.0 * validity_sum / n), "%")
    print("avg tool calls:", round(tools_sum / n, 1))
    print("avg iterations:", round(iters_sum / n, 1))
    print("cost/question: $" + str(round(total_cost / n, 5)))


def usd(usage, input_rate, output_rate):
    input_cost = (usage["input_tokens"] / 1000000.0) * input_rate
    output_cost = (usage["output_tokens"] / 1000000.0) * output_rate
    return input_cost + output_cost


def print_retrieval_table(table):
    types = ["semantic", "lexical"]
    print()
    print("hit@5 by retrieval method and query_type")
    header = "method".ljust(10)
    for name in types:
        header = header + name.ljust(14)
    print(header)
    for method in ["dense", "lexical", "hybrid"]:
        line = method.ljust(10)
        for name in types:
            scores = table[method].get(name, [])
            if scores:
                pct = round(100.0 * sum(scores) / len(scores))
                cell = str(pct) + "% (n=" + str(len(scores)) + ")"
            else:
                cell = "-"
            line = line + cell.ljust(14)
        print(line)


def print_answer_report(per_question):
    
    graded = 0
    faithful = 0
    total_cost = 0.0
    for row in per_question:
        total_cost = total_cost + usd(row["answer_usage"], STRONG_INPUT_PER_MTOK, STRONG_OUTPUT_PER_MTOK)
        total_cost = total_cost + usd(row["judge_usage"], CHEAP_INPUT_PER_MTOK, CHEAP_OUTPUT_PER_MTOK)
        if row["verdict"] is not None:
            graded = graded + 1
            if row["verdict"]["verdict"] == "YES":
                faithful = faithful + 1

    print()
    print("answers: n=" + str(len(per_question)))
    print("judge parsed:", graded, "of", len(per_question))
    if graded:
        print("faithfulness:", round(100.0 * faithful / graded), "%")
    print("total cost: $" + str(round(total_cost, 4)))
    print("cost/question: $" + str(round(total_cost / len(per_question), 5)))


if __name__ == "__main__":
    path = config.PROJECT_ROOT / "eval" / "golden_set.jsonl"
    rows = load_golden_set(path)
    print("loaded", len(rows), "questions")

    table = run_retrieval_eval(rows)
    print_retrieval_table(table)

    per_question = run_answer_eval(rows)
    print_answer_report(per_question)
    
    agent_results = run_agent_eval(rows)
    print_agent_report(agent_results)


    out = config.PROJECT_ROOT / "eval" / ("results_" + date.today().isoformat() + ".json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"retrieval": table, "answers": per_question, "agent": agent_results}, handle, indent=2)
    print("saved", out)