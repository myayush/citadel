import re


def verify_citations(answer_text, seen_chunk_ids):
    """Catches fabricated citations: ids the model wrote but was never shown this run.

    Pure Python on purpose. seen_chunk_ids came out of real tool results, which came
    out of the database, so membership in that set proves both existence and
    provenance. No LLM call, no DB call, nothing to trust.
    """
    cited = []
    for match in re.findall(r"\[(\d+)\]", answer_text):
        chunk_id = int(match)
        if chunk_id not in cited:
            cited.append(chunk_id)

    fabricated = []
    for chunk_id in cited:
        if chunk_id not in seen_chunk_ids:
            fabricated.append(chunk_id)

    if cited:
        validity = (len(cited) - len(fabricated)) / len(cited)
    else:
        # No citations at all: valid for a refusal, vacuous otherwise.
        # The caller can tell the difference by checking the answer text.
        validity = 1.0

    return {
        "citation_validity": validity,
        "cited_ids": cited,
        "fabricated_ids": fabricated,
    }