FAITHFULNESS = """You are grading whether an answer is supported by source excerpts.

{excerpts}

Answer being graded:
{answer}

Is every factual claim in the answer supported by the excerpts above? An answer
of exactly NOT_IN_CORPUS is correct only if the excerpts genuinely do not
contain the answer to the question.

Reply with exactly one JSON object and nothing else. No markdown, no code
fences, no text before or after it. Use this shape:

{{"verdict": "YES", "reason": "one short sentence"}}

verdict must be the string YES or the string NO.
"""