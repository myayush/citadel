from datetime import date

# In-memory on purpose: one container, one process. State resets on restart,
# which only ever makes the cap more generous. Fine for a demo whose only
# risk is exhausting the free-tier daily LLM quota; a real service would use
# Redis or a database counter.
_STATE = {"day": None, "count": 0}

DAILY_LIMIT = 100


def check_and_count():
    """Caps LLM-calling requests per day so one visitor cannot drain the free tier."""
    today = date.today().isoformat()
    if _STATE["day"] != today:
        _STATE["day"] = today
        _STATE["count"] = 0
    if _STATE["count"] >= DAILY_LIMIT:
        return False
    _STATE["count"] += 1
    return True