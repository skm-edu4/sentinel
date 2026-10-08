import re
import time

from agents.mcp import call_tool

AMOUNT_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")
DEMAND_RE = re.compile(
    r"refund(?:ed|ing)?\s+(?:me|my|to|the|one)\b|full refund|refund my|"
    r"i want.{0,40}refund|reimburse",
    re.IGNORECASE,
)


def _extract_amount_cents(text: str) -> int | None:
    match = AMOUNT_RE.search(text)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    return int(round(value * 100))


def action_node(state: dict) -> dict:
    start = time.perf_counter()
    cls = state["classification"]
    intent = next(iter(cls["intent"]))
    text = f"{state['subject']}\n{state['body']}"

    if intent != "refund":
        return {
            "action": None,
            "trace": [
                {"node": "action", "action": "none", "latency_ms": 0.0}
            ],
        }

    amount_cents = _extract_amount_cents(text)
    if amount_cents is None or not DEMAND_RE.search(text):
        return {
            "action": None,
            "trace": [
                {
                    "node": "action",
                    "action": "none",
                    "note": "refund intent without explicit actionable amount",
                    "latency_ms": 0.0,
                }
            ],
        }

    try:
        result = call_tool(
            "action_server.py",
            "issue_refund",
            {
                "ticket_id": state["ticket_id"],
                "amount_cents": amount_cents,
                "reason": state["subject"][:200],
            },
        )
        action = dict(result)
        action["mcp_error"] = None
    except Exception as exc:
        action = {
            "ticket_id": state["ticket_id"],
            "action_type": "issue_refund",
            "amount_cents": amount_cents,
            "status": "failed",
            "action_id": None,
            "detail": str(exc),
            "mcp_error": str(exc),
        }

    elapsed = (time.perf_counter() - start) * 1000
    return {
        "action": action,
        "trace": [
            {
                "node": "action",
                "action": action.get("status"),
                "amount_cents": amount_cents,
                "latency_ms": round(elapsed, 2),
            }
        ],
    }


def route_after_action(state: dict) -> str:
    action = state.get("action")
    if action and action.get("status") == "requires_human_approval":
        return "approve"
    return "proceed"
