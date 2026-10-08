import time

from agents.mcp import call_tool


def _mcp_enrich(state: dict, handoff: dict) -> None:
    try:
        created = call_tool(
            "action_server.py",
            "create_escalation",
            {
                "ticket_id": state["ticket_id"],
                "priority": handoff["priority"],
                "reasons": handoff["reasons"],
                "summary": handoff.get("research_summary", ""),
            },
        )
        handoff["escalation_id"] = created["escalation_id"]
    except Exception as exc:
        handoff["mcp_error"] = f"create_escalation failed: {exc}"
        return
    try:
        ticket = call_tool(
            "ticket_server.py", "get_ticket", {"ticket_id": state["ticket_id"]}
        )
        customer = call_tool(
            "crm_server.py", "get_customer", {"customer_id": ticket["customer_id"]}
        )
        handoff["customer"] = customer
    except Exception as exc:
        handoff["mcp_error"] = f"customer context failed: {exc}"


def escalation_node(state: dict) -> dict:
    start = time.perf_counter()
    cls = state["classification"]
    sentiment = next(iter(cls["sentiment"]))
    urgency = next(iter(cls["urgency"]))
    qa = state.get("qa") or {}
    action = state.get("action")
    supervisor_trace = next(
        (t for t in state.get("trace", []) if t["node"] == "supervisor"), {}
    )

    reasons = list(supervisor_trace.get("reasons", []))
    if action and action.get("status") == "requires_human_approval":
        amount = (action.get("amount_cents") or 0) / 100
        reasons.append(
            f"Refund of ${amount:.2f} exceeds the $50 auto-approval limit — "
            "human approval required"
        )
    if not reasons and not qa.get("passed", False):
        reasons.append(f"QA grounding failed (score {qa.get('score', 0.0)})")
    if not reasons and qa.get("passed") and cls["auto_resolvable_confidence"] < 0.8:
        reasons.append(
            "QA passed but auto-resolvable confidence "
            f"{cls['auto_resolvable_confidence']:.2f} < 0.80 — human review required"
        )

    priority = "high" if sentiment == "angry" or urgency == "critical" else "normal"
    if action and action.get("status") == "requires_human_approval":
        priority = "high"
    handoff = {
        "priority": priority,
        "reasons": reasons,
        "intent": cls["intent"],
        "sentiment": sentiment,
        "urgency": urgency,
        "auto_resolvable_confidence": cls["auto_resolvable_confidence"],
        "research_summary": (state.get("research") or {}).get("summary", ""),
        "suggested_draft": state.get("draft", ""),
        "qa": qa,
        "action": action,
    }
    _mcp_enrich(state, handoff)
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "outcome": "escalate",
        "handoff": handoff,
        "trace": [
            {
                "node": "escalation",
                "priority": priority,
                "reasons": reasons,
                "escalation_id": handoff.get("escalation_id"),
                "latency_ms": round(elapsed, 2),
            }
        ],
    }
