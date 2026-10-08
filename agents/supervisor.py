import time

DIRECT_ESCALATION_CONFIDENCE_FLOOR = 0.15


def _top(labelled: dict) -> str:
    return next(iter(labelled))


def supervisor_node(state: dict) -> dict:
    start = time.perf_counter()
    cls = state["classification"]
    confidence = cls["auto_resolvable_confidence"]
    sentiment = _top(cls["sentiment"])
    urgency = _top(cls["urgency"])

    override = state.get("override_route") or ""
    if override in {"research", "escalate"}:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "route": override,
            "trace": [
                {
                    "node": "supervisor",
                    "route": override,
                    "confidence": round(confidence, 4),
                    "reasons": ["policy override (eval)"],
                    "latency_ms": round(elapsed, 2),
                }
            ],
        }

    reasons = []
    if sentiment == "angry":
        reasons.append("angry sentiment")
    if urgency == "critical":
        reasons.append("critical urgency")
    if confidence < DIRECT_ESCALATION_CONFIDENCE_FLOOR:
        reasons.append(f"auto-resolvable confidence {confidence:.2f} < 0.15")

    route = "escalate" if reasons else "research"
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "route": route,
        "trace": [
            {
                "node": "supervisor",
                "route": route,
                "confidence": round(confidence, 4),
                "reasons": reasons,
                "latency_ms": round(elapsed, 2),
            }
        ],
    }


def route_after_supervisor(state: dict) -> str:
    return state["route"]
