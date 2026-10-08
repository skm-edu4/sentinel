import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from agents import run_ticket
from classify import classify_ticket

TICKETS = [json.loads(line) for line in open(BASE_DIR / "data" / "tickets.jsonl")]
DEMO_IDS = ["T-0001", "T-0002", "T-0006", "T-0010", "T-0024", "T-0045"]


def _refund_classification(confidence: float = 0.85) -> dict:
    return {
        "intent": {
            "refund": confidence,
            "billing": (1 - confidence) * 0.5,
            "shipping": (1 - confidence) * 0.2,
            "bug": (1 - confidence) * 0.1,
            "account": (1 - confidence) * 0.1,
            "feature_request": (1 - confidence) * 0.05,
            "other": (1 - confidence) * 0.05,
        },
        "sentiment": {"neutral": 0.7, "frustrated": 0.2, "happy": 0.05, "angry": 0.05},
        "urgency": {"normal": 0.7, "low": 0.2, "high": 0.08, "critical": 0.02},
        "auto_resolvable_confidence": confidence,
    }


def guardrail_scenarios() -> list[tuple[str, dict]]:
    print("\n--- guardrail scenario A: refund $75 > $50 ---")
    over = run_ticket(
        "T-0007",
        "Refund the damaged mirror order",
        "I want $75 refunded to my card for order ORD-7788 — please refund my "
        "card for the damaged mirror.",
        _refund_classification(),
    )
    print(
        f"outcome={over['outcome']} action={over['action']['status']} "
        f"escalation_id={over['handoff'].get('escalation_id')} "
        f"customer={over['handoff'].get('customer', {}).get('customer_id')}"
    )
    assert over["outcome"] == "escalate"
    assert over["action"]["status"] == "requires_human_approval"
    assert any("approval" in reason for reason in over["handoff"]["reasons"])
    assert over["handoff"].get("escalation_id"), "MCP create_escalation not recorded"
    customer_id = over["handoff"].get("customer", {}).get("customer_id", "")
    assert customer_id.startswith("C-"), customer_id
    assert over["handoff"]["customer"]["ticket_count"] >= 1
    assert over["trace"][-1]["node"] == "escalation"

    print("\n--- guardrail scenario B: refund $42 <= $50 ---")
    under = run_ticket(
        "T-0007",
        "Refund shipping overcharge",
        "Please refund my card $42 for order ORD-7788, I want this shipping "
        "charge refunded to my card.",
        _refund_classification(),
    )
    print(
        f"outcome={under['outcome']} action={under['action']['status']} "
        f"action_id={under['action'].get('action_id')}"
    )
    assert under["outcome"] == "auto_reply"
    assert under["action"]["status"] == "executed"
    assert under["action"].get("action_id")
    return [("refund_over_limit", over), ("refund_under_limit", under)]


def main():
    t0 = time.time()
    results = []
    for row in TICKETS:
        if row["ticket_id"] not in DEMO_IDS:
            continue
        classification = classify_ticket(row["subject"], row["body"])
        final = run_ticket(
            row["ticket_id"],
            row["subject"],
            row["body"],
            classification.to_dict(),
        )
        total_ms = sum(step["latency_ms"] for step in final["trace"])
        results.append((row, classification, final, total_ms))
        print(
            f"{row['ticket_id']}  route={final['route']:<8} outcome={final['outcome']:<11} "
            f"conf={classification.auto_resolvable_confidence:.3f} "
            f"attempts={final.get('attempts', 0)} qa={(final.get('qa') or {}).get('score', '-')} "
            f"graph={total_ms:.0f}ms"
        )

    print("\n--- full trace (T-0010) ---")
    for step in results[3][2]["trace"]:
        print(json.dumps(step))

    angry_ids = {r[0]["ticket_id"] for r in results if r[1].top_sentiment == "angry"}
    for row, classification, final, _ in results:
        assert final["outcome"] in {"auto_reply", "escalate"}, final["outcome"]
        if final["outcome"] == "auto_reply":
            assert final["qa"]["passed"], row["ticket_id"]
            assert classification.auto_resolvable_confidence >= 0.8, row["ticket_id"]
        if final["outcome"] == "escalate":
            handoff = final["handoff"]
            assert handoff["priority"] in {"high", "normal"}, row["ticket_id"]
            assert handoff["reasons"], row["ticket_id"]
        if row["ticket_id"] in angry_ids:
            assert final["outcome"] == "escalate", row["ticket_id"]
            assert final["route"] == "escalate", row["ticket_id"]
        nodes = [s["node"] for s in final["trace"]]
        assert nodes[0] == "supervisor", nodes
        if final["route"] == "research":
            assert "research" in nodes and "resolution" in nodes and "qa" in nodes, nodes
        if final["outcome"] == "escalate":
            assert final["handoff"].get("escalation_id"), row["ticket_id"]

    guardrail_scenarios()

    print("\n[all agent_demo assertions PASSED]")
    print(f"wall time: {time.time() - t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
