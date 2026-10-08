import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from agents import run_ticket
from classify import classify_ticket
from eval.costs import estimate_ticket_cost

REPORTS_DIR = BASE_DIR / "reports"
TICKETS = [json.loads(line) for line in open(BASE_DIR / "data" / "tickets.jsonl")]

LAYER_NODES = [
    "supervisor",
    "research",
    "resolution",
    "qa",
    "action",
    "escalation",
]


def node_ms(trace: list[dict], node: str) -> float:
    return sum(step["latency_ms"] for step in trace if step["node"] == node)


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def run_record(row: dict) -> dict:
    classification = classify_ticket(row["subject"], row["body"])
    start = time.perf_counter()
    final = run_ticket(
        row["ticket_id"], row["subject"], row["body"], classification.to_dict()
    )
    e2e_ms = classification.latency_ms + (time.perf_counter() - start) * 1000
    trace = final["trace"]
    llm_name = next(
        (step.get("llm") for step in trace if step["node"] == "resolution"),
        "template-extractive",
    )
    cost = estimate_ticket_cost(
        classification.backend,
        trace,
        draft_chars=len(final.get("draft", "")),
        llm_name=llm_name,
    )
    layers = {node: round(node_ms(trace, node), 2) for node in LAYER_NODES}
    layers["graph_ms"] = round(sum(step["latency_ms"] for step in trace), 2)
    return {
        "trace_id": uuid.uuid4().hex[:12],
        "ticket_id": row["ticket_id"],
        "classification_backend": classification.backend,
        "classification_ms": round(classification.latency_ms, 2),
        "route": final["route"],
        "outcome": final["outcome"],
        "attempts": final.get("attempts", 0),
        "qa_score": (final.get("qa") or {}).get("score"),
        "action": (final.get("action") or {}).get("status"),
        "escalation_id": (final.get("handoff") or {}).get("escalation_id"),
        "confidence": classification.auto_resolvable_confidence,
        "layers": layers,
        "e2e_ms": round(e2e_ms, 2),
        "estimated_cost_usd": round(cost, 6),
        "trace": trace,
    }


def summarize(records: list[dict]) -> dict:
    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record["outcome"]] = outcomes.get(record["outcome"], 0) + 1
    layer_summary = {}
    for node in LAYER_NODES + ["graph_ms"]:
        values = [record["layers"][node] for record in records]
        layer_summary[node] = percentiles(values)
    layer_summary["classify_ms"] = percentiles(
        [record["classification_ms"] for record in records]
    )
    layer_summary["e2e_ms"] = percentiles([record["e2e_ms"] for record in records])
    costs = [record["estimated_cost_usd"] for record in records]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(records),
        "backend": records[0]["classification_backend"] if records else "",
        "outcome_counts": outcomes,
        "auto_rate": outcomes.get("auto_reply", 0) / len(records) if records else 0.0,
        "layer_latency_ms": layer_summary,
        "cost_usd": {
            "mean": statistics.mean(costs) if costs else 0.0,
            "total": sum(costs),
        },
        "cost_table_note": (
            "Estimates from eval/costs.py; local prototype/embedding costs are "
            "amortized estimates, Jev and LLM rates are published list prices."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    rows = TICKETS if args.all else TICKETS[: args.n]
    t0 = time.time()
    records = []
    for row in rows:
        record = run_record(row)
        records.append(record)
        print(
            f"{record['ticket_id']} trace={record['trace_id']} "
            f"{record['outcome']:<11} e2e={record['e2e_ms']:.0f}ms "
            f"cost=${record['estimated_cost_usd']:.5f}"
        )

    summary = summarize(records)
    REPORTS_DIR.mkdir(exist_ok=True)
    traces_path = REPORTS_DIR / "traces.jsonl"
    summary_path = REPORTS_DIR / "summary.json"
    with traces_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n--- summary ---")
    print(json.dumps(summary, indent=2))

    assert traces_path.exists() and summary_path.exists()
    assert len(records) == summary["n"]
    assert all(r["trace_id"] and r["outcome"] in {"auto_reply", "escalate"} for r in records)
    assert summary["cost_usd"]["mean"] >= 0
    assert summary["layer_latency_ms"]["e2e_ms"]["p50"] <= summary["layer_latency_ms"]["e2e_ms"]["p95"]
    print("\n[all trace_report assertions PASSED]", flush=True)
    print(f"wall: {time.time() - t0:.1f}s", flush=True)
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
