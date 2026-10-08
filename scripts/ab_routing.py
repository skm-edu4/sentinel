import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from agents import run_ticket
from classify import classify_ticket
from eval.costs import classify_cost, estimate_ticket_cost

TICKETS = [json.loads(line) for line in open(BASE_DIR / "data" / "tickets.jsonl")]


def run_policy_a(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        classification = classify_ticket(row["subject"], row["body"])
        start = time.perf_counter()
        final = run_ticket(
            row["ticket_id"], row["subject"], row["body"], classification.to_dict()
        )
        e2e_ms = classification.latency_ms + (time.perf_counter() - start) * 1000
        records.append(
            {
                "row": row,
                "classification": classification,
                "outcome": final["outcome"],
                "e2e_ms": e2e_ms,
                "cost": estimate_ticket_cost(
                    classification.backend,
                    final["trace"],
                    draft_chars=len(final.get("draft", "")),
                ),
            }
        )
    return records


def run_policy_b(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        classification = classify_ticket(row["subject"], row["body"])
        start = time.perf_counter()
        final = run_ticket(
            row["ticket_id"],
            row["subject"],
            row["body"],
            classification.to_dict(),
            override_route="research",
        )
        e2e_ms = classification.latency_ms + (time.perf_counter() - start) * 1000
        records.append(
            {
                "row": row,
                "classification": classification,
                "outcome": final["outcome"],
                "e2e_ms": e2e_ms,
                "cost": estimate_ticket_cost(
                    classification.backend,
                    final["trace"],
                    draft_chars=len(final.get("draft", "")),
                ),
            }
        )
    return records


def run_policy_c(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        classification = classify_ticket(row["subject"], row["body"])
        records.append(
            {
                "row": row,
                "classification": classification,
                "outcome": "escalate",
                "e2e_ms": classification.latency_ms,
                "cost": classify_cost(classification.backend),
            }
        )
    return records


def score(records: list[dict]) -> dict:
    n = len(records)
    routine = [r for r in records if r["row"]["labels"]["routine"]]
    non_routine = [r for r in records if not r["row"]["labels"]["routine"]]
    auto = [r for r in records if r["outcome"] == "auto_reply"]
    unsafe = [r for r in non_routine if r["outcome"] == "auto_reply"]
    captured = [r for r in routine if r["outcome"] == "auto_reply"]
    return {
        "auto_rate": len(auto) / n,
        "unsafe_rate": len(unsafe) / len(non_routine) if non_routine else 0.0,
        "routine_capture": len(captured) / len(routine) if routine else 0.0,
        "human_load": len(records) - len(auto),
        "mean_e2e_ms": statistics.mean(r["e2e_ms"] for r in records),
        "mean_cost_usd": statistics.mean(r["cost"] for r in records),
    }


def print_table(results: dict[str, dict]) -> None:
    header = (
        f"{'policy':<26} {'auto':>6} {'unsafe':>7} {'capture':>8} "
        f"{'human':>6} {'e2e ms':>8} {'cost $':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, s in results.items():
        print(
            f"{name:<26} {s['auto_rate']:>6.0%} {s['unsafe_rate']:>7.0%} "
            f"{s['routine_capture']:>8.0%} {s['human_load']:>6d} "
            f"{s['mean_e2e_ms']:>8.0f} {s['mean_cost_usd']:>9.5f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    rows = TICKETS[: args.n]
    t0 = time.time()

    print(f"routing policy A/B on first {len(rows)} tickets (labels used for scoring only)\n")
    results = {}
    for name, runner in [
        ("A: current supervisor", run_policy_a),
        ("B: always research path", run_policy_b),
        ("C: escalate all (human-only)", run_policy_c),
    ]:
        policy_start = time.time()
        records = runner(rows)
        results[name] = score(records)
        print_table({name: results[name]})
        print(f"  [{name} ran in {time.time() - policy_start:.1f}s]\n", flush=True)

    print()
    print_table(results)

    if not JevBackend_available_note():
        print(
            "\n[Note: Jev-vs-LLM classification A/B activates when TYPESAFE_API_KEY "
            "or an LLM key is present — see classify_eval.py]"
        )

    from datetime import datetime, timezone

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "ab_routing.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n": len(rows),
                "policies": results,
            },
            indent=2,
        )
    )
    print("[reports/ab_routing.json written]")

    for s in results.values():
        assert 0.0 <= s["auto_rate"] <= 1.0
        assert 0.0 <= s["unsafe_rate"] <= 1.0
        assert 0.0 <= s["routine_capture"] <= 1.0
        assert s["human_load"] == int(round(len(rows) * (1 - s["auto_rate"])))
        assert s["mean_e2e_ms"] > 0 and s["mean_cost_usd"] > 0
    assert results["C: escalate all (human-only)"]["auto_rate"] == 0.0
    print("\n[all ab_routing assertions PASSED]")
    print(f"wall: {time.time() - t0:.1f}s")


def JevBackend_available_note() -> bool:
    from classify import JevBackend

    return JevBackend.available()


if __name__ == "__main__":
    main()
