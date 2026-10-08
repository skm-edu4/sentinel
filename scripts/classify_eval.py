import json
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from classify import JevBackend, PrototypeBackend, classify_ticket
from classify.schema import Classification

TICKETS = [json.loads(line) for line in open(BASE_DIR / "data" / "tickets.jsonl")]


def accuracy(preds: list[str], golds: list[str]) -> float:
    return sum(p == g for p, g in zip(preds, golds)) / len(golds)


def auc(scores: list[float], labels: list[bool]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2
        for k in range(i, j):
            if ranked[k][1]:
                rank_sum += avg_rank
        rank = j + 1
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def run_backend(backend, name: str) -> dict:
    intent_preds, sentiment_preds, urgency_preds = [], [], []
    gold_intents, gold_sentiments, gold_urgencies = [], [], []
    ticket_ids = []
    auto_scores, routine_labels = [], []
    latencies = []
    errors = []

    for row in TICKETS:
        gold = row["labels"]
        try:
            result = classify_ticket(row["subject"], row["body"], backend=backend)
        except Exception as exc:
            errors.append(f"{row['ticket_id']}: {exc}")
            continue
        ticket_ids.append(row["ticket_id"])
        intent_preds.append(result.top_intent)
        sentiment_preds.append(result.top_sentiment)
        urgency_preds.append(result.top_urgency)
        gold_intents.append(gold["intent"])
        gold_sentiments.append(gold["sentiment"])
        gold_urgencies.append(gold["urgency"])
        auto_scores.append(result.auto_resolvable_confidence)
        routine_labels.append(gold["routine"])
        latencies.append(result.latency_ms)

    metrics = {
        "name": name,
        "n": len(intent_preds),
        "errors": errors,
        "intent_acc": accuracy(intent_preds, gold_intents),
        "sentiment_acc": accuracy(sentiment_preds, gold_sentiments),
        "urgency_acc": accuracy(urgency_preds, gold_urgencies),
        "auto_auc": auc(auto_scores, routine_labels),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_p95_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "intent_confusion": [
            (tid, g, pred)
            for tid, g, pred in zip(ticket_ids, gold_intents, intent_preds)
            if pred != g
        ],
        "auto_mean_routine": statistics.mean(
            [s for s, lab in zip(auto_scores, routine_labels) if lab]
        ),
        "auto_mean_nonroutine": statistics.mean(
            [s for s, lab in zip(auto_scores, routine_labels) if not lab]
        ),
    }
    return metrics


def report(m: dict) -> None:
    print(f"\n=== {m['name']} ===")
    if m["errors"]:
        print(f"  errors ({len(m['errors'])}): {m['errors'][:3]}")
    print(f"  intent accuracy:    {m['intent_acc']:.2%}")
    print(f"  sentiment accuracy: {m['sentiment_acc']:.2%}")
    print(f"  urgency accuracy:   {m['urgency_acc']:.2%}")
    print(f"  auto-resolvable AUC vs routine: {m['auto_auc']:.3f}")
    print(
        f"  auto mean: routine={m['auto_mean_routine']:.3f}  "
        f"non-routine={m['auto_mean_nonroutine']:.3f}"
    )
    print(
        f"  latency: mean={m['latency_mean_ms']:.1f}ms  "
        f"p95={m['latency_p95_ms']:.1f}ms"
    )
    print(f"  intent errors ({len(m['intent_confusion'])}): {m['intent_confusion']}")


CANNED_JEV_ANSWERS = {
    "answers": {
        "intent": {
            "choice": "refund",
            "confidence": 0.91,
            "probabilities": {
                "refund": 0.91,
                "billing": 0.03,
                "shipping": 0.02,
                "bug": 0.01,
                "account": 0.01,
                "feature_request": 0.01,
                "other": 0.01,
            },
        },
        "sentiment": {
            "choice": "angry",
            "confidence": 0.8,
            "probabilities": {"angry": 0.8, "frustrated": 0.1, "neutral": 0.07, "happy": 0.03},
        },
        "urgency": {
            "score": 3,
            "confidence": 0.77,
            "probabilities": {"0": 0.02, "1": 0.05, "2": 0.16, "3": 0.77},
        },
        "auto_resolvable": {"noul": 0.35},
    }
}


def jev_parse_selftest() -> None:
    import httpx

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return CANNED_JEV_ANSWERS

    original_post = httpx.post
    httpx.post = lambda *args, **kwargs: FakeResponse()
    try:
        backend = JevBackend(api_key="test-key")
        result = classify_ticket(
            TICKETS[0]["subject"], TICKETS[0]["body"], backend=backend
        )
    finally:
        httpx.post = original_post

    assert result.top_intent == "refund", result.top_intent
    assert result.top_sentiment == "angry", result.top_sentiment
    assert result.top_urgency == "critical", result.top_urgency
    assert abs(result.auto_resolvable_confidence - 0.35) < 1e-9
    assert abs(sum(result.intent.values()) - 1.0) < 1e-6
    assert abs(sum(result.urgency.values()) - 1.0) < 1e-6
    print("\n[jev parse selftest PASSED: choice/score/noul mapping + renormalization]")


def main():
    prototype = PrototypeBackend()
    results = [run_backend(prototype, prototype.name)]
    if JevBackend.available():
        results.append(run_backend(JevBackend(), JevBackend.name))
    else:
        print("\n[JevBackend skipped: no TYPESAFE_API_KEY / JEV_API_KEY in environment]")

    for m in results:
        report(m)

    sample = classify_ticket(TICKETS[0]["subject"], TICKETS[0]["body"], backend=prototype)
    print("\n--- brief-shaped output (T-0001) ---")
    print(json.dumps(sample.to_dict(), indent=2))

    jev_parse_selftest()

    m = results[0]
    assert m["errors"] == [], m["errors"]
    assert m["intent_acc"] >= 0.72, m["intent_acc"]
    assert m["sentiment_acc"] >= 0.44, m["sentiment_acc"]
    assert m["urgency_acc"] >= 0.44, m["urgency_acc"]
    assert m["auto_auc"] >= 0.55, m["auto_auc"]
    assert m["latency_p95_ms"] < 500, m["latency_p95_ms"]

    from datetime import datetime, timezone

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "classify_eval.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "backends": [
                    {
                        "name": r["name"],
                        "n": r["n"],
                        "intent_acc": r["intent_acc"],
                        "sentiment_acc": r["sentiment_acc"],
                        "urgency_acc": r["urgency_acc"],
                        "auto_auc": r["auto_auc"],
                        "latency_mean_ms": r["latency_mean_ms"],
                        "latency_p95_ms": r["latency_p95_ms"],
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    print("[reports/classify_eval.json written]")
    print("\n[all classify_eval assertions PASSED]")
    return results


if __name__ == "__main__":
    main()
