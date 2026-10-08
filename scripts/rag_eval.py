import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from agents.research import _summarize
from eval.metrics import answer_relevance, context_precision, context_recall, faithfulness
from eval.qrels import QRELS
from rag import get_engine

K = 4


def _unique_docs(chunks) -> list[str]:
    seen = set()
    docs = []
    for chunk in chunks:
        if chunk.doc_name not in seen:
            seen.add(chunk.doc_name)
            docs.append(chunk.doc_name)
    return docs


def _dense_docs(engine, query: str, k: int) -> list[str]:
    seen = set()
    docs = []
    for chunk_id in engine.dense_search(query, n=k * 2):
        doc = engine.by_id[chunk_id].doc_name
        if doc not in seen:
            seen.add(doc)
            docs.append(doc)
        if len(docs) >= k:
            break
    return docs


def retrieval_scores(system_name: str, doc_lists: list[list[str]]) -> dict:
    precisions, recalls = [], []
    for docs, entry in zip(doc_lists, QRELS):
        precisions.append(context_precision(docs, entry["relevant"], K))
        recalls.append(context_recall(docs, entry["relevant"], K))
    return {
        "system": system_name,
        "context_precision": statistics.mean(precisions),
        "context_recall": statistics.mean(recalls),
    }


def main():
    t0 = time.time()
    engine = get_engine()

    dense_docs = [_dense_docs(engine, e["query"], K) for e in QRELS]
    hybrid_norerank_docs = [
        _unique_docs(engine.retrieve(e["query"], k=K, rerank=False)) for e in QRELS
    ]
    hybrid_docs = [
        _unique_docs(engine.retrieve(e["query"], k=K, rerank=True)) for e in QRELS
    ]

    print("--- retrieval A/B (doc-level, k=4) ---")
    rows = [
        retrieval_scores("dense-only", dense_docs),
        retrieval_scores("hybrid (no rerank)", hybrid_norerank_docs),
        retrieval_scores("hybrid + cross-encoder rerank", hybrid_docs),
    ]
    for row in rows:
        print(
            f"  {row['system']:<30} precision={row['context_precision']:.3f}  "
            f"recall={row['context_recall']:.3f}"
        )

    faiths, relevances = [], []
    for entry, chunks in zip(QRELS, hybrid_chunks(engine)):
        answer = _summarize(chunks, entry["query"])
        faiths.append(faithfulness(answer, [c.__dict__ for c in chunks]))
        relevances.append(answer_relevance(answer, entry["query"]))
    print("\n--- generation metrics (extractive answers, hybrid + rerank) ---")
    print(f"  faithfulness:    {statistics.mean(faiths):.3f}")
    print(f"  answer relevance: {statistics.mean(relevances):.3f}")

    hybrid_row = rows[2]
    dense_row = rows[0]
    assert hybrid_row["context_recall"] >= 0.70, hybrid_row
    assert hybrid_row["context_precision"] >= 0.50, hybrid_row
    assert hybrid_row["context_precision"] >= dense_row["context_precision"] - 0.05, (
        hybrid_row,
        dense_row,
    )
    assert statistics.mean(faiths) >= 0.80, statistics.mean(faiths)
    assert statistics.mean(relevances) >= 0.30, statistics.mean(relevances)

    import json
    from datetime import datetime, timezone

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": K,
        "n_queries": len(QRELS),
        "retrieval": rows,
        "generation": {
            "faithfulness": statistics.mean(faiths),
            "answer_relevance": statistics.mean(relevances),
        },
        "divergences": [
            {
                "id": entry["id"],
                "query": entry["query"],
                "relevant": entry["relevant"],
                "dense": dense,
                "hybrid": hybrid,
            }
            for entry, dense, hybrid, faith in zip(QRELS, dense_docs, hybrid_docs, faiths)
            if context_recall(dense, entry["relevant"], K)
            != context_recall(hybrid, entry["relevant"], K)
        ],
    }
    (reports_dir / "rag_eval.json").write_text(json.dumps(report, indent=2))
    print("[reports/rag_eval.json written]")
    print("\n[all rag_eval assertions PASSED]", flush=True)
    print(f"wall: {time.time() - t0:.1f}s", flush=True)
    import os

    os._exit(0)


def hybrid_chunks(engine):
    return [engine.retrieve(e["query"], k=K) for e in QRELS]


if __name__ == "__main__":
    main()
