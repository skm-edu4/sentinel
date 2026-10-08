import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag import decompose_query, get_engine


def show_results(label, results):
    print(f"\n--- {label} ---")
    print(
        f"{'rank':<5}{'ce':>8}{'rrf':>9}{'d_rank':>8}{'s_rank':>8}  doc"
    )
    for i, r in enumerate(results, 1):
        print(
            f"{i:<5}{r.ce_score:>8.2f}{r.rrf_score:>9.4f}"
            f"{str(r.dense_rank):>8}{str(r.sparse_rank):>8}  {r.doc_name}"
        )


def main():
    t0 = time.time()
    engine = get_engine()
    print(f"chunks indexed: {len(engine.chunks)} | collection: {engine._collection.count()}")

    q1 = "How many days do I have to request a refund?"
    subs1 = decompose_query(q1)
    print(f"\nsub-queries(q1): {subs1}")
    r1 = engine.retrieve(q1)
    show_results(f"q1: {q1}", r1)
    docs1 = [r.doc_name for r in r1]
    assert "refund-policy.md" in docs1[:3], f"q1 expected refund-policy in top-3, got {docs1}"

    q2 = "error code 4032"
    r2 = engine.retrieve(q2)
    show_results(f"q2: {q2}", r2)
    assert r2[0].doc_name == "troubleshooting-error-codes.md", f"q2 expected troubleshooting top-1, got {r2[0].doc_name}"

    q3 = "My screen is frozen and the app will not load anything"
    r3 = engine.retrieve(q3)
    show_results(f"q3: {q3}", r3)
    docs3 = [r.doc_name for r in r3]
    assert set(docs3[:3]) & {"known-issues-2026-10.md", "troubleshooting-error-codes.md"}, f"q3 expected crash docs in top-3, got {docs3}"

    q4 = "Where is order ORD-7788? Also, what is your return policy if it arrives damaged?"
    subs4 = decompose_query(q4)
    print(f"\nsub-queries(q4): {subs4}")
    assert len(subs4) == 2, f"q4 expected 2 sub-queries, got {subs4}"
    r4 = engine.retrieve(q4, k=6)
    show_results(f"q4 (multi-part): {q4}", r4)
    docs4 = {r.doc_name for r in r4}
    assert {"shipping-delivery.md", "returns-exchanges.md"} <= docs4, f"q4 expected shipping+returns in top-6, got {sorted(docs4)}"

    print(f"\nALL RAG CHECKS PASSED ({time.time() - t0:.1f}s)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
