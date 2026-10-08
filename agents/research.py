import re
import time

from rag import get_engine

TOP_SENTENCES = 3


def _sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


def _summarize(chunks: list, query: str) -> str:
    from rag.engine import tokenize

    query_tokens = set(tokenize(query))
    scored = []
    for chunk in chunks:
        for sentence in _sentences(chunk.text):
            tokens = set(tokenize(sentence))
            if len(tokens) < 5:
                continue
            overlap = len(tokens & query_tokens) / len(tokens)
            scored.append((overlap, sentence, chunk.doc_name))
    scored.sort(key=lambda item: -item[0])
    picked = []
    seen = set()
    for _, sentence, doc_name in scored:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(f"{sentence} [{doc_name}]")
        if len(picked) == TOP_SENTENCES:
            break
    return "\n".join(picked)


def research_node(state: dict) -> dict:
    start = time.perf_counter()
    engine = get_engine()
    query = f"{state['subject']}\n{state['body']}"
    feedback = state.get("qa", {}).get("feedback") or []
    if feedback:
        query = query + "\n" + "\n".join(feedback)

    chunks = engine.retrieve(query, k=4)
    summary = _summarize(chunks, query)
    attempts = state.get("attempts", 0) + 1
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "attempts": attempts,
        "research": {
            "query": query,
            "summary": summary,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "ce_score": round(c.ce_score, 4),
                    "text": c.text,
                }
                for c in chunks
            ],
        },
        "trace": [
            {
                "node": "research",
                "attempt": attempts,
                "chunks": len(chunks),
                "latency_ms": round(elapsed, 2),
            }
        ],
    }
