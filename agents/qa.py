import re
import time

from rag.engine import tokenize

PASS_RATIO = 0.5
GROUNDING_THRESHOLD = 0.45
MIN_CONTENT_TOKENS = 4
MAX_ATTEMPTS = 2
NUMBER_RE = re.compile(r"(?<![#\w])\d+(?:[.,]\d+)*(?!\w)")
CITATION_RE = re.compile(r"\[[^\]]*\]")


def _content_sentences(draft: str) -> list[str]:
    sentences = []
    for raw in re.split(r"(?<=[.!?])\s+", draft.replace("\n", " ")):
        sentence = raw.strip().lstrip("-").strip()
        if len(set(tokenize(sentence))) >= MIN_CONTENT_TOKENS:
            sentences.append(sentence)
    return sentences


def _numbers_grounded(sentence: str, contexts: list[str]) -> bool:
    cleaned = CITATION_RE.sub("", sentence)
    numbers = NUMBER_RE.findall(cleaned)
    if not numbers:
        return True
    return all(any(number in context for context in contexts) for number in numbers)


def score_faithfulness(draft: str, chunks: list[dict]) -> tuple[float, list[str]]:
    context_texts = [c["text"] for c in chunks]
    contexts = [set(tokenize(text)) for text in context_texts]
    sentences = _content_sentences(draft)
    if not sentences:
        return 0.0, []
    ungrounded = []
    grounded = 0
    for sentence in sentences:
        tokens = set(tokenize(sentence))
        best = 0.0
        for context in contexts:
            if not tokens:
                continue
            overlap = len(tokens & context) / len(tokens)
            best = max(best, overlap)
        if best >= GROUNDING_THRESHOLD and _numbers_grounded(sentence, context_texts):
            grounded += 1
        else:
            ungrounded.append(sentence)
    return grounded / len(sentences), ungrounded


def qa_node(state: dict) -> dict:
    start = time.perf_counter()
    chunks = (state.get("research") or {}).get("chunks", [])
    score, ungrounded = score_faithfulness(state.get("draft", ""), chunks)
    attempts = state.get("attempts", 1)
    passed = score >= PASS_RATIO and len(ungrounded) < len(_content_sentences(state.get("draft", "")))
    cls = state["classification"]
    confidence = cls["auto_resolvable_confidence"]

    outcome = ""
    if passed:
        outcome = "auto_reply" if confidence >= 0.8 else "escalate"
    elif attempts >= MAX_ATTEMPTS:
        outcome = "escalate"

    elapsed = (time.perf_counter() - start) * 1000
    return {
        "outcome": outcome,
        "qa": {
            "passed": passed,
            "score": round(score, 4),
            "ungrounded": ungrounded,
            "feedback": ungrounded,
            "attempts": attempts,
        },
        "trace": [
            {
                "node": "qa",
                "passed": passed,
                "score": round(score, 4),
                "attempts": attempts,
                "outcome": outcome,
                "latency_ms": round(elapsed, 2),
            }
        ],
    }


def route_after_qa(state: dict) -> str:
    qa = state.get("qa", {})
    if not qa.get("passed") and qa.get("attempts", MAX_ATTEMPTS) < MAX_ATTEMPTS:
        return "retry"
    if state.get("outcome") == "auto_reply":
        return "auto"
    return "escalate"
