import time

from agents.llm import get_llm

SYSTEM_PROMPT = (
    "You are a customer support agent at Sentinel. Answer the customer's "
    "question using ONLY the provided documentation context. Be concise, "
    "acknowledge their situation, and cite the source document for each claim. "
    "Never invent policies, dates, or account details not present in the context."
)

TONE_HINTS = {
    "frustrated": "Acknowledge the frustration and apologize for the trouble.",
    "neutral": "Be direct and factual.",
    "happy": "Match their positive tone briefly.",
    "angry": "Apologize sincerely and offer escalation immediately.",
}


def _prompt(state: dict) -> str:
    cls = state["classification"]
    sentiment = next(iter(cls["sentiment"]))
    research = state.get("research") or {}
    context_lines = []
    for chunk in research.get("chunks", []):
        text = chunk["text"].replace("\n", " ").strip()
        context_lines.append(f"[[{chunk['chunk_id']}]] {text}")
    context = "\n".join(context_lines) if context_lines else "(no context retrieved)"
    return (
        f"TONE: {TONE_HINTS.get(sentiment, 'Be professional.')}\n"
        f"GOAL: {state['subject']}\n"
        f"CUSTOMER MESSAGE:\n{state['body']}\n"
        f"CONTEXT:\n{context}\n"
        f"END_OF_CONTEXT\n"
        f"Write the reply the support agent should send to the customer."
    )


def resolution_node(state: dict) -> dict:
    start = time.perf_counter()
    llm = get_llm()
    draft = llm.complete(SYSTEM_PROMPT, _prompt(state))
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "draft": draft,
        "trace": [
            {
                "node": "resolution",
                "llm": llm.name,
                "chars": len(draft),
                "latency_ms": round(elapsed, 2),
            }
        ],
    }
