from agents.qa import score_faithfulness


def faithfulness(answer: str, context_chunks: list[dict]) -> float:
    score, _ = score_faithfulness(answer, context_chunks)
    return score


def answer_relevance(answer: str, question: str) -> float:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    ef = DefaultEmbeddingFunction()
    answer_vec, question_vec = ef([answer, question])
    dot = sum(float(a) * float(q) for a, q in zip(answer_vec, question_vec))
    norm_a = sum(float(a) * float(a) for a in answer_vec) ** 0.5
    norm_q = sum(float(q) * float(q) for q in question_vec) ** 0.5
    if norm_a == 0 or norm_q == 0:
        return 0.0
    return float(dot / (norm_a * norm_q))


def context_precision(retrieved: list[str], relevant: list[str], k: int) -> float:
    gold = set(relevant)
    hits = 0
    precision_sum = 0.0
    for rank, doc in enumerate(retrieved[:k], 1):
        if doc in gold:
            hits += 1
            precision_sum += hits / rank
    denominator = min(len(gold), k)
    if denominator == 0:
        return 0.0
    return precision_sum / denominator


def context_recall(retrieved: list[str], relevant: list[str], k: int) -> float:
    gold = set(relevant)
    if not gold:
        return 1.0
    hits = len({doc for doc in retrieved[:k]} & gold)
    return hits / len(gold)
