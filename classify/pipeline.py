import time

from classify.backends import get_default_backend
from classify.schema import Classification


def classify_ticket(subject: str, body: str, backend=None) -> Classification:
    backend = backend or get_default_backend()
    start = time.perf_counter()
    result = backend.classify(subject, body)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result.with_meta(backend=backend.name, latency_ms=elapsed_ms)
