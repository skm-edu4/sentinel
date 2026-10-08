from dataclasses import asdict, dataclass, field


@dataclass
class Classification:
    intent: dict[str, float]
    sentiment: dict[str, float]
    urgency: dict[str, float]
    auto_resolvable_confidence: float
    backend: str = field(default="")
    latency_ms: float = field(default=0.0)

    @property
    def top_intent(self) -> str:
        return max(self.intent, key=self.intent.get)

    @property
    def top_sentiment(self) -> str:
        return max(self.sentiment, key=self.sentiment.get)

    @property
    def top_urgency(self) -> str:
        return max(self.urgency, key=self.urgency.get)

    @property
    def intent_confidence(self) -> float:
        return self.intent[self.top_intent]

    def to_dict(self) -> dict:
        return {
            "intent": dict(sorted(self.intent.items(), key=lambda kv: -kv[1])),
            "sentiment": dict(sorted(self.sentiment.items(), key=lambda kv: -kv[1])),
            "urgency": dict(sorted(self.urgency.items(), key=lambda kv: -kv[1])),
            "auto_resolvable_confidence": round(self.auto_resolvable_confidence, 4),
        }

    def with_meta(self, backend: str, latency_ms: float) -> "Classification":
        return Classification(
            intent=self.intent,
            sentiment=self.sentiment,
            urgency=self.urgency,
            auto_resolvable_confidence=self.auto_resolvable_confidence,
            backend=backend,
            latency_ms=latency_ms,
        )

    def full_dict(self) -> dict:
        d = asdict(self)
        d.update({"top_intent": self.top_intent, "top_sentiment": self.top_sentiment, "top_urgency": self.top_urgency})
        return d
