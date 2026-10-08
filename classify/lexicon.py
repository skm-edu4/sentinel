LEXICON_WEIGHT = 0.5

LEXICONS = {
    "sentiment": {
        "angry": [
            "unacceptable",
            "ridiculous",
            "incompetent",
            "furious",
            "lawsuit",
            "legal action",
            "dispute",
            "chargeback",
            "cancel my",
            "cancel this",
            "speak to a manager",
            "worst",
            "done with",
            "never again",
            "scammer",
            "fraud",
            "damn",
        ],
        "frustrated": [
            "again",
            "still",
            "third time",
            "fourth time",
            "waiting",
            "no reply",
            "no response",
            "tired of",
            "frustrated",
            "disappointed",
            "wasted",
            "fed up",
            "chasing",
            "follow up",
            "days and",
        ],
        "happy": [
            "thank",
            "great",
            "love",
            "appreciate",
            "awesome",
            "perfect",
            "helpful",
            "kudos",
            "brilliant",
        ],
        "neutral": ["?"],
    },
    "urgency": {
        "low": [
            "no rush",
            "whenever you get a chance",
            "not urgent",
            "whenever possible",
            "just wondering",
        ],
        "normal": [],
        "high": [
            "today",
            "deadline",
            "asap",
            "urgent",
            "immediately",
            "blocked",
            "cannot access",
            "cannot work",
            "tomorrow",
            "launch",
            "payroll",
            "overdue",
            "failing",
        ],
        "critical": [
            "outage",
            "down",
            "locked out",
            "data loss",
            "breach",
            "leaking",
            "everyone on our team",
            "production",
            "security",
            "hacked",
        ],
    },
    "resolution": {
        "answer": [
            "how long",
            "how do",
            "what is",
            "what are",
            "where can",
            "does ",
            "do you offer",
            "policy",
            "window",
            "eligib",
            "what does",
            "is there a",
            "can i ",
        ],
        "action": [
            "change my",
            "change our",
            "change the",
            "update my",
            "update our",
            "update the",
            "merge",
            "delete my",
            "delete our",
            "cancel my",
            "terminate",
            "compensate",
            "chargeback",
            "partnership",
            "refund my",
            "where is my refund",
            "refund for ord",
            "delivery address",
            "card on file",
            "add sso",
            "filed",
            "investigate my",
            "waive",
            "reimburse",
        ],
    },
}

NO_CUE_DEFAULT = {"urgency": "normal"}


def lexicon_distribution(text: str, axis: str) -> dict[str, float]:
    lowered = text.lower()
    scores = {
        label: float(sum(1 for cue in cues if cue in lowered))
        for label, cues in LEXICONS[axis].items()
    }
    total = sum(scores.values())
    default = NO_CUE_DEFAULT.get(axis)
    if total <= 0:
        if default is not None:
            n = len(scores)
            return {label: (1.0 if label == default else 0.0) for label in scores}
        n = len(scores)
        return {label: 1.0 / n for label in scores}
    return {label: value / total for label, value in scores.items()}
