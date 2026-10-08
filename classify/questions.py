INTENT_CRITERIA = {
    "refund": {
        "what": "Customer wants money back, or asks about refund eligibility, amount or timing",
        "examples": [
            "I want my money back",
            "refund the duplicate charge",
            "how long do refunds take to arrive?",
        ],
        "not_for": "Billing questions where no money is returned to the customer",
    },
    "billing": {
        "what": (
            "Invoices, receipts, payment methods, coupons, plan prices or billing-cycle "
            "questions where the customer wants information or a correction, not cash back"
        ),
        "examples": [
            "why is my invoice higher than last month",
            "update my card",
            "what does the pro plan cost",
        ],
        "not_for": "Requests to return money to the customer",
    },
    "bug": {
        "what": "Something in the product is broken, crashing, erroring or not behaving as documented",
        "examples": [
            "the app crashes when I click save",
            "error 4032 on login",
            "the export button does nothing",
        ],
        "not_for": "Asks for new capabilities that do not exist yet",
    },
    "shipping": {
        "what": "Delivery status, tracking, delays, lost, stolen or wrong packages",
        "examples": [
            "where is order ORD-1234",
            "tracking has been stuck for a week",
            "the package says delivered but nothing arrived",
        ],
        "not_for": "Questions about charges or invoices",
    },
    "account": {
        "what": "Login, password, account access, profile settings, security, privacy or personal data requests",
        "examples": [
            "I cannot log in",
            "reset my password",
            "delete my personal data",
            "my two-factor code never arrives",
        ],
        "not_for": "Payment and invoice problems from customers who can log in",
    },
    "feature_request": {
        "what": "Customer asks for a capability, improvement or integration that does not exist yet",
        "examples": [
            "please add a dark mode",
            "it would be great if this supported SSO",
            "can you build an integration with X",
        ],
        "not_for": "Reports that an existing feature is broken",
    },
    "other": {
        "what": (
            "Greetings, sales or partnership questions, compliments without a request, "
            "or anything that fits none of the other categories"
        ),
        "examples": [
            "hello",
            "do you offer nonprofit discounts",
            "love your product",
        ],
        "not_for": "Any concrete support request covered by the other categories",
    },
}

SENTIMENT_CRITERIA = {
    "angry": {
        "what": (
            "Furious or demanding: forceful demands, ultimatums, all-caps emphasis, "
            "threats to leave or escalate, zero patience"
        ),
        "examples": [
            "I want this fixed NOW",
            "I want my money back immediately",
            "I am done shopping with you",
            "This is UNACCEPTABLE",
        ],
    },
    "frustrated": {
        "what": (
            "Disappointed or annoyed but civil: reports a real problem, delay, defect "
            "or wrong outcome in plain words, without abuse or ultimatums"
        ),
        "examples": [
            "checkout keeps failing with an error code",
            "the dashboard loads an error and nothing works",
            "my order has not moved in a week",
            "I was sent the wrong product",
        ],
    },
    "neutral": {
        "what": (
            "No complaint: asks a how-to, policy or account question in plain language "
            "without reporting a problem"
        ),
        "examples": [
            "How do I change my billing address?",
            "Where can I find my API keys?",
            "Do you offer annual billing?",
        ],
    },
    "happy": {
        "what": "Positive, thankful or praising",
        "examples": [
            "Thanks for the quick fix",
            "Love the new dashboard",
            "Appreciate your help",
        ],
    },
}

URGENCY_LABELS = ["low", "normal", "high", "critical"]

URGENCY_LEVELS = [
    {
        "what": "No time pressure; the customer expects to wait days for an answer",
        "examples": [
            "no rush",
            "whenever you get a chance",
            "just wondering if you offer annual billing",
        ],
    },
    {
        "what": "Ordinary support timeframe; no deadline and no active block",
        "examples": [
            "question about my plan",
            "how do I invite a teammate",
            "when you have a moment",
        ],
    },
    {
        "what": (
            "Time-sensitive; the customer is blocked from working, money moved incorrectly, "
            "or a deadline is near"
        ),
        "examples": [
            "I need access today to finish payroll",
            "charged twice and my billing cycle closes tomorrow",
            "our launch is tomorrow",
        ],
    },
    {
        "what": (
            "Happening now with severe impact; outage, security breach, data loss, "
            "legal or safety threat"
        ),
        "examples": [
            "everyone on our team is locked out",
            "we are leaking customer data",
            "the production server is down",
        ],
    },
]

INTENT_LABELS = list(INTENT_CRITERIA)
SENTIMENT_LABELS = list(SENTIMENT_CRITERIA)

RESOLUTION_CRITERIA = {
    "answer": {
        "what": "Answerable by quoting general documentation; no change to this customer's account",
        "examples": [
            "how long do refunds take",
            "what is your return policy",
            "where can I find the API keys",
        ],
    },
    "action": {
        "what": "Requires performing an action on this specific customer's account or granting an exception",
        "examples": [
            "update the card on file",
            "merge my two accounts",
            "delete all my data",
            "I demand compensation for the delay",
        ],
    },
}
RESOLUTION_LABELS = list(RESOLUTION_CRITERIA)


def prototype_texts(label: str, spec) -> list[str]:
    texts = [f"{label}: {spec}"] if isinstance(spec, str) else [f"{label}: {spec['what']}"]
    if isinstance(spec, dict):
        texts.extend(spec.get("examples", []))
    return texts


def axis_prototypes(criteria: dict | list, labels: list[str]) -> dict[str, list[str]]:
    specs = list(criteria.values()) if isinstance(criteria, dict) else criteria
    return {
        label: prototype_texts(label, spec)
        for label, spec in zip(labels, specs)
    }


INTENT_PROTOTYPES = axis_prototypes(INTENT_CRITERIA, INTENT_LABELS)
SENTIMENT_PROTOTYPES = axis_prototypes(SENTIMENT_CRITERIA, SENTIMENT_LABELS)
URGENCY_PROTOTYPES = axis_prototypes(URGENCY_LEVELS, URGENCY_LABELS)
RESOLUTION_PROTOTYPES = axis_prototypes(RESOLUTION_CRITERIA, RESOLUTION_LABELS)

INTENT_PRIOR = {
    "refund": 0.95,
    "shipping": 0.9,
    "account": 0.9,
    "billing": 0.85,
    "bug": 0.6,
    "other": 0.55,
    "feature_request": 0.5,
}
SENTIMENT_FACTOR = {"angry": 0.35, "frustrated": 0.65, "neutral": 1.0, "happy": 1.0}
URGENCY_FACTOR = {"critical": 0.25, "high": 0.6, "normal": 1.0, "low": 1.0}


def build_jev_questions() -> dict:
    return {
        "intent": {
            "type": "choice",
            "instructions": "What kind of support request is this?",
            "criteria": INTENT_CRITERIA,
        },
        "sentiment": {
            "type": "choice",
            "instructions": "What is the customer's emotional tone?",
            "criteria": SENTIMENT_CRITERIA,
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": URGENCY_LEVELS,
        },
        "auto_resolvable": {
            "type": "noul",
            "instructions": (
                "Can this ticket be fully resolved by an AI agent following the "
                "knowledge base, without human intervention, manual verification "
                "or an account-specific action?"
            ),
        },
    }
