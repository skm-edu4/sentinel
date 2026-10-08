COST_TABLE = {
    "classify_jev": 0.00010,
    "classify_llm": 0.00200,
    "classify_prototype": 0.00001,
    "llm_input_per_1k_tokens": 0.00015,
    "llm_output_per_1k_tokens": 0.00060,
    "template_llm": 0.0,
    "rag_call": 0.00002,
    "mcp_call": 0.00001,
}

MCP_CALLS_PER_ESCALATION = 3


def _tokens(chars: int) -> int:
    return max(1, chars // 4)


def classify_cost(backend_name: str) -> float:
    if backend_name == "jev-system-one":
        return COST_TABLE["classify_jev"]
    if backend_name.startswith("llm"):
        return COST_TABLE["classify_llm"]
    return COST_TABLE["classify_prototype"]


def estimate_ticket_cost(
    backend_name: str,
    trace: list[dict],
    draft_chars: int = 0,
    llm_name: str = "template-extractive",
) -> float:
    cost = classify_cost(backend_name)
    for step in trace:
        node = step.get("node")
        if node == "research":
            cost += COST_TABLE["rag_call"]
        elif node == "resolution":
            if llm_name == "template-extractive":
                cost += COST_TABLE["template_llm"]
            else:
                cost += (
                    _tokens(draft_chars) * COST_TABLE["llm_input_per_1k_tokens"]
                    + _tokens(draft_chars) * COST_TABLE["llm_output_per_1k_tokens"]
                ) / 1000
        elif node == "escalation":
            cost += MCP_CALLS_PER_ESCALATION * COST_TABLE["mcp_call"]
        elif node == "action" and step.get("action") not in {None, "none"}:
            cost += COST_TABLE["mcp_call"]
    return cost
