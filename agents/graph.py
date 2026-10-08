from langgraph.graph import END, START, StateGraph

from agents.actions import action_node, route_after_action
from agents.escalation import escalation_node
from agents.qa import qa_node, route_after_qa
from agents.research import research_node
from agents.resolution import resolution_node
from agents.state import AgentState
from agents.supervisor import route_after_supervisor, supervisor_node


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("resolution", resolution_node)
    graph.add_node("qa", qa_node)
    graph.add_node("action", action_node)
    graph.add_node("escalation", escalation_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"research": "research", "escalate": "escalation"},
    )
    graph.add_edge("research", "resolution")
    graph.add_edge("resolution", "qa")
    graph.add_conditional_edges(
        "qa",
        route_after_qa,
        {"retry": "research", "auto": "action", "escalate": "escalation"},
    )
    graph.add_conditional_edges(
        "action",
        route_after_action,
        {"proceed": END, "approve": "escalation"},
    )
    graph.add_edge("escalation", END)
    return graph.compile()


def run_ticket(
    ticket_id: str,
    subject: str,
    body: str,
    classification: dict,
    override_route: str = "",
) -> dict:
    app = _compiled()
    return app.invoke(
        {
            "ticket_id": ticket_id,
            "subject": subject,
            "body": body,
            "classification": classification,
            "override_route": override_route,
            "trace": [],
        }
    )


_app = None


def _compiled():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
