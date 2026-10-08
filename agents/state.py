import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    ticket_id: str
    subject: str
    body: str
    classification: dict
    route: str
    override_route: str
    research: dict
    draft: str
    qa: dict
    action: dict | None
    attempts: int
    outcome: str
    handoff: dict
    trace: Annotated[list[dict], operator.add]
