import json
import re
from typing import TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from db import SELECT_SUMMARY, VALID_STATUSES, ensure_seeded, execute, fetch_all, fetch_one

ensure_seeded()

server = MCPServer(
    name="sentinel-tickets",
    description="Ticket database access for the Sentinel support system.",
    instructions=(
        "Read and update customer support tickets stored in the local SQLite "
        "database. Read tools are side-effect free; update_ticket_status is the "
        "only write tool and accepts open, pending, or resolved."
    ),
)


class TicketSummary(TypedDict):
    ticket_id: str
    customer_id: str
    created_at: str
    channel: str
    subject: str
    status: str


class TicketDetail(TypedDict):
    ticket_id: str
    customer_id: str
    created_at: str
    channel: str
    subject: str
    body: str
    status: str


class TicketSearchHit(TypedDict):
    ticket_id: str
    subject: str
    status: str
    snippet: str
    bm25_score: float


class TicketStatusUpdate(TypedDict):
    ticket_id: str
    previous_status: str
    status: str
    changed: bool


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_tickets(status: str | None = None, limit: int = 20, offset: int = 0) -> list[TicketSummary]:
    if status is not None and status not in VALID_STATUSES:
        raise ToolError(f"status must be one of {VALID_STATUSES}")
    if not 1 <= limit <= 100:
        raise ToolError("limit must be between 1 and 100")
    if offset < 0:
        raise ToolError("offset must be >= 0")
    sql = SELECT_SUMMARY
    params: list = []
    if status is not None:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    return fetch_all(sql, tuple(params))


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_ticket(ticket_id: str) -> TicketDetail:
    row = fetch_one("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
    if row is None:
        raise ToolError(f"ticket not found: {ticket_id}")
    return TicketDetail(**row)


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def search_tickets(query: str, limit: int = 10) -> list[TicketSearchHit]:
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    if not tokens:
        raise ToolError("query must contain at least one alphanumeric token")
    if not 1 <= limit <= 50:
        raise ToolError("limit must be between 1 and 50")
    match = " ".join(f'"{token}"' for token in tokens[:12])
    rows = fetch_all(
        """
        SELECT t.ticket_id, t.subject, t.status,
               snippet(tickets_fts, 2, '[', ']', '...', 12) AS snippet,
               bm25(tickets_fts) AS bm25_score
        FROM tickets_fts
        JOIN tickets AS t ON t.ticket_id = tickets_fts.ticket_id
        WHERE tickets_fts MATCH ?
        ORDER BY bm25(tickets_fts)
        LIMIT ?
        """,
        (match, limit),
    )
    return [TicketSearchHit(**row) for row in rows]


@server.tool()
def update_ticket_status(ticket_id: str, status: str) -> TicketStatusUpdate:
    if status not in VALID_STATUSES:
        raise ToolError(f"status must be one of {VALID_STATUSES}")
    row = fetch_one("SELECT status FROM tickets WHERE ticket_id = ?", (ticket_id,))
    if row is None:
        raise ToolError(f"ticket not found: {ticket_id}")
    previous = row["status"]
    if previous == status:
        return TicketStatusUpdate(
            ticket_id=ticket_id, previous_status=previous, status=status, changed=False
        )
    execute("UPDATE tickets SET status = ? WHERE ticket_id = ?", (status, ticket_id))
    return TicketStatusUpdate(
        ticket_id=ticket_id, previous_status=previous, status=status, changed=True
    )


@server.resource("ticket://{ticket_id}", name="Ticket as JSON", mime_type="application/json")
def ticket_resource(ticket_id: str) -> str:
    row = fetch_one("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
    if row is None:
        raise ToolError(f"ticket not found: {ticket_id}")
    return json.dumps(row, indent=2)


@server.resource("queue://summary", name="Queue summary", mime_type="application/json")
def queue_summary() -> str:
    by_status = fetch_all("SELECT status, COUNT(*) AS count FROM tickets GROUP BY status ORDER BY status")
    by_channel = fetch_all("SELECT channel, COUNT(*) AS count FROM tickets GROUP BY channel ORDER BY channel")
    oldest = fetch_one("SELECT MIN(created_at) AS oldest, MAX(created_at) AS newest FROM tickets")
    return json.dumps(
        {"by_status": by_status, "by_channel": by_channel, "range": oldest},
        indent=2,
    )


@server.prompt(
    name="triage_guidance",
    description="Structured triage instructions with the requested ticket embedded.",
)
def triage_guidance(ticket_id: str) -> str:
    row = fetch_one("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
    if row is None:
        raise ToolError(f"ticket not found: {ticket_id}")
    return (
        f"You are triaging support ticket {row['ticket_id']} "
        f"(channel: {row['channel']}, created: {row['created_at']}).\n\n"
        f"Subject: {row['subject']}\n"
        f"Body: {row['body']}\n\n"
        "Classify intent, sentiment, and urgency. Then decide whether the ticket "
        "can be auto-resolved from the knowledge base or must be escalated to a "
        "human. Respond with a JSON object containing intent, sentiment, urgency, "
        "auto_resolvable (boolean), and a one-sentence rationale."
    )


if __name__ == "__main__":
    server.run(transport="stdio")
