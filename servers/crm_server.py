from typing import TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from db import ensure_seeded, fetch_all, fetch_one

ensure_seeded()

server = MCPServer(
    name="sentinel-crm",
    description="Customer history lookup for the Sentinel support system.",
    instructions=(
        "Read-only customer context derived from support ticket history: "
        "previous tickets, activity window, and a support-tier flag derived "
        "from contact frequency. Never returns payment or authentication data."
    ),
)


class CustomerHistory(TypedDict):
    customer_id: str
    ticket_count: int
    open_tickets: int
    first_contact_at: str
    last_contact_at: str
    support_tier: str
    prior_tickets: list[dict]


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_customer(customer_id: str) -> CustomerHistory:
    rows = fetch_all(
        "SELECT ticket_id, created_at, subject, status FROM tickets "
        "WHERE customer_id = ? ORDER BY created_at DESC",
        (customer_id,),
    )
    if not rows:
        raise ToolError(f"customer not found: {customer_id}")
    count = len(rows)
    tier = "gold" if count >= 3 else "silver" if count == 2 else "standard"
    return CustomerHistory(
        customer_id=customer_id,
        ticket_count=count,
        open_tickets=sum(1 for row in rows if row["status"] != "resolved"),
        first_contact_at=rows[-1]["created_at"],
        last_contact_at=rows[0]["created_at"],
        support_tier=tier,
        prior_tickets=[
            {
                "ticket_id": row["ticket_id"],
                "subject": row["subject"],
                "status": row["status"],
            }
            for row in rows[:5]
        ],
    )


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_customer_for_ticket(ticket_id: str) -> CustomerHistory:
    row = fetch_one("SELECT customer_id FROM tickets WHERE ticket_id = ?", (ticket_id,))
    if row is None:
        raise ToolError(f"ticket not found: {ticket_id}")
    return get_customer(row["customer_id"])


@server.resource(
    "customer://{customer_id}", name="Customer history as JSON", mime_type="application/json"
)
def customer_resource(customer_id: str) -> str:
    import json

    return json.dumps(get_customer(customer_id), indent=2)


if __name__ == "__main__":
    server.run(transport="stdio")
