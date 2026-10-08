import json
from datetime import datetime, timezone
from typing import TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from db import ensure_seeded, execute, fetch_all, fetch_one

ensure_seeded()

HUMAN_APPROVAL_LIMIT_CENTS = 5000

server = MCPServer(
    name="sentinel-actions",
    description="Action executor for the Sentinel support system.",
    instructions=(
        "Execute side-effecting customer actions: issue_refund, reset_password, "
        "and create_escalation. Every attempt is written to the audit ledger "
        "regardless of outcome. Refunds above $50 (5000 cents) are never "
        "executed here; they return requires_human_approval for a human agent "
        "to confirm."
    ),
)


class ActionResult(TypedDict):
    ticket_id: str
    action_type: str
    amount_cents: int | None
    status: str
    action_id: int | None
    detail: str


class EscalationCreated(TypedDict):
    escalation_id: int
    ticket_id: str
    priority: str
    status: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_ticket(ticket_id: str) -> None:
    row = fetch_one("SELECT ticket_id FROM tickets WHERE ticket_id = ?", (ticket_id,))
    if row is None:
        raise ToolError(f"ticket not found: {ticket_id}")


@server.tool()
def issue_refund(
    ticket_id: str, amount_cents: int, reason: str = "customer request"
) -> ActionResult:
    _require_ticket(ticket_id)
    if amount_cents <= 0:
        raise ToolError("amount_cents must be positive")
    if len(reason) > 500:
        raise ToolError("reason must be 500 characters or fewer")

    status = (
        "requires_human_approval"
        if amount_cents > HUMAN_APPROVAL_LIMIT_CENTS
        else "executed"
    )
    detail = (
        f"Refund of ${amount_cents / 100:.2f} exceeds the $50.00 auto-approval "
        "limit; a human agent must confirm."
        if status == "requires_human_approval"
        else f"Refund of ${amount_cents / 100:.2f} issued to the original payment method."
    )
    execute(
        "INSERT INTO actions (ticket_id, action_type, amount_cents, status, detail, created_at) "
        "VALUES (?, 'issue_refund', ?, ?, ?, ?)",
        (ticket_id, amount_cents, status, detail, _now()),
    )
    row = fetch_one("SELECT action_id FROM actions ORDER BY action_id DESC LIMIT 1")
    return ActionResult(
        ticket_id=ticket_id,
        action_type="issue_refund",
        amount_cents=amount_cents,
        status=status,
        action_id=row["action_id"],
        detail=detail,
    )


@server.tool()
def reset_password(email: str) -> ActionResult:
    if "@" not in email or len(email) > 254:
        raise ToolError("email must be a valid address")
    masked = email[:2] + "***" + email[email.index("@") :]
    execute(
        "INSERT INTO actions (ticket_id, action_type, amount_cents, status, detail, created_at) "
        "VALUES ('-', 'reset_password', NULL, 'executed', ?, ?)",
        (f"Password reset link sent to {masked}", _now()),
    )
    row = fetch_one("SELECT action_id FROM actions ORDER BY action_id DESC LIMIT 1")
    return ActionResult(
        ticket_id="-",
        action_type="reset_password",
        amount_cents=None,
        status="executed",
        action_id=row["action_id"],
        detail=f"Password reset link sent to {masked}",
    )


@server.tool()
def create_escalation(
    ticket_id: str, priority: str, reasons: list[str], summary: str = ""
) -> EscalationCreated:
    _require_ticket(ticket_id)
    if priority not in {"normal", "high"}:
        raise ToolError("priority must be 'normal' or 'high'")
    if not reasons:
        raise ToolError("reasons must contain at least one entry")
    execute(
        "INSERT INTO escalations (ticket_id, priority, reasons, summary, status, created_at) "
        "VALUES (?, ?, ?, ?, 'open', ?)",
        (ticket_id, priority, json.dumps(reasons), summary[:1000], _now()),
    )
    row = fetch_one("SELECT escalation_id FROM escalations ORDER BY escalation_id DESC LIMIT 1")
    return EscalationCreated(
        escalation_id=row["escalation_id"],
        ticket_id=ticket_id,
        priority=priority,
        status="open",
    )


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_actions(ticket_id: str, limit: int = 20) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ToolError("limit must be between 1 and 100")
    return fetch_all(
        "SELECT * FROM actions WHERE ticket_id = ? ORDER BY action_id DESC LIMIT ?",
        (ticket_id, limit),
    )


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_escalations(ticket_id: str, limit: int = 20) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ToolError("limit must be between 1 and 100")
    return fetch_all(
        "SELECT * FROM escalations WHERE ticket_id = ? ORDER BY escalation_id DESC LIMIT ?",
        (ticket_id, limit),
    )


@server.resource("actions://audit", name="Recent action audit log", mime_type="application/json")
def audit_resource() -> str:
    rows = fetch_all("SELECT * FROM actions ORDER BY action_id DESC LIMIT 20")
    return json.dumps(rows, indent=2)


if __name__ == "__main__":
    server.run(transport="stdio")
