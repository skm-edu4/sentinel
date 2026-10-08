import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TICKETS_JSONL = BASE_DIR / "data" / "tickets.jsonl"
TICKETS_DB = BASE_DIR / "data" / "tickets.db"

VALID_STATUSES = ("open", "pending", "resolved")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    channel TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets (customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets (status);
CREATE VIRTUAL TABLE IF NOT EXISTS tickets_fts USING fts5(
    ticket_id UNINDEXED,
    subject,
    body
);
CREATE TABLE IF NOT EXISTS actions (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    amount_cents INTEGER,
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    priority TEXT NOT NULL,
    reasons TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);
"""

SELECT_SUMMARY = (
    "SELECT ticket_id, customer_id, created_at, channel, subject, status "
    "FROM tickets"
)


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(TICKETS_DB)
    con.row_factory = sqlite3.Row
    return con


def fetch_all(sql: str, params: tuple = ()) -> list[dict]:
    con = connect()
    try:
        return [dict(row) for row in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def fetch_one(sql: str, params: tuple = ()) -> dict | None:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> int:
    con = connect()
    try:
        cursor = con.execute(sql, params)
        con.commit()
        return cursor.rowcount
    finally:
        con.close()


def _initial_status(created_at: str) -> str:
    return "resolved" if created_at < "2026-09-20" else "open"


def ensure_seeded() -> int:
    con = connect()
    try:
        con.executescript(SCHEMA)
        count = con.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        if count:
            return count
        lines = [line for line in TICKETS_JSONL.read_text().splitlines() if line.strip()]
        for line in lines:
            ticket = json.loads(line)
            con.execute(
                "INSERT INTO tickets "
                "(ticket_id, customer_id, created_at, channel, subject, body, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ticket["ticket_id"],
                    ticket["customer_id"],
                    ticket["created_at"],
                    ticket["channel"],
                    ticket["subject"],
                    ticket["body"],
                    _initial_status(ticket["created_at"]),
                ),
            )
            con.execute(
                "INSERT INTO tickets_fts (ticket_id, subject, body) VALUES (?, ?, ?)",
                (ticket["ticket_id"], ticket["subject"], ticket["body"]),
            )
        con.commit()
        return len(lines)
    finally:
        con.close()


if __name__ == "__main__":
    import sys

    if "--force" in sys.argv:
        con = connect()
        try:
            con.executescript("DROP TABLE IF EXISTS tickets_fts; DROP TABLE IF EXISTS tickets;")
        finally:
            con.close()
    total = ensure_seeded()
    print(f"tickets.db ready: {total} tickets")
