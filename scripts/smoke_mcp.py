import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE = Path(__file__).resolve().parent.parent


def server_params(script: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[str(BASE / "servers" / script)],
        cwd=str(BASE),
    )


def payload(result) -> dict:
    return result.structured_content or {}


async def test_ticket_server() -> None:
    async with stdio_client(server_params("ticket_server.py")) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print("tools:", [t.name for t in tools])
            resources = (await session.list_resources()).resources
            templates = (await session.list_resource_templates()).resource_templates
            prompts = (await session.list_prompts()).prompts
            print("resources:", [str(r.uri) for r in resources])
            print("templates:", [t.uri_template for t in templates])
            print("prompts:", [p.name for p in prompts])

            res = await session.call_tool("list_tickets", {"status": "open", "limit": 3})
            open_tickets = payload(res)["result"]
            print(f"list_tickets(open): {len(open_tickets)} rows, first={open_tickets[0]['ticket_id']}")

            res = await session.call_tool("get_ticket", {"ticket_id": "T-0001"})
            ticket = payload(res)
            print("get_ticket(T-0001):", ticket["subject"], "| status:", ticket["status"])

            res = await session.call_tool("search_tickets", {"query": "error 4032"})
            hits = payload(res)["result"]
            print("search('error 4032'):", [(h["ticket_id"], h["bm25_score"]) for h in hits])

            res = await session.call_tool("search_tickets", {"query": "ORD-7788"})
            hits = payload(res)["result"]
            print("search('ORD-7788'):", [h["ticket_id"] for h in hits])

            res = await session.call_tool(
                "update_ticket_status", {"ticket_id": "T-0001", "status": "pending"}
            )
            print("update -> pending:", payload(res))
            res = await session.call_tool(
                "update_ticket_status", {"ticket_id": "T-0001", "status": "resolved"}
            )
            print("restore -> resolved:", payload(res))

            res = await session.call_tool(
                "update_ticket_status", {"ticket_id": "T-0001", "status": "bogus"}
            )
            print("invalid status rejected:", res.is_error)

            res = await session.read_resource("ticket://T-0001")
            data = json.loads(res.contents[0].text)
            print("resource ticket://T-0001:", data["ticket_id"], data["status"])

            res = await session.read_resource("queue://summary")
            print("resource queue://summary:", json.loads(res.contents[0].text)["by_status"])

            res = await session.get_prompt("triage_guidance", {"ticket_id": "T-0009"})
            text = res.messages[0].content.text
            print("prompt triage_guidance:", "T-0009" in text, "| mentions crash:", "crash" in text)


async def test_kb_server() -> None:
    async with stdio_client(server_params("kb_server.py")) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print("tools:", [t.name for t in tools])
            prompts = (await session.list_prompts()).prompts
            print("prompts:", [p.name for p in prompts])

            res = await session.call_tool("list_documents", {})
            docs = payload(res)["result"]
            print(f"list_documents: {len(docs)} docs")

            res = await session.call_tool("read_document", {"name": "refund-policy.md"})
            text = res.content[0].text
            print("read_document(refund-policy):", "30 days" in text, "| $50 rule:", "$50" in text)

            res = await session.call_tool("find_documents", {"query": "refund window days"})
            matches = payload(res)["result"]
            print("find('refund window days'):", [(m["name"], m["score"]) for m in matches])

            res = await session.call_tool(
                "read_document", {"name": "../../etc/passwd"}
            )
            print("path traversal rejected:", res.is_error)

            res = await session.read_resource("kb://data-privacy-gdpr.md")
            print("resource kb://data-privacy-gdpr.md:", "do not sell your data" in res.contents[0].text.lower())

            res = await session.get_prompt("grounded_reply_guidelines", {"tone": "direct"})
            text = res.messages[0].content.text
            print("prompt grounded_reply_guidelines:", "never invent" in text.lower())


async def test_action_server() -> None:
    async with stdio_client(server_params("action_server.py")) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print("tools:", [t.name for t in tools])

            res = await session.call_tool(
                "issue_refund",
                {"ticket_id": "T-0007", "amount_cents": 7500, "reason": "damaged mirror"},
            )
            over = payload(res)
            print("refund $75 ->", over["status"])
            assert over["status"] == "requires_human_approval"

            res = await session.call_tool(
                "issue_refund",
                {"ticket_id": "T-0007", "amount_cents": 4200, "reason": "goodwill"},
            )
            under = payload(res)
            print("refund $42 ->", under["status"])
            assert under["status"] == "executed"

            res = await session.call_tool(
                "issue_refund",
                {"ticket_id": "T-0007", "amount_cents": -5, "reason": "bad"},
            )
            print("negative amount rejected:", res.is_error)

            res = await session.call_tool(
                "issue_refund",
                {"ticket_id": "T-9999", "amount_cents": 100, "reason": "ghost"},
            )
            print("unknown ticket rejected:", res.is_error)

            res = await session.call_tool(
                "create_escalation",
                {
                    "ticket_id": "T-0007",
                    "priority": "high",
                    "reasons": ["refund over limit"],
                    "summary": "smoke test",
                },
            )
            created = payload(res)
            print("create_escalation ->", created["escalation_id"])
            assert created["status"] == "open"

            res = await session.call_tool(
                "create_escalation",
                {"ticket_id": "T-0007", "priority": "urgent", "reasons": ["x"]},
            )
            print("invalid priority rejected:", res.is_error)

            res = await session.call_tool("list_actions", {"ticket_id": "T-0007"})
            rows = payload(res)["result"]
            print(f"list_actions(T-0007): {len(rows)} rows")
            assert len(rows) >= 2

            res = await session.call_tool("list_escalations", {"ticket_id": "T-0007"})
            escs = payload(res)["result"]
            print(f"list_escalations(T-0007): {len(escs)} rows")
            assert len(escs) >= 1

            res = await session.read_resource("actions://audit")
            audit = json.loads(res.contents[0].text)
            print("resource actions://audit:", len(audit), "entries")
            assert len(audit) >= 2


async def test_crm_server() -> None:
    async with stdio_client(server_params("crm_server.py")) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print("tools:", [t.name for t in tools])

            res = await session.call_tool(
                "get_customer_for_ticket", {"ticket_id": "T-0001"}
            )
            customer = payload(res)
            print(
                f"customer {customer['customer_id']}: {customer['ticket_count']} tickets, "
                f"tier={customer['support_tier']}"
            )
            assert customer["ticket_count"] >= 1

            res = await session.call_tool("get_customer", {"customer_id": "C-9999"})
            print("unknown customer rejected:", res.is_error)

            res = await session.read_resource(
                "customer://" + customer["customer_id"]
            )
            data = json.loads(res.contents[0].text)
            print("resource customer://", data["customer_id"])
            assert data["customer_id"] == customer["customer_id"]


async def main() -> None:
    print("=== ticket_server ===")
    await test_ticket_server()
    print("\n=== kb_server ===")
    await test_kb_server()
    print("\n=== action_server ===")
    await test_action_server()
    print("\n=== crm_server ===")
    await test_crm_server()
    print("\nALL MCP SMOKE CHECKS PASSED", flush=True)
    os._exit(0)


asyncio.run(main())
