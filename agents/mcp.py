import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE_DIR = Path(__file__).resolve().parent.parent


class McpCallError(RuntimeError):
    pass


async def _call(server_script: str, tool: str, args: dict, timeout: float) -> object:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(BASE_DIR / "servers" / server_script)],
        cwd=str(BASE_DIR),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            if result.is_error:
                messages = [c.text for c in result.content if hasattr(c, "text")]
                raise McpCallError(f"{server_script}.{tool}: {' '.join(messages)}")
            structured = result.structured_content
            if isinstance(structured, dict):
                if set(structured) == {"result"}:
                    return structured["result"]
                return structured
            text = result.content[0].text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text


def call_tool(server_script: str, tool: str, args: dict, timeout: float = 20.0) -> object:
    return asyncio.run(asyncio.wait_for(_call(server_script, tool, args, timeout), timeout))
