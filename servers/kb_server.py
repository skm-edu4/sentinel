import json
import re
from pathlib import Path
from typing import TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "kb"
DOC_NAME = re.compile(r"^[a-z0-9-]+\.md$")

server = MCPServer(
    name="sentinel-knowledge-base",
    description="Access to Sentinel internal documentation.",
    instructions=(
        "Read-only access to the internal knowledge base. Documents are markdown "
        "policy and product docs. cite the document name when using its content."
    ),
)


class DocumentInfo(TypedDict):
    name: str
    title: str
    word_count: int


class DocumentMatch(TypedDict):
    name: str
    title: str
    score: int
    excerpt: str


def _doc_path(name: str) -> Path:
    if not DOC_NAME.match(name):
        raise ToolError(f"invalid document name: {name!r} (expected e.g. 'refund-policy.md')")
    path = (KB_DIR / name).resolve()
    if path.parent != KB_DIR.resolve():
        raise ToolError(f"invalid document name: {name!r}")
    if not path.is_file():
        raise ToolError(f"document not found: {name}")
    return path


def _title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_documents() -> list[DocumentInfo]:
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            DocumentInfo(
                name=path.name,
                title=_title(text),
                word_count=len(text.split()),
            )
        )
    return docs


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def read_document(name: str) -> str:
    return _doc_path(name).read_text(encoding="utf-8")


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def find_documents(query: str, limit: int = 3) -> list[DocumentMatch]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)]
    if not tokens:
        raise ToolError("query must contain at least one alphanumeric token")
    if not 1 <= limit <= 10:
        raise ToolError("limit must be between 1 and 10")

    matches: list[DocumentMatch] = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = _title(text)
        body_lower = text.lower()
        title_lower = title.lower()
        score = sum(body_lower.count(tok) for tok in tokens) + 3 * sum(
            title_lower.count(tok) for tok in tokens
        )
        if score == 0:
            continue
        excerpt = ""
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if any(tok in sentence.lower() for tok in tokens):
                excerpt = sentence.strip()[:200]
                break
        matches.append(
            DocumentMatch(name=path.name, title=title, score=score, excerpt=excerpt)
        )
    matches.sort(key=lambda m: -m["score"])
    return matches[:limit]


@server.resource("kb://{doc_name}", name="Knowledge base document", mime_type="text/markdown")
def kb_document(doc_name: str) -> str:
    return _doc_path(doc_name).read_text(encoding="utf-8")


@server.resource("index://documents", name="Document index", mime_type="application/json")
def document_index() -> str:
    return json.dumps(list_documents(), indent=2)


@server.prompt(
    name="grounded_reply_guidelines",
    description="Rules for drafting a customer reply grounded in retrieved documents.",
)
def grounded_reply_guidelines(tone: str) -> str:
    if tone not in ("empathetic", "neutral", "direct"):
        raise ToolError("tone must be one of: empathetic, neutral, direct")
    return (
        f"Draft a customer support reply in a {tone} tone, following these rules:\n"
        "1. Use only facts found in the retrieved knowledge base documents.\n"
        "2. Name the document each key fact comes from, e.g. (refund-policy).\n"
        "3. Never invent policies, dates, amounts, or timelines that are not in the documents.\n"
        "4. If the retrieved documents do not answer the question, say a human agent "
        "will follow up — do not guess.\n"
        "5. End with one clear next step for the customer."
    )


if __name__ == "__main__":
    server.run(transport="stdio")
