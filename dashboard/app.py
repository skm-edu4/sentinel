import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
REPORTS_DIR = BASE_DIR / "reports"
TICKETS = [json.loads(line) for line in open(BASE_DIR / "data" / "tickets.jsonl")]
TICKET_BY_ID = {row["ticket_id"]: row for row in TICKETS}

st.set_page_config(page_title="Sentinel", page_icon="🛡️", layout="wide")


def _read_json(name: str):
    path = REPORTS_DIR / name
    if path.exists():
        return json.loads(path.read_text())
    return None


def _read_jsonl(name: str) -> list[dict]:
    path = REPORTS_DIR / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


@st.cache_data
def load_reports() -> dict:
    return {
        "summary": _read_json("summary.json"),
        "traces": _read_jsonl("traces.jsonl"),
        "rag_eval": _read_json("rag_eval.json"),
        "ab": _read_json("ab_routing.json"),
        "classify": _read_json("classify_eval.json"),
    }


def latency_table(summary: dict) -> pd.DataFrame:
    rows = []
    for layer, stats in summary["layer_latency_ms"].items():
        rows.append(
            {
                "layer": layer,
                "p50 ms": round(stats["p50"], 1),
                "p95 ms": round(stats["p95"], 1),
                "max ms": round(stats["max"], 1),
            }
        )
    return pd.DataFrame(rows)


def trace_rows(record: dict) -> pd.DataFrame:
    rows = []
    for step in record["trace"]:
        details = {
            key: value
            for key, value in step.items()
            if key not in {"node", "latency_ms"}
        }
        rows.append(
            {
                "node": step["node"],
                "latency ms": step["latency_ms"],
                "details": json.dumps(details) if details else "",
            }
        )
    return pd.DataFrame(rows)


def policy_table(ab: dict) -> pd.DataFrame:
    rows = []
    for name, scores in ab["policies"].items():
        rows.append(
            {
                "policy": name,
                "auto rate": f"{scores['auto_rate']:.0%}",
                "unsafe rate": f"{scores['unsafe_rate']:.0%}",
                "routine capture": f"{scores['routine_capture']:.0%}",
                "human load": scores["human_load"],
                "mean e2e ms": round(scores["mean_e2e_ms"]),
                "mean cost $": f"{scores['mean_cost_usd']:.5f}",
            }
        )
    return pd.DataFrame(rows)


def retrieval_table(rag: dict) -> pd.DataFrame:
    rows = []
    for row in rag["retrieval"]:
        rows.append(
            {
                "system": row["system"],
                "context precision": round(row["context_precision"], 3),
                "context recall": round(row["context_recall"], 3),
            }
        )
    return pd.DataFrame(rows)


def run_live(ticket_id: str) -> dict:
    from agents import run_ticket
    from classify import classify_ticket

    row = TICKET_BY_ID[ticket_id]
    classification = classify_ticket(row["subject"], row["body"])
    final = run_ticket(
        ticket_id, row["subject"], row["body"], classification.to_dict()
    )
    return {"classification": classification, "final": final}


reports = load_reports()

st.title("Sentinel — Multi-Agent Support Intelligence")
st.caption(
    "Layer 1 MCP tools · Layer 2 System-1 classification · Layer 3 hybrid RAG · "
    "Layer 4 LangGraph orchestration · Layer 5 actions & guardrails · Layer 6 evaluation"
)

tab_overview, tab_ticket, tab_eval, tab_arch = st.tabs(
    ["Overview", "Ticket Explorer", "Evaluations", "Architecture"]
)

with tab_overview:
    summary = reports["summary"]
    if summary is None:
        st.info("No reports yet — run `python scripts/trace_report.py` first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tickets traced", summary["n"])
        col2.metric("Auto-resolved", f"{summary['auto_rate']:.0%}")
        col3.metric("Mean cost / ticket", f"${summary['cost_usd']['mean']:.5f}")
        col4.metric(
            "E2E latency p50",
            f"{summary['layer_latency_ms']['e2e_ms']['p50']:.0f} ms",
        )
        st.subheader("Latency by layer")
        st.dataframe(latency_table(summary), width="stretch", hide_index=True)
        st.subheader("Outcomes")
        st.json(summary["outcome_counts"])
        st.caption(
            "Escalation latency is dominated by MCP spawn-per-call (~700ms × 3 "
            "calls); production fix is persistent sessions per server."
        )

with tab_ticket:
    if not reports["traces"]:
        st.info("No traces yet — run `python scripts/trace_report.py` first.")
    else:
        ids = [record["ticket_id"] for record in reports["traces"]]
        selected = st.selectbox("Traced ticket", ids)
        record = next(r for r in reports["traces"] if r["ticket_id"] == selected)
        ticket = TICKET_BY_ID[selected]

        st.markdown(f"**{ticket['subject']}**")
        st.caption(ticket["body"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Outcome", record["outcome"])
        col2.metric("Route", record["route"])
        col3.metric("Confidence", f"{record['confidence']:.3f}")
        col4.metric("QA score", record["qa_score"])
        st.subheader("Trace")
        st.dataframe(trace_rows(record), width="stretch", hide_index=True)

        with st.expander("Handoff / action details"):
            st.json(
                {
                    "action": record["action"],
                    "escalation_id": record["escalation_id"],
                    "attempts": record["attempts"],
                }
            )

        if st.button("Run live", key=f"live_{selected}"):
            with st.spinner("Classifying and running the agent graph..."):
                live = run_live(selected)
            st.subheader("Live run")
            st.json(live["classification"].to_dict())
            st.dataframe(
                trace_rows(live["final"]), width="stretch", hide_index=True
            )
            if live["final"].get("draft"):
                st.text_area(
                    "Draft", live["final"]["draft"], height=200, disabled=True
                )
            if live["final"].get("handoff"):
                st.json(live["final"]["handoff"].get("reasons", []))

with tab_eval:
    rag = reports["rag_eval"]
    if rag is None:
        st.info("No rag_eval report — run `python scripts/rag_eval.py` first.")
    else:
        st.subheader("Retrieval A/B (qrels, doc-level)")
        st.dataframe(retrieval_table(rag), width="stretch", hide_index=True)
        col1, col2 = st.columns(2)
        col1.metric("Faithfulness", f"{rag['generation']['faithfulness']:.3f}")
        col2.metric(
            "Answer relevance", f"{rag['generation']['answer_relevance']:.3f}"
        )
        if rag["divergences"]:
            st.subheader("Dense vs hybrid divergences")
            for item in rag["divergences"]:
                st.write(
                    f"`{item['id']}` {item['query']} → dense {item['dense']} | "
                    f"hybrid {item['hybrid']}"
                )

    ab = reports["ab"]
    if ab is None:
        st.info("No ab_routing report — run `python scripts/ab_routing.py` first.")
    else:
        st.subheader("Routing policy A/B")
        st.dataframe(policy_table(ab), width="stretch", hide_index=True)

    classify = reports["classify"]
    if classify:
        st.subheader("Classification backends")
        st.dataframe(pd.DataFrame(classify["backends"]), width="stretch", hide_index=True)

with tab_arch:
    st.markdown(
        """
**6-layer production architecture**

```
L1  MCP tools      ticket · knowledge base · CRM · action executor
L2  System-1 route intent / sentiment / urgency / auto-confidence (<50ms class target)
L3  Hybrid RAG     dense + BM25 → RRF fusion → cross-encoder rerank (+ query decomposition)
L4  LangGraph      supervisor → research → resolution → QA ⇄ retry → escalation
L5  Actions        issue_refund (>$50 human approval) · reset_password · create_escalation
L6  Evaluation     faithfulness / relevance / context P+R · traces · cost · policy A/B
```

**Routing guardrails**
- Auto-reply only when QA grounding passes **and** auto-resolvable confidence ≥ 0.8
- Angry sentiment or critical urgency → direct priority escalation
- Refunds > $50 rejected server-side by the MCP action executor → human approval
- QA failure loops back to Research with ungrounded sentences as feedback (max 2 attempts)

**Interview one-liner:** *cascading architecture — cheap System-1 routing before
expensive reasoning, hybrid retrieval with measured (not assumed) gains, and a
state-machine agent graph with cyclic QA feedback and server-enforced financial
guardrails.*
"""
    )
