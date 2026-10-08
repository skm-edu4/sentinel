# Sentinel — Production-Grade Multi-Agent Customer Intelligence & Autonomous Resolution Engine

An AI system that reads incoming customer complaints, classifies them in
milliseconds, searches internal knowledge with hybrid RAG, decides whether to
auto-resolve or escalate to a human, executes guarded actions through MCP, and
measures everything — all with typed outputs, cyclic QA feedback, and
server-enforced financial guardrails.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ L1  MCP TOOL LAYER                                          │
│     ticket server · knowledge base · CRM · action executor  │
├─────────────────────────────────────────────────────────────┤
│ L2  SYSTEM-1 CLASSIFICATION                                 │
│     intent / sentiment / urgency / auto-confidence          │
│     prototype-embedding stand-in + Jev backend (API-ready)  │
├─────────────────────────────────────────────────────────────┤
│ L3  HYBRID RAG                                              │
│     dense (MiniLM/Chroma) + BM25 → RRF fusion →             │
│     cross-encoder rerank · query decomposition              │
├─────────────────────────────────────────────────────────────┤
│ L4  LANGGRAPH ORCHESTRATION                                 │
│     supervisor → research → resolution → QA ⇄ retry         │
│                     └→ escalation (priority handoff)        │
├─────────────────────────────────────────────────────────────┤
│ L5  RESPONSE & ACTIONS                                      │
│     auto-reply (QA pass ∧ confidence ≥ 0.8)                 │
│     issue_refund (>$50 → server-side human approval)        │
│     reset_password · create_escalation (audited ledger)     │
├─────────────────────────────────────────────────────────────┤
│ L6  EVALUATION & OBSERVABILITY                              │
│     faithfulness · answer relevance · context P/R (qrels)   │
│     trace export (per-layer p50/p95) · cost estimates       │
│     routing policy A/B · retrieval A/B                      │
└─────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/smoke_mcp.py        # all 4 MCP servers
.venv/bin/python scripts/rag_demo.py         # hybrid RAG assertions
.venv/bin/python scripts/classify_eval.py    # classification metrics + Jev mock test
.venv/bin/python scripts/agent_demo.py       # graph demo + $50 guardrail scenarios
.venv/bin/python scripts/rag_eval.py         # RAGAS-style metrics + retrieval A/B
.venv/bin/python scripts/trace_report.py     # reports/traces.jsonl + summary.json
.venv/bin/python scripts/ab_routing.py       # routing policy A/B
.venv/bin/python scripts/dashboard_smoke.py  # dashboard render check

.venv/bin/streamlit run dashboard/app.py     # interactive dashboard
```

## Layout

```
servers/    MCP servers (ticket, kb, crm, action) + shared SQLite/FTS5 db
classify/   L2 classification: shared criteria, prototype + Jev backends, eval lexicons
rag/        L3 hybrid RAG engine (dense, BM25, RRF, cross-encoder, decomposition)
agents/     L4+L5 LangGraph state machine: supervisor, research, resolution,
            qa, action, escalation + MCP client + LLM abstraction
eval/       L6 qrels, metrics, cost model
dashboard/  Streamlit app (overview, ticket explorer, evaluations, architecture)
scripts/    smoke tests, evals, report generators
data/       50 labeled tickets (labels are eval-only — never used at runtime)
kb/         10 knowledge-base documents (75 chunks)
reports/    generated traces + metric JSON
```

## Measured results (offline, no API keys)

| Area | Result |
|---|---|
| Classification (50 tickets) | intent 78% · sentiment 50% · urgency 50% · auto-AUC 0.613 |
| Jev parse selftest | choice / score / noul mapping + renormalization pass (mocked HTTP) |
| RAG faithfulness / answer relevance | 0.911 / 0.648 |
| Context precision / recall @4 | hybrid 0.969 / 0.969 (dense-only 0.990 / 1.000 — corpus saturates) |
| Graph sweep (50 tickets) | 4 auto-reply · 46 escalate · 0 unsafe · 0 MCP errors |
| Guardrails | $75 refund → requires_human_approval → escalate; $42 → executed |
| Routing A/B (20 tickets) | current 15% auto / 0% unsafe / capture 25% · escalate-all 0% auto |
| Cost estimate | ~$0.00005/ticket (prototype + template paths) |

## Key design decisions

1. **System-1 before System-2.** Classification is a cheap typed stand-in today
   and a Jev API call when `TYPESAFE_API_KEY` is set — never an LLM for routing.
2. **Hybrid RAG, measured.** Dense + BM25 + RRF + cross-encoder with a CE
   relevance gate. On our qrels the corpus saturates, so we report the honest
   A/B (including the one dense-win divergence) instead of a marketing number.
3. **Cyclic, not linear.** QA scores draft sentences against retrieved context
   (token grounding + numeric containment); failures loop back to Research with
   ungrounded sentences as feedback, capped at 2 attempts.
4. **Defense in depth on money.** The >$50 refund rule is enforced inside the
   MCP action server — clients cannot bypass it — and the graph converts
   `requires_human_approval` into a priority escalation.
5. **Offline-first.** Every external dependency (Jev, OpenAI, Anthropic) has an
   `available()` switch with a local fallback, so the full pipeline runs with
   zero API keys and upgrades in place when keys appear.

## Activation switches

| Env var | Activates |
|---|---|
| `TYPESAFE_API_KEY` / `JEV_API_KEY` | Jev System-1 backend (real classification A/B) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Generative Resolution agent instead of template |

## Known bottlenecks & next steps

- Escalation latency ≈ 2.2s p50 from MCP spawn-per-call (3 calls) — fix:
  persistent sessions per server (async loop + shared `ClientSession`).
- Sentiment/urgency accuracy (50%) is the stand-in's ceiling; Jev calibration
  is the intended upgrade and the reason the backend interface exists.
- Streamlit dashboard reads `reports/` — regenerate after each eval run.
