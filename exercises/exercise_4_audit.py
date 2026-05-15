"""Exercise 4 - Structured SQLite audit trail + durable checkpointer."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)

console = Console()
AGENT_ID = "pr-review-agent@v0.1"


async def audit(state, entry: AuditEntry) -> None:
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


def _entry(
    *,
    action: str,
    confidence: float,
    decision: str,
    reason: str | None,
    t0: float,
    reviewer_id: str | None = None,
    risk_level: str | None = None,
) -> AuditEntry:
    return AuditEntry(
        agent_id=AGENT_ID,
        action=action,
        confidence=confidence,
        risk_level=risk_level or risk_level_for(confidence),
        reviewer_id=reviewer_id,
        decision=decision,
        reason=reason,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    )


async def node_fetch_pr(state):
    console.print("[cyan]-> fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


async def node_analyze(state):
    console.print("[cyan]-> analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": (
                "Senior reviewer. Structured output. Calibrate confidence carefully. "
                "If confidence is below 60%, populate escalation_questions with 2-4 "
                "specific, context-rich questions."
            )},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]OK[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, _entry(
        action="analyze",
        confidence=a.confidence,
        decision="pending",
        reason=a.confidence_reasoning,
        t0=t0,
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]-> route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, _entry(
        action="route",
        confidence=c,
        decision=decision,
        reason=f"Routed by confidence thresholds: {c:.2f}",
        t0=t0,
    ))
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]
    await audit(state, _entry(
        action="human_approval",
        confidence=a.confidence,
        decision="pending",
        reason="Waiting for reviewer approval",
        t0=t0,
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    await audit(state, _entry(
        action="human_approval",
        confidence=a.confidence,
        decision=resp.get("choice", "pending"),
        reason=resp.get("feedback") or "Reviewer submitted a decision",
        reviewer_id=os.environ.get("GITHUB_USER"),
        t0=t0,
    ))
    return {"human_choice": resp.get("choice"), "human_feedback": resp.get("feedback")}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` - {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]OK[/green] posted comment to {state['pr_url']}")
        return "committed"
    except Exception as e:
        console.print(f"  [red]FAIL[/red] post failed: {e}")
        return "commit_failed"


async def node_commit(state):
    console.print("[cyan]-> commit[/cyan]")
    t0 = time.monotonic()
    if state.get("escalation_answers") or state.get("human_choice") == "approve":
        action = _post(state)
    else:
        console.print(f"  [yellow]skip[/yellow] skipping comment (choice={state.get('human_choice')})")
        action = "rejected"

    a = state["analysis"]
    await audit(state, _entry(
        action="commit",
        confidence=a.confidence,
        decision=action,
        reason=f"Final action: {action}",
        reviewer_id=os.environ.get("GITHUB_USER") if state.get("human_choice") or state.get("escalation_answers") else None,
        t0=t0,
    ))
    return {"final_action": action, "posted_comment_body": _render_comment_body(state) if action == "committed" else None}


async def node_auto_approve(state):
    console.print("[cyan]-> auto_approve[/cyan]  [dim]high confidence - posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action = _post(state)
    await audit(state, _entry(
        action="auto_approve",
        confidence=a.confidence,
        decision="auto",
        reason=f"High confidence auto path; post result={action}",
        t0=t0,
    ))
    return {"final_action": f"auto_{action}", "posted_comment_body": _render_comment_body(state) if action == "committed" else None}


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    await audit(state, _entry(
        action="escalate",
        confidence=a.confidence,
        decision="pending",
        reason="Waiting for reviewer answers: " + " | ".join(questions),
        t0=t0,
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    answer_summary = "; ".join(f"{q}: {answer}" for q, answer in answers.items())
    await audit(state, _entry(
        action="escalate",
        confidence=a.confidence,
        decision="escalate",
        reason=answer_summary or "Reviewer answered escalation questions",
        reviewer_id=os.environ.get("GITHUB_USER"),
        t0=t0,
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]-> synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    initial = state["analysis"]
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": "Refine the PR review with reviewer answers and return structured PRAnalysis."},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}\n\n"
                f"Initial summary: {initial.summary}\n"
                f"Initial risks: {initial.risk_factors}\n"
                f"Initial reasoning: {initial.confidence_reasoning}\n\n"
                f"Q&A:\n{qa}"
            )},
        ])
    console.print(f"  [green]OK[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, _entry(
        action="synthesize",
        confidence=refined.confidence,
        decision="pending",
        reason=refined.confidence_reasoning,
        reviewer_id=os.environ.get("GITHUB_USER"),
        t0=t0,
    ))
    return {"analysis": refined}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("commit", node_commit),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"], title=f"conf={payload['confidence']:.0%}", border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 - SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
