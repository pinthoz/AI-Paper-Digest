#!/usr/bin/env python3
"""Structural checks on the exported n8n workflows.

An n8n workflow is a JSON graph, and the two ways it goes wrong in a
repository are both invisible to the eye:

1. **A dangling connection.** Renaming a node in the editor and hand-editing
   the JSON afterwards leaves `connections` pointing at a name that no longer
   exists. n8n imports it without complaint and the branch silently never
   runs.

2. **A committed secret.** The files here are templates: every credential is
   read from the environment with `$env`. But the natural way to update one is
   to export from a working instance — where the values have been filled in —
   and the diff looks like noise unless you are reading for it. A webhook URL
   in a public repo is live until somebody regenerates it.

Run with no arguments to check every workflow beside this file:

    python workflows/validate.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Node types that legitimately have no inbound connection.
ENTRY_POINTS = {
    "n8n-nodes-base.scheduleTrigger",
    "n8n-nodes-base.manualTrigger",
    "n8n-nodes-base.errorTrigger",
    "n8n-nodes-base.webhook",
    "n8n-nodes-base.executeWorkflowTrigger",
}
DECORATIVE = {"n8n-nodes-base.stickyNote"}

# Shapes that are secrets wherever they appear. Each is anchored on something
# structural rather than on entropy, so a placeholder never trips them.
SECRETS = (
    (
        "Discord webhook URL",
        re.compile(r"discord(app)?\.com/api/webhooks/\d{5,}/[\w-]{20,}", re.I),
    ),
    (
        "Telegram bot token",
        re.compile(r"\b\d{8,12}:AA[\w-]{30,}\b"),
    ),
    (
        "Discord bot token",
        re.compile(r"\b[A-Za-z0-9_-]{24,28}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}\b"),
    ),
    (
        "Google API key",
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ),
)

# Config fields that must be read from the environment, never typed in.
FROM_ENV = ("discord_webhook_url", "telegram_chat_id", "ranker_token", "store_url")


def check(path: Path) -> list[str]:
    problems: list[str] = []
    raw = path.read_text(encoding="utf-8")

    try:
        wf = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"is not valid JSON: {exc}"]

    nodes = wf.get("nodes", [])
    if not nodes:
        return ["has no nodes"]

    names = {n["name"] for n in nodes}
    by_name = {n["name"]: n for n in nodes}

    # --- 1. the graph actually connects -------------------------------------
    #
    # Every connection kind counts, not just "main". A LangChain sub-node - the
    # model, the output parser - hangs off its chain node through `ai_*` and has
    # no inbound main edge at all; checking only `main` reports the two
    # best-attached nodes in the workflow as orphans.
    targets: set[str] = set()
    attached: set[str] = set()  # sources of a non-main edge: sub-nodes
    for source, outputs in wf.get("connections", {}).items():
        if source not in names:
            problems.append(f"connection from unknown node {source!r}")
        for kind, branches in outputs.items():
            if kind != "main":
                attached.add(source)
            for branch in branches or []:
                for conn in branch or []:
                    if conn["node"] not in names:
                        problems.append(f"{source!r} connects to unknown node {conn['node']!r}")
                    targets.add(conn["node"])

    for node in nodes:
        if node["type"] in DECORATIVE:
            continue
        if node["type"] in ENTRY_POINTS:
            if node["name"] in targets:
                problems.append(f"{node['name']!r} is a trigger but something connects into it")
            continue
        if node["name"] not in targets and node["name"] not in attached:
            problems.append(f"{node['name']!r} has no inbound connection - it will never run")

    # --- 2. no secrets ------------------------------------------------------
    for label, pattern in SECRETS:
        for hit in pattern.finditer(raw):
            shown = hit.group(0)[:12]
            problems.append(f"contains what looks like a {label} ({shown}...) - this file is a template")

    # --- 3. credentials come from the environment ---------------------------
    config = by_name.get("Config")
    if config:
        for item in config["parameters"].get("assignments", {}).get("assignments", []):
            if item["name"] in FROM_ENV:
                value = str(item.get("value", ""))
                if "$env" not in value:
                    problems.append(
                        f"Config.{item['name']} does not read from $env - "
                        "an export from a live instance has overwritten the template"
                    )

    # --- 4. an error workflow is wired ---------------------------------------
    settings = wf.get("settings", {})
    if not any(n["type"] == "n8n-nodes-base.errorTrigger" for n in nodes):
        if not settings.get("errorWorkflow"):
            problems.append("no errorWorkflow set - a failed run would be silent")

    return problems


def main() -> int:
    files = sorted(HERE.glob("*.json"))
    if not files:
        print("no workflows found", file=sys.stderr)
        return 1

    failed = 0
    for path in files:
        problems = check(path)
        if problems:
            failed += 1
            print(f"FAIL {path.name}")
            for p in problems:
                print(f"       {p}")
        else:
            wf = json.loads(path.read_text(encoding="utf-8"))
            real = [n for n in wf["nodes"] if n["type"] not in DECORATIVE]
            print(f"ok   {path.name:<28} {len(real)} nodes")

    print()
    print(f"{len(files) - failed}/{len(files)} workflows pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
