"""Does the cheap ranker agree with the expensive one?

The whole premise of the first stage is that a 130MB bi-encoder can pick which
papers deserve an LLM call. That is a claim, and it is measurable: every paper
the pipeline processes ends up with both a cosine `score` and an LLM
`relevance_score`, so the agreement between them can be checked directly.

Feed it a JSONL file with one object per paper, each holding at least
`arxiv_id`, `score` and `relevance_score` — the n8n workflow's own output
shape. `Collect Result` items can be exported straight from an execution.

    python eval.py --db                    # the usual way, once it has been running
    python eval.py --db --since 2026-09-01
    python eval.py history.jsonl --k 10    # or from an exported file

What the numbers mean:

  Spearman ρ    rank correlation. Above ~0.4 the pre-filter is carrying real
                signal; near 0 it is an expensive shuffle and you should feed
                the LLM the newest papers instead.
  Recall@k      of the papers the LLM scored highest, how many the ranker put
                in the top k. This is the number that matters — it is literally
                "how often does the cheap stage hand the expensive stage the
                right paper".
  Regret        average LLM score lost by trusting the ranker's top k over an
                oracle. In LLM points, so it is readable: 0.4 means the
                shortlist is on average 0.4 points worse than perfect.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, ties averaged. No scipy: this is the only stat needed."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def load(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.exit(f"{path}:{n}: not valid JSON — {exc}")
        if "score" not in row or "relevance_score" not in row:
            continue  # papers the ranker never saw, or that the LLM skipped
        rows.append(row)
    return rows


def evaluate(rows: list[dict], k: int) -> dict[str, float]:
    cheap = [float(r["score"]) for r in rows]
    expensive = [float(r["relevance_score"]) for r in rows]

    by_cheap = sorted(rows, key=lambda r: -float(r["score"]))[:k]
    by_expensive = sorted(rows, key=lambda r: -float(r["relevance_score"]))[:k]

    chosen = {r["arxiv_id"] for r in by_cheap}
    ideal = {r["arxiv_id"] for r in by_expensive}

    got = sum(float(r["relevance_score"]) for r in by_cheap) / k
    best = sum(float(r["relevance_score"]) for r in by_expensive) / k

    return {
        "papers": len(rows),
        "spearman": spearman(cheap, expensive),
        f"recall@{k}": len(chosen & ideal) / k,
        "mean_llm_score_of_shortlist": got,
        "mean_llm_score_of_oracle": best,
        "regret": best - got,
        "baseline_mean_llm_score": statistics.fmean(expensive),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("history", type=Path, nargs="?", help="JSONL with arxiv_id, score, relevance_score")
    ap.add_argument("--db", action="store_true", help="read from the ranker store instead of a JSONL file")
    ap.add_argument("--since", help="ISO date; only papers analysed on or after it")
    ap.add_argument("--k", type=int, default=10, help="shortlist size, i.e. max_papers_per_run")
    args = ap.parse_args()

    if args.db:
        # The store is the normal source once the workflow has been running:
        # it already holds both scores for every paper, with no export step.
        sys.path.insert(0, str(Path(__file__).parent))
        from store import Store

        with Store() as s:
            rows = s.scored_pairs(since=args.since)
    elif args.history:
        rows = load(args.history)
    else:
        ap.error("give a JSONL file, or --db to read the store")
    if len(rows) < args.k * 2:
        sys.exit(f"only {len(rows)} usable rows; need at least {args.k * 2} for the numbers to mean anything")

    report = evaluate(rows, args.k)
    width = max(len(k) for k in report)
    for key, value in report.items():
        print(f"{key:<{width}}  {value:.3f}" if isinstance(value, float) else f"{key:<{width}}  {value}")

    print()
    rho = report["spearman"]
    if rho > 0.4:
        print(f"Spearman {rho:.2f}: the pre-filter is carrying real signal. Keep it.")
    elif rho > 0.15:
        print(f"Spearman {rho:.2f}: weak agreement. Try a larger model, or raise negative_weight.")
    else:
        print(f"Spearman {rho:.2f}: the ranker is close to noise. Feeding the LLM the newest papers")
        print("would do about as well, and the honest move is to say so rather than keep the box.")


if __name__ == "__main__":
    main()
