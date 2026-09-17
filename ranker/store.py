"""Persistence for AI Paper Digest.

Everything the pipeline learns about a paper ends up here: the feed metadata,
the embedding that ranked it, the model's analysis, and — eventually — whether
the reader actually opened it. Three questions become answerable once it does:

  * Is the cheap ranker any good?      cosine score vs LLM score  (eval.py)
  * Have I seen this idea before?      nearest neighbours          (/similar)
  * What am I actually reading about?  clustering over embeddings  (clustering.py)

SQLite on purpose. Ten papers a day is a few thousand a year, and a brute-force
cosine over a few thousand 768-dimension vectors is milliseconds in numpy.
Postgres with pgvector would be a server to maintain in exchange for nothing at
this scale. Every statement lives in this module, so swapping engines later is
one file rather than a project.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

DEFAULT_PATH = Path(__file__).parent / "data" / "papers.db"

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id         TEXT PRIMARY KEY,
    version          TEXT,
    title            TEXT NOT NULL,
    abstract         TEXT,
    authors_line     TEXT,
    primary_category TEXT,
    categories       TEXT,          -- json array
    announce_type    TEXT,
    published        TEXT,
    abs_url          TEXT,
    pdf_url          TEXT,
    first_seen       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    arxiv_id          TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    analysed_at       TEXT NOT NULL,
    rank_score        REAL,          -- cosine, the cheap stage
    rank_matched      TEXT,          -- which profile interest it hit
    relevance_score   INTEGER,       -- the LLM, the expensive stage
    relevance_reason  TEXT,
    tldr              TEXT,
    why_it_matters    TEXT,
    method            TEXT,
    results           TEXT,
    limitations       TEXT,
    topics            TEXT,          -- json array
    paper_type        TEXT,
    reading_priority  TEXT,
    posted            INTEGER NOT NULL DEFAULT 0,
    -- Where the card landed. Without these a posted paper cannot be found
    -- again, and its reactions are unreachable.
    discord_channel_id TEXT,
    discord_message_id TEXT
);

-- Kept apart from analyses: vectors are bulky, and most queries never want them.
CREATE TABLE IF NOT EXISTS embeddings (
    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL          -- float32, little-endian, L2-normalised
);

-- Append-only. A paper can be marked read today and archived next week, and
-- the sequence is itself a signal.
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id   TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    signal     TEXT NOT NULL,       -- read | useful | not-useful | archived
    source     TEXT,                -- discord-reaction | manual
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_date  ON analyses(analysed_at);
CREATE INDEX IF NOT EXISTS idx_analyses_score ON analyses(relevance_score);
CREATE INDEX IF NOT EXISTS idx_feedback_paper ON feedback(arxiv_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_blob(vec: Sequence[float]) -> bytes:
    """Store as normalised float32.

    Normalising on the way in means every similarity query is a plain dot
    product, and half the precision costs nothing at this scale — the
    difference between float32 and float64 cosine is far below the noise in
    the embeddings themselves.
    """
    a = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(a))
    if n:
        a = a / n
    return a.tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


@dataclass(frozen=True)
class Neighbour:
    arxiv_id: str
    title: str
    similarity: float
    published: str
    relevance_score: int | None


class Store:
    """Thin, explicit, and deliberately not an ORM.

    The queries here are the whole data layer; hiding them behind a mapper
    would make the one interesting operation — brute-force similarity over
    every stored vector — harder to read, not easier.
    """

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn

    # ------------------------------------------------------------------ write

    def record(self, paper: dict, embedding: Sequence[float] | None = None) -> bool:
        """Upsert one paper and its analysis. Returns True if it was new.

        Takes the n8n item verbatim — same field names the workflow already
        carries — so the HTTP layer stays a pass-through and there is no
        second vocabulary to keep in sync.
        """
        arxiv_id = str(paper["arxiv_id"]).strip()
        if not arxiv_id:
            raise ValueError("paper has no arxiv_id")

        with self._tx() as c:
            existed = c.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()

            c.execute(
                """
                INSERT INTO papers (arxiv_id, version, title, abstract, authors_line,
                                    primary_category, categories, announce_type,
                                    published, abs_url, pdf_url, first_seen)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    version = excluded.version,
                    title   = excluded.title
                """,
                (
                    arxiv_id,
                    paper.get("version", ""),
                    paper.get("title", ""),
                    paper.get("abstract", ""),
                    paper.get("authors_line", ""),
                    paper.get("primary_category", ""),
                    json.dumps(paper.get("categories", [])),
                    paper.get("announce_type", ""),
                    paper.get("published", ""),
                    paper.get("abs_url", ""),
                    paper.get("pdf_url", ""),
                    _now(),
                ),
            )

            # An analyses row is written in two stages, and either can come
            # first. The ranker scores every candidate and knows nothing about
            # the LLM; the LLM sees only the shortlist. Requiring a relevance
            # score before writing anything meant the cosine had nowhere to
            # land, so most papers were stored with no score at all.
            if paper.get("relevance_score") is not None or paper.get("rank_score") is not None:
                c.execute(
                    """
                    INSERT INTO analyses (arxiv_id, analysed_at, rank_score, rank_matched,
                                          relevance_score, relevance_reason, tldr, why_it_matters,
                                          method, results, limitations, topics, paper_type,
                                          reading_priority, posted,
                                          discord_channel_id, discord_message_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    -- COALESCE throughout: the ranking stage and the LLM stage
                    -- write the same row at different times, and whichever runs
                    -- second must not blank the other's columns.
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        analysed_at      = excluded.analysed_at,
                        rank_score       = COALESCE(excluded.rank_score, analyses.rank_score),
                        rank_matched     = COALESCE(NULLIF(excluded.rank_matched, ''), analyses.rank_matched),
                        relevance_score  = COALESCE(excluded.relevance_score, analyses.relevance_score),
                        relevance_reason = COALESCE(NULLIF(excluded.relevance_reason, ''), analyses.relevance_reason),
                        tldr             = COALESCE(NULLIF(excluded.tldr, ''), analyses.tldr),
                        topics           = COALESCE(NULLIF(excluded.topics, '[]'), analyses.topics),
                        posted           = MAX(excluded.posted, analyses.posted),
                        discord_channel_id = COALESCE(excluded.discord_channel_id, analyses.discord_channel_id),
                        discord_message_id = COALESCE(excluded.discord_message_id, analyses.discord_message_id)
                    """,
                    (
                        arxiv_id,
                        paper.get("analysed_at") or _now(),
                        paper.get("rank_score"),
                        paper.get("rank_matched", ""),
                        int(paper["relevance_score"]) if paper.get("relevance_score") is not None else None,
                        paper.get("relevance_reason", ""),
                        paper.get("tldr", ""),
                        paper.get("why_it_matters", ""),
                        paper.get("method", ""),
                        paper.get("results", ""),
                        paper.get("limitations", ""),
                        json.dumps(paper.get("topics", [])),
                        paper.get("paper_type", ""),
                        paper.get("reading_priority", ""),
                        1 if paper.get("posted_to_discord") else 0,
                        paper.get("discord_channel_id"),
                        paper.get("discord_message_id"),
                    ),
                )

            if embedding is not None:
                blob = _to_blob(embedding)
                c.execute(
                    """
                    INSERT INTO embeddings (arxiv_id, dim, vec) VALUES (?,?,?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET dim = excluded.dim, vec = excluded.vec
                    """,
                    (arxiv_id, len(blob) // 4, blob),
                )

        return existed is None

    def record_many(self, papers: Iterable[dict]) -> dict[str, int]:
        new = seen = 0
        for p in papers:
            if self.record(p, p.get("embedding")):
                new += 1
            else:
                seen += 1
        return {"new": new, "updated": seen}

    def add_feedback(self, arxiv_id: str, signal: str, source: str = "manual") -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO feedback (arxiv_id, signal, source, created_at) VALUES (?,?,?,?)",
                (arxiv_id, signal, source, _now()),
            )

    # ------------------------------------------------------------------- read

    def count(self) -> dict[str, int]:
        c = self._conn
        one = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
        return {
            "papers": one("SELECT COUNT(*) FROM papers"),
            "ranked": one("SELECT COUNT(*) FROM analyses WHERE rank_score IS NOT NULL"),
            "analysed": one("SELECT COUNT(*) FROM analyses WHERE relevance_score IS NOT NULL"),
            "embedded": one("SELECT COUNT(*) FROM embeddings"),
            # Distinct reactions, not rows. The table is append-only on
            # purpose - re-reading a card records the same tap again, which is
            # what preserves *when* a reaction arrived - so COUNT(*) answers
            # "how many times did the sync run" rather than "how many
            # reactions are there". Once the sync is on a schedule the two
            # diverge fast: a fortnight of four-hourly runs turns two
            # reactions into a hundred and sixty-eight.
            "feedback": one(
                "SELECT COUNT(*) FROM (SELECT DISTINCT arxiv_id, signal FROM feedback)"
            ),
        }

    def vectors(self) -> tuple[list[str], np.ndarray]:
        """Every stored vector as one matrix, ids in matching order.

        Loading the lot is the right move at this scale and keeps the maths
        obvious; it is also the first thing to change if this ever grows.
        """
        rows = self._conn.execute(
            "SELECT arxiv_id, vec FROM embeddings ORDER BY arxiv_id"
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [r["arxiv_id"] for r in rows]
        return ids, np.vstack([_from_blob(r["vec"]) for r in rows])

    def similar(self, arxiv_id: str, k: int = 5, min_similarity: float = 0.0) -> list[Neighbour]:
        """Nearest neighbours of a stored paper, itself excluded.

        This is the semantic deduplication the arXiv ID cannot do: the same
        group reposting under a new number, or two teams landing the same idea
        in the same week.
        """
        row = self._conn.execute("SELECT vec FROM embeddings WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        if row is None:
            return []

        ids, matrix = self.vectors()
        if not ids:
            return []

        sims = matrix @ _from_blob(row["vec"])  # vectors are normalised on write
        order = np.argsort(-sims)

        out: list[Neighbour] = []
        for i in order:
            other = ids[i]
            if other == arxiv_id or float(sims[i]) < min_similarity:
                continue
            meta = self._conn.execute(
                """
                SELECT p.title, p.published, a.relevance_score
                FROM papers p LEFT JOIN analyses a USING (arxiv_id)
                WHERE p.arxiv_id = ?
                """,
                (other,),
            ).fetchone()
            out.append(
                Neighbour(
                    arxiv_id=other,
                    title=meta["title"] if meta else "",
                    similarity=round(float(sims[i]), 6),
                    published=meta["published"] if meta else "",
                    relevance_score=meta["relevance_score"] if meta else None,
                )
            )
            if len(out) == k:
                break
        return out

    def posted_cards(self, days: int = 14) -> list[dict]:
        """Cards worth checking for reactions.

        Bounded on purpose. Reactions arrive within days of a card being
        posted, and re-reading a year of history every hour would spend a lot
        of Discord quota to learn nothing.
        """
        return [
            dict(r)
            for r in self._conn.execute(
                """
                SELECT arxiv_id, discord_channel_id, discord_message_id
                FROM analyses
                WHERE discord_message_id IS NOT NULL
                  AND analysed_at >= datetime('now', ?)
                ORDER BY analysed_at DESC
                """,
                (f"-{int(days)} days",),
            )
        ]

    def feedback_summary(self) -> list[dict]:
        """What the reader actually did, next to what the model predicted.

        This is the join the whole store was built for: it turns "the model
        said 8" into "the model said 8 and you archived it unread".
        """
        return [
            dict(r)
            for r in self._conn.execute(
                """
                SELECT f.signal,
                       COUNT(DISTINCT f.arxiv_id) AS papers,
                       ROUND(AVG(a.relevance_score), 2) AS mean_llm_score,
                       ROUND(AVG(a.rank_score), 4) AS mean_rank_score
                FROM feedback f JOIN analyses a USING (arxiv_id)
                GROUP BY f.signal
                ORDER BY papers DESC
                """
            )
        ]

    def scored_pairs(self, since: str | None = None) -> list[dict]:
        """Rows carrying both scores — exactly what eval.py needs."""
        q = """
            SELECT arxiv_id, rank_score AS score, relevance_score
            FROM analyses
            WHERE rank_score IS NOT NULL AND relevance_score IS NOT NULL
        """
        params: tuple = ()
        if since:
            q += " AND analysed_at >= ?"
            params = (since,)
        return [dict(r) for r in self._conn.execute(q + " ORDER BY analysed_at", params)]

    def topics_since(self, since: str | None = None) -> list[tuple[str, int]]:
        rows = self._conn.execute(
            "SELECT topics FROM analyses" + (" WHERE analysed_at >= ?" if since else ""),
            (since,) if since else (),
        ).fetchall()
        counts: dict[str, int] = {}
        for r in rows:
            for t in json.loads(r["topics"] or "[]"):
                counts[t] = counts.get(t, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])
