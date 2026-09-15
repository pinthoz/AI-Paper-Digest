"""HTTP surface for the AI Paper Digest ranker and store.

Two jobs in one service, because they share the same data:

  * **Ranking** (`/rank`) — the first stage of the two-stage pipeline, for
    anyone running a local embedding model instead of the Gemini API. Optional.
  * **Memory** (`/papers`, `/similar`, `/clusters`, `/metrics`) — where the
    pipeline keeps what it has learned, so it can be asked questions later.

The second is the interesting one. Without it the workflow is amnesiac: it
scores a paper, posts it, and forgets. With it, three questions become
answerable — is the cheap ranker any good, have I seen this idea before, and
what is the field actually doing.

n8n talks to this over HTTP, exactly as it does to Discord. Python keeps the
numpy, n8n keeps the orchestration, and neither has to pretend to be the other.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Annotated, AsyncIterator

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from clustering import cluster_papers
from discord import SEED_EMOJI, DiscordError, seed as seed_reactions, sync as sync_reactions
from ranker import Profile, Scored, rank, split_profile
from store import Store

MODEL_NAME = os.getenv("RANKER_MODEL", "BAAI/bge-small-en-v1.5")
AUTH_TOKEN = os.getenv("RANKER_TOKEN", "")
# A whole day across five arXiv categories is about 900 announcements. The
# old ceiling of 400 was sized for a single category and would have rejected
# most real runs with a 413.
MAX_PAPERS = int(os.getenv("RANKER_MAX_PAPERS", "2000"))
DB_PATH = os.getenv("RANKER_DB", "")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ranker")

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Announce what the service is holding, and whether it is unlocked."""
    if not AUTH_TOKEN:
        log.warning("RANKER_TOKEN is unset - every endpoint is open to anyone who can reach this port")
    log.info("store at %s", get_store().path)
    log.info("counts: %s", get_store().count())
    yield


app = FastAPI(
    title="AI Paper Digest — ranker and store",
    description="Bi-encoder relevance scoring, plus the memory the pipeline queries later.",
    version="2.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------- plumbing

@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store(DB_PATH) if DB_PATH else Store()


@lru_cache(maxsize=1)
def _model():
    """Loaded on first use so /health answers during a cold start."""
    from sentence_transformers import SentenceTransformer

    started = time.monotonic()
    model = SentenceTransformer(MODEL_NAME)
    log.info("loaded %s in %.1fs", MODEL_NAME, time.monotonic() - started)
    return model


def encode(texts: list[str]) -> np.ndarray:
    return _model().encode(list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def require_token(x_ranker_token: Annotated[str | None, Header()] = None) -> None:
    """A shared secret, because this service holds a reading history.

    Nothing here is dangerous, but an endpoint that loads a model on demand and
    stores data is not one to leave open if it is reachable from anywhere.
    """
    if not AUTH_TOKEN:
        return
    if not x_ranker_token or not secrets.compare_digest(x_ranker_token, AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="bad or missing X-Ranker-Token")


guard = [Depends(require_token)]


@dataclass(frozen=True)
class _Paper:
    arxiv_id: str
    title: str
    abstract: str


# ------------------------------------------------------------------ contracts

class PaperIn(BaseModel):
    arxiv_id: str
    title: str
    abstract: str = ""


class RankRequest(BaseModel):
    profile: str = Field(min_length=20)
    papers: list[PaperIn]
    negative_weight: float = Field(0.5, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1)
    # Ranking embeds every paper anyway, so keeping the vectors costs nothing
    # here - and they are exactly what the clustering and the near-duplicate
    # search need later. Returning them instead would be megabytes of floats.
    persist: bool = True


class RecordRequest(BaseModel):
    """Whatever the workflow has on the item, passed straight through.

    Deliberately permissive: the n8n item grows fields over time and this
    endpoint should not need editing every time it does.
    """

    papers: list[dict]


class FeedbackRequest(BaseModel):
    arxiv_id: str
    signal: str = Field(pattern="^(read|useful|not-useful|archived)$")
    source: str = "manual"


# --------------------------------------------------------------------- routes

@app.get("/health")
def health() -> dict[str, object]:
    """Liveness only — deliberately does not touch the model or the disk."""
    return {"status": "ok", "model": MODEL_NAME, "model_loaded": _model.cache_info().currsize > 0}


@app.post("/warm", dependencies=guard)
def warm() -> dict[str, object]:
    started = time.monotonic()
    encode(["warm"])
    return {"status": "warm", "took_ms": int((time.monotonic() - started) * 1000)}


@app.post("/rank", dependencies=guard)
def rank_papers(req: RankRequest) -> dict[str, object]:
    if not req.papers:
        raise HTTPException(400, "no papers to rank")
    if len(req.papers) > MAX_PAPERS:
        raise HTTPException(413, f"at most {MAX_PAPERS} papers per request")

    profile: Profile = split_profile(req.profile)
    if not profile:
        raise HTTPException(422, "profile produced no usable concepts")

    started = time.monotonic()
    vectors: dict[str, list[float]] = {}
    scored: list[Scored] = rank(
        [_Paper(p.arxiv_id, p.title, p.abstract) for p in req.papers],
        profile,
        encode,
        negative_weight=req.negative_weight,
        collect=vectors if req.persist else None,
    )

    stored = 0
    if req.persist:
        s = get_store()
        by_id = {x.arxiv_id: x for x in scored}
        for p in req.papers:
            hit = by_id.get(p.arxiv_id)
            # No analysis yet - the LLM has not seen these. record() writes the
            # paper and its vector and leaves the analyses row for later.
            s.record(
                {**p.model_dump(), "rank_score": hit.score if hit else None,
                 "rank_matched": hit.matched if hit else ""},
                vectors.get(p.arxiv_id),
            )
            stored += 1

    return {
        "stored": stored,
        "model": MODEL_NAME,
        "took_ms": int((time.monotonic() - started) * 1000),
        "positive_concepts": len(profile.positive),
        "negative_concepts": len(profile.negative),
        "results": [asdict(s) for s in (scored[: req.top_k] if req.top_k else scored)],
    }


@app.post("/papers", dependencies=guard)
def record_papers(req: RecordRequest) -> dict[str, object]:
    """Record what a run produced. Idempotent: re-posting updates in place."""
    if not req.papers:
        raise HTTPException(400, "nothing to record")
    try:
        result = get_store().record_many(req.papers)
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    log.info("recorded %s", result)
    return {**result, "totals": get_store().count()}


@app.post("/feedback", dependencies=guard)
def add_feedback(req: FeedbackRequest) -> dict[str, str]:
    """The signal that makes the LLM's scores checkable against reality."""
    import sqlite3

    try:
        get_store().add_feedback(req.arxiv_id, req.signal, req.source)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(404, f"unknown paper {req.arxiv_id}") from exc
    return {"status": "recorded"}


@app.get("/similar/{arxiv_id}", dependencies=guard)
def similar(
    arxiv_id: str,
    k: int = Query(5, ge=1, le=50),
    min_similarity: float = Query(0.0, ge=-1.0, le=1.0),
) -> dict[str, object]:
    """Semantic deduplication: what the arXiv ID cannot catch.

    The same group reposting under a new number, or two teams landing the same
    idea in the same week.
    """
    neighbours = get_store().similar(arxiv_id, k=k, min_similarity=min_similarity)
    return {"arxiv_id": arxiv_id, "neighbours": [asdict(n) for n in neighbours]}


@app.get("/clusters", dependencies=guard)
def clusters(min_cluster_size: int = Query(3, ge=2, le=50)) -> dict[str, object]:
    """Topics the papers fall into, discovered rather than declared.

    The workflow's own tags answer "which of my interests is this?". This
    answers "what is the field doing?", and is allowed to disagree.
    """
    s = get_store()
    ids, matrix = s.vectors()
    if len(ids) < min_cluster_size:
        return {"clusters": [], "unclustered": ids, "note": "not enough embedded papers yet"}

    rows = {
        r["arxiv_id"]: r
        for r in s._conn.execute(  # noqa: SLF001 - the store is this module's own
            """
            SELECT p.arxiv_id, p.title, a.topics, a.relevance_score
            FROM papers p LEFT JOIN analyses a USING (arxiv_id)
            """
        )
    }
    texts = {i: f"{rows[i]['title']} {rows[i]['topics'] or ''}" for i in ids if i in rows}
    relevance = {
        i: rows[i]["relevance_score"] for i in ids if i in rows and rows[i]["relevance_score"] is not None
    }

    try:
        found, noise = cluster_papers(
            ids, matrix, texts, min_cluster_size=min_cluster_size, relevance=relevance
        )
    except RuntimeError as exc:  # umap/hdbscan not installed
        raise HTTPException(501, str(exc)) from exc

    return {
        "papers": len(ids),
        "clusters": [asdict(c) for c in found],
        "unclustered": noise,
    }


@app.get("/metrics", dependencies=guard)
def metrics(k: int = Query(10, ge=1, le=100), since: str | None = None) -> dict[str, object]:
    """Is the cheap stage earning its place?

    The whole premise of ranking before spending an LLM call is that a cosine
    similarity predicts what the model would have said. That is a claim, and
    every paper carries both numbers, so it is measurable rather than assumed.
    """
    from eval import evaluate

    rows = get_store().scored_pairs(since=since)
    if len(rows) < k * 2:
        return {
            "rows": len(rows),
            "note": f"need at least {k * 2} scored papers before the numbers mean anything",
        }
    return {"since": since, **evaluate(rows, k)}


@app.post("/feedback/seed", dependencies=guard)
def seed_feedback(days: int = Query(3, ge=1, le=30)) -> dict[str, object]:
    """Put the reaction menu on recent cards so there is something to tap.

    Discord reactions are not buttons: nothing appears under a message until
    someone adds an emoji. Asking the reader to hover, open the picker and
    find the right symbol is three steps, and three steps is how a feedback
    mechanism ends up collecting nothing. The bot places the three emoji up
    front, and giving feedback becomes one tap.

    Needs ADD_REACTIONS on top of the read permissions. Safe to re-run: adding
    a reaction the bot already added changes nothing.
    """
    if not DISCORD_BOT_TOKEN:
        raise HTTPException(501, "DISCORD_BOT_TOKEN is not set - seeding reactions needs a bot")

    cards = get_store().posted_cards(days=days)
    if not cards:
        return {"cards": 0, "note": "no posted cards in that window"}

    try:
        done = seed_reactions(cards, DISCORD_BOT_TOKEN)
    except DiscordError as exc:
        raise HTTPException(502, str(exc)) from exc

    return {"cards": len(cards), "seeded": done, "emoji": list(SEED_EMOJI)}


@app.post("/feedback/sync", dependencies=guard)
def sync_feedback(days: int = Query(14, ge=1, le=90)) -> dict[str, object]:
    """Read reactions off the recent cards and record what they mean.

    Needs a **bot** token, which is a different thing from the webhook the
    workflow posts with: on Discord, writing and reading are separate
    permissions. The bot needs View Channel and Read Message History on the
    channel holding the cards, and nothing else.

    Idempotent in effect rather than in storage: feedback is append-only, so
    running this hourly records the same reaction repeatedly. That is
    deliberate - the metrics count distinct papers per signal, and keeping
    every observation leaves the door open to asking when a reaction arrived,
    which a deduplicating table would throw away.
    """
    if not DISCORD_BOT_TOKEN:
        raise HTTPException(501, "DISCORD_BOT_TOKEN is not set - reactions need a bot, not the webhook")

    s = get_store()
    cards = s.posted_cards(days=days)
    if not cards:
        return {"cards_checked": 0, "signals": 0, "note": "no posted cards in that window"}

    try:
        found = sync_reactions(cards, DISCORD_BOT_TOKEN)
    except DiscordError as exc:
        raise HTTPException(502, str(exc)) from exc

    for r in found:
        s.add_feedback(r["arxiv_id"], r["signal"], r["source"])

    log.info("synced %d cards, %d signals", len(cards), len(found))
    return {"cards_checked": len(cards), "signals": len(found), "summary": s.feedback_summary()}


@app.get("/stats", dependencies=guard)
def stats() -> dict[str, object]:
    s = get_store()
    return {"counts": s.count(), "top_topics": s.topics_since()[:15], "feedback": s.feedback_summary()}


