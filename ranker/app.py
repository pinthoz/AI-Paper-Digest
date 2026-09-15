"""HTTP surface for the first-stage ranker.

Deployed as a Hugging Face Space (Docker SDK), called by the n8n workflow
between `Recent Window` and `Cap Papers Per Run`.

The service is stateless apart from one cache: the profile changes maybe once
a month, and re-embedding it on every request would triple the work for no
benefit.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ranker import Profile, Scored, rank, split_profile

MODEL_NAME = os.getenv("RANKER_MODEL", "BAAI/bge-small-en-v1.5")
AUTH_TOKEN = os.getenv("RANKER_TOKEN", "")
MAX_PAPERS = int(os.getenv("RANKER_MAX_PAPERS", "400"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ranker")

app = FastAPI(
    title="AI Paper Digest — first-stage ranker",
    description="Bi-encoder relevance scoring, so the LLM only sees the shortlist.",
    version="1.0.0",
)


# --------------------------------------------------------------------- model

@lru_cache(maxsize=1)
def _model():
    """Loaded once, on first use.

    Import is deferred so that `/health` answers, and the tests run, without
    dragging in torch.
    """
    from sentence_transformers import SentenceTransformer

    started = time.monotonic()
    model = SentenceTransformer(MODEL_NAME)
    log.info("loaded %s in %.1fs", MODEL_NAME, time.monotonic() - started)
    return model


def encode(texts: list[str]) -> np.ndarray:
    # normalize_embeddings makes the dot product in ranker.rank a cosine.
    return _model().encode(
        list(texts),
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )


@lru_cache(maxsize=8)
def _parse_profile(profile_hash: str, text: str) -> Profile:
    """Cached on the profile text, not on the request."""
    parsed = split_profile(text)
    log.info(
        "profile %s parsed: %d positive, %d negative concepts",
        profile_hash[:8],
        len(parsed.positive),
        len(parsed.negative),
    )
    return parsed


# ---------------------------------------------------------------------- auth

def require_token(x_ranker_token: Annotated[str | None, Header()] = None) -> None:
    """A Space on the free tier is public, so the endpoint needs a shared secret.

    Nothing here is sensitive — arXiv abstracts and a research profile — but an
    open endpoint that loads a model on demand is an invitation to burn someone
    else's CPU quota.
    """
    if not AUTH_TOKEN:
        return  # explicitly unset: open, and the startup log says so
    if not x_ranker_token or not secrets.compare_digest(x_ranker_token, AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="bad or missing X-Ranker-Token")


# ------------------------------------------------------------------ contract

class PaperIn(BaseModel):
    arxiv_id: str
    title: str
    abstract: str = ""


class RankRequest(BaseModel):
    profile: str = Field(min_length=20, description="The reader profile, verbatim from the n8n Config node.")
    papers: list[PaperIn]
    negative_weight: float = Field(0.5, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1, description="Truncate the response. The caller usually caps it anyway.")


class RankedOut(BaseModel):
    arxiv_id: str
    score: float
    positive_similarity: float
    negative_similarity: float
    matched: str


class RankResponse(BaseModel):
    model: str
    took_ms: int
    positive_concepts: int
    negative_concepts: int
    results: list[RankedOut]


# ------------------------------------------------------------------- routes

@app.get("/health")
def health() -> dict[str, object]:
    """Liveness only — deliberately does not touch the model.

    A health check that loads 130MB of weights turns every cold start into a
    timeout, which is the opposite of what a health check is for.
    """
    return {"status": "ok", "model": MODEL_NAME, "loaded": _model.cache_info().currsize > 0}


@app.post("/warm", dependencies=[Depends(require_token)])
def warm() -> dict[str, object]:
    """Load the model now rather than during the first real request.

    A free Space sleeps when idle. Calling this before the run turns a 60s
    first request into a 60s wait nobody is watching.
    """
    started = time.monotonic()
    encode(["warm"])
    return {"status": "warm", "took_ms": int((time.monotonic() - started) * 1000)}


@app.post("/rank", response_model=RankResponse, dependencies=[Depends(require_token)])
def rank_papers(req: RankRequest) -> RankResponse:
    if not req.papers:
        raise HTTPException(status_code=400, detail="no papers to rank")
    if len(req.papers) > MAX_PAPERS:
        raise HTTPException(status_code=413, detail=f"at most {MAX_PAPERS} papers per request")

    profile_hash = hashlib.sha256(req.profile.encode()).hexdigest()
    profile = _parse_profile(profile_hash, req.profile)
    if not profile:
        raise HTTPException(status_code=422, detail="profile produced no usable concepts")

    started = time.monotonic()
    scored: list[Scored] = rank(
        [_Paper(p.arxiv_id, p.title, p.abstract) for p in req.papers],
        profile,
        encode,
        negative_weight=req.negative_weight,
    )
    took_ms = int((time.monotonic() - started) * 1000)

    log.info("ranked %d papers in %dms", len(scored), took_ms)

    return RankResponse(
        model=MODEL_NAME,
        took_ms=took_ms,
        positive_concepts=len(profile.positive),
        negative_concepts=len(profile.negative),
        results=[RankedOut(**vars(s)) for s in (scored[: req.top_k] if req.top_k else scored)],
    )


@dataclass(frozen=True)
class _Paper:
    arxiv_id: str
    title: str
    abstract: str


@app.on_event("startup")
def _startup() -> None:
    if not AUTH_TOKEN:
        log.warning("RANKER_TOKEN is unset - /rank is open to anyone who finds this Space")
    log.info("ready; model %s will load on first use", MODEL_NAME)
