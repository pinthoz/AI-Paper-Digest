"""First-stage ranking for AI Paper Digest.

The pipeline can only afford a handful of LLM calls per run. Without a cheap
way to order candidates, the papers that get those calls are simply the newest
ones — which is a sampling strategy, not a selection one.

This module is the cheap stage: a bi-encoder scores every candidate abstract
against the reader's profile, and only the top few reach the model. It is the
standard retrieve-then-rerank arrangement, with the roles filled by a 130MB
sentence transformer and an LLM respectively.

Everything here is pure and takes its encoder by injection, so the logic is
testable without loading a model. `app.py` supplies the real one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

# An encoder turns texts into L2-normalised row vectors, so a dot product is a
# cosine similarity.
Encoder = Callable[[Sequence[str]], np.ndarray]

# Phrases that flip the profile from what the reader wants to what they don't.
# A profile is prose, so this is deliberately forgiving.
NEGATIVE_MARKERS = (
    "not interested",
    "no interest",
    "not relevant",
    "don't care",
    "do not care",
    "uninterested",
    "avoid",
    "exclude",
    "ignore",
)

POSITIVE_MARKERS = (
    "interested in",
    "actively working",
    "working on",
    "focus",
    "care about",
    "looking for",
)

# Negations that appear *inside* a sentence rather than as a heading. A line
# like "Ships production systems; does not run large-scale pretraining" is
# positive and negative at once, so classification happens per fragment and
# these decide the exceptions.
INLINE_NEGATIONS = (
    "does not",
    "do not",
    "doesn't",
    "don't",
    "never ",
    "rather than",
    "instead of",
    "no interest",
)

# Left over once a marker is stripped: "Not interested in: X" -> " in: X".
_LEADING_CONNECTOR = re.compile(r"^\s*(?:in|on|about|with|to|for|by)?\s*:?\s*", re.IGNORECASE)

# Shorter than this and a fragment is punctuation debris rather than a concept.
# Kept low on purpose: "rag", "agents" and "tool-use" are all real entries in
# the topic vocabulary, and an aggressive threshold silently drops the reader's
# most important interests while appearing to work.
_MIN_FRAGMENT = 3


class Paper(Protocol):
    arxiv_id: str
    title: str
    abstract: str


@dataclass(frozen=True)
class Profile:
    """A research profile split into what the reader wants and what they don't.

    Embeddings do not represent negation: "not interested in reinforcement
    learning for games" embeds close to "interested in reinforcement learning
    for games", because the content words dominate. Folding the whole profile
    into one vector therefore *attracts* exactly the papers it was meant to
    repel.

    Splitting the two and subtracting is the cheap fix, and it is why this
    class exists at all.
    """

    positive: tuple[str, ...]
    negative: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.positive or self.negative)


@dataclass(frozen=True)
class Scored:
    arxiv_id: str
    score: float
    positive_similarity: float
    negative_similarity: float
    matched: str


def _fragments(text: str) -> list[str]:
    """Break a clause into individual concepts.

    One vector for "RAG, agent orchestration, evaluation, and sports CV" is a
    blurred average that matches none of them well. Four vectors, scored with a
    max, match each of them precisely.
    """
    parts = (p.strip(" \t-–—:()") for p in re.split(r"[,;.\n]| and (?=[a-z])", text))
    # A letter has to be in there: "1.", "-" and "()" are not concepts.
    return [p for p in parts if len(p) >= _MIN_FRAGMENT and any(c.isalpha() for c in p)]


def split_profile(text: str) -> Profile:
    """Parse a free-text profile into positive and negative concept lists.

    The profile is written for a human and for the LLM prompt; this reads the
    same text without asking the user to maintain a second, structured copy.
    Lines carry the mode forward, so a "Not interested in:" heading applies to
    the lines that continue it.
    """
    positive: list[str] = []
    negative: list[str] = []
    line_mode = positive

    # Greedy, so the *last* marker in the line is the one stripped: with a
    # non-greedy prefix, "Actively working on:" matches "actively working" and
    # leaves a stray " on:" glued to the first concept.
    marker_re = re.compile(
        r"^.*(?:" + "|".join(re.escape(m) for m in NEGATIVE_MARKERS + POSITIVE_MARKERS) + r")",
        re.IGNORECASE,
    )

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        lowered = stripped.lower()
        # A heading marker sets the mode for this line and the ones continuing
        # it, so "Not interested in:" still governs its wrapped second line.
        if any(m in lowered for m in NEGATIVE_MARKERS):
            line_mode = negative
        elif any(m in lowered for m in POSITIVE_MARKERS):
            line_mode = positive

        # Drop the marker itself. "Not interested in: X" should contribute X;
        # keeping "interested" would pull the vector towards everything.
        body = marker_re.sub("", stripped)
        body = _LEADING_CONNECTOR.sub("", body) if body != stripped else stripped

        for fragment in _fragments(body or stripped):
            # Per fragment, not per line: one sentence can hold both a want and
            # a don't-want, and the don't-want is the half that matters most.
            inline_negative = any(neg in fragment.lower() for neg in INLINE_NEGATIONS)
            (negative if inline_negative else line_mode).append(fragment)

    return Profile(tuple(dict.fromkeys(positive)), tuple(dict.fromkeys(negative)))


def paper_text(title: str, abstract: str) -> str:
    """What actually gets embedded.

    The title is repeated because it is the densest signal in the document and
    a 512-token window truncates long abstracts from the right.
    """
    return f"{title.strip()}\n\n{title.strip()}. {abstract.strip()}"


def rank(
    papers: Sequence[Paper],
    profile: Profile,
    encode: Encoder,
    *,
    negative_weight: float = 0.5,
    collect: dict[str, list[float]] | None = None,
) -> list[Scored]:
    """Score every paper against the profile, best first.

    score = max similarity to any positive concept
          - negative_weight * max similarity to any negative concept

    Max rather than mean on both sides: a paper is relevant because it matches
    one of the reader's interests strongly, not because it is vaguely close to
    the average of all of them. The same logic makes a single strong match
    against a negative concept enough to push a paper down.

    Pass a dict as *collect* to get the paper vectors back, keyed by arXiv ID.
    The encoder has already computed them and the caller usually wants to store
    them: re-encoding later would pay twice, and returning them through the
    HTTP response would put megabytes of floats on the wire for nothing.
    """
    if not papers:
        return []
    if not profile:
        raise ValueError("profile has no usable concepts")

    paper_vecs = encode([paper_text(p.title, p.abstract) for p in papers])

    if collect is not None:
        for p, v in zip(papers, paper_vecs):
            collect[p.arxiv_id] = [float(x) for x in v]

    pos_vecs = encode(list(profile.positive)) if profile.positive else None
    neg_vecs = encode(list(profile.negative)) if profile.negative else None

    n = len(papers)
    if pos_vecs is not None:
        pos_sim = paper_vecs @ pos_vecs.T
        pos_best = pos_sim.max(axis=1)
        pos_idx = pos_sim.argmax(axis=1)
    else:
        pos_best = np.zeros(n)
        pos_idx = np.zeros(n, dtype=int)

    neg_best = (paper_vecs @ neg_vecs.T).max(axis=1) if neg_vecs is not None else np.zeros(n)

    scores = pos_best - negative_weight * neg_best

    out = [
        Scored(
            arxiv_id=p.arxiv_id,
            score=round(float(s), 6),
            positive_similarity=round(float(pb), 6),
            negative_similarity=round(float(nb), 6),
            # Which interest this paper actually hit. Makes a surprising
            # ranking explainable instead of a black box.
            matched=profile.positive[int(i)] if profile.positive else "",
        )
        for p, s, pb, nb, i in zip(papers, scores, pos_best, neg_best, pos_idx)
    ]
    out.sort(key=lambda r: r.score, reverse=True)
    return out
