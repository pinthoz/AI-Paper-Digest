"""Tests for the ranking logic.

No model is loaded anywhere here. `rank` takes its encoder by injection, so the
behaviour that actually matters — profile parsing, negation handling, ordering —
is tested against a toy encoder in milliseconds. Whether `bge-small` is a good
embedding model is not this suite's problem; `eval.py` answers that.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranker import Profile, _fragments, paper_text, rank, split_profile  # noqa: E402


@dataclass(frozen=True)
class P:
    arxiv_id: str
    title: str
    abstract: str


# A bag-of-words encoder over a fixed vocabulary: deterministic, and similarity
# means "shares vocabulary", which is enough to test ordering.
VOCAB = ("retrieval", "agents", "tracking", "football", "reinforcement", "games", "theory", "quantum")


def toy_encode(texts):
    rows = []
    for t in texts:
        low = t.lower()
        v = np.array([float(low.count(w)) for w in VOCAB])
        norm = np.linalg.norm(v)
        rows.append(v / norm if norm else v)
    return np.vstack(rows)


PROFILE = Profile(positive=("retrieval augmented generation", "agents"), negative=("reinforcement learning for games",))


class TestSplitProfile:
    def test_splits_on_the_negative_marker(self):
        p = split_profile(
            "Actively working on: retrieval augmented generation, agent orchestration.\n"
            "Not interested in: reinforcement learning for games, pure theory."
        )
        assert any("retrieval" in c for c in p.positive)
        assert any("reinforcement" in c for c in p.negative)
        assert not any("reinforcement" in c for c in p.positive)

    def test_marker_carries_across_continuation_lines(self):
        p = split_profile(
            "Interested in: evaluation of LLM systems.\n"
            "Not interested in: incremental leaderboard deltas,\n"
            "hardware and systems papers."
        )
        assert any("hardware" in c for c in p.negative), p

    def test_marker_words_are_stripped_from_the_concept(self):
        p = split_profile("Not interested in: quantum computing for chemistry")
        assert p.negative
        assert not any("interested" in c for c in p.negative)

    def test_deduplicates(self):
        p = split_profile("Interested in: agents, agents, agents")
        assert len(p.positive) == 1

    def test_empty_profile_is_falsy(self):
        assert not split_profile("")
        assert not split_profile("   \n  \n")

    def test_untagged_prose_counts_as_positive(self):
        p = split_profile("Applied ML engineer working with object tracking in sports footage.")
        assert p.positive
        assert not p.negative


class TestFragments:
    def test_splits_a_list_into_concepts(self):
        assert len(_fragments("retrieval augmented generation, agent orchestration, evaluation")) == 3

    def test_drops_punctuation_debris(self):
        assert _fragments("a, b, c") == []

    def test_strips_bullets_and_brackets(self):
        assert _fragments("- (multi-object tracking)") == ["multi-object tracking"]


class TestRank:
    def test_orders_by_similarity_to_the_profile(self):
        papers = [
            P("1", "Quantum chemistry", "quantum quantum theory"),
            P("2", "Retrieval agents", "retrieval retrieval agents"),
        ]
        out = rank(papers, PROFILE, toy_encode)
        assert [r.arxiv_id for r in out] == ["2", "1"]
        assert out[0].score > out[1].score

    def test_negation_pushes_a_paper_down(self):
        """The point of splitting the profile.

        Both papers mention the reader's interests; one is also squarely inside
        what they said they do not want. A single blended profile vector would
        rank them nearly identically.
        """
        wanted = P("keep", "Agents", "agents agents retrieval")
        unwanted = P("drop", "Agents for games", "agents reinforcement reinforcement games games")

        out = {r.arxiv_id: r.score for r in rank([wanted, unwanted], PROFILE, toy_encode)}
        assert out["keep"] > out["drop"]

    def test_negative_weight_zero_disables_the_penalty(self):
        papers = [P("x", "Agents for games", "agents reinforcement games")]
        with_penalty = rank(papers, PROFILE, toy_encode, negative_weight=1.0)[0]
        without = rank(papers, PROFILE, toy_encode, negative_weight=0.0)[0]
        assert without.score > with_penalty.score
        assert without.score == pytest.approx(without.positive_similarity)

    def test_reports_which_interest_matched(self):
        out = rank([P("1", "Tracking", "retrieval retrieval")], PROFILE, toy_encode)
        assert out[0].matched == "retrieval augmented generation"

    def test_empty_input_is_not_an_error(self):
        assert rank([], PROFILE, toy_encode) == []

    def test_empty_profile_is_an_error(self):
        with pytest.raises(ValueError):
            rank([P("1", "t", "a")], Profile((), ()), toy_encode)

    def test_profile_with_no_negatives_still_scores(self):
        out = rank([P("1", "Agents", "agents")], Profile(("agents",), ()), toy_encode)
        assert out[0].negative_similarity == 0.0
        assert out[0].score > 0

    def test_scores_are_json_safe(self):
        out = rank([P("1", "Agents", "agents")], PROFILE, toy_encode)
        assert isinstance(out[0].score, float)
        assert not isinstance(out[0].score, np.floating)

    def test_one_paper_per_result(self):
        papers = [P(str(i), "Agents", "agents") for i in range(20)]
        assert len({r.arxiv_id for r in rank(papers, PROFILE, toy_encode)}) == 20


class TestPaperText:
    def test_repeats_the_title(self):
        assert paper_text("Title", "Body").count("Title") == 2

    def test_survives_an_empty_abstract(self):
        assert "Title" in paper_text("Title", "")


class TestInlineNegation:
    """A sentence can hold a want and a don't-want at once."""

    def test_mixed_sentence_splits_both_ways(self):
        p = split_profile("Ships production systems; does not run large-scale pretraining.")
        assert any("production systems" in c for c in p.positive)
        assert any("pretraining" in c for c in p.negative), p
        assert not any("pretraining" in c for c in p.positive)

    def test_marker_leaves_no_stray_preposition(self):
        p = split_profile("Actively working on: retrieval-augmented generation, agents.")
        assert p.positive[0] == "retrieval-augmented generation", p.positive

    def test_negative_heading_leaves_no_stray_preposition(self):
        p = split_profile("Not interested in: incremental leaderboard deltas, pure theory.")
        assert p.negative[0] == "incremental leaderboard deltas", p.negative

    def test_short_vocabulary_terms_survive(self):
        p = split_profile("Interested in: rag, agents, tool-use")
        assert set(p.positive) == {"rag", "agents", "tool-use"}
