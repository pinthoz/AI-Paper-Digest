"""Tests for the persistence layer.

Every test gets its own temporary database, so they are order-independent and
leave nothing behind. The interesting ones are not "does INSERT work" but the
two things the store exists to make possible: keeping both scores side by side
for evaluation, and finding a paper's neighbours by meaning rather than by ID.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from store import Store  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    with Store(tmp_path / "test.db") as s:
        yield s


def paper(pid: str, **over) -> dict:
    base = {
        "arxiv_id": pid,
        "version": "v1",
        "title": "Paper " + pid,
        "abstract": "An abstract about attention heads.",
        "authors_line": "A. Pinto",
        "primary_category": "cs.CL",
        "categories": ["cs.CL", "cs.LG"],
        "announce_type": "new",
        "published": "2026-09-15T04:00:00Z",
        "abs_url": "https://arxiv.org/abs/" + pid,
        "pdf_url": "https://arxiv.org/pdf/" + pid,
        "relevance_score": 8,
        "relevance_reason": "Squarely on interpretability.",
        "tldr": "Does a thing.",
        "topics": ["attention-heads", "probing"],
        "paper_type": "method",
        "reading_priority": "read-this-week",
        "rank_score": 0.42,
        "rank_matched": "attention head analysis",
    }
    base.update(over)
    return base


class TestRecord:
    def test_first_write_is_new_second_is_not(self, store):
        assert store.record(paper("2609.0001")) is True
        assert store.record(paper("2609.0001")) is False
        assert store.count()["papers"] == 1

    def test_rejects_a_paper_with_no_id(self, store):
        with pytest.raises(ValueError):
            store.record(paper(""))

    def test_a_paper_can_be_stored_before_the_llm_sees_it(self, store):
        store.record(paper("2609.0002", relevance_score=None))
        c = store.count()
        assert c["papers"] == 1 and c["analysed"] == 0

    def test_reanalysis_updates_rather_than_duplicates(self, store):
        store.record(paper("2609.0003", relevance_score=4))
        store.record(paper("2609.0003", relevance_score=9))
        rows = store.scored_pairs()
        assert len(rows) == 1
        assert rows[0]["relevance_score"] == 9

    def test_record_many_reports_new_versus_updated(self, store):
        store.record(paper("2609.0001"))
        out = store.record_many([paper("2609.0001"), paper("2609.0002"), paper("2609.0003")])
        assert out == {"new": 2, "updated": 1}

    def test_missing_optional_fields_do_not_crash(self, store):
        store.record({"arxiv_id": "2609.0009", "title": "Bare minimum"})
        assert store.count()["papers"] == 1


class TestEmbeddings:
    def test_vectors_are_normalised_on_write(self, store):
        store.record(paper("2609.0001"), embedding=[3.0, 4.0])  # norm 5
        _, m = store.vectors()
        assert np.isclose(np.linalg.norm(m[0]), 1.0)

    def test_a_zero_vector_does_not_divide_by_zero(self, store):
        store.record(paper("2609.0001"), embedding=[0.0, 0.0, 0.0])
        _, m = store.vectors()
        assert np.all(m[0] == 0)

    def test_vectors_and_ids_stay_aligned(self, store):
        for i, v in enumerate([[1, 0], [0, 1], [1, 1]]):
            store.record(paper(f"2609.000{i}"), embedding=v)
        ids, m = store.vectors()
        assert ids == ["2609.0000", "2609.0001", "2609.0002"]
        assert m.shape == (3, 2)

    def test_empty_store_returns_an_empty_matrix(self, store):
        ids, m = store.vectors()
        assert ids == [] and m.shape[0] == 0


class TestSimilar:
    """Semantic deduplication: what the arXiv ID cannot catch."""

    def setup_papers(self, store):
        # two near-identical vectors, one orthogonal
        store.record(paper("2609.0001", title="Attention heads in BERT"), embedding=[1.0, 0.0, 0.0])
        store.record(paper("2609.0002", title="BERT attention head analysis"), embedding=[0.98, 0.2, 0.0])
        store.record(paper("2609.0003", title="Football xG models"), embedding=[0.0, 0.0, 1.0])

    def test_finds_the_near_duplicate_first(self, store):
        self.setup_papers(store)
        out = store.similar("2609.0001")
        assert out[0].arxiv_id == "2609.0002"
        assert out[0].similarity > 0.9

    def test_never_returns_itself(self, store):
        self.setup_papers(store)
        assert all(n.arxiv_id != "2609.0001" for n in store.similar("2609.0001", k=10))

    def test_respects_k(self, store):
        self.setup_papers(store)
        assert len(store.similar("2609.0001", k=1)) == 1

    def test_min_similarity_filters_the_unrelated(self, store):
        self.setup_papers(store)
        out = store.similar("2609.0001", k=10, min_similarity=0.9)
        assert [n.arxiv_id for n in out] == ["2609.0002"]

    def test_carries_metadata_for_a_useful_answer(self, store):
        self.setup_papers(store)
        n = store.similar("2609.0001")[0]
        assert n.title == "BERT attention head analysis"
        assert n.relevance_score == 8

    def test_unknown_paper_returns_nothing_rather_than_raising(self, store):
        self.setup_papers(store)
        assert store.similar("9999.9999") == []


class TestEvalInputs:
    def test_scored_pairs_only_returns_rows_with_both_scores(self, store):
        store.record(paper("2609.0001", rank_score=0.5, relevance_score=9))
        store.record(paper("2609.0002", rank_score=None, relevance_score=7))
        store.record(paper("2609.0003", rank_score=0.1, relevance_score=None))
        rows = store.scored_pairs()
        assert [r["arxiv_id"] for r in rows] == ["2609.0001"]

    def test_scored_pairs_shape_matches_what_eval_expects(self, store):
        store.record(paper("2609.0001"))
        assert set(store.scored_pairs()[0]) == {"arxiv_id", "score", "relevance_score"}

    def test_since_filters_by_date(self, store):
        store.record(paper("2609.0001", analysed_at="2026-09-01T00:00:00Z"))
        store.record(paper("2609.0002", analysed_at="2026-09-20T00:00:00Z"))
        assert len(store.scored_pairs(since="2026-09-10")) == 1


class TestTopicsAndFeedback:
    def test_topics_are_counted_and_ranked(self, store):
        store.record(paper("2609.0001", topics=["rag", "agents"]))
        store.record(paper("2609.0002", topics=["rag"]))
        assert store.topics_since()[0] == ("rag", 2)

    def test_feedback_is_append_only(self, store):
        store.record(paper("2609.0001"))
        store.add_feedback("2609.0001", "read", source="discord-reaction")
        store.add_feedback("2609.0001", "archived")
        assert store.count()["feedback"] == 2

    def test_feedback_requires_a_known_paper(self, store):
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            store.add_feedback("9999.9999", "read")


class TestPersistence:
    def test_data_survives_reopening(self, tmp_path):
        path = tmp_path / "persist.db"
        with Store(path) as s:
            s.record(paper("2609.0001"), embedding=[1.0, 0.0])
        with Store(path) as s:
            assert s.count()["papers"] == 1
            assert s.vectors()[0] == ["2609.0001"]


class TestTwoStageAnalysis:
    """The ranker scores every candidate; the LLM sees only the shortlist.
    Either can write the row first, and neither may erase the other."""

    def test_a_rank_only_paper_is_recorded(self, store):
        store.record({"arxiv_id": "2609.0001", "title": "t", "rank_score": 0.41, "rank_matched": "rag"})
        c = store.count()
        assert c["ranked"] == 1
        assert c["analysed"] == 0, "no LLM score, so it is ranked but not analysed"

    def test_the_llm_stage_fills_in_without_losing_the_cosine(self, store):
        store.record({"arxiv_id": "2609.0001", "title": "t", "rank_score": 0.41, "rank_matched": "rag"})
        store.record(paper("2609.0001", rank_score=0.41, relevance_score=9))
        row = store.scored_pairs()[0]
        assert row["score"] == 0.41 and row["relevance_score"] == 9

    def test_a_later_rank_only_write_does_not_blank_the_llm_fields(self, store):
        """Re-ranking a paper a second day must not erase what the model said."""
        store.record(paper("2609.0001", relevance_score=9, tldr="A real summary."))
        store.record({"arxiv_id": "2609.0001", "title": "t", "rank_score": 0.55})
        row = store._conn.execute(
            "SELECT relevance_score, tldr, rank_score FROM analyses WHERE arxiv_id = ?", ("2609.0001",)
        ).fetchone()
        assert row["relevance_score"] == 9
        assert row["tldr"] == "A real summary."
        assert row["rank_score"] == 0.55

    def test_posted_is_never_unset_by_a_later_write(self, store):
        store.record(paper("2609.0001", posted_to_discord=True))
        store.record({"arxiv_id": "2609.0001", "title": "t", "rank_score": 0.2})
        assert store._conn.execute(
            "SELECT posted FROM analyses WHERE arxiv_id = ?", ("2609.0001",)
        ).fetchone()["posted"] == 1

    def test_a_paper_with_neither_score_writes_no_analysis(self, store):
        store.record({"arxiv_id": "2609.0001", "title": "just metadata"})
        assert store.count()["papers"] == 1
        assert store.count()["ranked"] == 0

    def test_rank_only_rows_stay_out_of_the_eval_set(self, store):
        """scored_pairs needs both numbers; a half-filled row is not a data point."""
        store.record({"arxiv_id": "2609.0001", "title": "t", "rank_score": 0.4})
        assert store.scored_pairs() == []
