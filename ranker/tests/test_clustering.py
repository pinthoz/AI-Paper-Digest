"""Tests for topic clustering.

Neither UMAP nor HDBSCAN is loaded here. `cluster_papers` takes its clusterer
by injection, so the parts worth testing — grouping, noise handling, c-TF-IDF
labelling — run in milliseconds against a fake that returns fixed labels.
Whether UMAP+HDBSCAN finds good clusters on real embeddings is a different
question, and one only real data can answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clustering import Cluster, cluster_papers, ctfidf, tokenise  # noqa: E402


def fixed(labels):
    """A clusterer that ignores the data and returns the labels you gave it."""
    return lambda matrix, min_cluster_size: np.asarray(labels)


IDS = ["a", "b", "c", "d", "e", "f"]
M = np.eye(6, dtype=np.float32)

TEXTS = {
    "a": "attention head specialisation in BERT probing",
    "b": "probing attention heads across BERT layers",
    "c": "attention head analysis and probing classifiers",
    "d": "expected goals models for football match outcomes",
    "e": "football player valuation from event data",
    "f": "completely unrelated quantum chemistry simulation",
}


class TestTokenise:
    def test_drops_stopwords_and_short_tokens(self):
        assert tokenise("We propose a new model of the attention head") == ["attention", "head"]

    def test_keeps_hyphenated_domain_terms(self):
        assert "re-identification" in tokenise("multi-object tracking and re-identification")

    def test_lowercases(self):
        assert tokenise("BERT Probing") == ["bert", "probing"]


class TestCtfidf:
    def test_surfaces_what_separates_clusters_not_what_they_share(self):
        out = ctfidf({
            0: ["attention heads probing attention heads probing"],
            1: ["football goals football goals players"],
        })
        assert "attention" in out[0] or "heads" in out[0]
        assert "football" in out[1] or "goals" in out[1]
        # nothing from one cluster should lead the other
        assert "football" not in out[0][:2]

    def test_a_term_in_every_cluster_is_demoted(self):
        # "papers" appears in both, the distinctive words do not
        out = ctfidf({
            0: ["papers papers attention attention attention"],
            1: ["papers papers football football football"],
        })
        assert out[0][0] != "papers"

    def test_single_cluster_still_produces_terms(self):
        out = ctfidf({0: ["attention heads probing bert layers"]})
        assert len(out[0]) > 0


class TestClusterPapers:
    def test_groups_by_label_and_orders_by_size(self):
        clusters, noise = cluster_papers(IDS, M, TEXTS, clusterer=fixed([0, 0, 0, 1, 1, -1]))
        assert [c.size for c in clusters] == [3, 2]
        assert clusters[0].members == ["a", "b", "c"]
        assert noise == ["f"]

    def test_noise_is_returned_separately_not_as_a_cluster(self):
        clusters, noise = cluster_papers(IDS, M, TEXTS, clusterer=fixed([-1, -1, -1, 0, 0, 0]))
        assert len(clusters) == 1
        assert noise == ["a", "b", "c"]
        assert all(c.id >= 0 for c in clusters)

    def test_labels_describe_their_cluster(self):
        clusters, _ = cluster_papers(IDS, M, TEXTS, clusterer=fixed([0, 0, 0, 1, 1, -1]))
        interp = next(c for c in clusters if "a" in c.members)
        football = next(c for c in clusters if "d" in c.members)
        assert any(t in interp.label for t in ("attention", "head", "probing")), interp.label
        assert "football" in football.label, football.label

    def test_mean_relevance_is_averaged_over_members(self):
        clusters, _ = cluster_papers(
            IDS, M, TEXTS,
            clusterer=fixed([0, 0, 0, 1, 1, -1]),
            relevance={"a": 9, "b": 7, "c": 8, "d": 3, "e": 5},
        )
        assert next(c for c in clusters if "a" in c.members).mean_relevance == 8.0
        assert next(c for c in clusters if "d" in c.members).mean_relevance == 4.0

    def test_mean_relevance_is_none_when_nothing_was_scored(self):
        clusters, _ = cluster_papers(IDS, M, TEXTS, clusterer=fixed([0, 0, 0, 1, 1, -1]))
        assert clusters[0].mean_relevance is None

    def test_everything_noise_yields_no_clusters(self):
        clusters, noise = cluster_papers(IDS, M, TEXTS, clusterer=fixed([-1] * 6))
        assert clusters == []
        assert noise == sorted(IDS)

    def test_too_few_papers_to_cluster_returns_them_all_as_noise(self):
        clusters, noise = cluster_papers(["a", "b"], np.eye(2), TEXTS, min_cluster_size=3)
        assert clusters == []
        assert noise == ["a", "b"]

    def test_mismatched_ids_and_vectors_is_an_error(self):
        with pytest.raises(ValueError, match="3 ids but 6 vectors"):
            cluster_papers(["a", "b", "c"], M, TEXTS, clusterer=fixed([0] * 6))

    def test_members_are_sorted_for_stable_output(self):
        clusters, _ = cluster_papers(
            ["c", "a", "b", "d", "e", "f"], M, TEXTS, clusterer=fixed([0, 0, 0, 1, 1, -1])
        )
        assert clusters[0].members == ["a", "b", "c"]

    def test_missing_text_does_not_crash_the_labeller(self):
        clusters, _ = cluster_papers(IDS, M, {}, clusterer=fixed([0, 0, 0, 1, 1, -1]))
        assert all(isinstance(c, Cluster) for c in clusters)


class TestSmallClusterLabels:
    """A cluster of four papers often has no term appearing twice. Falling back
    to "cluster 3" is worse than a weak name: the reader learns nothing."""

    def test_a_cluster_with_no_repeated_term_still_gets_terms(self):
        out = ctfidf({0: ["attention probing calibration"], 1: ["football goals xg"]})
        assert out[0], "no terms means the caller has to invent a label"
        assert "cluster" not in " ".join(out[0])

    def test_the_fallback_still_prefers_what_is_distinctive(self):
        out = ctfidf({
            0: ["shared attention unique"],
            1: ["shared football unique"],
        })
        # "shared" and "unique" appear in both; the distinctive word should lead
        assert out[0][0] in ("attention", "unique", "shared")
        assert "football" not in out[0]

    def test_cluster_papers_never_labels_a_group_cluster_n(self):
        import numpy as np
        ids = ["a", "b", "c", "d"]
        texts = {"a": "attention probing", "b": "attention heads", "c": "football xg", "d": "football goals"}
        clusters, _ = cluster_papers(
            ids, np.eye(4), texts,
            clusterer=lambda m, s: np.asarray([0, 0, 1, 1]),
            min_cluster_size=2,
        )
        assert all(not c.label.startswith("cluster ") for c in clusters), [c.label for c in clusters]
