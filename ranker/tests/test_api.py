"""Tests for the HTTP surface.

The embedding model is never loaded: these exercise the endpoints n8n actually
calls in the pipeline — recording papers, finding neighbours, reading metrics —
and those touch the store, not the model. `/rank` is covered by its own unit
tests in test_ranker.py, which is where the interesting logic lives.

Each test gets a fresh database via RANKER_DB, so they can run in any order.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RANKER_DB", str(tmp_path / "api.db"))
    monkeypatch.setenv("RANKER_TOKEN", "")  # auth has its own test below

    for mod in ("app", "store"):
        sys.modules.pop(mod, None)
    import app as app_module

    app_module.get_store.cache_clear()
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as c:
        yield c


def paper(pid: str, **over) -> dict:
    base = {
        "arxiv_id": pid,
        "title": "Attention head specialisation in BERT",
        "abstract": "We probe attention heads.",
        "topics": ["attention-heads", "probing"],
        "relevance_score": 8,
        "rank_score": 0.42,
        "published": "2026-09-15T04:00:00Z",
    }
    base.update(over)
    return base


class TestHealth:
    def test_health_does_not_load_the_model(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False


class TestRecording:
    def test_records_a_batch_and_reports_totals(self, client):
        r = client.post("/papers", json={"papers": [paper("2609.0001"), paper("2609.0002")]})
        assert r.status_code == 200
        body = r.json()
        assert body["new"] == 2 and body["updated"] == 0
        assert body["totals"]["analysed"] == 2

    def test_reposting_updates_instead_of_duplicating(self, client):
        client.post("/papers", json={"papers": [paper("2609.0001", relevance_score=4)]})
        r = client.post("/papers", json={"papers": [paper("2609.0001", relevance_score=9)]})
        assert r.json() == {"new": 0, "updated": 1, "totals": r.json()["totals"]}
        assert r.json()["totals"]["papers"] == 1

    def test_empty_batch_is_a_400_not_a_silent_success(self, client):
        assert client.post("/papers", json={"papers": []}).status_code == 400

    def test_a_paper_without_an_id_is_a_422(self, client):
        r = client.post("/papers", json={"papers": [{"title": "no id"}]})
        assert r.status_code == 422

    def test_unknown_fields_are_accepted(self, client):
        """The n8n item grows over time; this endpoint should not need editing."""
        r = client.post("/papers", json={"papers": [paper("2609.0001", something_new="x")]})
        assert r.status_code == 200


class TestSimilar:
    def test_finds_the_near_duplicate(self, client):
        client.post("/papers", json={"papers": [
            {**paper("2609.0001"), "embedding": [1.0, 0.0, 0.0]},
            {**paper("2609.0002", title="BERT head analysis"), "embedding": [0.97, 0.24, 0.0]},
            {**paper("2609.0003", title="Football xG"), "embedding": [0.0, 0.0, 1.0]},
        ]})
        n = client.get("/similar/2609.0001").json()["neighbours"]
        assert n[0]["arxiv_id"] == "2609.0002"
        assert n[0]["similarity"] > 0.9
        assert n[0]["title"] == "BERT head analysis"

    def test_min_similarity_filters(self, client):
        client.post("/papers", json={"papers": [
            {**paper("2609.0001"), "embedding": [1.0, 0.0]},
            {**paper("2609.0002"), "embedding": [0.0, 1.0]},
        ]})
        assert client.get("/similar/2609.0001?min_similarity=0.5").json()["neighbours"] == []

    def test_unknown_paper_is_an_empty_list_not_an_error(self, client):
        r = client.get("/similar/9999.9999")
        assert r.status_code == 200 and r.json()["neighbours"] == []


class TestMetrics:
    def test_says_so_when_there_is_not_enough_data(self, client):
        client.post("/papers", json={"papers": [paper("2609.0001")]})
        body = client.get("/metrics?k=10").json()
        assert "note" in body and "need at least" in body["note"]

    def test_computes_once_there_is_enough(self, client):
        # cheap score correlated with the expensive one
        papers = [
            paper(f"2609.{i:04d}", rank_score=i / 10, relevance_score=min(10, i))
            for i in range(1, 11)
        ]
        client.post("/papers", json={"papers": papers})
        body = client.get("/metrics?k=3").json()
        assert body["papers"] == 10
        assert body["spearman"] > 0.9
        assert body["recall@3"] == 1.0
        assert body["regret"] == 0.0


class TestClusters:
    def test_reports_honestly_when_there_is_too_little_data(self, client):
        client.post("/papers", json={"papers": [{**paper("2609.0001"), "embedding": [1.0, 0.0]}]})
        body = client.get("/clusters").json()
        assert body["clusters"] == []
        assert "not enough" in body["note"]


class TestFeedback:
    def test_records_a_signal(self, client):
        client.post("/papers", json={"papers": [paper("2609.0001")]})
        assert client.post("/feedback", json={"arxiv_id": "2609.0001", "signal": "read"}).status_code == 200
        assert client.get("/stats").json()["counts"]["feedback"] == 1

    def test_unknown_paper_is_a_404(self, client):
        r = client.post("/feedback", json={"arxiv_id": "9999.9999", "signal": "read"})
        assert r.status_code == 404

    def test_an_invented_signal_is_rejected(self, client):
        client.post("/papers", json={"papers": [paper("2609.0001")]})
        r = client.post("/feedback", json={"arxiv_id": "2609.0001", "signal": "loved-it"})
        assert r.status_code == 422


class TestAuth:
    def test_token_is_enforced_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANKER_DB", str(tmp_path / "auth.db"))
        monkeypatch.setenv("RANKER_TOKEN", "s3cret")
        for mod in ("app", "store"):
            sys.modules.pop(mod, None)
        import app as app_module

        app_module.get_store.cache_clear()
        from fastapi.testclient import TestClient

        with TestClient(app_module.app) as c:
            assert c.get("/stats").status_code == 401
            assert c.get("/stats", headers={"X-Ranker-Token": "wrong"}).status_code == 401
            assert c.get("/stats", headers={"X-Ranker-Token": "s3cret"}).status_code == 200
            # health stays open so a container probe works without the secret
            assert c.get("/health").status_code == 200
