---
title: AI Paper Digest Ranker
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# First-stage ranker

A bi-encoder that decides which arXiv papers are worth an LLM call.

> **This service is optional.** The workflow ships with the same algorithm running inside n8n against the Gemini embeddings API — same free tier, nothing to host. Deploy this instead when you want a stronger or local embedding model, or somewhere to run `eval.py` against real data.

The [AI Paper Digest](../README.md) pipeline can afford a handful of model calls per run on a free tier. Without a cheap way to order candidates, those go to the *newest* papers — a sampling strategy dressed up as a selection one. This service is the cheap stage of a retrieve-then-rerank arrangement: it scores every candidate against the reader's profile in one pass, and only the shortlist reaches the expensive model.

```
64 papers ──▶ bge-small-en-v1.5 ──▶ top 10 ──▶ Gemini ──▶ digest
              ~1s, free, local        shortlist    10 calls
```

## The one idea worth stealing

**Embeddings cannot represent negation.** "Not interested in reinforcement learning for games" embeds close to "interested in reinforcement learning for games", because the content words dominate and `not` is a rounding error. Fold a whole profile into one vector and it will faithfully *attract* every paper the reader asked to avoid.

So the profile is split, and the negative half is subtracted:

```
score = max cosine(paper, any positive concept)
      − w · max cosine(paper, any negative concept)
```

Max rather than mean on both sides: a paper is relevant because it strongly matches *one* interest, not because it is vaguely near the average of all of them. The same logic makes a single strong hit against a negative concept enough to sink it.

Splitting happens per fragment, not per line, because one sentence often holds both — `Ships production systems; does not run large-scale pretraining` contributes a want and a don't-want.

## API

`POST /rank`

```jsonc
{
  "profile": "Actively working on: RAG, agents…\nNot interested in: …",
  "papers": [{ "arxiv_id": "2609.10540", "title": "…", "abstract": "…" }],
  "negative_weight": 0.5,
  "top_k": null
}
```

```jsonc
{
  "model": "BAAI/bge-small-en-v1.5",
  "took_ms": 812,
  "positive_concepts": 15,
  "negative_concepts": 6,
  "results": [
    {
      "arxiv_id": "2609.10540",
      "score": 0.481,
      "positive_similarity": 0.532,
      "negative_similarity": 0.102,
      "matched": "computer vision applied to sports footage"
    }
  ]
}
```

`matched` is the interest the paper actually hit, so a surprising ranking is explainable rather than a black box.

`POST /warm` loads the model without ranking anything — call it before a run so the cold start happens while nobody is waiting. `GET /health` is liveness only and deliberately does **not** touch the model: a health check that loads 130MB of weights turns every cold start into a timeout.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `RANKER_TOKEN` | *(unset)* | Shared secret checked against `X-Ranker-Token`. Unset means open, and the service says so in its startup log. |
| `RANKER_MODEL` | `BAAI/bge-small-en-v1.5` | Baked into the image at build time. Changing it means rebuilding. |
| `RANKER_MAX_PAPERS` | `400` | Request size ceiling. |

A Space on the free tier is public. Nothing here is sensitive — arXiv abstracts and a research profile — but an open endpoint that loads a model on demand is an invitation to burn someone else's CPU quota, so set `RANKER_TOKEN` under **Settings → Variables and secrets**.

## Deploying

Create a Space with the **Docker** SDK and push this directory. The `Dockerfile` installs CPU-only torch (the default wheel bundles CUDA and is ~2GB larger, for hardware a free Space does not have) and downloads the model at build time so the first request does not.

Free Spaces sleep when idle. The daily cron wakes it; the n8n node allows 90 seconds and falls back to chronological order if it never answers.

## Development

```bash
pip install -r requirements.txt
pytest tests/ -q          # 24 tests, no model loaded
uvicorn app:app --reload
```

`ranker.py` takes its encoder by injection, so the logic that matters — profile parsing, negation, ordering — is tested against a toy bag-of-words encoder in under a second. Whether `bge-small` is a *good* embedding model is a different question, and `eval.py` is what answers it.

## Is the pre-filter actually earning its place?

That is a measurable claim, not an article of faith. Every paper the pipeline processes ends up with both a cosine `score` and an LLM `relevance_score`, so export a few runs and check:

```bash
python eval.py history.jsonl --k 5
```

```
papers                       40
spearman                     0.888
recall@5                     0.600
mean_llm_score_of_shortlist  8.000
mean_llm_score_of_oracle     9.000
regret                       1.000
baseline_mean_llm_score      4.625
```

Read `recall@5` first — of the papers the LLM rated highest, how many did the ranker put in the shortlist. `regret` is the same thing in LLM points: how much quality the shortlist costs against a perfect oracle.

If Spearman comes out near zero, the ranker is an expensive shuffle and the honest move is to delete it and feed the LLM the newest papers instead. `eval.py` says so in as many words.
