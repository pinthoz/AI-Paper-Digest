"""Topic discovery over stored paper embeddings.

The pipeline already tags papers, but those tags come from a controlled
vocabulary written in advance — they answer "which of my interests is this?",
not "what is the field actually doing?". Clustering the embeddings answers the
second question, and the answer is allowed to surprise you.

The recipe is UMAP → HDBSCAN → c-TF-IDF, which is what BERTopic does, assembled
by hand here because it is about a hundred lines and the alternative is a heavy
dependency that hides the three decisions that matter.

**Why reduce first.** HDBSCAN degrades in high dimensions: as dimensionality
grows, the distances between all pairs of points converge, and a density-based
method has no density left to find. Projecting 768 dimensions down to about 5
restores the contrast. This is the step people skip, and skipping it is the
usual reason "HDBSCAN found one giant cluster and a lot of noise".

**Why HDBSCAN rather than k-means.** No k to guess, clusters may have different
densities and shapes, and — the part that matters most here — it is allowed to
say "this paper belongs to nothing". A daily arXiv feed is mostly noise by
construction, and a method that forces every point into a cluster will invent
themes that are not there.

**Why c-TF-IDF for labels.** Plain TF-IDF over a cluster surfaces words that are
rare overall, which in a corpus of ML abstracts means typos and author names.
c-TF-IDF treats each cluster as one document and asks which terms are frequent
*here* and rare *in the other clusters* — which is the actual question.

UMAP and HDBSCAN are imported lazily so the rest of the service runs without
them, and the clusterer is injectable so the logic is testable without either.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# Words that carry no signal in a corpus of machine-learning abstracts. The
# generic English stopwords plus the ones that are everywhere in this domain
# specifically — "model" and "learning" describe every cluster equally, which
# means they describe none of them.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those with without within for from to of in on at by
is are was were be been being do does did have has had can could may might will would should
we our us they their them it its as into over under via using use used based new novel approach
paper work study results result show shows shown propose proposed present presents method methods
model models learning deep neural network networks training train trained data dataset datasets
performance state art achieve achieves improve improves task tasks large small
""".split())

TOKEN = re.compile(r"[a-z][a-z0-9-]{2,}")


@dataclass
class Cluster:
    id: int
    size: int
    label: str
    terms: list[str]
    members: list[str] = field(default_factory=list)
    mean_relevance: float | None = None


def tokenise(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def ctfidf(docs_by_cluster: dict[int, list[str]], top_n: int = 6) -> dict[int, list[str]]:
    """Class-based TF-IDF.

    Each cluster is collapsed into a single document. A term scores highly when
    it is frequent inside its cluster and rare across the others — so shared
    vocabulary cancels out and what remains is what makes each cluster itself.
    """
    counts = {cid: Counter(tokenise(" ".join(docs))) for cid, docs in docs_by_cluster.items()}
    totals = {cid: max(sum(c.values()), 1) for cid, c in counts.items()}

    # How many clusters use each term at all.
    spread: Counter[str] = Counter()
    for c in counts.values():
        spread.update(set(c))

    n_clusters = max(len(counts), 1)
    out: dict[int, list[str]] = {}
    for cid, c in counts.items():
        scored = {
            term: (freq / totals[cid]) * math.log(1 + n_clusters / spread[term])
            for term, freq in c.items()
            if freq > 1 or n_clusters == 1
        }
        # A small cluster can have no term appearing twice, which left it with an
        # empty label and a caller falling back to "cluster 3". A weak name beats
        # a meaningless one: drop the repetition requirement rather than the
        # cluster, and let the shared-vocabulary penalty still do the ranking.
        if not scored:
            scored = {
                term: (freq / totals[cid]) * math.log(1 + n_clusters / spread[term])
                for term, freq in c.items()
            }
        out[cid] = [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:top_n]]
    return out


def _default_clusterer(matrix: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """UMAP then HDBSCAN. Imported here so the module loads without either."""
    try:
        import hdbscan
        import umap
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "clustering needs umap-learn and hdbscan: pip install umap-learn hdbscan"
        ) from exc

    n = matrix.shape[0]
    # UMAP needs more neighbours than it has points; below that, cluster the
    # raw vectors and accept the dimensionality penalty rather than crashing.
    if n >= 15:
        reduced = umap.UMAP(
            n_neighbors=min(15, n - 1),
            n_components=min(5, n - 2),
            metric="cosine",
            random_state=42,  # reproducible runs matter more here than speed
        ).fit_transform(matrix)
    else:
        reduced = matrix

    return hdbscan.HDBSCAN(
        min_cluster_size=max(2, min_cluster_size),
        metric="euclidean",  # UMAP has already put things in a euclidean space
        cluster_selection_method="eom",
    ).fit_predict(reduced)


# ---------------------------------------------------------------- choosing it

def reduce_for_clustering(matrix: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """The projection HDBSCAN actually clusters on.

    Pulled out of the default clusterer so a parameter sweep can reduce once
    and cluster many times. Re-reducing per candidate would be slower and,
    worse, would compare groupings found in *different* spaces.
    """
    n = matrix.shape[0]
    if n < 15:
        return matrix
    import umap

    return umap.UMAP(
        n_neighbors=min(15, n - 1),
        n_components=min(5, n - 2),
        metric="cosine",
        random_state=seed,
    ).fit_transform(matrix)


def _silhouette(points: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette over the clustered points only.

    Noise is excluded rather than treated as its own cluster: HDBSCAN's -1 is
    "belongs to nothing", and scoring it as a group would reward a run that
    dumped everything there. Needs at least two surviving clusters.
    """
    keep = labels >= 0
    if keep.sum() < 3 or len(set(labels[keep])) < 2:
        return None
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(points[keep], labels[keep]))


def sweep_min_cluster_size(
    matrix: np.ndarray, sizes: Sequence[int] = range(2, 16)
) -> list[dict]:
    """Try every candidate min_cluster_size and report what each one found.

    This is the honest analogue of an elbow plot for HDBSCAN, which has no k to
    choose. The knob is how many papers make a theme, and the trade it controls
    is visible directly: a small value finds many tiny groups and little noise,
    a large one finds a few broad groups and calls most papers noise.

    Silhouette is the tie-breaker rather than the goal. A grouping that scores
    0.6 while discarding 90% of the corpus has not found six good themes; it
    has found six outliers. Read the noise fraction alongside it.
    """
    import hdbscan

    reduced = reduce_for_clustering(matrix)
    out = []
    for size in sizes:
        if size >= len(matrix):
            break
        labels = hdbscan.HDBSCAN(
            min_cluster_size=max(2, int(size)),
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(reduced)
        clustered = labels >= 0
        out.append(
            {
                "min_cluster_size": int(size),
                "clusters": int(len(set(labels[clustered]))),
                "noise_fraction": round(float(1 - clustered.mean()), 4),
                "silhouette": _silhouette(reduced, labels),
            }
        )
    return out


def kmeans_scan(matrix: np.ndarray, ks: Sequence[int] = range(2, 13)) -> list[dict]:
    """Inertia and silhouette over k, for the elbow plot.

    Offered as a cross-check, not as the method. k-means assigns every point to
    a cluster, which on a daily arXiv feed is the wrong assumption - most papers
    genuinely belong to no theme, and a method that cannot say so will invent
    one. Useful mainly to see whether HDBSCAN's answer is in the same
    neighbourhood as the classical one.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    reduced = reduce_for_clustering(matrix)
    out = []
    for k in ks:
        if k >= len(reduced):
            break
        km = KMeans(n_clusters=int(k), n_init=10, random_state=42).fit(reduced)
        out.append(
            {
                "k": int(k),
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(reduced, km.labels_)),
            }
        )
    return out


def knee(xs: Sequence[float], ys: Sequence[float]) -> int | None:
    """The elbow, found rather than eyeballed.

    The point furthest from the straight line joining the first and last -
    the Kneedle idea, stripped to its geometry. An elbow read by eye is a
    judgement call dressed as a measurement; this one is reproducible.
    """
    if len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    # Normalise both axes: inertia and k live on wildly different scales, and
    # distance-to-line is meaningless until they do not.
    x = (x - x.min()) / (np.ptp(x) or 1)
    y = (y - y.min()) / (np.ptp(y) or 1)
    p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = p1 - p0
    line = line / (np.linalg.norm(line) or 1)
    dists = [float(np.linalg.norm((np.array([xi, yi]) - p0) - np.dot(np.array([xi, yi]) - p0, line) * line))
             for xi, yi in zip(x, y)]
    return int(np.argmax(dists))


def recommend(sweep: list[dict], *, max_noise: float = 0.6) -> dict | None:
    """Pick a min_cluster_size from a sweep.

    Best silhouette among the candidates that keep noise under the ceiling and
    find at least two clusters. The ceiling is the point: without it the rule
    reliably picks the run that clustered four papers beautifully and threw the
    rest away.
    """
    usable = [
        r for r in sweep
        if r["silhouette"] is not None and r["clusters"] >= 2 and r["noise_fraction"] <= max_noise
    ]
    if not usable:
        return None
    return max(usable, key=lambda r: r["silhouette"])


def project_2d(matrix: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """Squash the embeddings to two dimensions, for looking at.

    A separate projection from the one clustering uses. Clustering reduces to
    about five dimensions because that is where HDBSCAN works best; a picture
    needs exactly two, and forcing the clusterer to use two would trade
    cluster quality for a chart. Running UMAP twice costs a second and keeps
    each answer honest.

    The axes mean nothing in themselves - UMAP preserves neighbourhoods, not
    distances - so the chart should never carry numbered axis ticks.
    """
    n = matrix.shape[0]
    if n < 4:
        return np.zeros((n, 2), dtype=np.float32)
    try:
        import umap
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("projection needs umap-learn: pip install umap-learn") from exc

    return umap.UMAP(
        n_neighbors=min(15, n - 1),
        n_components=2,
        metric="cosine",
        random_state=seed,
    ).fit_transform(matrix)


def cluster_papers(
    ids: Sequence[str],
    matrix: np.ndarray,
    texts: dict[str, str],
    *,
    min_cluster_size: int = 3,
    relevance: dict[str, int] | None = None,
    clusterer: Callable[[np.ndarray, int], np.ndarray] = _default_clusterer,
) -> tuple[list[Cluster], list[str]]:
    """Group papers by meaning. Returns (clusters, unclustered ids).

    HDBSCAN labels noise as -1; those ids come back separately rather than as a
    cluster called "miscellaneous", because a bucket of leftovers presented as
    a theme is worse than admitting there is no theme.
    """
    if len(ids) != matrix.shape[0]:
        raise ValueError(f"{len(ids)} ids but {matrix.shape[0]} vectors")
    if len(ids) < min_cluster_size:
        return [], list(ids)

    labels = np.asarray(clusterer(matrix, min_cluster_size))

    grouped: dict[int, list[str]] = {}
    noise: list[str] = []
    for arxiv_id, label in zip(ids, labels):
        if int(label) < 0:
            noise.append(arxiv_id)
        else:
            grouped.setdefault(int(label), []).append(arxiv_id)

    if not grouped:
        return [], list(ids)

    terms = ctfidf({cid: [texts.get(i, "") for i in members] for cid, members in grouped.items()})

    clusters = []
    for cid, members in grouped.items():
        scores = [relevance[i] for i in members if relevance and i in relevance]
        clusters.append(
            Cluster(
                id=cid,
                size=len(members),
                # The label is the top terms joined: honest about being derived,
                # and cheap. An LLM call per cluster would read better, and is
                # the obvious upgrade once there are enough clusters to matter.
                label=" · ".join(terms.get(cid, [])[:3]) or f"cluster {cid}",
                terms=terms.get(cid, []),
                members=sorted(members),
                mean_relevance=round(sum(scores) / len(scores), 2) if scores else None,
            )
        )

    clusters.sort(key=lambda c: -c.size)
    return clusters, sorted(noise)
