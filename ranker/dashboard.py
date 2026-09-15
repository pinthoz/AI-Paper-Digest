"""A window onto what the pipeline has learned.

Every number here already existed as an HTTP endpoint; this renders it. The tabs
answer five questions, in the order you would actually ask them:

  1. What is the field doing?         the topic map
  2. How many themes are there?       parameter sweep, elbow, silhouette
  3. Is the cheap ranker any good?    cosine against LLM score
  4. Does the model know me?          its scores against my reactions
  5. What has it been feeding me?     volume, scores, topics

Reads the SQLite store directly rather than through the API — same process
boundary in this deployment, and skipping the round trip keeps the charts
responsive when a tab is refiltered.

    streamlit run dashboard.py
"""

from __future__ import annotations

import json
from itertools import count

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from clustering import cluster_papers, kmeans_scan, knee, project_2d, recommend, sweep_min_cluster_size
from eval import evaluate
from store import Store
from theme import Theme, css

st.set_page_config(page_title="AI Paper Digest", page_icon="📄", layout="wide")

# Not detected. .streamlit/config.toml pins the theme and this matches it;
# guessing is what painted near-black labels onto a near-black page.
T = Theme()
st.markdown(css(), unsafe_allow_html=True)


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def waiting(message: str) -> None:
    """One consistent empty state.

    Says what is missing and why a number is withheld, rather than rendering a
    chart of three points that looks like an answer.
    """
    st.markdown(
        f"<div style='border:1px dashed {T.border};border-radius:10px;padding:26px 22px;"
        f"color:{T.ink_3};font-size:.9rem;line-height:1.6'>{message}</div>",
        unsafe_allow_html=True,
    )


_verdicts = count()


def verdict(state: str, message: str) -> None:
    """A banner whose left rule carries the state.

    Streamlit paints every alert identically under this theme, which loses the
    only thing a verdict banner is for. The key puts a stable hook in the DOM —
    Streamlit's own kind classes are emotion hashes — and the stylesheet colours
    the rule from the reserved status palette. The sentence still names the
    verdict, so the colour is a second reading of it rather than the only one.
    """
    with st.container(key=f"verdict-{state}-{next(_verdicts)}"):
        {"good": st.success, "warn": st.warning, "bad": st.error}[state](message)


# --- data --------------------------------------------------------------------

@st.cache_resource
def store() -> Store:
    return Store()


@st.cache_data(ttl=120)
def load() -> pd.DataFrame:
    rows = store()._conn.execute(  # noqa: SLF001 - the dashboard ships with the store
        """
        SELECT p.arxiv_id, p.title, p.primary_category, p.abs_url,
               a.analysed_at, a.rank_score, a.rank_matched, a.relevance_score,
               a.relevance_reason, a.tldr, a.topics, a.paper_type, a.reading_priority, a.posted
        FROM papers p JOIN analyses a USING (arxiv_id)
        ORDER BY a.analysed_at DESC
        """
    ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["analysed_at"] = pd.to_datetime(df["analysed_at"], format="mixed", utc=True)
    df["day"] = df["analysed_at"].dt.date
    df["topics"] = df["topics"].apply(lambda t: json.loads(t or "[]"))
    return df


@st.cache_data(ttl=600)
def topic_map(min_cluster_size: int) -> pd.DataFrame | None:
    s = store()
    ids, matrix = s.vectors()
    if len(ids) < max(8, min_cluster_size):
        return None

    # dict(r), not the Row: sqlite3.Row indexes but has no .get(), and the
    # difference only surfaces on the first id that is missing.
    meta = {
        r["arxiv_id"]: dict(r)
        for r in s._conn.execute(  # noqa: SLF001
            "SELECT p.arxiv_id, p.title, a.topics, a.relevance_score, a.rank_score "
            "FROM papers p LEFT JOIN analyses a USING (arxiv_id)"
        )
    }
    texts = {i: f"{meta[i]['title']} {meta[i]['topics'] or ''}" for i in ids if i in meta}
    relevance = {i: meta[i]["relevance_score"] for i in ids if i in meta and meta[i]["relevance_score"] is not None}

    clusters, _ = cluster_papers(ids, matrix, texts, min_cluster_size=min_cluster_size, relevance=relevance)
    label_of = {m: c.label for c in clusters for m in c.members}
    xy = project_2d(matrix)

    return pd.DataFrame({
        "arxiv_id": ids,
        "x": xy[:, 0], "y": xy[:, 1],
        "title": [meta.get(i, {}).get("title", "") for i in ids],
        # The cosine: every ranked paper has one. The LLM score reaches only the
        # daily shortlist, so colouring by it would leave most of the map blank.
        "cosine": [meta.get(i, {}).get("rank_score") for i in ids],
        "llm": [relevance.get(i) for i in ids],
        "cluster": [label_of.get(i, "unclustered") for i in ids],
    })


@st.cache_data(ttl=900)
def sweep(max_size: int) -> list[dict]:
    return sweep_min_cluster_size(store().vectors()[1], range(2, max_size + 1))


@st.cache_data(ttl=900)
def kmeans(max_k: int) -> list[dict]:
    return kmeans_scan(store().vectors()[1], range(2, max_k + 1))



def hull(points: np.ndarray, pad: float = 0.55) -> tuple[np.ndarray, np.ndarray] | None:
    """A padded convex hull around a cluster, closed and ready to fill.

    Drawing the region is what lets colour mean something here. Without it,
    identity rests on hue alone, which caps a scatter at three distinguishable
    series; with an outline and a label doing the same job, colour becomes
    redundant encoding and the palette can open up.

    The padding pushes the boundary off the outermost points so the fill reads
    as a region containing them rather than a shape cutting through them.
    """
    if len(points) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return None

    try:
        h = ConvexHull(points)
    except Exception:
        # Collinear or degenerate: no area to enclose, so no region to draw.
        return None

    ring = points[h.vertices]
    centre = ring.mean(axis=0)
    ring = centre + (ring - centre) * (1 + pad)
    closed = np.vstack([ring, ring[:1]])
    return closed[:, 0], closed[:, 1]


# --- header ------------------------------------------------------------------

st.title("AI Paper Digest")
st.caption("What the pipeline read, what it thought, and whether it was right.")

df = load()
if df.empty:
    waiting(
        "Nothing stored yet. Run the workflow once with <code>STORE_URL</code> set — "
        "the store fills from the pipeline, never from this page."
    )
    st.stop()

counts = store().count()
st.write("")
# Ranked and analysed are different populations, and the gap between them is the
# whole design: everything gets the cheap score, only the shortlist gets the
# expensive one.
#
# The key is what the stylesheet hooks each tile's hue on — four quantities,
# four colours, rather than one accent repeated four times. It is chrome, not
# an encoding: nothing is being compared across the four, so a categorical hue
# says "these are different things" without implying an order between them.
tiles = (
    ("Ranked", f"{counts['ranked']:,}", "Scored by the bi-encoder — every candidate, free."),
    ("Analysed", f"{counts['analysed']:,}", "Sent to the LLM — the daily shortlist only."),
    ("Posted", f"{int(df['posted'].sum()):,}", "Cards published to Discord."),
    ("Reactions", f"{counts['feedback']:,}", "Your taps, read back off those cards."),
)
for i, (col, (label, value, hint)) in enumerate(zip(st.columns(4), tiles)):
    with col.container(key=f"tile-{i}"):
        st.metric(label, value, help=hint)
st.write("")

tab_map, tab_tuning, tab_ranker, tab_feedback, tab_flow, tab_data = st.tabs(
    ["Topic map", "How many themes?", "Ranker quality", "Does it know me?", "What it feeds me", "Papers"]
)

# --- 1. topic map ------------------------------------------------------------

with tab_map:
    st.caption(
        "Papers placed by meaning — UMAP over the embeddings, HDBSCAN for the groups, c-TF-IDF for "
        "the labels. Each theme gets a region, a colour and a name. Grey points belong to no theme; "
        "on a daily arXiv feed most papers legitimately do not."
    )
    ctl, _ = st.columns([1, 2])
    size = ctl.slider("Minimum papers per theme", 2, 12, 4, help="Smaller finds more, tinier themes.")

    try:
        tm = topic_map(size)
    except RuntimeError as exc:
        verdict("warn", str(exc))
        tm = None

    if tm is None:
        waiting("Not enough embedded papers yet. The map needs a few dozen before it says anything.")
    else:
        noise, rest = tm[tm["cluster"] == "unclustered"], tm[tm["cluster"] != "unclustered"]
        fig = go.Figure()

        # Order by size so the biggest theme always takes slot 1. Colour follows
        # the theme, not the row it happened to land on.
        order = rest["cluster"].value_counts().index.tolist() if not rest.empty else []
        colour = {name: T.hue(i) for i, name in enumerate(order)}

        # Regions first, underneath everything.
        for name in order:
            grp = rest[rest["cluster"] == name]
            ring = hull(grp[["x", "y"]].to_numpy())
            if ring is None:
                continue
            hx, hy = ring
            fig.add_trace(go.Scatter(
                x=hx, y=hy, mode="lines", fill="toself",
                fillcolor=T.wash(colour[name], 0.12),
                line=dict(color=T.wash(colour[name], 0.55), width=2, shape="spline", smoothing=1.1),
                hoverinfo="skip", showlegend=False,
            ))

        # Papers with no theme: present, recessive, never the story.
        if not noise.empty:
            fig.add_trace(go.Scattergl(
                x=noise["x"], y=noise["y"], mode="markers", name="no theme",
                marker=dict(size=6, color=T.muted, line=dict(width=0)),
                customdata=noise[["title"]],
                hovertemplate="%{customdata[0]}<extra>no theme</extra>",
            ))

        # One trace per theme, so the legend is real and the hover names it.
        for name in order:
            grp = rest[rest["cluster"] == name]
            cos = pd.to_numeric(grp["cosine"], errors="coerce").fillna(0)
            llm = grp["llm"].apply(lambda v: f"{int(v)}/10" if pd.notna(v) else "not analysed")
            fig.add_trace(go.Scattergl(
                x=grp["x"], y=grp["y"], mode="markers", name=name,
                marker=T.marker(colour[name], size=11),
                customdata=np.stack([grp["title"], cos, llm], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>" + name +
                    "<br>cosine %{customdata[1]:.3f} · LLM %{customdata[2]}<extra></extra>"
                ),
            ))

        # Direct labels on the regions: the secondary encoding that lets colour
        # be redundant rather than load-bearing.
        for name in order:
            grp = rest[rest["cluster"] == name]
            fig.add_annotation(
                x=grp["x"].mean(), y=grp["y"].max(), text=f"<b>{name}</b>",
                showarrow=False, yshift=12, font=dict(color=T.ink, size=12),
                bgcolor=T.background, bordercolor=T.wash(colour[name], 0.8),
                borderwidth=1, borderpad=5, opacity=0.96,
            )

        T.apply(fig, height=620, legend=True)
        # UMAP preserves neighbourhoods, not distances. Numbered ticks would
        # invite readings the projection cannot support.
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        show(fig)
        st.caption(
            f"**{len(order)} themes**, {len(noise)} papers in none. Hover any point for its title and "
            "scores. The regions are convex hulls around each group — they carry the grouping, so the "
            "colour is a second way of saying the same thing rather than the only one."
        )

        if not rest.empty:
            with st.expander("Themes as a table"):
                st.dataframe(
                    rest.groupby("cluster").agg(
                        papers=("arxiv_id", "count"),
                        mean_similarity=("cosine", "mean"),
                        analysed=("llm", "count"),
                        mean_llm=("llm", "mean"),
                    ).sort_values("papers", ascending=False).round(3),
                    use_container_width=True,
                )
                st.caption("`analysed` counts the papers a model actually scored — a small fraction, by design.")

# --- 2. how many themes ------------------------------------------------------

with tab_tuning:
    st.caption(
        "**HDBSCAN has no k to choose**, so the elbow method answers a question it does not ask. Its "
        "knob is how many papers make a theme, and the honest sweep is over that. The k-means elbow "
        "is below as a cross-check, not as the method."
    )
    ids, matrix = store().vectors()

    if len(ids) < 20:
        waiting(f"{len(ids)} embedded papers. A sweep needs a few dozen before its curves mean anything.")
    else:
        ctl, _ = st.columns([1, 2])
        upper = ctl.slider("Sweep up to", 5, min(30, max(6, len(ids) // 2)), min(15, max(6, len(ids) // 3)))
        with st.spinner("Clustering at every candidate size…"):
            rows = sweep(upper)
        sw = pd.DataFrame(rows)
        pick = recommend(rows)

        if pick:
            g1, g2, g3 = st.columns(3)
            g1.metric("Recommended size", pick["min_cluster_size"], help="Best silhouette while noise stays under 60%.")
            g2.metric("Themes found", pick["clusters"])
            g3.metric("Left unclustered", f"{pick['noise_fraction']:.0%}")
        else:
            waiting(
                "No setting found two or more themes while keeping noise under 60%. "
                "Usually means there are not enough papers yet for stable structure."
            )

        st.write("")
        st.markdown("### Quality against how much gets discarded")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sw["min_cluster_size"], y=sw["silhouette"], mode="lines+markers", name="silhouette",
            line=T.line(), marker=T.marker(),
            hovertemplate="%{x} papers minimum<br>silhouette %{y:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=sw["min_cluster_size"], y=sw["noise_fraction"], mode="lines+markers", name="left unclustered",
            line=T.line(T.series_2, dash="dot"), marker=T.marker(T.series_2),
            hovertemplate="%{x} papers minimum<br>%{y:.0%} unclustered<extra></extra>",
        ))
        if pick:
            fig.add_vline(x=pick["min_cluster_size"], line=dict(color=T.ink_3, width=1, dash="dot"))
        T.apply(fig, height=400, legend=True)
        fig.update_xaxes(title="minimum papers per theme", dtick=1)
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        show(fig)
        st.caption(
            "Both sit on the same 0–1 scale, which is the only reason they share an axis. **Read them "
            "together**: a silhouette of 0.6 that discards 90% of the corpus has not found six good "
            "themes, it has found six outliers."
        )

        st.markdown("### Themes found at each setting")
        fig = go.Figure(go.Bar(
            x=sw["min_cluster_size"], y=sw["clusters"], marker=T.bar(),
            hovertemplate="%{x} papers minimum<br>%{y} themes<extra></extra>",
        ))
        T.apply(fig, height=250)
        fig.update_xaxes(title="minimum papers per theme", dtick=1)
        fig.update_yaxes(dtick=2)
        show(fig)

        with st.expander("The sweep as a table"):
            st.dataframe(sw, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### The k-means elbow, for comparison")
        st.caption(
            "k-means puts **every** paper in a cluster. On a daily arXiv feed that assumption is wrong — "
            "most papers belong to no theme — so this checks whether HDBSCAN's answer is in the same "
            "neighbourhood as the classical one, nothing more."
        )
        with st.spinner("Fitting k-means at every k…"):
            km = pd.DataFrame(kmeans(min(12, max(4, len(ids) // 3))))

        if km.empty:
            waiting("Not enough papers for a k sweep.")
        else:
            elbow_i = knee(km["k"].tolist(), km["inertia"].tolist())
            elbow_k = int(km["k"].iloc[elbow_i]) if elbow_i is not None else None
            best_k = int(km.loc[km["silhouette"].idxmax(), "k"])

            k1, k2 = st.columns(2)
            with k1:
                fig = go.Figure(go.Scatter(
                    x=km["k"], y=km["inertia"], mode="lines+markers",
                    line=T.line(), marker=T.marker(),
                    hovertemplate="k %{x}<br>inertia %{y:.1f}<extra></extra>",
                ))
                if elbow_k is not None:
                    T.pin(fig, elbow_k, float(km["inertia"].iloc[elbow_i]), f"elbow · k={elbow_k}")
                T.apply(fig, height=300)
                fig.update_xaxes(title="k — within-cluster spread", dtick=1)
                show(fig)

            with k2:
                fig = go.Figure(go.Scatter(
                    x=km["k"], y=km["silhouette"], mode="lines+markers",
                    line=T.line(), marker=T.marker(),
                    hovertemplate="k %{x}<br>silhouette %{y:.3f}<extra></extra>",
                ))
                T.pin(fig, best_k, float(km["silhouette"].max()), f"best · k={best_k}")
                T.apply(fig, height=300)
                fig.update_xaxes(title="k — silhouette", dtick=1)
                show(fig)

            agree = [v for v in (elbow_k, best_k, pick["clusters"] if pick else None) if v]
            st.caption(
                f"Elbow says **{elbow_k}**, silhouette says **{best_k}**"
                + (f", HDBSCAN found **{pick['clusters']}**. " if pick else ". ")
                + ("They agree, which is a good sign the structure is real."
                   if len(set(agree)) == 1
                   else "They rarely agree exactly; when they land far apart the corpus probably has no "
                        "strong theme structure yet, which is itself the answer.")
            )

# --- 3. ranker quality -------------------------------------------------------

with tab_ranker:
    st.caption(
        "The premise of ranking before spending a model call is that a cosine similarity predicts what "
        "the model would have said. Every analysed paper carries both numbers, so it is measurable "
        "rather than assumed."
    )
    pairs = store().scored_pairs()

    if len(pairs) < 20:
        waiting(
            f"<b>{len(pairs)} papers carry both scores.</b> The numbers need about 20 before they mean "
            "anything — a correlation over five points is noise with a decimal place. Withholding it "
            "is the honest rendering."
        )
    else:
        k = min(10, max(2, len(pairs) // 4))
        report = evaluate(pairs, k)
        rho = report["spearman"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Spearman ρ", f"{rho:.2f}", help="Rank agreement between the cheap and the expensive stage.")
        m2.metric(f"Recall@{k}", f"{report[f'recall@{k}']:.0%}", help="Of the papers the LLM rated highest, how many the ranker shortlisted.")
        m3.metric("Regret", f"{report['regret']:.2f}", help="LLM points lost against a perfect shortlist.")

        if rho > 0.4:
            verdict("good", "The pre-filter is carrying real signal. Keep it.")
        elif rho > 0.15:
            verdict("warn", "Weak agreement. A stronger embedding model, or more negative weight, might help.")
        else:
            verdict("bad", "Close to noise. Feeding the LLM the newest papers would do about as well.")

        st.write("")
        pdf = pd.DataFrame(pairs)
        fig = go.Figure()
        if len(pdf) > 3:
            z = np.polyfit(pdf["score"], pdf["relevance_score"], 1)
            xs = np.linspace(pdf["score"].min(), pdf["score"].max(), 60)
            # Drawn first so the points sit on top of it.
            fig.add_trace(go.Scatter(
                x=xs, y=np.polyval(z, xs), mode="lines",
                line=T.line(T.ink_3, dash="dot"), hoverinfo="skip", showlegend=False,
            ))
        # Emphasis, not a second category: the papers that reached Discord take
        # the series hue and the rest recede, which shows where the cut actually
        # fell across the cloud. Size moves with it so the split survives CVD.
        sent = set(df.loc[df["posted"] == 1, "arxiv_id"]) if "posted" in df else set()
        made_it = pdf["arxiv_id"].isin(sent)
        for name, subset, tone, size in (
            ("not posted", pdf[~made_it], T.muted, 9),
            ("posted to Discord", pdf[made_it], T.series, 12),
        ):
            if subset.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=subset["score"], y=subset["relevance_score"], mode="markers", name=name,
                marker=T.marker(tone, size=size), customdata=subset[["arxiv_id"]],
                hovertemplate="%{customdata[0]}<br>cosine %{x:.3f} · LLM %{y}/10<extra>"
                              + name + "</extra>",
            ))
        # A legend only when there are genuinely two groups to tell apart;
        # a legend box over a single series is noise naming the obvious.
        T.apply(fig, height=420, legend=bool(made_it.any() and (~made_it).any()))
        fig.update_xaxes(title="cosine similarity to the profile")
        fig.update_yaxes(title="LLM relevance score", range=[-0.6, 10.6], dtick=2)
        show(fig)
        st.caption("A rising cloud means the cheap stage predicts the expensive one. A round blob means it does not.")

# --- 4. does it know me ------------------------------------------------------

with tab_feedback:
    st.caption(
        "What the model predicted, beside what you actually did. This is the only chart here that can "
        "tell you the research profile is wrong."
    )
    summary = store().feedback_summary()

    if not summary:
        waiting(
            "No reactions yet. React to the cards in Discord — ✅ useful, ❌ not useful, 👀 read — "
            "then run <code>POST /feedback/sync</code>."
        )
    else:
        fdf = pd.DataFrame(summary)
        by = fdf.set_index("signal")["mean_llm_score"]
        useful = by.get("useful")
        not_useful = by.get("not-useful")

        if useful is not None and not_useful is not None:
            gap = float(useful) - float(not_useful)
            g1, g2 = st.columns([1, 2])
            g1.metric(
                "Score gap", f"{gap:+.1f}",
                help="Mean LLM score of what you found useful, minus what you did not. Positive means it knows you.",
            )
            with g2:
                st.write("")
                if gap >= 1.5:
                    verdict("good", "The model's scores line up with your judgement.")
                elif gap > 0:
                    verdict("warn", "Pointing the right way, but weakly. More reactions will sharpen it.")
                else:
                    verdict("bad", "Inverted or flat: the scores are not predicting your behaviour. Rewrite the profile.")

        st.write("")
        # Status colours, because these are states and not arbitrary categories.
        # Green against red measures deltaE 6.2 under deuteranopia - inside the
        # 6-8 band that is legal ONLY with secondary encoding - so the emoji
        # rides in the tick label and carries the identity without the hue.
        # They are the same three emoji the Discord card is seeded with.
        faces = {"useful": "✅ useful", "not-useful": "❌ not useful",
                 "read": "👀 read", "archived": "🗑 archived"}
        tones = {"useful": T.good, "not-useful": T.critical,
                 "read": T.series, "archived": T.muted}
        # Not a column on fdf: the expander below renders that frame as-is.
        faced = [faces.get(s, s) for s in fdf["signal"]]

        fig = go.Figure(go.Bar(
            x=faced, y=fdf["mean_llm_score"], width=0.5,
            marker=dict(color=[tones.get(s, T.series) for s in fdf["signal"]],
                        cornerradius=5, line=dict(width=0)),
            customdata=fdf[["papers"]],
            hovertemplate="%{x}<br>mean score %{y:.2f} · %{customdata[0]} papers<extra></extra>",
        ))
        for face, score in zip(faced, fdf["mean_llm_score"]):
            T.label(fig, face, score, f"{score:.1f}")
        T.apply(fig, height=340)
        fig.update_yaxes(title="mean score the model gave", range=[0, 11], dtick=2)
        show(fig)
        st.caption("Read it as a gap, not as levels: **useful** should sit clearly above **not useful**.")
        with st.expander("Feedback as a table"):
            st.dataframe(fdf, use_container_width=True, hide_index=True)

# --- 5. what it feeds me -----------------------------------------------------

with tab_flow:
    left, right = st.columns(2)

    with left:
        st.markdown("### Papers analysed per day")
        per_day = df.groupby("day").size().reset_index(name="papers")
        fig = go.Figure(go.Bar(
            x=per_day["day"], y=per_day["papers"], marker=T.bar(),
            hovertemplate="%{x|%a %d %b}<br>%{y} papers<extra></extra>",
        ))
        T.apply(fig, height=300)
        fig.update_yaxes(dtick=2)
        show(fig)

    with right:
        st.markdown("### Score distribution")
        hist = df["relevance_score"].value_counts().sort_index().reindex(range(11), fill_value=0)
        peak = int(hist.idxmax())
        # An ordinal ramp: 0-10 is an ordered scale, so a dim low end reading
        # as "less" is correct. This is not a value-ramp on nominal categories -
        # the bands have a real order, which is what makes it legal here.
        fig = go.Figure(go.Bar(
            x=hist.index, y=hist.values,
            marker=dict(color=[T.step(v) for v in hist.index], cornerradius=5, line=dict(width=0)),
            hovertemplate="score %{x}/10<br>%{y} papers<extra></extra>",
        ))
        T.label(fig, peak, int(hist.max()), f"{int(hist.max())} at {peak}")
        T.apply(fig, height=300)
        fig.update_xaxes(dtick=1, title="relevance score")
        show(fig)
        st.caption("A healthy shape leans left with a thin right tail. A bulge at 7–8 means the rubric has drifted upward.")

    st.markdown("### Most frequent topics")
    top = store().topics_since()[:15][::-1]
    if top:
        # Length is how often a topic appears; colour is how those papers scored.
        # Two different variables, so the hue is not the bar length said twice -
        # it answers "what am I seeing a lot of that is actually any good?"
        by_topic = df.explode("topics").groupby("topics")["relevance_score"].mean()
        scores = [float(by_topic.get(t, float("nan"))) for t, _ in top]
        fig = go.Figure(go.Bar(
            x=[c for _, c in top], y=[t for t, _ in top], orientation="h",
            marker=dict(
                color=[0 if pd.isna(v) else v for v in scores],
                colorscale=T.ordinal_scale, cmin=0, cmax=10,
                cornerradius=5, line=dict(width=0),
                colorbar=dict(
                    title=dict(text="mean<br>score", side="right", font=dict(color=T.ink_3, size=11)),
                    thickness=8, len=0.55, outlinewidth=0, y=0.5,
                    tickfont=dict(color=T.ink_3, size=11),
                ),
            ),
            customdata=np.array([["—" if pd.isna(v) else f"{v:.1f}"] for v in scores]),
            hovertemplate="%{y}<br>%{x} papers · mean score %{customdata[0]}<extra></extra>",
        ))
        T.apply(fig, height=max(320, 30 * len(top)))
        fig.update_xaxes(dtick=1, title="papers")
        show(fig)
        st.caption(
            "Bar length is how often a topic shows up; colour is how those papers scored. A long dark "
            "bar is a subject the feed is full of and you keep not wanting."
        )
    else:
        waiting("No topics yet — they arrive with the first analysed paper.")

# --- 6. the papers -----------------------------------------------------------

with tab_data:
    col1, col2 = st.columns([3, 1])
    q = col1.text_input("Search title or summary", placeholder="attention, retrieval, calibration…")
    floor = col2.slider("Minimum score", 0, 10, 0)

    view = df[df["relevance_score"].fillna(-1) >= floor] if floor else df
    if q:
        needle = q.lower()
        view = view[
            view["title"].str.lower().str.contains(needle, na=False)
            | view["tldr"].fillna("").str.lower().str.contains(needle, na=False)
        ]

    st.caption(f"{len(view)} of {len(df)} papers")
    st.dataframe(
        view[["analysed_at", "relevance_score", "rank_score", "title", "rank_matched", "tldr", "abs_url"]]
        .rename(columns={"analysed_at": "analysed", "relevance_score": "LLM", "rank_score": "cosine",
                         "rank_matched": "matched interest"}),
        use_container_width=True, hide_index=True,
        column_config={
            "abs_url": st.column_config.LinkColumn("arXiv", display_text="open"),
            "LLM": st.column_config.ProgressColumn("LLM", min_value=0, max_value=10, format="%d"),
        },
    )
