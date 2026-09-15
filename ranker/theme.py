"""Chart and page styling for the dashboard.

Kept apart from the charts so every figure is built against roles — surface,
ink, series — rather than raw hex. The values come from a validated palette: the
categorical slots clear the colour-vision separation gates, and the sequential
ramp is a single hue, light to dark.

**The theme is declared, not detected.** `st.get_option("theme.base")` reports
what was *configured*, which is None when the viewer is on "auto" — so the app
rendered dark while the palette was built for light, and every label came out
near-black on near-black. `.streamlit/config.toml` pins it; this module matches
those exact values. Change one and you must change the other.

The mark specs below are not taste. Thin bars with a rounded data-end, 2px
lines, markers with a surface-coloured ring, and hairline recessive gridlines
are what make a chart read as considered rather than default — and the ring in
particular is load-bearing: without it, overlapping dots merge into a blob.
"""

from __future__ import annotations

import plotly.graph_objects as go


class Theme:
    """The one resolved palette. Mirrors .streamlit/config.toml."""

    # Page chrome — these must equal the config.toml values.
    #
    # The surfaces carry a blue cast rather than being neutral grey. A neutral
    # near-black reads as "dark mode was switched on"; a tinted ground reads as
    # chosen, and gives the accents something to sit against. The tint lives in
    # the surfaces only — the ink stays near-neutral, so text does not look
    # coloured. Re-validated after the change: the categorical order still
    # passes every gate against this surface, and all three ink steps clear
    # 4.5:1 (ink 16.2:1, ink_2 8.8:1, ink_3 5.6:1) — every one of them better
    # than on the neutral ground it replaced.
    background = "#0b0f16"
    raised = "#121824"

    # Ink, on a scale from "read this" to "it is there if you look".
    ink = "#e8ecf2"
    ink_2 = "#a5b0c0"
    ink_3 = "#7f8b9d"

    border = "#1f2937"
    # One step off the background: present when looked for, invisible otherwise.
    grid = "#161d29"

    # Categorical slots, in the order the palette validated. Never cycled, and
    # never generated: a ninth series folds into "other" or gets its own facet.
    series = "#4d93e8"
    series_2 = "#e07a4a"
    series_3 = "#2fb488"
    muted = "#33415a"

    # The categorical order, re-validated against the tinted background: worst
    # adjacent CVD deltaE 8.4, normal-vision 19.3, every slot clear of 3:1.
    #
    # Six is the cap, and it is only legal because the topic map gives each
    # theme a drawn region and a direct label. Colour is redundant there, not
    # the sole channel - which is what the 6-8 CVD band requires. A seventh
    # theme folds into Other rather than getting a generated hue.
    categorical = (
        "#3987e5",  # blue
        "#d95926",  # orange
        "#199e70",  # aqua
        "#c98500",  # yellow
        "#d55181",  # magenta
        "#9085e9",  # violet
    )

    # Status. Reserved — never reused as a fourth series.
    good = "#3fbf6f"
    warning = "#e0a92c"
    critical = "#e05c5c"

    # Ordinal: discrete ORDERED categories - score bands, tiers, priorities.
    # Validated separately from the sequential ramp because it has a different
    # job: every step has to be legible on its own, so the darkest one still
    # clears the background at 2.75:1. The sequential ramp's dark end does not,
    # which is fine under a colorbar and wrong as a bar you have to read.
    ordinal = ("#1d5c9c", "#2a78d6", "#4d93e8", "#86b6ef", "#c2dbfa")

    # Sequential: one hue, dark to light, so "more" means further from the
    # background rather than closer to it.
    sequential = ["#123a63", "#17497d", "#1d5c9c", "#2a78d6", "#4d93e8", "#86b6ef", "#c2dbfa"]

    @property
    def surface(self) -> str:
        """What a mark's ring and a label's plate are painted with."""
        return self.background

    @property
    def colorscale(self) -> list[list]:
        n = len(self.sequential) - 1
        return [[i / n, c] for i, c in enumerate(self.sequential)]

    @property
    def ordinal_scale(self) -> list[list]:
        """The ordinal steps as a continuous scale, for a colorbar on marks
        that also have to be readable one at a time."""
        n = len(self.ordinal) - 1
        return [[i / n, c] for i, c in enumerate(self.ordinal)]

    # ------------------------------------------------------------------ marks

    def bar(self, color: str | None = None) -> dict:
        """Thin, capped, with a rounded data-end and no stroke."""
        return dict(color=color or self.series, cornerradius=5, line=dict(width=0))

    def line(self, color: str | None = None, dash: str | None = None, width: int = 2) -> dict:
        return dict(color=color or self.series, width=width, dash=dash)

    def marker(self, color: str | None = None, size: int = 9) -> dict:
        """The surface ring keeps overlapping dots legible where they cross."""
        return dict(size=size, color=color or self.series, line=dict(width=2, color=self.background))


    def step(self, value: float, lo: float = 0.0, hi: float = 10.0) -> str:
        """The ordinal step for a value in a known range."""
        if hi <= lo:
            return self.ordinal[-1]
        frac = min(max((value - lo) / (hi - lo), 0.0), 1.0)
        return self.ordinal[round(frac * (len(self.ordinal) - 1))]

    def hue(self, index: int) -> str:
        """Slot *index* of the categorical order. Never cycles past the end."""
        return self.categorical[index % len(self.categorical)]

    @staticmethod
    def mix(hex_colour: str, onto: str, amount: float) -> str:
        """*amount* of one colour laid over another, resolved to a hex.

        Computed here rather than written as CSS color-mix(), which is recent
        enough that a browser without it would drop the whole declaration and
        take the banner's tint with it. A hex works everywhere.
        """
        a = [int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)]
        b = [int(onto[i : i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x * amount + y * (1 - amount)):02x}" for x, y in zip(a, b))

    @staticmethod
    def wash(hex_colour: str, alpha: float) -> str:
        """The same hue as a translucent fill, for a region behind its points."""
        r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{alpha})"

    # ----------------------------------------------------------------- layout

    def apply(self, fig: go.Figure, *, height: int = 380, legend: bool = False) -> go.Figure:
        fig.update_layout(
            height=height,
            showlegend=legend,
            margin=dict(l=0, r=0, t=30 if legend else 10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=self.ink_2, size=13, family="ui-sans-serif, -apple-system, Segoe UI, sans-serif"),
            hovermode="closest",
            hoverlabel=dict(bgcolor=self.raised, bordercolor=self.border, font=dict(color=self.ink, size=13)),
            bargap=0.5,
            legend=dict(
                orientation="h", y=1.14, x=0, xanchor="left",
                font=dict(color=self.ink_2, size=12), bgcolor="rgba(0,0,0,0)",
                itemsizing="constant",
            ),
        )
        axis = dict(
            showgrid=True, gridcolor=self.grid, gridwidth=1, griddash="solid",
            zeroline=False, showline=False, ticks="",
            tickfont=dict(color=self.ink_3, size=12),
            title_font=dict(color=self.ink_3, size=12),
        )
        fig.update_xaxes(**axis)
        fig.update_yaxes(**axis)
        return fig

    def label(self, fig: go.Figure, x, y, text: str, *, shift: int = 16) -> None:
        """A direct label — in ink, on a plate, never in the series colour."""
        fig.add_annotation(
            x=x, y=y, text=f"<b>{text}</b>", showarrow=False, yshift=shift,
            font=dict(color=self.ink, size=12),
            bgcolor=self.background, bordercolor=self.border, borderwidth=1, borderpad=4,
        )

    def pin(self, fig: go.Figure, x, y, text: str) -> None:
        """A label for a point worth singling out: highlight dot plus a plate."""
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", showlegend=False, hoverinfo="skip",
            marker=self.marker(self.series_2, size=14),
        ))
        self.label(fig, x, y, text, shift=20)


CSS = """
<style>
  :root {
    --bg: {bg}; --raised: {raised}; --border: {border};
    --ink: {ink}; --ink-2: {ink2}; --ink-3: {ink3}; --accent: {accent};
    --good: {good}; --warning: {warning}; --critical: {critical};
    --radius: 14px;
  }

  /* A low wash behind the masthead so the page has a light source. Under a
     tenth of an alpha: enough that the top is not the same flat field as the
     bottom, nowhere near enough to tint what sits on it. */
  .stApp {
    background:
      radial-gradient(1100px 420px at 16% -14%, {washa}, transparent 70%),
      radial-gradient(900px 380px at 88% -18%, {washb}, transparent 72%),
      var(--bg);
  }

  .block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1480px; }

  h1 {
    font-size: 1.6rem !important; font-weight: 640 !important;
    letter-spacing: -0.021em; margin-bottom: .15rem !important;
  }
  h3 {
    font-size: .95rem !important; font-weight: 600 !important;
    letter-spacing: -0.005em; color: var(--ink) !important;
    margin: .2rem 0 .1rem !important;
  }

  /* Stat tiles. Four different quantities, so four hues rather than one accent
     four times: a 2px rule on top and a dot beside the label. Small areas on
     purpose — the colour tells the tiles apart, the number is still the
     content, and the number stays in ink. */
  div[data-testid="stMetric"] {
    position: relative; overflow: hidden;
    background: var(--raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 17px 18px 13px;
  }
  div[data-testid="stMetric"]::before {
    content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
    background: var(--tile, var(--accent));
  }
  div[data-testid="stMetricLabel"] p {
    font-size: .74rem !important; font-weight: 500 !important;
    text-transform: uppercase; letter-spacing: .07em;
    color: var(--ink-3) !important;
  }
  div[data-testid="stMetricLabel"] p::before {
    content: ""; display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; margin-right: 7px; vertical-align: .08em;
    background: var(--tile, var(--accent));
  }
  div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: 640 !important;
    letter-spacing: -0.025em; color: var(--ink) !important; line-height: 1.15;
  }
  div[class*="st-key-tile-0"] div[data-testid="stMetric"] { --tile: {cat0}; }
  div[class*="st-key-tile-1"] div[data-testid="stMetric"] { --tile: {cat1}; }
  div[class*="st-key-tile-2"] div[data-testid="stMetric"] { --tile: {cat2}; }
  div[class*="st-key-tile-3"] div[data-testid="stMetric"] { --tile: {cat3}; }

  /* Tabs: quiet until chosen, then the section's own colour. The shared moving
     highlight is hidden and each button carries its own rule, which is what
     lets the six differ at all. */
  div[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
  button[data-baseweb="tab"] {
    font-weight: 540 !important; font-size: .9rem !important;
    letter-spacing: -0.004em; color: var(--ink-3) !important;
    padding: 8px 14px !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px;
  }
  button[data-baseweb="tab"]:hover { color: var(--ink-2) !important; }
  button[data-baseweb="tab"][aria-selected="true"] { color: var(--ink) !important; }
  div[data-baseweb="tab-highlight"] { display: none !important; }
  div[data-baseweb="tab-border"] { display: none; }
  div[data-baseweb="tab-list"] button:nth-of-type(1)[aria-selected="true"] { border-bottom-color: {cat0} !important; }
  div[data-baseweb="tab-list"] button:nth-of-type(2)[aria-selected="true"] { border-bottom-color: {cat1} !important; }
  div[data-baseweb="tab-list"] button:nth-of-type(3)[aria-selected="true"] { border-bottom-color: {cat2} !important; }
  div[data-baseweb="tab-list"] button:nth-of-type(4)[aria-selected="true"] { border-bottom-color: {cat3} !important; }
  div[data-baseweb="tab-list"] button:nth-of-type(5)[aria-selected="true"] { border-bottom-color: {cat4} !important; }
  div[data-baseweb="tab-list"] button:nth-of-type(6)[aria-selected="true"] { border-bottom-color: {cat5} !important; }

  /* Captions carry the reasoning, so make them readable rather than tiny. */
  div[data-testid="stCaptionContainer"] p {
    font-size: .875rem !important; line-height: 1.6;
    color: var(--ink-2) !important; max-width: 82ch;
  }
  div[data-testid="stCaptionContainer"] strong { color: var(--ink) !important; font-weight: 600; }

  /* Controls: a slider should not be the loudest thing on the page. */
  div[data-testid="stSlider"] label p, div[data-testid="stTextInput"] label p {
    font-size: .78rem !important; font-weight: 500 !important;
    text-transform: uppercase; letter-spacing: .06em; color: var(--ink-3) !important;
  }
  div[data-testid="stTextInput"] input {
    background: var(--raised) !important; border-color: var(--border) !important;
    border-radius: 9px !important;
  }

  div[data-testid="stExpander"] details {
    border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
    background: var(--raised);
  }
  div[data-testid="stExpander"] summary { font-size: .85rem; color: var(--ink-2); }
  div[data-testid="stExpander"] summary:hover { color: var(--ink) !important; }

  hr { border-color: var(--border) !important; margin: 1.6rem 0 !important; }

  /* Verdict banners: a left rule and a faint tint of the same status colour,
     instead of a saturated wash. The colour is the one thing the banner exists
     to say, and a uniform grey threw it away. It is redundant, never sole:
     every banner spells the verdict out in words, so it survives greyscale and
     a CVD reader. Keyed containers, because Streamlit's own kind classes are
     emotion hashes that change between releases. */
  div[data-testid="stAlert"] {
    background: var(--raised) !important; border: 1px solid var(--border) !important;
    border-left-width: 3px !important; border-radius: 10px !important;
    color: var(--ink) !important;
  }
  div[data-testid="stAlert"] p { font-size: .88rem !important; }
  div[class*="st-key-verdict-good"] div[data-testid="stAlert"] {
    border-left-color: var(--good) !important; background: {goodtint} !important;
  }
  div[class*="st-key-verdict-warn"] div[data-testid="stAlert"] {
    border-left-color: var(--warning) !important; background: {warntint} !important;
  }
  div[class*="st-key-verdict-bad"] div[data-testid="stAlert"] {
    border-left-color: var(--critical) !important; background: {badtint} !important;
  }

  div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: var(--radius); }
</style>
"""


def css(t: Theme | None = None) -> str:
    """The stylesheet, filled from the palette.

    The custom properties are substituted from the class rather than typed a
    second time. They had already drifted once — the chrome was still using the
    old muted grey after the charts moved to a lighter one that clears 4.5:1 —
    and a colour defined twice is a colour that will disagree with itself.

    Every token is replaced globally, because {cat0}..{cat5} each appear
    several times over: the title rule, a tile, a tab.
    """
    t = t or Theme()
    out = CSS
    for token, value in (
        ("{bg}", t.background), ("{raised}", t.raised), ("{border}", t.border),
        ("{ink2}", t.ink_2), ("{ink3}", t.ink_3), ("{accent}", t.series),
        ("{good}", t.good), ("{warning}", t.warning), ("{critical}", t.critical),
        ("{goodtint}", t.mix(t.good, t.raised, 0.08)),
        ("{warntint}", t.mix(t.warning, t.raised, 0.08)),
        ("{badtint}", t.mix(t.critical, t.raised, 0.08)),
        ("{washa}", t.wash(t.categorical[0], 0.085)),
        ("{washb}", t.wash(t.categorical[5], 0.065)),
        *((f"{{cat{i}}}", c) for i, c in enumerate(t.categorical)),
        # Last: "{ink}" is a prefix of "{ink2}" and "{ink3}", so replacing it
        # first would turn "{ink2}" into "#e8ecf22" and paint nothing.
        ("{ink}", t.ink),
    ):
        out = out.replace(token, value)
    return out
