"""Tests for the palette and the generated stylesheet.

None of this checks whether the colours are *nice* — that is what the palette
validator is for, and it runs outside the suite. What is tested here is the
machinery that has actually broken: a colour defined in two places drifting
apart, and a substitution silently leaving the page half-painted.

**Run these in the container**, which is the only place the dashboard's own
dependencies exist:

    docker compose exec dashboard python -m pytest /home/user/app/tests -q

On a bare host they report SKIPPED rather than failing collection — without
the guard, one missing import takes the other suites down with it.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("plotly", reason="dashboard-only dependency; run this suite in the container")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from theme import Theme, css  # noqa: E402


@pytest.fixture
def t():
    return Theme()


class TestStylesheet:
    def test_every_placeholder_is_substituted(self, t):
        """A leftover {token} renders as literal text in a CSS value, which the
        browser drops — the rule silently does nothing and the page looks
        almost right."""
        left = re.findall(r"\{[a-z0-9_]+\}", css(t))
        assert left == [], f"por substituir: {left}"

    def test_ink_does_not_eat_ink2_and_ink3(self, t):
        """{ink} is a prefix of {ink2} and {ink3}. Substituted in the wrong
        order it turns {ink2} into '<hex>2', a colour the browser rejects, and
        every secondary label loses its rule."""
        out = css(t)
        assert f"--ink-2: {t.ink_2};" in out
        assert f"--ink-3: {t.ink_3};" in out
        assert f"{t.ink}2" not in out
        assert f"{t.ink}3" not in out

    def test_each_categorical_slot_reaches_the_page(self, t):
        """The tabs, the tiles and the title rule all key off the same order.
        A slot missing from the stylesheet means a tab underlines in nothing."""
        for hue in t.categorical:
            assert hue in css(t), f"{hue} nao chegou ao CSS"

    def test_a_hue_that_appears_more_than_once_is_replaced_everywhere(self, t):
        """The first four slots paint a stat tile *and* a tab. str.replace with
        a count would do the tile and leave the tab holding the literal token —
        one coloured underline and five that silently render as nothing."""
        out = css(t)
        for i in range(4):
            assert out.count(t.categorical[i]) >= 2, f"slot {i} so aparece uma vez"

    def test_status_colours_are_present_for_the_verdict_banners(self, t):
        out = css(t)
        for hue in (t.good, t.warning, t.critical):
            assert hue in out


class TestConfigAgreement:
    """The chart palette and Streamlit's own chrome are two different systems
    painting the same page. They agreed once and drifted; now it is a test."""

    @pytest.fixture
    def config(self):
        with open(ROOT / ".streamlit" / "config.toml", "rb") as fh:
            return tomllib.load(fh)["theme"]

    def test_background_matches(self, t, config):
        assert config["backgroundColor"].lower() == t.background

    def test_raised_matches(self, t, config):
        assert config["secondaryBackgroundColor"].lower() == t.raised

    def test_text_colour_matches(self, t, config):
        assert config["textColor"].lower() == t.ink

    def test_theme_is_pinned_not_auto(self, config):
        """On "auto", st.get_option("theme.base") returns None and the app
        rendered dark against a light palette — near-black text on near-black."""
        assert config["base"] == "dark"


class TestMix:
    def test_none_of_the_overlay_returns_the_ground(self, t):
        assert t.mix("#ffffff", "#121824", 0.0) == "#121824"

    def test_all_of_the_overlay_returns_the_overlay(self, t):
        assert t.mix("#3fbf6f", "#121824", 1.0) == "#3fbf6f"

    def test_a_tint_stays_close_to_the_ground(self, t):
        """8% of a status colour is a hint that the banner has a state, not a
        wash that fights the text sitting on it."""
        tint = t.mix(t.good, t.raised, 0.08)
        ground = [int(t.raised[i : i + 2], 16) for i in (1, 3, 5)]
        mixed = [int(tint[i : i + 2], 16) for i in (1, 3, 5)]
        assert max(abs(a - b) for a, b in zip(ground, mixed)) < 25

    def test_output_is_always_a_six_digit_hex(self, t):
        for amount in (0.0, 0.03, 0.5, 0.97, 1.0):
            assert re.fullmatch(r"#[0-9a-f]{6}", t.mix(t.critical, t.background, amount))


class TestOrdinalRamp:
    def test_step_spans_the_whole_ramp(self, t):
        assert t.step(0) == t.ordinal[0]
        assert t.step(10) == t.ordinal[-1]

    def test_every_step_is_reachable(self, t):
        """A ramp with an unreachable step is a ramp with fewer steps."""
        assert {t.step(v) for v in range(11)} == set(t.ordinal)

    def test_values_outside_the_range_clamp(self, t):
        assert t.step(-5) == t.ordinal[0]
        assert t.step(99) == t.ordinal[-1]

    def test_a_degenerate_range_does_not_divide_by_zero(self, t):
        assert t.step(5, lo=3, hi=3) == t.ordinal[-1]

    def test_the_scale_runs_from_zero_to_one(self, t):
        assert t.ordinal_scale[0][0] == 0.0
        assert t.ordinal_scale[-1][0] == 1.0

    def test_status_colours_are_not_in_the_categorical_order(self, t):
        """Reserved means reserved: a status colour reused as a series makes
        'this one is broken' indistinguishable from 'this one is the third'."""
        assert not ({t.good, t.warning, t.critical} & set(t.categorical))
