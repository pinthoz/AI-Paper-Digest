"""Tests for reading reactions off the Discord cards.

Discord is never called: `sync` takes its fetcher by injection, so the parts
worth testing — the emoji vocabulary, skipping cards that were never posted,
and surviving a card that has since been deleted — run offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discord import SIGNALS, DiscordError, Reaction, sync  # noqa: E402
from store import Store  # noqa: E402


def card(pid: str, channel="111", message="222") -> dict:
    return {"arxiv_id": pid, "discord_channel_id": channel, "discord_message_id": message}


class TestVocabulary:
    def test_ticks_and_crosses_map_to_opposite_signals(self):
        assert SIGNALS["✅"] == "useful"
        assert SIGNALS["❌"] == "not-useful"

    def test_several_emoji_share_a_meaning(self):
        """People reach for whichever tick their client offers first."""
        assert SIGNALS["👍"] == SIGNALS["✅"] == "useful"
        assert SIGNALS["👀"] == SIGNALS["📖"] == "read"


class TestSync:
    def test_collects_one_record_per_signal(self):
        def fetch(channel, message, token):
            return [Reaction("✅", "useful", 1), Reaction("👀", "read", 1)]

        out = sync([card("2609.0001")], "tok", fetch=fetch, pause=0)
        assert {r["signal"] for r in out} == {"useful", "read"}
        assert all(r["source"] == "discord-reaction" for r in out)
        assert all(r["arxiv_id"] == "2609.0001" for r in out)

    def test_a_card_with_no_reactions_yields_nothing(self):
        out = sync([card("2609.0001")], "tok", fetch=lambda *a: [], pause=0)
        assert out == []

    def test_papers_that_were_never_posted_are_skipped(self):
        """A paper below the threshold has no card, so there is nothing to read."""
        calls = []

        def fetch(channel, message, token):
            calls.append(message)
            return []

        rows = [card("2609.0001"), {"arxiv_id": "2609.0002"}, card("2609.0003", message=None)]
        sync(rows, "tok", fetch=fetch, pause=0)
        assert calls == ["222"]

    def test_one_deleted_card_does_not_abort_the_batch(self):
        """A deleted message is a normal thing to find, not a reason to stop."""

        def fetch(channel, message, token):
            if message == "bad":
                raise DiscordError("message not found")
            return [Reaction("✅", "useful", 1)]

        rows = [card("2609.0001", message="bad"), card("2609.0002", message="ok")]
        out = sync(rows, "tok", fetch=fetch, pause=0)
        assert [r["arxiv_id"] for r in out] == ["2609.0002"]

    def test_unknown_emoji_are_dropped_by_the_fetcher_contract(self):
        """A party popper is not an opinion this system can act on."""
        out = sync([card("2609.0001")], "tok", fetch=lambda *a: [], pause=0)
        assert out == []


class TestStoreIntegration:
    @pytest.fixture()
    def store(self, tmp_path):
        with Store(tmp_path / "fb.db") as s:
            yield s

    def paper(self, pid, **over):
        base = {
            "arxiv_id": pid,
            "title": "A paper",
            "relevance_score": 8,
            "rank_score": 0.4,
            "discord_channel_id": "111",
            "discord_message_id": "msg-" + pid,
        }
        base.update(over)
        return base

    def test_posted_cards_only_returns_papers_with_a_message(self, store):
        store.record(self.paper("2609.0001"))
        store.record(self.paper("2609.0002", discord_message_id=None))
        assert [c["arxiv_id"] for c in store.posted_cards()] == ["2609.0001"]

    def test_message_id_is_not_erased_by_a_later_update(self, store):
        """Re-recording a paper without the id must not lose the card."""
        store.record(self.paper("2609.0001"))
        store.record(self.paper("2609.0001", discord_message_id=None, discord_channel_id=None))
        assert store.posted_cards()[0]["discord_message_id"] == "msg-2609.0001"

    def test_feedback_summary_joins_behaviour_to_prediction(self, store):
        """The join the store exists for."""
        store.record(self.paper("2609.0001", relevance_score=9))
        store.record(self.paper("2609.0002", relevance_score=3))
        store.add_feedback("2609.0001", "useful", "discord-reaction")
        store.add_feedback("2609.0002", "not-useful", "discord-reaction")

        summary = {r["signal"]: r for r in store.feedback_summary()}
        assert summary["useful"]["mean_llm_score"] == 9.0
        assert summary["not-useful"]["mean_llm_score"] == 3.0

    def test_summary_counts_a_paper_once_per_signal(self, store):
        store.record(self.paper("2609.0001"))
        store.add_feedback("2609.0001", "read")
        store.add_feedback("2609.0001", "read")
        assert next(r for r in store.feedback_summary() if r["signal"] == "read")["papers"] == 1

    def test_a_sync_result_can_be_written_straight_back(self, store):
        store.record(self.paper("2609.0001"))
        records = sync(store.posted_cards(), "tok", fetch=lambda *a: [Reaction("✅", "useful", 1)], pause=0)
        for r in records:
            store.add_feedback(r["arxiv_id"], r["signal"], r["source"])
        assert store.count()["feedback"] == 1


class TestSeeding:
    """Discord reactions are not buttons: nothing is tappable until someone
    puts an emoji there. Seeding is what turns three steps into one."""

    def test_places_every_emoji_on_every_card(self):
        from discord import SEED_EMOJI, seed

        calls = []
        done = seed(
            [card("2609.0001"), card("2609.0002", message="333")],
            "tok",
            pause=0,
            put=lambda ch, msg, e, tok: calls.append((msg, e)),
        )
        assert done == 2
        assert len(calls) == 2 * len(SEED_EMOJI)
        assert {m for m, _ in calls} == {"222", "333"}

    def test_every_seeded_emoji_maps_to_a_signal(self):
        """Seeding an emoji the sync cannot read back would be a dead button."""
        from discord import SEED_EMOJI, SIGNALS

        assert all(e in SIGNALS for e in SEED_EMOJI)
        assert {SIGNALS[e] for e in SEED_EMOJI} == {"useful", "not-useful", "read"}

    def test_skips_papers_that_were_never_posted(self):
        from discord import seed

        calls = []
        done = seed(
            [{"arxiv_id": "2609.0001"}, card("2609.0002")],
            "tok",
            pause=0,
            put=lambda *a: calls.append(a),
        )
        assert done == 1

    def test_one_failing_card_does_not_stop_the_batch(self):
        from discord import DiscordError, seed

        def put(channel, message, emoji, token):
            if message == "bad":
                raise DiscordError("message not found")

        done = seed([card("2609.0001", message="bad"), card("2609.0002")], "tok", pause=0, put=put)
        assert done == 1

    def test_emoji_are_percent_encoded_for_the_url_path(self):
        """A raw check mark in the path returns a 404 that looks like a missing
        message, which is a miserable thing to debug."""
        import urllib.parse

        from discord import SEED_EMOJI

        assert urllib.parse.quote(SEED_EMOJI[0], safe="") == "%E2%9C%85"


class TestSeedingIsNotFeedback:
    """The bot seeds the emoji itself. Counting its own taps would mark every
    card useful, not-useful and read at once - data that looks real and is not."""

    def _message(self, reactions):
        return {"reactions": reactions}

    def test_a_reaction_only_the_bot_made_is_ignored(self, monkeypatch):
        import discord

        monkeypatch.setattr(discord, "_request", lambda *a, **k: self._message(
            [{"emoji": {"name": "✅"}, "count": 1, "me": True}]
        ))
        assert discord.reactions_for("1", "2", "tok") == []

    def test_a_human_tap_on_a_seeded_emoji_counts(self, monkeypatch):
        import discord

        monkeypatch.setattr(discord, "_request", lambda *a, **k: self._message(
            [{"emoji": {"name": "✅"}, "count": 2, "me": True}]
        ))
        out = discord.reactions_for("1", "2", "tok")
        assert len(out) == 1 and out[0].count == 1

    def test_a_reaction_the_bot_never_made_counts_in_full(self, monkeypatch):
        import discord

        monkeypatch.setattr(discord, "_request", lambda *a, **k: self._message(
            [{"emoji": {"name": "👍"}, "count": 3, "me": False}]
        ))
        assert discord.reactions_for("1", "2", "tok")[0].count == 3

    def test_all_three_seeded_emoji_on_one_card_produce_nothing(self, monkeypatch):
        """The exact shape that poisoned the first real sync."""
        import discord

        monkeypatch.setattr(discord, "_request", lambda *a, **k: self._message(
            [{"emoji": {"name": e}, "count": 1, "me": True} for e in discord.SEED_EMOJI]
        ))
        assert discord.reactions_for("1", "2", "tok") == []
