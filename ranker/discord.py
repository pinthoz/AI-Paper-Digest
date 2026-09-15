"""Reading reactions back off the Discord cards.

The pipeline scores papers; this is how it finds out whether those scores were
any good. A reaction is the cheapest feedback that exists — the reader is
already looking at the card, and it costs one tap.

That cheapness is the whole argument. A feedback mechanism nobody uses produces
no data, and every design that asks the reader to go somewhere else to record
an opinion is a design that collects nothing.

Two Discord facts make this possible:

  * A webhook returns the created message when the URL carries `?wait=true`.
    Without it the response is an empty 204 and the card is unfindable
    afterwards — there is no "list my webhook's messages" endpoint.
  * `GET /channels/{id}/messages/{id}` returns every reaction with its count in
    one call, so syncing a week of papers is one request per card rather than
    one per card per emoji.

Reading messages needs a **bot token**, not the webhook. Posting and reading are
different permissions, which is why the workflow can stay webhook-only and this
module is the one place that needs more.
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

log = logging.getLogger("ranker.discord")

API = "https://discord.com/api/v10"

# What a tap means. Deliberately forgiving: people reach for whichever tick or
# cross their client offers first, and an unrecognised emoji is ignored rather
# than guessed at.
SIGNALS: dict[str, str] = {
    "✅": "useful",
    "👍": "useful",
    "⭐": "useful",
    "❌": "not-useful",
    "👎": "not-useful",
    "👀": "read",
    "📖": "read",
    "🗑️": "archived",
    "🗑": "archived",
}


@dataclass(frozen=True)
class Reaction:
    emoji: str
    signal: str
    count: int


class DiscordError(RuntimeError):
    pass


def _request(url: str, token: str, method: str = "GET", timeout: float = 10.0) -> dict:
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Authorization": f"Bot {token}",
            # Discord asks bots to identify themselves and will rate-limit
            # harder if they do not.
            "User-Agent": "ai-paper-digest (+https://github.com/pinthoz/AI-Paper-Digest, 1.0)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            import json

            body = r.read()
            # PUT /reactions answers 204 with no body.
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise DiscordError("rate limited by Discord; slow the sync down") from exc
        if exc.code in (401, 403):
            raise DiscordError(
                "Discord refused the bot token. The bot needs View Channel and Read "
                "Message History to read reactions, plus Add Reactions to place "
                "them. Posting through a webhook grants none of the three."
            ) from exc
        if exc.code == 404:
            raise DiscordError("message not found - deleted, or the bot cannot see the channel") from exc
        raise DiscordError(f"Discord returned {exc.code}") from exc


# What the bot puts on each card so the reader has something to tap. Three is
# deliberate: a longer menu is a decision, and a decision is the thing that
# stops people from giving feedback at all.
SEED_EMOJI = ("✅", "❌", "👀")


def add_reaction(channel_id: str, message_id: str, emoji: str, token: str) -> None:
    """Put one emoji on a card, as the bot.

    The emoji goes in the URL path and has to be percent-encoded - a raw
    check mark there produces a 404 that looks like a missing message.

    Idempotent by nature: adding a reaction the bot already added is a no-op,
    so re-seeding a card costs a request and changes nothing.
    """
    quoted = urllib.parse.quote(emoji, safe="")
    url = f"{API}/channels/{channel_id}/messages/{message_id}/reactions/{quoted}/@me"
    _request(url, token, method="PUT")


def seed(rows: list[dict], token: str, *, emoji=SEED_EMOJI, pause: float = 0.25, put=add_reaction) -> int:
    """Pre-place the reaction menu on a batch of cards.

    Needs ADD_REACTIONS on top of the read permissions. One card that fails -
    deleted, or in a channel the bot lost access to - does not stop the rest.
    """
    done = 0
    for i, row in enumerate(rows):
        channel, message = row.get("discord_channel_id"), row.get("discord_message_id")
        if not channel or not message:
            continue
        try:
            for e in emoji:
                put(str(channel), str(message), e, token)
                if pause:
                    time.sleep(pause)
            done += 1
        except DiscordError as exc:
            log.warning("could not seed %s: %s", row.get("arxiv_id"), exc)
    return done


def reactions_for(channel_id: str, message_id: str, token: str) -> list[Reaction]:
    """Every reaction on one card, mapped to a signal.

    Emoji the vocabulary does not know are dropped: someone reacting with a
    party popper is not expressing an opinion this system can act on, and
    inventing a meaning for it would put noise in the training signal.
    """
    msg = _request(f"{API}/channels/{channel_id}/messages/{message_id}", token)
    out = []
    for r in msg.get("reactions", []):
        name = (r.get("emoji") or {}).get("name") or ""
        signal = SIGNALS.get(name)
        if not signal:
            continue

        # Discord reports a total count, and the bot seeded these emoji itself.
        # Counting its own tap as feedback marks every card useful, not-useful
        # and read at once - which is not a weak signal, it is a wrong one that
        # looks like data. "me" is true when the authenticated bot reacted.
        count = int(r.get("count", 0)) - (1 if r.get("me") else 0)
        if count > 0:
            out.append(Reaction(emoji=name, signal=signal, count=count))
    return out


def sync(
    rows: list[dict],
    token: str,
    *,
    fetch=reactions_for,
    pause: float = 0.25,
) -> list[dict]:
    """Collect reactions for a batch of posted cards.

    `rows` carry `arxiv_id`, `discord_channel_id` and `discord_message_id`.
    Returns one record per (paper, signal) found.

    A quarter-second between calls keeps a week's worth of cards well under
    Discord's limits without needing a rate-limit parser. One card that fails
    does not stop the sync: a deleted message is a normal thing to find, and
    the rest of the batch is still worth collecting.
    """
    found: list[dict] = []
    for i, row in enumerate(rows):
        channel, message = row.get("discord_channel_id"), row.get("discord_message_id")
        if not channel or not message:
            continue
        try:
            for r in fetch(str(channel), str(message), token):
                found.append({"arxiv_id": row["arxiv_id"], "signal": r.signal, "source": "discord-reaction"})
        except DiscordError as exc:
            log.warning("skipping %s: %s", row.get("arxiv_id"), exc)
        if pause and i < len(rows) - 1:
            time.sleep(pause)
    return found
