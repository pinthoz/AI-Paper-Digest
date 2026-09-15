# Discord

One rich embed per paper in a channel that becomes the archive. This document covers the webhook, the shape of the embed, and the limits the workflow enforces before it makes the request.

## Creating the webhook

Server settings → **Integrations → Webhooks → New Webhook**. Pick the channel (`#ai-papers`), name it, copy the URL.

That URL is the whole credential. It goes in `.env` as `DISCORD_WEBHOOK_URL`, and the `Config` node reads it from there — not into a credential, and never into the workflow JSON.

Anyone holding the URL can post to that channel. `.env` is gitignored; the workflow JSON is not, which is the whole reason the `Config` node reads it through `$env` instead of storing it inline.

## Why a webhook and not a bot

| | Webhook | Bot |
| --- | --- | --- |
| Setup | One URL | Application, token, invite, scopes, permissions |
| Node needed | HTTP Request | Discord node |
| Connection | Plain HTTPS POST | Gateway connection |
| Can post embeds | yes | yes |
| Can create threads, react, read history | no | yes |

The workflow only ever posts, so the bot buys nothing and costs a permissions model to maintain.

The post itself is an **HTTP Request** node rather than the Discord node. A webhook is a documented plain `POST` with a JSON body and no auth header, the payload has to be assembled in code anyway — the Discord node's embed UI is a fixed collection and cannot express a variable number of fields — and going direct removes a layer whose parameter names change between node versions. If you later want threads or reactions, that is the point to bring in the Discord node with Bot auth; the embed payload itself is unchanged.

The body is exactly:

```jsonc
{
  "content": "`rag` `agents`",        // tags, shown in the channel preview
  "embeds": [ { /* the card */ } ],     // max 10 per message; this sends 1
  "allowed_mentions": { "parse": [] }   // see below
}
```

`allowed_mentions` is not politeness. arXiv titles are not a trusted input, and one containing `@everyone` would otherwise ping the entire server.

## Anatomy of a card

```
┌────────────────────────────────────────────────┐
│ Zheng-Hui Huang, Guixu Lin, …          ← author │
│ Programmable World Model               ← title, links to /abs/
│                                                │
│ Introduces a programmable world model that     │  ← tldr
│ accepts explicit physical constraints…         │
│ Constraint-conditioned generation is the       │  ← why_it_matters, italic
│ missing piece for using world models as…       │
│                                                │
│ Key contributions   Method                     │
│ Results             Limitations                │  ← fields
│ Type   Priority   Categories                   │  ← inline fields
│ Links  abstract · pdf                          │
│                                                │
│ 2609.10540 v1 · cs.CV · scored 8/10   ← footer │
└────────────────────────────────────────────────┘
  `world-models` `video-understanding` `evaluation`   ← message content
```

The left stripe is colour-coded, so the channel is scannable before you read a single score:

| Score | Colour |
| --- | --- |
| 9–10 | red `#d94f4f` |
| 8 | amber `#e0913e` |
| 7 | blue `#3f7fd9` |
| below threshold | grey `#6b7280` (never posted, kept for manual runs) |

Tags sit in the **message content** rather than inside the embed, because that is the part Discord shows in the channel list preview and in notifications.

A field the model left empty — `results` on a paper whose abstract states no numbers — arrives as *"Not stated in the abstract."*, filled in by `Merge Analysis`. That is deliberate: an absent field is ambiguous, since the reader cannot tell whether the abstract was silent or the model failed. Genuinely empty values, like an empty category list, are still dropped rather than posted blank.

## Limits

Discord rejects the entire message with a `400` if any single limit is exceeded, and a webhook returns no useful error body to debug it with. `Build Paper Embed` enforces all of them before the request leaves n8n:

| Limit | Value | Handling |
| --- | --- | --- |
| Embed title | 256 | truncated with an ellipsis |
| Embed description | 4096 | truncated |
| Field value | 1024 | truncated |
| Fields per embed | 25 | sliced (the workflow emits 8) |
| **Total across title, description, author, footer and every field** | **6000** | longest field halved repeatedly until it fits |
| Message content | 2000 | tags only; nowhere near it |

The 6000 cap is the one that catches people: it is measured across the whole embed, not per field, so four fields that are each legal can still fail together. The trimming loop halves the longest field rather than dropping fields, which would silently lose a whole section.

Verified against live arXiv data, including a case where every field arrives at 9000 characters — the loop converges in about a millisecond and lands at 5961 of 6000.

## Rate limits

Discord allows roughly **5 requests per 2 seconds per webhook**. The workflow posts one paper per loop iteration, and each iteration is gated behind an LLM call of several seconds, so it sits far below the ceiling and needs no `Wait` node. If you ever remove the LLM step or raise the batch size, that stops being true.

## What Discord is not

It is a good log and a fine search surface. It is not a database: you cannot sort a channel by relevance score, filter to unread, or mark a paper as read. If that turns out to matter, the honest fix is to add a store rather than to fight Discord — the workflow already produces the structured record that would go into one.

Ranking is what the Telegram digest is for.
