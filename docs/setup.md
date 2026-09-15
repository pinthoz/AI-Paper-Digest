# Setup

About ten minutes end to end, and it costs nothing to run. Do the steps in order — the workflow will not run until the credentials exist.

Tested on n8n 1.6x (self-hosted and Cloud). **1.62 is the minimum**: earlier builds do not have the cross-execution mode of `Remove Duplicates`.

---

## Where to run n8n

This is the first decision, because it settles two things everything else depends on: whether the schedule fires unattended, and whether `$env` works in expressions.

| | On your machine | Always-free VM | n8n Cloud |
| --- | --- | --- | --- |
| Cost | free | free | trial, then paid |
| Fires on schedule, unattended | only while the machine is awake | yes | yes |
| `$env` in expressions | yes | yes | **no** — see step 6 |
| arXiv rate limit | yours alone | yours alone | shared with other tenants |

**A cron only fires while n8n is running, and n8n does not replay missed schedules.** A laptop that was asleep when the schedule fired simply has no digest that day. That single fact is usually what decides this.

### On your machine — best for building

```powershell
docker run -d --name n8n -p 5678:5678 `
  -v n8n_data:/home/node/.n8n `
  -e GENERIC_TIMEZONE=Europe/Lisbon -e TZ=Europe/Lisbon `
  -e N8N_BLOCK_ENV_ACCESS_IN_NODE=false `
  -e DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..." `
  -e TELEGRAM_CHAT_ID="123456789" `
  docker.n8n.io/n8nio/n8n
```

Then open <http://localhost:5678>.

**`N8N_BLOCK_ENV_ACCESS_IN_NODE=false` is not optional either.** Without it the `Config` node fails with *access to env vars denied* — on n8n 2.x, self-hosted, with the variable unset. Do not rely on the default being permissive; set it and move on.

**The `-v n8n_data:` volume is not optional.** Without it, recreating the container loses your credentials and — worse, because it is silent — the deduplication history, so the next run re-sends papers you have already read.

Prefer Docker over `npx n8n`: n8n targets Node 20 and 22, and a newer Node on your machine will either refuse to start or misbehave in ways that waste an afternoon.

### An always-free VM — best for actually running it

The same Docker command on a small always-on machine gets you both halves: free, and awake whenever the schedule fires. Oracle Cloud's Always Free ARM instances are the usual choice, being genuinely free rather than trial-free, with far more memory than n8n needs.

Put a reverse proxy with TLS in front before exposing it to the internet, or keep it on a private network and reach it over a tunnel. n8n holds your API keys.

### n8n Cloud

Nothing to install, and it is awake when you are not. The catch is that it blocks `$env` in expressions, so the two secrets have to be pasted into the `Config` node — see step 6, including why that means you should not export the workflow back into this repo.

### Moving between them

Export the workflow from one and import it into the other. Two things do not travel:

- **Credentials.** Recreate them; they are deliberately not part of a workflow export.
- **The deduplication history.** It lives in the instance, so the first run on a new host re-sends papers the old host had already shown you. Once, then it settles.

---
## 1. The model key (free)

<https://aistudio.google.com/apikey> → **Create API key**. That is the free tier, and it is a different thing from a Gemini Advanced / Google One AI Premium subscription — a consumer subscription does not grant API access, and an API key does not need one.

Free-tier keys are capped in **requests per minute** as well as per day. Google publishes the current limits per model and changes them, so check them against your model; the workflow's `Throttle (free tier)` node is what keeps the run underneath, and `llm_throttle_seconds` in `Config` is the dial.

One thing to know rather than discover: free-tier traffic is subject to different data-use terms than paid. Everything this workflow sends is a public arXiv abstract, so it does not matter here — it would if you pointed the same pipeline at anything private.

## 2. First-stage ranking — nothing to do

The pipeline ranks papers by relevance before spending model calls on them, using the **embeddings API on the same Gemini key** from step 1. No extra service, no extra account, no extra key.

If the embedding call fails the run continues on chronological order, so there is nothing here that can break the digest.

[`ranker/`](../ranker/) holds the same algorithm as a standalone Python service, for running a stronger local model instead. It is optional and not wired in by default.

## 3. Discord

Server settings → **Integrations → Webhooks → New Webhook**. Point it at the channel that will hold the archive (`#ai-papers` is what the docs assume), name it, copy the URL.

That URL *is* the credential — anyone holding it can post to the channel, so it goes in `.env` (step 6) and never into the workflow JSON or a screenshot. More on the embed shape and the webhook-versus-bot trade-off in [discord.md](discord.md).

## 4. Telegram

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token.
2. Send your new bot any message (a bot cannot open a conversation with you).
3. Get your chat ID:

```bash
curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates" | grep -o '"chat":{"id":[-0-9]*'
```

A personal chat ID is a positive integer; a group is negative. For a channel, add the bot as an administrator and use `@channelname` instead.

> **The number before the colon in your token is not your chat ID.** A bot token is `<bot_id>:<hash>`, so a token like `8844684439:AAEE…` belongs to bot **8844684439**. Use that as `telegram_chat_id` and the bot tries to message itself, which Telegram refuses with `403 Forbidden: bots can't send messages to bots`. Your own chat ID is a different number, and it only appears in `getUpdates` after you have messaged the bot.

---

## 5. Credentials in n8n

Two credentials, both under **Credentials → New**. Discord needs none — its webhook URL is not a credential type here, it goes in `.env`.

Create them under **Credentials → New**. Search the credential type, paste the secret, then rename it: the name is the **editable title at the top of the dialog** (it defaults to something like *Google Gemini(PaLM) account*) — click it, or the pencil beside it. There is no field labelled *Name*, which is a reliable five minutes lost the first time.

The names below are a convention, not a requirement. This export ships placeholder credential IDs, so you pick the credential in each node on import either way — matching names just makes it obvious which goes where.

| Credential type | Name to use | Holds |
| --- | --- | --- |
| Google Gemini(PaLM) Api | `Google Gemini - AI Paper Digest` | the AI Studio key from step 1 |

| Telegram API | `Telegram - Digest Bot` | the bot token from step 4 |

On import, each node shows its credential field highlighted. Pick the credential once per node and save. Two nodes need one: `Gemini 2.5 Flash` and `Send Telegram Digest`.

---

## 6. Environment variables

The `Config` node reads both of these from the environment and falls back to a visible placeholder if either is unset, which is what keeps the exported JSON free of secrets.

Copy `.env.example` to `.env`:

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/123.../abc...
TELEGRAM_CHAT_ID=123456789
```

For Docker, pass it through in your compose file:

```yaml
services:
  n8n:
    environment:
      - DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
```

**On n8n Cloud** — and on any instance with `N8N_BLOCK_ENV_ACCESS_IN_NODE=true` — expressions cannot read `$env` at all. The node does not fall back to the placeholder; it fails outright with *access to env vars denied*.

The fix is to open `Config` and paste the two values literally, switching each field from **Expression** to **Fixed** so the `{{ }}` stops being evaluated. Only `discord_webhook_url` and `telegram_chat_id` use `$env`; nothing else in the workflow does.

One consequence worth keeping in mind: the webhook URL then lives inside the workflow JSON. Do not overwrite `workflows/ai-paper-digest.json` in this repo with an export from your instance — that file is the clean template.

Restart n8n after adding the variable — it is read at process start.

---

## 7. Import

**Workflows → Import from File**, twice:

1. `workflows/error-handler.json`
2. `workflows/ai-paper-digest.json`

Order matters: the error handler has to exist as a saved workflow before the main one can point at it.

Then wire them together. Open **AI Paper Digest → ⋯ (top right) → Settings → Error Workflow** and pick **AI Paper Digest - Error Handler** from the dropdown. It lists your workflows by name — there is no ID to copy anywhere, and the `REPLACE_WITH_ERROR_HANDLER_WORKFLOW_ID` string in the JSON is only a placeholder for a value n8n assigns on import. Save.

If the dropdown is empty, the error handler was imported but never saved: open it and hit **Save** once.

Set the timezone in the same panel if you are not in `Europe/Lisbon`. The cron expression `30 17 * * 1-5` is evaluated in the workflow's timezone, not the server's.

---

## 8. Tune the profile

This is the part that decides whether the thing is useful. Open **Config** and rewrite `research_profile` to describe your own work. Be specific and include what you *don't* want — the negative half does most of the filtering:

```
Actively working on: mechanistic interpretability of transformers, attention
head analysis, probing what BERT and GPT-2 class models encode …
Not interested in: large-scale pretraining, incremental leaderboard deltas,
pure theory with no implementation …
```

Two rules that matter, because the ranker splits this text and embeds each fragment on its own:

- **Separate concepts with commas, not "and".** "image and video generation" splits into `image` and `video generation`, and a bare `image` on the negative side pushes down every multimodal paper you actually wanted. "image generation, video generation" gives two complete concepts.
- **Never put a negation inside a positive clause.** "ships systems rather than training them" is classified as negative in full, taking "ships systems" with it.

While you are in there:

| Field | Default | What it does |
| --- | --- | --- |
| `arxiv_categories` | `cs.AI, cs.LG, cs.CL, cs.CV, stat.ML` | [arXiv taxonomy](https://arxiv.org/category_taxonomy). Joined with `+` into the RSS URL. More categories, more volume. |
| `relevance_threshold` | `6` | How selective the digest is. It has to move with `max_papers_per_run`: a high threshold against a small cap gives you an empty digest most days. |
| `digest_size` | `8` | Papers in the Telegram message. Discord still gets everything above the threshold. |
| `digest_language` | `English` | Output language. The prompts stay in English regardless. |
| `max_papers_per_run` | `10` | **The free-tier budget, in papers.** How many reach the model each run. Each one costs `llm_throttle_seconds` of wall time, so 10 papers means a run of roughly 80 seconds. |
| `embedding_model` | `text-embedding-004` | Model used for relevance ranking. Same key as the chat model. |
| `ranker_negative_weight` | `0.5` | How hard the ranker pushes down papers matching your "not interested in" list. `0` disables the penalty. |
| `llm_throttle_seconds` | `8` | Seconds between model calls: 8s is 7.5/min, safely under a 10/min ceiling. `0` on a paid key. |
| `topic_vocabulary` | see node | Preferred tags, so the tag line stays consistent between runs. |

---

## 9. First run

Click **Test workflow** and let it run against the Manual Trigger.

Expect roughly: ~115 items out of `Fetch arXiv Feed`, ~64 after `New and Cross-Lists Only`, exactly `max_papers_per_run` after `Cap Papers Per Run`, the same count after `Unseen Papers Only` on the very first run, then one Discord card per paper at or above the threshold, and one Telegram message. At the default cap the run takes a bit over 80 seconds, almost all of it the deliberate throttle.

A Telegram message arrives **even when nothing clears the threshold** — it says so, and gives the counts. Silence means something is wrong; it is never the normal quiet-day outcome.

Three things worth checking on that first run:

- **`Unseen Papers Only` outputs zero on the second manual run.** That is correct — it means cross-execution dedup is live. Use the node's *Clear history* option if you want to replay.
- **The Discord cards render as embeds**, with a coloured stripe and named fields — not as a wall of plain text. Plain text means the embed JSON did not reach the node; check `Build Paper Embed`'s output for `embed_json`.
- **The Telegram message renders formatting.** Raw `<b>` tags mean `parse_mode` did not survive the import; re-select `HTML` in the node's additional fields.

Once it looks right, **Activate** the workflow. The schedule takes over from the next weekday at 17:30.

---

## Swapping the model

The chain is provider-agnostic — only the sub-node attached to `Analyse Paper` is Gemini-specific.

Delete the `Gemini 2.5 Flash` node, drag in **OpenAI Chat Model**, **Anthropic Chat Model**, **Ollama Chat Model** or any other, and connect it to the chain's *Model* input. Nothing else changes; the structured output parser sits on the chain, not on the model.

The model dropdown lists what your key actually has access to, so pick from it rather than trusting the name in the export — Google renames and retires model IDs on its own schedule. A `flash`-class model is the right size for this task; a `lite` variant is cheaper still and usually adequate, and a `pro` model is paying for reasoning this workflow does not use.

Two caveats if you go local: a model that cannot reliably emit JSON will hit the parser's retry and double your call count, and the prompt assumes a context window that comfortably holds an abstract plus the profile — roughly 2k tokens.

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Discord returns `400` with no detail | An embed limit was exceeded. `Build Paper Embed` caps all of them, so this means the node was edited — compare against the table in [discord.md](discord.md#limits). |
| Cards post as plain text, no embed | The `embeds` field is empty or the JSON is malformed. Inspect `embed_json` in the previous node's output. |
| `401` / `404` from Discord | The webhook was deleted or regenerated in server settings. Update `DISCORD_WEBHOOK_URL` and restart n8n. |
| Discord posts but pings nobody expected | Working as intended: `allowed_mentions: { parse: [] }` blocks `@everyone` arriving in a paper title. |
| `Bad Request: can't parse entities` | An unescaped character reached Telegram. `Build Telegram Digest` escapes `&`, `<`, `>`; if you edited the template, keep `esc()` around every model-generated string. |
| `403 Forbidden` from Telegram | Almost always `telegram_chat_id` holding the **bot's** id — the digits before the colon in the token — instead of yours. Telegram refuses bot-to-bot messages. Get the real id from `getUpdates` after messaging the bot; it will be a different number. Can also mean you blocked the bot. |
| `401 Unauthorized` from Telegram | The token is wrong or was revoked in BotFather. Paste the current one into the credential. |
| `chat not found` | You never sent the bot a message, or the chat ID has a typo. |
| `429` / "Rate exceeded." from arXiv | The workflow uses `rss.arxiv.org`, which is far more forgiving than the query API, but it is still per IP — and on n8n Cloud that IP is shared with other tenants. Wait a few minutes; no `Retry-After` is sent. While iterating downstream, **pin the output of `Fetch arXiv Feed`** (open the node, run once, click the pin icon) so n8n replays the data instead of re-fetching. |
| Zero items after `New and Cross-Lists Only` | The feed held only revisions — rare, but possible on a quiet day. Nothing is wrong. |
| The loop never finishes | A node between `Relevant Enough?` and `Loop Over Papers` was replaced with a Filter, or its return edge to the loop is missing. |
| `Model output doesn't fit required format` | Read the raw model output in the failed node. **If the JSON stops mid-string, it is truncation** — raise `maxOutputTokens` on the model node; a thinking model spends that budget on reasoning before it writes anything. If the JSON is complete but a value is out of range, someone tightened the schema: it ships with types only, and value constraints belong in `Merge Analysis`. The `"output"` wrapper around the JSON is n8n's own convention and is correct. |
| Retries seem not to wait as long as documented | They do not. n8n caps **Wait Between Tries at 5000ms**, whatever the JSON says. Real spacing comes from the `Throttle (free tier)` node, not from the retry. |
| `429` / `RESOURCE_EXHAUSTED` from Gemini | Too many requests. First lower `max_papers_per_run`, which is the direct control; then raise `llm_throttle_seconds` if the failures come in bursts within a single run. Repeated 429s on the very first call of a run means the *daily* cap is spent, and only waiting or a paid key fixes that. |
| `access to env vars denied` in `Config` | Self-hosted: set `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` on the container and recreate it. n8n 2.x denies env access even with the variable unset, so setting it explicitly is the fix. n8n Cloud: you cannot change it — paste the two values literally instead, per step 6. |
| `⚠ ranked by recency` in the digest header | The rank call did not succeed, or returned the wrong number of scores. The run continues on chronological order rather than failing; open `Rank via Service` in the execution for the reason. The warning is deliberate — an earlier version degraded silently and nobody noticed the pre-filter had stopped running. |
| `Rank via Service` returns `401` | `RANKER_TOKEN` differs between the ranker container and n8n. It is sent as the `X-Ranker-Token` header; both sides read the same variable from `.env`, so a mismatch usually means one container was not recreated after the file changed. |
| `Rank via Service` cannot connect | Both endpoints are derived from one variable: `STORE_URL`, plus `/rank` or `/store`. From inside n8n the host is the compose service name — `http://ranker:8000` — never `localhost`, which there means the n8n container itself. An empty `STORE_URL` disables the store *and* the ranking, and the digest arrives ordered by recency. |
| Run takes a while | Expected: `max_papers_per_run` × `llm_throttle_seconds` of deliberate waiting, plus the calls. It runs on a schedule and nobody is waiting on it. |
| Digest arrives empty most days | `relevance_threshold` is too high for the number of papers the model sees, or `research_profile` does not describe your actual work. Open `Merge Analysis` in a finished execution and read `relevance_reason` — it says in one sentence why each paper scored what it did, which settles the question immediately. |
| Dashboard raises `AttributeError` on something you just added to `theme.py` | Streamlit re-runs `dashboard.py` on every save but keeps already-imported local modules in `sys.modules`, so the page runs new chart code against the old `Theme`. Testing it with `docker compose exec dashboard python …` will not reproduce it — a fresh interpreter imports the current file. Run `docker compose restart dashboard`. Editing `dashboard.py` alone never needs this. |
