# Architecture

A node-by-node walkthrough of `workflows/ai-paper-digest.json`, with the data shape at each hop. Read this if you want to modify the pipeline rather than just run it.

The workflow is one graph in four stages: **ingest → analyse → store → deliver**. Stages 1 and 4 run once per execution; stages 2 and 3 run once per paper.

---

## 1 · Ingest

### `Schedule 16:30 (Mon-Fri)` / `Run Manually`

`30 16 * * 1-5`, evaluated in the workflow timezone — not the server's, which is why `GENERIC_TIMEZONE` is set on the container as well. The arXiv RSS feed is rebuilt around 04:00 UTC each weekday, so any run after that sees a complete batch; the hour itself is just when you want to read it. The Manual Trigger shares the same downstream path, so a test run and a scheduled run are the same code path — worth keeping, because a "test-only" branch is a branch that rots.

### `Config`

A Set node holding 12 fields. Every downstream node reads it through `$('Config').first().json.<field>`, which means there is exactly one place to change behaviour and no configuration hidden inside a Code node three screens to the right.

Two of those fields are secrets, and both read the environment:

```
{{ $env.DISCORD_WEBHOOK_URL || 'PASTE_DISCORD_WEBHOOK_URL' }}
{{ $env.TELEGRAM_CHAT_ID    || 'PASTE_TELEGRAM_CHAT_ID' }}
```

So this export contains nothing identifying. The Gemini key is not here either — both nodes that need it use the n8n credential.

**The `||` fallback is decoration, not a safety net**, and it is worth being clear about why. When an instance blocks env access in expressions — n8n Cloud always does, self-hosted does under `N8N_BLOCK_ENV_ACCESS_IN_NODE=true` — the expression does not evaluate to the right-hand side. It raises *access to env vars denied* and the node fails. On such an instance the two values have to be pasted literally, with the fields switched from Expression to Fixed.

### `Build arXiv Query`

arXiv's `search_query` is a boolean expression over field prefixes, not a list, so the comma-separated config value has to be expanded:

```
"cs.AI, cs.LG"  →  "cat:cs.AI OR cat:cs.LG"
```

Throws on an empty category list rather than fetching the whole of arXiv.

### `Fetch arXiv Feed`

`GET https://rss.arxiv.org/rss/<categories joined with +>`. Response format is forced to **text** — left on autodetect, n8n sees `application/rss+xml` and hands back a binary buffer that the XML node cannot read.

**Why RSS and not `export.arxiv.org/api/query`.** The feed is the day's announcement list, which is exactly the question this workflow asks, so there is no date window to tune and no paging. It also carries an `announce_type` per item that the query API does not expose. And it is a separate service with separate rate limits: the query API throttles per IP, which on n8n Cloud is shared with every other tenant, so its `429` can be caused by traffic that is not yours and cannot be avoided by backing off.

The trade is that `pubDate` is the announcement timestamp, identical for every item in a feed, rather than the submission date. Nothing downstream needs submission order, and the deduplicator handles repetition.

One request per execution. Three retries five seconds apart, and a `User-Agent` that identifies the client, because arXiv asks callers to.

A scheduled run makes one request a day and never comes close to any limit. The way to get throttled is re-running *Test workflow* while building, and the n8n-native answer is to **pin** this node's output while iterating downstream — it replays the last response and stops calling arXiv at all.

### `Parse Atom Feed`

XML → JSON with `explicitArray: false` and `mergeAttrs: true`. Attributes get flattened into their parent rather than sitting under a nested `$` key.

The cost of `explicitArray: false` is that a feed with exactly one entry yields an object where every other feed yields an array. That is handled in the next node.

### `Normalise Papers`

One input item, N output items. A Code node rather than a Split Out node, precisely because of that single-entry case — Split Out on a non-array field errors, and this pipeline runs unattended.

Two RSS-specific shapes need handling. `mergeAttrs` folds attributes into their parent, so an element carrying both attributes and text — `<guid isPermaLink="false">oai:…</guid>` — arrives as `{ isPermaLink: 'false', _: 'oai:…' }` rather than a string. And the abstract is buried in `description` behind an `arXiv:2609.11955v1 Announce Type: new` preamble that has to be stripped, or every prompt and every Discord card would open with it.

Output shape, one item per paper:

```jsonc
{
  "arxiv_id": "2609.11955",          // version stripped — the stable key
  "version": "v1",
  "announce_type": "new",             // new | cross | replace | replace-cross
  "title": "…",                       // whitespace collapsed
  "abstract": "…",                    // the "arXiv:… Announce Type:" prefix removed
  "authors": ["…"],
  "authors_line": "A, B, C et al.",
  "primary_category": "cs.CL",
  "categories": ["cs.CL", "cs.LG"],
  "published": "2026-09-14T04:00:00.000Z",   // announcement, not submission
  "abs_url": "https://arxiv.org/abs/2609.11955",
  "pdf_url": "https://arxiv.org/pdf/2609.11955",
  "comment": "",                      // not carried by RSS; kept for shape
  "journal_ref": ""
}
```

### `New and Cross-Lists Only`

Filter: `announce_type` does not start with `replace`.

arXiv announces four kinds of item, and a daily feed of 115 breaks down roughly as **44 new, 20 cross, 35 replace, 16 replace-cross**. The last two are revisions of papers announced earlier — usually a typo fix or a camera-ready — and they are not news. Dropping them removes about 45% of the feed before anything is spent on it.

**`notStartsWith`, not `notEquals`.** `replace-cross` is a distinct fourth value, and an equality test against `"replace"` lets all sixteen of them through while appearing to work. That is exactly the shape of bug that survives code review and is caught only by running the parser over a real feed and printing the value counts.

### `Split Profile` → `Embed with Gemini` → `Rank by Similarity`

The first stage of a two-stage ranking, and the reason the papers that reach the chat model are the most relevant rather than the most recent.

**`Split Profile`** parses the profile into wants and don't-wants. Embeddings do not represent negation — *"not interested in reinforcement learning for games"* embeds close to *"interested in reinforcement learning for games"*, because the content words dominate — so a single profile vector would faithfully attract everything the reader asked to avoid. The halves are embedded separately and the negative one is subtracted.

Classification is per fragment, not per line: *"Ships production systems; does not run large-scale pretraining"* is a want and a don't-want in one sentence.

**One batch call** carries every concept and every abstract. `batchEmbedContents` caps a request at 100 texts, so concepts go first — without them nothing can be scored at all — and papers take what is left, with any overflow keeping its chronological place further down rather than disappearing.

The scoring:

    score = max cosine(paper, any positive concept)
          - w * max cosine(paper, any negative concept)

Max rather than mean on both sides: a paper is relevant because it matches one interest strongly, not because it is vaguely near the average of all of them.

**The HTTP node continues on error.** It authenticates with the same `googlePalmApi` credential as the chat model, so the key never enters the workflow JSON.

**`Rank by Similarity` is where the graceful degradation lives.** It re-sorts the papers when the response is usable, and otherwise returns them in the feed's own chronological order, stamped `ranked_by: 'recency'` so the provenance shows in the execution.

It insists the response holds *exactly* the expected number of vectors. A short response could still be sliced, but the concept and paper vectors would be misaligned and every score silently wrong — which is worse than not ranking at all. Papers that overflowed the batch keep their place at the back rather than being dropped.

The ranking is an optimisation, not a dependency: every failure path ends in a working digest.

### `Cap Papers Per Run`

A `Limit` node keeping the first `max_papers_per_run` items. The feed arrives newest-first and every node since has preserved that order, so this is the N most recent submissions.

**It sits before the deduplicator, not after, and that ordering is the entire point.** The deduplicator records every item it sees, permanently. Cap after it and the papers past the cap are marked seen without ever being analysed — they are gone, silently, and raising the cap next week does not bring them back. Cap before it and the surplus is simply not considered on this run.

With the ranker in front of it, this takes the N most *relevant* papers rather than the N newest — which is the entire reason the ranker exists. Without it (or when it is unreachable) the fallback is recency, which at a small cap is closer to sampling than to selection.

### `Unseen Papers Only`

`Remove Duplicates` in **Remove Items Seen in Previous Executions** mode, keyed on `arxiv_id`, history 10,000.

This is the node that makes the workflow economical. arXiv re-lists cross-posted papers for days; without persistent dedup a daily job re-summarises the same paper four or five times and pays for it each time. History is stored by n8n outside the execution, so it survives restarts.

Its *Clear history* option is the replay button during development.

---

## 2 · Analyse

### `Loop Over Papers`

`Split In Batches`, batch size 1, `reset: false`.

- **output 0 (`done`)** → the digest branch. Fires once, after the last iteration.
- **output 1 (`loop`)** → the analysis branch, once per paper.

Batch size 1 is a blast-radius decision: a failure costs one API call instead of the whole batch, and it pins concurrency at one, which keeps every rate limit involved comfortably satisfied.

Everything routed back into this node's input accumulates and is emitted on `done` — which is why both branches of the threshold gate return here, and why the digest branch can filter the full set afterwards.

### `Throttle (free tier)`

A `Wait` node between the loop and the model, set from `llm_throttle_seconds`.

Gemini free-tier keys are capped in requests **per minute**, not only per day. Eight seconds per iteration holds the run at 7.5 requests a minute, comfortably under a 10/min ceiling. At the default cap of 10 papers the loop therefore spends about 80 seconds waiting — unattended, while you are doing something else, which is the whole point of doing this on a schedule.

This is the second of two defences and the weaker one. `Cap Papers Per Run` decides *how many* calls happen at all; the throttle only decides how fast. If you are seeing `429`s, lower the cap first.

Set it to `0` on a paid key. Waits under 65 seconds are handled in-process rather than by suspending the execution, so this costs wall time and nothing else.

Throttling rather than relying on the retry is deliberate, and n8n settles the argument anyway: **Wait Between Tries is capped at 5000ms**, so a retry cannot back off far enough to clear a per-minute quota on its own. A retry after a `429` has also already spent a request against that quota, so on a long run the failures compound instead of clearing. Spacing the calls in the first place is the only thing that actually works.

### `Analyse Paper` + `Gemini 2.5 Flash` + `Analysis Schema`

A Basic LLM Chain with two sub-nodes attached.

The **system prompt** interpolates `research_profile` and `topic_vocabulary` from `Config` and carries an anchored rubric (what a 3, 5, 7 and 9 mean), plus hard rules against inventing numbers the abstract does not state. The **user prompt** is the paper: title, IDs, categories, authors, the author comment, and the abstract in a delimited block.

The **structured output parser** enforces this schema:

| Field | Schema type | Constrained to | Enforced where |
| --- | --- | --- | --- |
| `relevance_score` | number | whole number 0–10 | `Merge Analysis` |
| `relevance_reason` | string | one sentence | prompt |
| `tldr` | string | ≤30 words | prompt, clipped in the digest |
| `why_it_matters` | string | ≤2 sentences | prompt, clipped in the digest |
| `key_contributions` | string[] | 1–4 entries | `Merge Analysis` |
| `method` / `results` / `limitations` | string | non-empty | `Merge Analysis` |
| `topics` | string[] | 2–5 slugs | `Merge Analysis` |
| `paper_type` | string | 7 allowed values | `Merge Analysis` |
| `reading_priority` | string | 4 allowed values | `Merge Analysis` |

**The schema carries types and required keys, and nothing else.** That is the correction to an obvious first instinct: a strict schema — `enum`, `minItems`, `minimum`, `additionalProperties` — is worse than a loose one here. Gemini supports a subset of JSON Schema and silently drops those keywords, but the parser still validates the response against them, so the run fails on output the model was never asked to constrain. `Model output doesn't fit required format` is that mismatch, not a bad model.

Ranges therefore live in the field descriptions and the prompt, where the model reads them, and enforcement lives one node downstream, where it is deterministic and survives swapping the provider. The parser still earns its place: it guarantees `topics` is an array rather than a comma-separated string, which is the failure the code cannot cheaply recover from.

Temperature 0.2: this is a classification task with a fixed output shape, not a writing task.

**`maxOutputTokens` is 8192, not the ~700 the answer needs.** Gemini 2.5 Flash reasons before it answers, and those thinking tokens are billed against the same output budget. A limit sized for the visible JSON gets spent on reasoning and the response is cut off mid-string — which surfaces as a parser error, not as a truncation warning, and sends you hunting through the schema for a problem that is not there.

**On failure the paper is skipped, not the run.** After three retries the chain's *error output* routes to `Skipped` and back into the loop. One abstract the model cannot handle should not cost the other forty-four, and a digest that arrives short is better than a digest that does not arrive.

A flash-class model is the right size for it. Scoring an abstract against a profile and reformatting it needs no long-horizon reasoning, and the schema does the structural work that a larger model would otherwise be paid to infer.

### `Merge Analysis`

Runs once per item. Re-attaches the paper metadata to the model output and normalises the two fields that get rendered:

- **`topics`** are slugged (`lowercase`, non-alphanumerics → `-`), deduplicated, capped at 5. Across three runs the same concept comes back as `RAG`, `rag ` and `Rag`, which makes the tag line useless for searching. Doing it here rather than in the prompt makes it deterministic — and it is the kind of thing a prompt will get right 95% of the time, which is the worst possible hit rate for something you stop checking.
- **`relevance_score`** is coerced and clamped to 0–10. The schema constrains it, but defence in depth costs one line.

Also emits `contributions_text` (dash-prefixed lines) for the Discord embed field.

### `Relevant Enough?`

IF: `relevance_score >= relevance_threshold`.

**An IF, not a Filter.** A Filter drops non-matching items, and a loop iteration whose branch produces no items does not return control to `Split In Batches` — the run stalls. Here both branches lead back to the loop: the true branch through Discord, the false branch through a No-Op. The failure mode a Filter introduces only appears on a day when nothing is relevant, which is exactly the day nobody is watching.

---

## 3 · Store

### `Build Paper Embed`

Assembles the Discord embed and returns it as a JSON string on `embed_json`, plus the tag line on `discord_content`.

This is a Code node rather than the Discord node's field-by-field embed UI for the same reason the digest is: **the limits have to be enforced, not hoped for.** Discord rejects the whole message with a `400` if any single one is exceeded, and a webhook gives back no error body worth reading.

| Limit | Value | Handling |
| --- | --- | --- |
| Title | 256 | truncated with an ellipsis |
| Description | 4096 | truncated |
| Field value | 1024 | truncated |
| Fields | 25 | sliced; the workflow emits 8 |
| **Whole embed** | **6000** | longest field halved until it fits |

The 6000 budget is the one that catches people: it counts title, description, author, footer and every field name and value *together*, so four individually legal fields can still fail as a set. The loop halves the longest field rather than dropping fields, because dropping would silently lose a whole section — the reader would never know `Limitations` had been there.

Verified against live arXiv data including a pathological case where every field arrives at 9000 characters: it converges in about a millisecond and lands at 5961.

Two smaller decisions. The stripe colour encodes the score, so the channel is scannable before you read a number. And tags go in the message **content** rather than inside the embed, because that is the part Discord surfaces in the channel preview and in notifications.

### `Discord - Post Paper Card`

Webhook authentication: the URL is the entire credential. The workflow only ever posts, so a bot would buy nothing and cost an application, a token, an invite and a permissions model. Switching to Bot auth is a one-field change if threads or reactions ever become worth it — the embed payload is identical.

Three retries. Discord's webhook ceiling is roughly 5 requests per 2 seconds and each iteration sits behind a multi-second LLM call, so no `Wait` node is needed; that stops being true if the batch size is ever raised.

### `Collect Result`

Runs once for all items, and returns exactly one.

The Discord node's response replaces the item, so the paper has to be put back before it returns to the loop and, from there, into the digest. Because the batch size is 1, `.first()` on `Merge Analysis` is exact rather than a heuristic — worth stating plainly, because this node stops being correct the moment someone raises that batch size.

### `Skipped`

A No-Op whose only job is to close the iteration. It receives two kinds of paper: those that scored below the threshold, and those whose analysis failed after every retry.

Visible on the canvas rather than implicit, so the return path is obvious to anyone reading the graph. Both kinds return to the loop and are filtered out of the digest afterwards — the score filter drops the low scorers, and `Build Telegram Digest` additionally checks for an `arxiv_id` and a numeric score, because an n8n error object has neither and a malformed entry in the digest is the one failure nobody would notice.

---

## 4 · Deliver

Reached from the loop's `done` output, once, after every paper is filed.

### `Kept Papers Only` → `Rank by Relevance` → `Top N`

`done` emits everything that came back through the loop, kept and skipped alike, so the first filter re-applies the threshold. Then sort by `relevance_score` descending and take `digest_size`.

Discord gets everything above the threshold; Telegram gets the top slice. A digest nobody finishes is a digest nobody opens.

### `Build Telegram Digest`

Renders the message and returns **one item per chunk**.

Two constraints drive the code. Telegram rejects any message over 4096 characters, so entries are packed into ≤3500-character chunks and split only on paper boundaries — cutting mid-entry would also cut an HTML tag in half. And Telegram's HTML parse mode is strict: one unescaped `<` in a paper title returns a `400` and the whole digest is lost, so every model-generated string passes through `esc()`.

Title, tldr and *why it matters* are additionally clipped **before** the entry is assembled, never after — truncating the finished string would slice through a tag. The caps only bite when a model ignores its "max 30 words" instruction, but without them one run-on field produces a single entry the chunker cannot place anywhere, and the whole digest fails.

Screened/kept counts come from `$('Unseen Papers Only').all().length`, reaching back across the loop rather than being threaded through it.

### `Send Telegram Digest`

One call per chunk, `parse_mode: HTML`, previews disabled, n8n attribution off. Three retries.

---

## Error handling

`workflows/error-handler.json` is a separate workflow registered under **Workflow settings → Error workflow**. n8n invokes it once per failed execution with the failing node, the error message and a deep link, and it forwards all three to the same Telegram chat.

Retry and error policy per node:

| Node | Policy | Why |
| --- | --- | --- |
| `Fetch arXiv Feed` | 3 retries, 5s | arXiv throttles bursts. |
| `Analyse Paper` | 3 retries, then skip the paper | After the retries the error output closes the iteration rather than failing the run. |
| `Discord - Post Paper Card` | 3 retries, 2s | Transient 5xx and rate-limit bounces. |

| `Send Telegram Digest` | 3 retries, 2s | Network. |

Execution data is saved on both success and failure, and `saveExecutionProgress` is on — so a run that dies at paper 34 shows exactly which paper and why.

---

## Extending it

Places the design deliberately leaves open:

- **A second source.** Papers With Code, Hugging Face Papers or a Semantic Scholar feed can join the graph immediately after `Normalise Papers`, provided they emit the same item shape. Everything downstream is source-agnostic.
- **Embedding-based filtering.** A cheap pre-filter before the LLM — embed the abstract, cosine against a profile vector, drop the bottom half — would cut the LLM call volume substantially at the cost of a vector store.
- **Weekly synthesis.** A separate workflow reading the week back out of the Discord channel and asking for themes across papers rather than per paper.
- **Reading the loop closed.** Reactions on the cards are the cheapest signal of what actually got read. Collecting them needs Bot auth, and feeding them back into the profile is the step that would make the scoring improve on its own.
