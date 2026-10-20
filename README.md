# News provenance tracer (hackathon prototype)

Given a statement from a news story, trace it back to the first report that carried it, and
let a self-improvement loop tune the tracer against a benchmark whose answers are known.

All LLM calls run through headless Claude Code (`claude -p`) on a Claude subscription, and are
cached in `data/cache/llm.sqlite`, so re-runs are free.

## Demo

**[docs/demo/demo.html](docs/demo/demo.html)**: a self-contained offline page, with no server and
no model calls. It has 80 precomputed traces from the held-out sets (40 quotes and 40 news events).
Search or pick one to see the first report the tracer found, the benchmark's answer and how the
story spread. Download it and open it in a browser, since GitHub shows HTML files as source. The
longer write-up, covering the improvement loop's history and the scores, is
[docs/demo/index.html](docs/demo/index.html). See [Running the demo](#running-the-demo) to rebuild them.

## Presentation

*Who said it first?* is a 10-slide deck covering the problem, the data, the tracer, the
self-improvement loop (with a dashboard screenshot) and where this leads.

- [PDF](docs/slides/who-said-it-first.pdf) (GitHub shows it inline)
- [HTML](docs/slides/who-said-it-first.html): open it locally and use ← → to move between slides.

The figures come from the dashboard at 5:20 PM on 2026-10-11 (round 9 of the all-knobs line). The
locked test set was scored afterwards — see Results. Prepared by Aleks Jakulin (data.Flowers) with Kevin Chiu (Plastachio) at an
RSIHOUSE × Sundai Club hack.

## Results (final, 2026-10-11)

First-appearance accuracy (the tracer names the article that first carried the quote; ground
truth v2). The loop ran four lines; the best config is the main line's v3
(`experiments/configs/v3.json`), reached in 3 accepted changes and unchanged over the last 9 rounds.

| Split | Baseline v0 | Best v3 |
|---|---|---|
| Train (145) | 0.621 | 0.834 |
| Validation (73) | 0.644 | 0.849 |
| Watch set (100, held out, never seen by the loop) | 0.60 | 0.85 |
| **Test (146, scored once at the end)** | **0.705** | **0.884** (95% CI 0.82–0.93; 32 items fixed, 6 broken) |

What the loop found (Claude Opus 5.5 proposing, Haiku 5.5 verifying, 2 Haiku calls per claim):
1. **Speaker-aware search** (index quote + speaker, query = content + speaker): search recall
   0.72 → 0.92 — found independently by the main and the retrieval-only lines.
2. **Stricter verifier prompt** (the quote must contain every part of the claim; stricter test for
   unattributed quotes): wrong earlier matches 0.23 → ~0.01.

Caveats: about 5 points of the early gain came from the v1 ground truth's blind spot (earlier
renderings of the same quote counted as wrong) and were removed by ground truth v2. On harder
queries (`paraphrase_hard`: speaker described by role, gist reworded) v3 drops to 0.45 on
validation, and search recall tops out at 47.6% (exhaustive sweep) — the obvious next target.

## Running the demo

**Offline demo: [docs/demo/demo.html](docs/demo/demo.html).** One self-contained page (no server, no
model calls) in the style of the Firstline app, with 80 precomputed traces from the held-out sets (40
quotes, 40 events): search or pick an example to see the tracer's first report, the benchmark answer
and how the story spread. Download it and open it in a browser (GitHub shows HTML as source). Its
event examples use the post-loop fixed tracer. Rebuild:

```
.venv/bin/python -I scripts/build_demo.py      # docs/demo/index.html: longer write-up, loop's final settings
.venv/bin/python -I scripts/build_demo.py --event-best experiments_event/configs/fix_retrieval.json --out /tmp/demo_fix
.venv/bin/python -I scripts/build_firstline_offline.py --source /tmp/demo_fix/index.html   # docs/demo/demo.html
```

Live app (needs the local database):

```
.venv/bin/python -I scripts/dashboard.py                   # loop dashboard on 127.0.0.1:8765
.venv/bin/python -I scripts/app.py --host 0.0.0.0          # Firstline app on :8780, dashboard tab proxied in
```

Open http://<host>:8780/ — "Trace a statement" (`#trace`) and "Self-improvement loop" (`#loop`).
The app always serves the loop's current best config.

## Setup

The datasets are not in the repo (tens of GB). Fetch and rebuild them with:

```
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -I scripts/get_data.py            # Quotebank 2020 + Reuters-21578, then build data/news.duckdb
.venv/bin/python -I scripts/get_data.py --gdelt    # optional: also GDELT wire rows (Jan-Apr 2020)
```

Downloads resume and are checked against the published MD5s; the build needs ~25 GB free disk,
takes ~20-30 minutes and makes no LLM calls. The tracer, loop and app call Claude through headless
Claude Code (`claude -p`), so a logged-in Claude Code install is required to run them.

## Data

| Source | What | Where |
|---|---|---|
| Quotebank 2020 (Jan–Apr), EPFL, CC-BY-4.0 | 5.24M unique quotes; 16.7M quote mentions with timestamp, outlet, URL, speaker, surrounding text | `data/news.duckdb` (`quotes`, `quote_mentions`, `quote_urls`) |
| Quote variants | 8-gram links between trimmed/edited versions of a quote | `quote_links`, `quote_clusters` |
| Domain labels | hand-curated wire / press_release / aggregator / feed + ownership groups | `data/domain_labels.csv` → `domains` view |
| Reuters-21578 (1987) | dev set only (single outlet, too old/narrow); its text is not redistributed here — `scripts/make_bench.py` rebuilds the early dev benchmark locally | `reuters_*` tables |
| GDELT 2.0 Events / Mentions / GKG 2.1 (Jan–Apr 2020, raw zips in `data/raw/gdelt/`) | 17.2M events, 55.0M mentions (with GDELT crawl time), 20.5M GKG article records (headline, persons, organisations, themes) | `gdelt_events`, `gdelt_mentions`, `gdelt_gkg`; headline index `gkg_titles` |
| GDELT GKG wire rows (Jan–Apr 2020, all 11,615 files fetched) | AP/Reuters/AFP/… articles with GDELT-extracted quotes | `gdelt_wire_articles`, `wire_quote_matches` (not used by the tracer by default) |

GDELT: `scripts/fetch_gdelt_raw.py` (manifest + resumable MD5-checked download),
`scripts/load_gdelt_events.py`, `scripts/load_gdelt_gkg.py`, then `scripts/trace_event.py --build-index`.

Rebuild: `scripts/parse_quotebank_quotes.py`, `scripts/parse_quotebank_parallel.py`,
`scripts/build_db.py`, `scripts/quote_variants.py`.

## Tracer

`scripts/trace_quote.py "statement" --as-of <ISO time> [--config config/base.json]`

1. BM25 full-text search over quotes first seen before `--as-of`.
2. Claude Haiku 5.5 checks which candidate quotes state the claim.
3. Matched quotes are expanded to their direct variants.
4. Timeline from `quote_mentions`: first appearance, first original outlet, spread by domain.
   "First" is ranked only among mentions with a real time of day (date-only stamps sort artificially early).

All tunables live in `config/` (`scripts/qb_config.py` defines what the loop may change).

## Benchmark and self-improvement loop

- `bench/qb_bench.jsonl` (`scripts/make_qb_bench.py`): quotes carried by ≥5 outlets; query = a
  Haiku paraphrase of a later outlet's report; answer = the earliest timed mention before it.
  Splits: train 145 / val 73 / test 146 (test is scored only at baseline and at the end).
  Ground truth v2 (`scripts/fix_ground_truth.py`): an item's quote also counts near-identical
  same-speaker versions that the 8-gram links missed (≥70% word containment, a shared 5-word run,
  first seen within 14 days), so an earlier rendering of the same quote is a correct answer, not a
  false positive. v1 files are kept as `*.jsonl.v1`; `improve.py rebaseline` re-scores after the change.
- `bench/qb_watch.jsonl` (`scripts/make_watch_set.py`): 100 extra held-out items, scored live by
  `scripts/watch_scorer.py` for the dashboard only; the loop never reads it.
- `scripts/evaluate.py`: recall@k, verifier recall, false-positive-earlier, first appearance
  (headline), first original outlet, Haiku calls per claim.
- `scripts/improve.py`: each round Claude Opus 5.5 reads failures and proposes up to three config
  changes; candidates are screened (retrieval changes free via recall@k), the top two are scored on
  train (`--train-n`, 145 in the final runs) with val in parallel, and one is accepted only if it fixes
  ≥ threshold more items than it breaks (threshold from measured run-to-run noise) and doesn't lose on
  val. With a train subset, borderline candidates (+1/+2) get a second look on the full split. Lines can
  run in parallel with a knob focus (`--line/--focus`), on a query form (`--mode paraphrase_hard`), and
  be combined with `merge`. Log: `experiments/**/ledger.jsonl`; `retrieval_sweep.py` maps the full
  retrieval grid for the dashboard's search-space view.
- Dashboard: `scripts/dashboard.py` → http://127.0.0.1:8765/

Anti-gaming choices: the loop cannot change the outlet-count filter, time window, variant
expansion or domain labels (they leak how the benchmark is built or feed the ground truth), and
prompts that add names/numbers from train/val items are rejected.

## Event tracer (post-hackathon, GDELT)

The same idea for events rather than quotes: given a sentence reporting an event and a time,
find the first article that reported it. `scripts/trace_event.py "sentence" --as-of <ISO time>
[--config experiments_event/configs/final.json]`: BM25 over GDELT headlines from the 7 days
before `--as-of` (syndicated copies grouped by normalised headline), Haiku 5.5 picks the headlines
reporting the same event, and the earliest matched article is the first report. Times are when
GDELT saw the article (15-minute steps), an upper bound on publication.

**Benchmark** (`bench/ev_bench.jsonl` train 130 / val 63 / test 130; `bench/ev_watch.jsonl` 88):
items start from GDELT events reported by many outlets (`scripts/make_event_bench.py`); the query
is a later outlet's report. Structural ground truth (GDELT event ids + shared headlines) failed an
Opus audit (precision 0.76–0.82, ~half the earlier reports missed), so the labels are model-made:
`scripts/label_event_gt.py` (Claude Sonnet 5.5, low effort, 8 items per call) judges candidates
from the structure, IDF headline search, BM25 and GKG person/organisation matches, and keeps only
discrete events. A label can only be as early as its candidates, so `scripts/adjudicate_event_gt.py`
lets the same labeller judge tracer predictions that come out earlier than the label: v3.2 used the
baseline's, v3.3 those of every config the loop scored in full (35 of 112 accepted). An independent
Fable audit of 50 v3.3 adjudications agreed on 45 (accepted precision ~89%). Correct = an article
of the first wave (the first report and same-event articles up to 30 minutes after it).
Scoring: `scripts/evaluate_event.py`; loop: `improve.py ... --task event` (`experiments_event/`).

**Results** (ground truth v3.4; Haiku 5.5 verifier, Opus 5.5 improver):

| Split | Baseline (k 30, pool 4) | Loop final (k 60, pool 8) | + fixes (headline query) |
|---|---|---|---|
| Train (130) | 0.485 | 0.585 | 0.646 |
| Validation (63) | 0.492 | 0.540 | 0.667 |
| Watch (88) | 0.500 | 0.557 | 0.625 |
| **Test (130, scored once per config)** | **0.492** | **0.608** | **0.662** (95% CI 0.58–0.74; vs baseline 28 fixed, 6 broken) |

The loop (two lines: all knobs, stopped at round 3 on its Haiku budget; retrieval only, 10 rounds)
accepted one change (k 30→50, pool 4→6). After v3.3 every scored config was re-scored on train/val
and the best (k 60, pool 8, rejected at +2 before) became the loop's final config — a pick among 18
configs; test was not used to choose. Inspecting held-out misses then showed three problems, fixed
by hand (`experiments_event/configs/fix_*.json`, logged as a "fix" entry in the ledger):
1. **Scoring:** same-event answers inside the 30-minute first wave that the labeller had never seen
   were scored "late"; v3.4 sends those to the labeller too (all configs' answers, symmetrically).
2. **Search:** reworded statements drift from headline vocabulary (the B.C. "eight fresh infections"
   statement ranked the true "Eight new COVID-19 cases" headline ~2,000th). `query_template
   "raw+headline"` adds a Haiku headline rewrite of the statement and interleaves both searches:
   recall@k 0.79 → 0.88 on train, 0.71 → 0.87 on val, at the same ~4 Haiku calls per item.
3. **Checker:** it accepts earlier occurrences of recurring events ("8th case" for "eight new cases").
   A stricter prompt plus the claim's own time (`verify_event_v2.txt`, `show_claim_date`) cut early
   answers but lost more true matches; it was not kept (train+val: 115 vs 126 for the search fix alone).

The fixes were motivated by held-out examples, so the test number above is less clean than a
loop-only result.

Tried afterwards and not kept (train+val only, test not scored):
- **Local models instead of Haiku.** Classifiers on TF-IDF, number-mismatch, timing and `bge-base`
  embedding features, with or without Haiku's verdict, and local LLMs (Qwen2.5-14B,
  Qwen3-Next-80B via MLX) as the checker all did worse than Haiku (val 19–40 of 63 vs 41): they
  accept on-topic earlier occurrences, which are most of the remaining errors.
- **An anchored second check** (`recheck` in `trace_event.py`, `experiments_event/configs/recheck_*.json`):
  re-ask Haiku about candidates seen before reference headlines taken from the first check's
  matches. Best variant 126 vs 123 of 193 (labels v3.5: its new early answers adjudicated, 3 of 8
  accepts reverted after a Fable audit), 9 fixed / 6 broken, for ~3 more Haiku calls per item —
  within noise. Labels are model-made (~10% disagreement with an independent audit).

## Backlog (not built yet)

1. **Wire and press-release copy detection from shared text.** Quotebank keeps ~500 characters
   around each quote. Near-identical surrounding text across unrelated outlets means a shared
   source (wire copy or a press release), not independent reporting. Plan: MinHash/LSH over
   `left_context || quotation || right_context` per mention, cluster near-duplicates, label each
   cluster as wire / press release / syndication group / independent, and attribute the cluster
   to an inferred source (e.g. "AP (inferred)", timed at its earliest member). Explicit markers
   ("(AP)", "Reuters", "PA") appear in only ~5% of contexts, so they can confirm but not drive it.
   Why it matters: AP and Reuters' own sites are absent from Quotebank, so "first appearance" is
   often just the fastest member site (seattletimes.com is the benchmark's most common "first").
2. **Who-copies-whom between outlets.** From the clusters above plus timing, build an outlet
   network (sources, sinks, main routes). Use text overlap as the evidence of copying; timing alone
   mostly reflects crawl order. NetRate/NetInf-style inference is an option once evidence exists;
   a simple "who carries it within 24h after whom, with shared text" table is the first step.
3. **GDELT wire matching (built, switched off).** `scripts/fetch_gdelt_wire.py` streams GDELT 2.1
   GKG files and keeps wire/press-release rows; `scripts/match_wire_quotes.py` links their quotes
   to Quotebank quotes (6-gram overlap, ≥8-word fragments, within a week). All 11,615 files
   for Jan–Apr 2020 are loaded: 409k wire articles, 90k GDELT quotes, 57.8k matches to 19.6k
   Quotebank quotes (for 6.1k of them the wire copy is >1h before Quotebank's first timed mention),
   but only 16 of the 464 benchmark/watch items have a wire match. `trace_quote.py --with-wire` merges them into timelines;
   GDELT times are when GDELT saw the article (15-min resolution), an upper bound on publication.
4. **Paraphrased facts beyond quotes.** (Events: see Event tracer.) Index the surrounding text as passages so the tracer can
   follow reported facts that never quote anyone; link facts to quotes (fact → quote → speaker).
5. **More data.** Other Quotebank years (2008–2019, ~185 GB); Reuters TRC2 (2008–09, NIST
   agreement) for true wire timestamps alongside Quotebank 2008–09.
6. **Loop improvements.** Self-generated harder benchmark items when train saturates; a GEPA
   comparison. (Borderline second look: done.)

## Licence

This project's code is released under the MIT License (see `LICENSE`). Slides in
`docs/slides/` are © their authors.

## Licences of what we use

No datasets are stored in this repo; `scripts/get_data.py` downloads them from their sources.

| Used | Licence / terms |
|---|---|
| Quotebank (Vaucher et al., WSDM 2021), via Zenodo | CC BY 4.0 — the benchmark and experiment files here include Quotebank quotations under it |
| Reuters-21578, Distribution 1.0 (UCI) | © Reuters Ltd.; free distribution for research purposes only (not included here) |
| GDELT 2.1 GKG (gdeltproject.org) | Unlimited use and redistribution with a citation to the GDELT Project and a link to https://www.gdeltproject.org/ |
| duckdb | MIT |
| pyarrow | Apache-2.0 |
| numpy | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) |
| scipy | BSD-3-Clause |
| anthropic (Python SDK) | MIT |
| rank-bm25 | Apache-2.0 |
| Chart.js 4.4.1, Plotly.js 2.35.2 (loaded from jsDelivr) | MIT |
| Claude models via Claude Code | Anthropic's terms of service |

