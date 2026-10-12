# News provenance tracer (hackathon prototype)

Given a statement from a news story, trace it back to the first report that carried it, and
let a self-improvement loop tune the tracer against a benchmark whose answers are known.

All LLM calls run through headless Claude Code (`claude -p`) on a Claude subscription, and are
cached in `data/cache/llm.sqlite`, so re-runs are free.

## Presentation

*Who said it first?* is a 10-slide deck covering the problem, the data, the tracer, the
self-improvement loop (with a dashboard screenshot) and where this leads.

- [PDF](docs/slides/who-said-it-first.pdf) (GitHub shows it inline)
- [HTML](docs/slides/who-said-it-first.html): open it locally and use ← → to move between slides.

The figures come from the dashboard at 5:20 PM on 2026-10-11 (round 9 of the all-knobs line). The
locked test set hasn't been scored again yet. Prepared by Aleks Jakulin (data.Flowers) with Kevin Chiu (Plastachio) at an
RSIHOUSE × Sundai Club hack.

## Data

| Source | What | Where |
|---|---|---|
| Quotebank 2020 (Jan–Apr), EPFL, CC-BY-4.0 | 5.24M unique quotes; 16.7M quote mentions with timestamp, outlet, URL, speaker, surrounding text | `data/news.duckdb` (`quotes`, `quote_mentions`, `quote_urls`) |
| Quote variants | 8-gram links between trimmed/edited versions of a quote | `quote_links`, `quote_clusters` |
| Domain labels | hand-curated wire / press_release / aggregator / feed + ownership groups | `data/domain_labels.csv` → `domains` view |
| Reuters-21578 (1987) | dev set only (single outlet, too old/narrow) | `reuters_*` tables |
| GDELT GKG wire rows (partial, 754/11,615 files) | AP/Reuters/AFP/… articles with GDELT-extracted quotes | `gdelt_wire_articles`, `wire_quote_matches` (paused) |

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
- `bench/qb_watch.jsonl` (`scripts/make_watch_set.py`): 100 extra held-out items, scored live by
  `scripts/watch_scorer.py` for the dashboard only; the loop never reads it.
- `scripts/evaluate.py`: recall@k, verifier recall, false-positive-earlier, first appearance
  (headline), first original outlet, Haiku calls per claim.
- `scripts/improve.py`: each round Claude Opus 5.5 reads failures and proposes up to three config
  changes; candidates are screened (retrieval changes free via recall@k), the best is scored on a
  60-item train subset and accepted only if it fixes ≥ threshold more items than it breaks (threshold
  from measured run-to-run noise) and doesn't lose on val. Lines can run in parallel with a knob
  focus (`--line/--focus`) and be combined with `merge`. Log: `experiments/**/ledger.jsonl`.
- Dashboard: `scripts/dashboard.py` → http://127.0.0.1:8765/

Anti-gaming choices: the loop cannot change the outlet-count filter, time window, variant
expansion or domain labels (they leak how the benchmark is built or feed the ground truth), and
prompts that add names/numbers from train/val items are rejected.

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
   to Quotebank quotes (6-gram overlap, ≥8-word fragments, within a week). Fetch paused at
   754/11,615 files for Jan–Apr 2020. `trace_quote.py --with-wire` merges them into timelines;
   GDELT times are when GDELT saw the article (15-min resolution), an upper bound on publication.
4. **Paraphrased facts beyond quotes.** Index the surrounding text as passages so the tracer can
   follow reported facts that never quote anyone; link facts to quotes (fact → quote → speaker).
5. **More data.** Other Quotebank years (2008–2019, ~185 GB); Reuters TRC2 (2008–09, NIST
   agreement) for true wire timestamps alongside Quotebank 2008–09.
6. **Loop improvements.** Borderline second look (re-score +1/+2 candidates on all 145 train
   items); self-generated harder benchmark items when train saturates; a GEPA comparison.
