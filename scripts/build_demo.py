"""Build the self-contained demo page: docs/demo/index.html (claude.ai Artifact form) and
docs/demo/standalone.html (the same page wrapped in a full HTML document, for opening locally).

  python -I scripts/build_demo.py [--quote-best experiments/configs/v3.json] [--quote-base experiments/configs/v0.json]
                                  [--event-best <experiments_event/state.json best_config>]
                                  [--event-base experiments_event/configs/v0.json] [--n 40] [--workers 3]

Every item of the two held-out watch sets (bench/qb_watch.jsonl, bench/ev_watch.jsonl; the loops never
read them) is traced with the best and the baseline config, which gives the watch-set scores; ~40 items
per task, picked by a fixed hash, are embedded with their full traces. Verifier calls go through the LLM
cache (data/cache/llm.sqlite), so configs that were already scored on the watch sets cost nothing; any
miss is a Haiku call (the builder refuses configs whose verifier is not a Haiku model). The loop ledgers
(experiments/**/ledger.jsonl, experiments_event/**/ledger.jsonl) give the per-round history and the
train/val/test numbers. Page template: scripts/demo_template.html.
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev_config  # noqa: E402
import qb_config  # noqa: E402
import trace_event  # noqa: E402
import trace_quote  # noqa: E402
from evaluate_event import gt_sets, in_wave  # noqa: E402
from verify import LLMCache  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "demo_template.html"
SPREAD_CAP = 30

# ---------- plain-words descriptions of config changes ----------

WORDS = {
    "retrieval.index": {"quotation": "the quote text only", "quotation_speaker": "the quote text plus the speaker's name"},
    "retrieval.query_template": {"raw": "the whole statement",
                                 "content": "the statement without its reporting frame (“X said that…”)",
                                 "content_plus_speaker": "the statement's content plus the speaker's name"},
    "retrieval.speaker_filter": {"off": "no speaker filter",
                                 "surname": "only quotes whose speaker shares the statement's surname (or is unknown)"},
    "verify.order": {"score": "by search score", "time": "earliest first"},
}
LABEL = {
    "retrieval.k": "Candidates passed to the checker",
    "retrieval.pool": "Headlines fetched per candidate slot",
    "retrieval.index": "Search index",
    "retrieval.query_template": "Search query",
    "retrieval.speaker_filter": "Speaker filter",
    "verify.effort": "Checker effort",
    "verify.batch": "Candidates per checker call",
    "verify.show_date": "Show the checker each candidate's date",
    "verify.show_source": "Show the checker each headline's outlet",
    "verify.show_copies": "Show the checker how many outlets carried a headline",
    "verify.speaker_hint": "Tell the checker whether the speaker matches",
    "verify.order": "Order of candidates shown to the checker",
    "verify.prompt": "Checker instructions",
}


def fmt_val(path, v):
    if isinstance(v, bool):
        return "on" if v else "off"
    return WORDS.get(path, {}).get(v, str(v))


def describe(path, old, new, has_old=True):
    label = LABEL.get(path, path)
    if path == "verify.prompt":
        if has_old and isinstance(old, str):
            return f"Rewrote the checker's instructions ({len(old):,} → {len(new):,} characters)"
        return "Rewrote the checker's instructions"
    if has_old:
        return f"{label}: {fmt_val(path, old)} → {fmt_val(path, new)}"
    return f"{label}: {fmt_val(path, new)}"


# ---------- ledgers ----------

def read_ledger(path):
    return [json.loads(line) for line in open(path) if line.strip()] if path.exists() else []


def metric_entry(m, metric):
    d = {"score": m[metric], "n": m["n"]}
    for k in ("recall_at_k", "calls_per_claim"):
        if m.get(k) is not None:
            d[k] = m[k]
    return d


def loop_history(root, metric, name):
    """Rounds, incumbent series and per-version metrics from one line's ledger.

    Also returns the line's "final" (held-out test) and "reselect" ledger entries. A reselect entry
    means the final config was picked after the loop (re-scoring every fully scored config on
    train/val under a revised ground truth), so the final test score belongs to that config, not
    to the line's incumbent best_version, whichever order the two entries were logged in.
    """
    entries = read_ledger(root / "ledger.jsonl")
    state = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {}
    # a line that starts from another line's best (e.g. the quote task's "hard" line starts from main v3
    # on harder queries) inherits the shared baseline entry; its own start is its first re-baseline
    started_from = state.get("started_from")
    versions, rounds, series, marks = {}, {}, [], []
    accepted_count, final, reselect, last_rebase = 0, None, None, None
    start = None

    def put(v, split, m):
        if m:
            versions.setdefault(v, {})[split] = metric_entry(m, metric)

    cur = None
    for e in entries:
        t = e["type"]
        if t == "baseline":
            if started_from:
                continue
            for s in ("train", "val", "test"):
                put(0, s, e.get(s))
            cur = {"round": 0, "train": e["train"][metric], "val": e["val"][metric], "n": e["train"]["n"]}
            series.append(dict(cur, version=0, why="baseline"))
        elif t == "rebaseline":
            last_rebase = e
            for s in ("train", "val", "test"):
                put(0, s, e["v0"].get(s))
            b = e["best"]
            put(b["version"], "train", b["train"])
            put(b["version"], "val", b["val"])
            if started_from and not series:
                cur = {"round": 0, "train": b["train"][metric], "val": b["val"][metric], "n": b["train"]["n"]}
                series.append(dict(cur, version=b["version"], why="start"))
                start = {"from": started_from, "mode": state.get("mode") or ("paraphrase_hard" if name == "hard" else None),
                         "gt_version": e.get("gt_version"), "n": b["train"]["n"],
                         "recall_at_k": b["train"].get("recall_at_k")}
                continue
            cur = {"round": e["round"] + 0.5, "train": b["train"][metric], "val": b["val"][metric], "n": b["train"]["n"]}
            note = f"re-scored: ground truth v{e.get('gt_version')}, {b['train']['n']} train items"
            if marks and marks[-1]["round"] == e["round"] + 0.5:
                marks[-1]["note"] = note
                series[-1] = dict(cur, version=b["version"], why="rescored")
            else:
                marks.append({"round": e["round"] + 0.5, "note": note})
                series.append(dict(cur, version=b["version"], why="rescored"))
        elif t == "proposal":
            r = rounds.setdefault(e["round"], {"round": e["round"], "proposals": [], "accepted": None})
            has_diff = e.get("diff") is not None
            changes = e.get("diff") if has_diff else [{"path": c["path"], "new": c["value"]} for c in e.get("changes", [])]
            p = {"decision": e["decision"], "reason": e.get("reason", ""), "category": e.get("category"), "rank": e.get("rank"),
                 "rationale": e.get("rationale", ""),
                 "changes": [describe(c["path"], c.get("old"), c["new"], has_diff) for c in changes],
                 "train": e["train"][metric] if e.get("train") else None,
                 "train_n": e["train"]["n"] if e.get("train") else None,
                 "val": e["val"][metric] if e.get("val") else None,
                 "fixed": e.get("fixed"), "broken": e.get("broken")}
            prompt = next((c for c in changes if c["path"] == "verify.prompt"), None)
            if prompt and e["decision"] == "accepted":
                p["prompt_old"], p["prompt_new"] = prompt.get("old"), prompt["new"]
            r["proposals"].append(p)
            if e["decision"] == "accepted":
                accepted_count += 1
                put(accepted_count, "train", e["train"])
                put(accepted_count, "val", e["val"])
                p["version"] = accepted_count
                cur = {"round": e["round"], "train": e["train"][metric], "val": e["val"][metric], "n": e["train"]["n"]}
                series.append(dict(cur, version=accepted_count, why="accepted"))
        elif t == "round_end":
            r = rounds.setdefault(e["round"], {"round": e["round"], "proposals": [], "accepted": None})
            r["accepted"] = e["accepted"]
        elif t == "final":
            final = e
        elif t == "reselect":
            reselect = e
    if state.get("baseline_test"):
        put(0, "test", state["baseline_test"])
    if state.get("best_version") is not None and state.get("train_metrics"):
        put(state["best_version"], "train", state["train_metrics"])
        put(state["best_version"], "val", state["val_metrics"])
    final_test = None
    if final:
        final_test = dict(metric_entry(final["best_metrics"], metric), ci95=final.get("best_ci95"),
                          fixed_broken=[final.get("fixed"), final.get("broken")])
        if not reselect:  # otherwise the test score belongs to the reselected config (see task_block)
            versions.setdefault(final["best_version"], {})["test"] = final_test
    hist = {"name": name, "rounds": [rounds[k] for k in sorted(rounds)], "series": series, "marks": marks,
            "best_version": state.get("best_version"), "n_rounds": state.get("round", 0),
            "final": bool(final), "start": start}
    extras = {"final": final, "final_test": final_test, "reselect": reselect, "last_rebase": last_rebase}
    return hist, versions, extras


def line_dirs(task_root):
    out = [("main", task_root)]
    if (task_root / "lines").is_dir():
        out += [(p.name, p) for p in sorted((task_root / "lines").iterdir()) if (p / "ledger.jsonl").exists()]
    return out


def config_line_version(path):
    """experiments/configs/v3.json -> (Path('experiments'), 3); lines/<x>/configs/vN.json likewise."""
    p = (ROOT / path).resolve()
    try:
        return p.parent.parent, int(p.stem.lstrip("v"))
    except ValueError:
        return None, None


# ---------- tracing ----------

def domain_of(url):
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except Exception:
        return ""


def short(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def check_haiku(cfg, path):
    model = cfg["verify"]["model"]
    if "haiku" not in model:
        sys.exit(f"{path}: verifier model {model} is not a Haiku model; refusing to run it")


class Tracer:
    def __init__(self, workers):
        self.con = duckdb.connect(str(ROOT / "data/news.duckdb"), read_only=True)
        self.con.execute("SET threads = 3")
        self.con.execute("LOAD fts")
        self.cache, self.local, self.workers = LLMCache(str(ROOT / "data/cache/llm.sqlite")), threading.local(), workers
        self.lock = threading.Lock()

    def cursor(self):
        if not hasattr(self.local, "con"):
            self.local.con = self.con.cursor()
            self.local.con.execute("LOAD fts")
            self.local.con.execute("SET threads = 3")
        return self.local.con

    def map(self, fn, items):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(fn, items))

    def sql(self, q, params):
        return self.cursor().execute(q, params).fetchall()


def quote_task(tr, items, cfg_best, cfg_base):
    def one(it):
        gt, out = it["gt_first_any"], {}
        for key, cfg in (("best", cfg_best), ("base", cfg_base)):
            res = trace_quote.trace(tr.cursor(), tr.cache, it["paraphrase"], it["as_of"], cfg)
            pred = res["first_any"]
            correct = bool(pred and (pred["url"] == gt["url"] or pred["date"][:19] == gt["date"][:19]))
            verdict = "right" if correct else "none" if not pred else \
                "early" if pred["date"][:19] < gt["date"][:19] else "late"
            out[key] = {"res": res, "correct": correct, "verdict": verdict,
                        "tie": bool(correct and pred["url"] != gt["url"])}
        return out
    t = time.time()
    traced = tr.map(one, items)
    print(f"quotes: traced {len(items)} items x 2 configs in {time.time() - t:.0f}s", flush=True)
    return traced


def quote_record(tr, it, tb):
    gt = it["gt_first_any"]
    ids = set(it["variant_ids"])
    title = tr.sql("SELECT title FROM quote_mentions WHERE url = ? AND title IS NOT NULL LIMIT 1", [gt["url"]])
    best, base = tb["best"], tb["base"]
    res = best["res"]
    texts = {m["quote_id"]: m["text"] for m in res["matched_quotes"]}

    def quote_text(qid):
        if qid not in texts:
            row = tr.sql("SELECT quotation FROM quotes WHERE quote_id = ?", [qid])
            texts[qid] = row[0][0] if row else ""
        return texts[qid]

    def mention(m, with_quote=False):
        if not m:
            return None
        d = {"date": m["date"][:16], "outlet": m["domain"], "url": m["url"], "title": short(m.get("title"), 200),
             "kind": m.get("kind"), "timed": bool(m.get("has_time"))}
        if with_quote:
            d["quote"] = short(quote_text(m["quote_id"]), 600)
        return d
    spread = res["spread"]
    gt_in = any(s["url"] == gt["url"] for s in spread[:SPREAD_CAP])
    first_url = res["first_any"]["url"] if res["first_any"] else None
    rows = [{"t": s["date"][:16], "o": s["domain"], "u": s["url"], "k": s["kind"], "timed": bool(s["has_time"]),
             "first": s["url"] == first_url, "gt": s["url"] == gt["url"]} for s in spread[:SPREAD_CAP]]
    if first_url and not any(r["first"] for r in rows):
        f = res["first_any"]
        rows.append({"t": f["date"][:16], "o": f["domain"], "u": f["url"], "k": f["kind"], "timed": True,
                     "first": True, "gt": f["url"] == gt["url"]})
    if not gt_in and not any(r["gt"] for r in rows):
        rows.append({"t": gt["date"][:16], "o": gt["domain"], "u": gt["url"], "k": None, "timed": True,
                     "first": False, "gt": True, "extra": True})
    rows.sort(key=lambda r: (r["t"], not r["first"]))
    return {
        "id": it["quote_id"], "statement": it["paraphrase"], "as_of": it["as_of"][:16],
        "source_outlet": domain_of(it.get("query_url", "")), "source_url": it.get("query_url"),
        "speaker": it.get("speaker"),
        "gt": {"date": gt["date"][:16], "outlet": gt["domain"], "url": gt["url"],
               "title": short(title[0][0] if title else "", 200), "quote": short(it["quotation"], 600)},
        "best": {"verdict": best["verdict"], "correct": best["correct"], "tie": best["tie"],
                 "first": mention(res["first_any"], True),
                 "matched": [{"speaker": m["speaker"], "text": short(m["text"], 400), "date": m["date"],
                              "articles": m["num_urls"], "gt": m["quote_id"] in ids} for m in res["matched_quotes"][:12]],
                 "n_matched": len(res["matched_quotes"]), "n_candidates": len(res["candidates"]),
                 "n_mentions": res["n_mentions"], "n_outlets": len(spread),
                 "calls": res["llm_calls"]["made"] + res["llm_calls"]["cached"]},
        "base": {"verdict": base["verdict"], "correct": base["correct"],
                 "first": mention(base["res"]["first_any"]), "n_matched": len(base["res"]["matched_quotes"])},
        "spread": rows, "spread_more": max(0, len(spread) - SPREAD_CAP),
    }


def event_task(tr, items, cfg_best, cfg_base):
    def one(it):
        t0, wave, same, keys = gt_sets(it)
        out = {}
        for key, cfg in (("best", cfg_best), ("base", cfg_base)):
            res = trace_event.trace(tr.cursor(), tr.cache, it["query"], it["as_of"], cfg)
            pred = res["first"]
            correct = bool(pred and in_wave(pred, t0, wave, same, keys))
            verdict = "right" if correct else "none" if not pred else \
                "early" if dt.datetime.fromisoformat(pred["date"]) < t0 else "late"
            out[key] = {"res": res, "correct": correct, "verdict": verdict}
        return out
    t = time.time()
    traced = tr.map(one, items)
    print(f"events: traced {len(items)} items x 2 configs in {time.time() - t:.0f}s", flush=True)
    return traced


def event_record(it, tb):
    t0, wave, same, keys = gt_sets(it)
    slot = set(it.get("gt_first_slot") or [])
    first_wave = sorted(it["gt_first_wave"], key=lambda a: (a["date"], a["url"] not in slot, a["url"]))
    g = first_wave[0]
    best, base = tb["best"], tb["base"]
    res = best["res"]

    def art(a):
        return a and {"date": a["date"][:16], "outlet": a["source"], "url": a["url"], "title": short(a["title"], 220)}
    hit = lambda a: in_wave(a, t0, wave, same, keys)  # noqa: E731
    spread = res["spread"]
    first_url = res["first"]["url"] if res["first"] else None
    rows = [{"t": a["date"][:16], "o": a["source"], "u": a["url"], "first": a["url"] == first_url,
             "gt": a["url"] == g["url"], "wave": hit(a), "timed": True} for a in spread[:SPREAD_CAP]]
    if first_url and not any(r["first"] for r in rows):
        f = res["first"]
        rows.append({"t": f["date"][:16], "o": f["source"], "u": f["url"], "first": True, "gt": f["url"] == g["url"],
                     "wave": hit(f), "timed": True})
    if not any(r["gt"] for r in rows):
        rows.append({"t": g["date"][:16], "o": g["source"], "u": g["url"], "first": False, "gt": True, "wave": True,
                     "timed": True, "extra": True})
    rows.sort(key=lambda r: (r["t"], not r["first"]))
    adjudicated = None
    if res["first"] and it.get("gt_adjudicated"):
        adj = next((a for a in it["gt_adjudicated"] if a["url"] == res["first"]["url"]), None)
        if adj:
            adjudicated = bool(adj.get("same_event"))
    return {
        "id": str(it["event_id"]), "statement": it["query"], "as_of": it["as_of"][:16],
        "source_outlet": it.get("query_source"), "source_url": it.get("query_url"),
        "source_title": it.get("query_title"),
        "gt": dict(art(g), wave=len(first_wave)),
        "best": {"verdict": best["verdict"], "correct": best["correct"], "first": art(res["first"]),
                 "adjudicated": adjudicated,
                 "matched": [{"title": short(m["title"], 220), "outlet": m["source"], "date": m["date"][:16],
                              "copies": m["n_copies"], "gt": any(hit(a) for a in m["articles"])} for m in res["matched"][:12]],
                 "n_matched": len(res["matched"]), "n_candidates": len(res["candidates"]),
                 "n_mentions": res["n_articles"], "n_outlets": len(spread),
                 "calls": res["llm_calls"]["made"] + res["llm_calls"]["cached"]},
        "base": {"verdict": base["verdict"], "correct": base["correct"], "first": art(base["res"]["first"]),
                 "n_matched": len(base["res"]["matched"])},
        "spread": rows, "spread_more": max(0, len(spread) - SPREAD_CAP),
    }


def pick(items, key, n):
    return sorted(items, key=lambda it: hashlib.sha1(f"demo:{key(it)}".encode()).hexdigest())[:n]


def config_label(path):
    """experiments/configs/v3.json -> 'v3'; experiments_event/configs/final.json -> 'final'."""
    stem = Path(path).stem
    return stem if re.fullmatch(r"v\d+|final", stem) else str(path)


def flat(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


LINE_SHORT = {"main": "Main", "retrieval": "Search", "verify": "Checker", "hard": "Harder-queries"}


def reselect_block(task_root, lines, reselect, final, final_test, last_rebase, base_cfg):
    """The post-hoc pick: every fully scored config re-scored on train/val under a revised ground truth.

    Annotates the matching proposals in `lines` (keys like 'retrieval_r2_1' = line, round, rank;
    'v0'/'v1' = accepted versions) and returns the table shown on the page plus the final config's
    train/val/test metrics."""
    scores, winner = reselect["scores"], reselect["winner"]
    n_train = last_rebase["v0"]["train"]["n"] if last_rebase else None
    n_val = last_rebase["v0"]["val"]["n"] if last_rebase else None
    cfgs_path = task_root / "runs/adj/configs.json"
    cfgs = json.loads(cfgs_path.read_text()) if cfgs_path.exists() else {}
    base_flat = flat(base_cfg)
    found = {}
    for L in lines:
        for r in L["rounds"]:
            for p in r["proposals"]:
                keys = [f"{L['name']}_r{r['round']}_{p.get('rank')}"]
                if p.get("version"):
                    keys.append(f"v{p['version']}")
                key = next((k for k in keys if k in scores), None)
                if key:
                    p["rescored"] = scores[key]
                    p["final_pick"] = key == winner
                    found[key] = {"line": L["name"], "round": r["round"], "decision": p["decision"],
                                  "reason": p["reason"], "changes": p["changes"]}
    rows = []
    for key, s in scores.items():
        info = found.get(key, {})
        if key in cfgs:
            diff = [(k, base_flat.get(k), v) for k, v in flat(cfgs[key]).items() if base_flat.get(k) != v]
            desc = [describe(k, o, v) for k, o, v in diff] or ["No change (the baseline)"]
        else:
            desc = info.get("changes") or [key]
        tr = round(s["train"] * n_train) if n_train else None
        va = round(s["val"] * n_val) if n_val else None
        if key == "v0":
            origin = "Baseline"
        elif not info:
            origin = key
        else:
            origin = f"{LINE_SHORT.get(info['line'], info['line'])} line, round {info['round']}"
        rows.append({"key": key, "origin": origin, "settings": desc, "train": s["train"], "val": s["val"],
                     "items": (tr + va) if tr is not None and va is not None else None,
                     "old_decision": info.get("decision") if key != "v0" else "baseline",
                     "old_reason": info.get("reason", ""), "winner": key == winner})
    rows.sort(key=lambda r: (-(r["items"] or 0), -r["train"]))
    best = {"train": {"score": scores[winner]["train"], "n": n_train},
            "val": {"score": scores[winner]["val"], "n": n_val}}
    if final_test:
        best["test"] = final_test
    win = found.get(winner, {})
    return {"note": reselect.get("note"), "gt_version": reselect.get("gt_version"), "winner": winner,
            "winner_origin": next(r["origin"] for r in rows if r["key"] == winner),
            "winner_old": {"decision": win.get("decision"), "reason": win.get("reason"),
                           "line": LINE_SHORT.get(win.get("line"), win.get("line")), "round": win.get("round")},
            "config": reselect.get("config"),
            "changes": [describe(c["path"], base_flat.get(c["path"]), c["value"]) for c in reselect.get("changes", [])],
            "n_configs": len(scores), "n_train": n_train, "n_val": n_val, "rows": rows,
            # the test entry may be logged before the reselect entry; the page says the pick used train/val only
            "final_logged_first": bool(final) and reselect.get("ts", "") > final.get("ts", "")}, best


def task_block(task, task_root, metric, best_path, base_path, items, traced, records, prompts, base_cfg):
    lines, versions_by_root, by_path, sel = [], {}, {}, None
    for name, root in line_dirs(task_root):
        hist, versions, extras = loop_history(root, metric, name)
        lines.append(hist)
        versions_by_root[root.resolve()] = versions
        if extras["reselect"]:
            sel = extras
    reselect = None
    if sel:
        reselect, best = reselect_block(task_root, lines, sel["reselect"], sel["final"], sel["final_test"],
                                        sel["last_rebase"], base_cfg)
        by_path[(ROOT / sel["reselect"]["config"]).resolve()] = best
        if (ROOT / best_path).resolve() != (ROOT / sel["reselect"]["config"]).resolve():
            print(f"warning: {task}: best config {best_path} is not the reselected {sel['reselect']['config']}")

    def metrics_for(path):
        if (ROOT / path).resolve() in by_path:
            return by_path[(ROOT / path).resolve()]
        root, v = config_line_version(path)
        return versions_by_root.get(root, {}).get(v, {}) if root else {}
    mb, m0 = metrics_for(best_path), metrics_for(base_path)
    n = len(traced)
    best_ok = sum(t["best"]["correct"] for t in traced)
    base_ok = sum(t["base"]["correct"] for t in traced)
    fixed = sum(t["best"]["correct"] and not t["base"]["correct"] for t in traced)
    broken = sum(t["base"]["correct"] and not t["best"]["correct"] for t in traced)
    watch = {"n": n, "best": best_ok / n, "base": base_ok / n, "fixed": fixed, "broken": broken}
    results = {s: {"base": m0.get(s), "best": mb.get(s)} for s in ("train", "val", "test")}
    results["watch"] = {"base": {"score": watch["base"], "n": n}, "best": {"score": watch["best"], "n": n,
                                                                         "fixed_broken": [fixed, broken]}}
    return {"task": task, "best_config": best_path, "base_config": base_path,
            "best_label": config_label(best_path), "base_label": config_label(base_path),
            "same_config": Path(best_path).resolve() == Path(base_path).resolve(),
            "results": results, "watch": watch, "lines": lines, "reselect": reselect,
            # work after the loop (hand fixes, experiments not kept), told after the loop's story
            "after": [e for e in read_ledger(task_root / "ledger.jsonl") if e["type"] in ("fix", "experiment")],
            "examples": records, "prompts": prompts}


def main():
    # the event task's final config was picked after the loop (ledger "reselect" entry); fall back to
    # the main line's incumbent when no such pick exists
    ev_final = "experiments_event/configs/final.json"
    if not (ROOT / ev_final).exists():
        ev_final = json.loads((ROOT / "experiments_event/state.json").read_text())["best_config"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quote-best", default="experiments/configs/v3.json")
    ap.add_argument("--quote-base", default="experiments/configs/v0.json")
    ap.add_argument("--event-best", default=ev_final)
    ap.add_argument("--event-base", default="experiments_event/configs/v0.json")
    ap.add_argument("--n", type=int, default=40, help="examples per task embedded in the page")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="docs/demo")
    ap.add_argument("--reuse-data", action="store_true",
                    help="re-render the page from the data already embedded in <out>/index.html (template edits only)")
    a = ap.parse_args()
    out = ROOT / a.out
    if a.reuse_data:
        page = (out / "index.html").read_text()
        m = re.search(r'^var DEMO_DATA = (.*);$', page, re.M)
        write_pages(json.loads(m.group(1).replace("<\\/", "</")), out)
        return
    if a.workers > 4:
        sys.exit("--workers is capped at 4 (shared machine)")

    cq_best, cq_base = qb_config.load(ROOT / a.quote_best), qb_config.load(ROOT / a.quote_base)
    ce_best, ce_base = ev_config.load(ROOT / a.event_best), ev_config.load(ROOT / a.event_base)
    for cfg, p in ((cq_best, a.quote_best), (cq_base, a.quote_base), (ce_best, a.event_best), (ce_base, a.event_base)):
        check_haiku(cfg, p)
    tr = Tracer(a.workers)

    q_items = [json.loads(line) for line in open(ROOT / "bench/qb_watch.jsonl")]
    q_traced = quote_task(tr, q_items, cq_best, cq_base)
    q_by_id = {it["quote_id"]: (it, t) for it, t in zip(q_items, q_traced)}
    q_pick = pick(q_items, lambda it: it["quote_id"], a.n)
    q_records = [quote_record(tr, it, q_by_id[it["quote_id"]][1]) for it in q_pick]

    e_items = [json.loads(line) for line in open(ROOT / "bench/ev_watch.jsonl")]
    e_traced = event_task(tr, e_items, ce_best, ce_base)
    e_by_id = {str(it["event_id"]): (it, t) for it, t in zip(e_items, e_traced)}
    e_pick = pick(e_items, lambda it: str(it["event_id"]), a.n)
    e_records = [event_record(it, e_by_id[str(it["event_id"])][1]) for it in e_pick]

    new_calls = sum(t[k]["res"]["llm_calls"]["made"] for t in q_traced + e_traced for k in ("best", "base"))
    data = {
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "new_llm_calls": new_calls,
        "quotes": task_block("quotes", ROOT / "experiments", "first_any", a.quote_best, a.quote_base, q_items,
                             q_traced, q_records, {"best": cq_best["verify"]["prompt"]}, cq_base),
        "events": task_block("events", ROOT / "experiments_event", "first_report", a.event_best, a.event_base,
                             e_items, e_traced, e_records, {"best": ce_best["verify"]["prompt"]}, ce_base),
        # spot check of 20 event labels by two Fable agents, before earlier answers were sent back to the labeller
        "gt_audit": {"n": 20, "right": 11, "wrong_event": 1, "not_event": 1, "earlier_missed": 7, "within_hour": 5},
        # ground truth v3.3: the labeller (Sonnet 5.5, low effort) reviewed every answer earlier than the label
        # from all fully scored configs (train/val) plus baseline and final config (all splits); an independent
        # Fable audit re-judged 35 accepted + 15 rejected of those reviews
        "gt_review": {"gt_version": "3.3", "reviewed": 112, "accepted": 35, "audit_n": 50, "audit_accepted": 35,
                      "audit_rejected": 15, "audit_agree": 45, "accepted_precision": 0.89},
    }
    write_pages(data, ROOT / a.out)
    print(f"watch: quotes {data['quotes']['watch']}  events {data['events']['watch']}  new Haiku calls {new_calls}")


def write_pages(data, out):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("__DEMO_DATA__", payload)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)
    # standalone copy: the artifact's <title>, meta and <style> go in <head>, the rest in <body>
    style_end = page.index("</style>") + len("</style>")
    standalone = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                  "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                  f"{page[:style_end]}\n</head>\n<body>\n{page[style_end:].lstrip()}</body>\n</html>\n")
    (out / "standalone.html").write_text(standalone)
    for f in ("index.html", "standalone.html"):
        print(f"{out / f}: {(out / f).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
