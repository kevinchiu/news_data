"""Self-improvement loop for the Quotebank tracer.

Each round an improver model (Claude Opus 5.5) reads the current config, the experiment
history and categorised failure cases from a fixed train subset, and proposes up to three
small config changes. Candidates are screened cheaply (retrieval changes by recall@k with no
LLM calls; verifier changes on the failing items plus passing "sentinels"), the most promising
one is scored on the full train subset, and it is accepted only if it fixes clearly more items
than it breaks (threshold set from the measured run-to-run noise) and does not lose on val.
Everything is logged to experiments/ledger.jsonl with versioned configs in experiments/configs/.

  python -I scripts/improve.py baseline          # score base config, measure noise, set threshold
  python -I scripts/improve.py run --rounds 10   # run rounds (resumable)
  python -I scripts/improve.py final             # score the best config on the held-out test split

Several improvement lines can run side by side, each restricted to a group of knobs and kept in
experiments/lines/<name>/ (started from the main baseline); `merge` combines their best configs:
  python -I scripts/improve.py run --line retrieval --focus retrieval --workers 8
  python -I scripts/improve.py merge --lines all,retrieval,verify
"""
import argparse
import json
import math
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qb_config  # noqa: E402
from evaluate import evaluate as _evaluate, load_items  # noqa: E402
from verify import claude_json  # noqa: E402

ROOT = Path("experiments")
EXP, LEDGER, STATE, FOCUS = ROOT, ROOT / "ledger.jsonl", ROOT / "state.json", None
TRAIN_N, SENTINELS, MAX_PROPOSALS, FULL_EVALS = 60, 15, 3, 2
MODE = "paraphrase"  # which query form of each benchmark item is traced (see --mode)
WORKERS = 12  # concurrent traces (each runs headless `claude -p` Haiku calls); the time is in those calls
evaluate = partial(_evaluate, workers=WORKERS, mode=MODE)
IMPROVER_MODEL, IMPROVER_EFFORT = "claude-opus-5-5", "high"
KNOB_HELP = {
    "retrieval.k": "number of BM25 candidates passed to the verifier (5-60)",
    "retrieval.index": "'quotation' (quote text only) or 'quotation_speaker' (quote text + speaker name)",
    "retrieval.query_template": "'raw' statement, 'content' (strip 'X said that'), or 'content_plus_speaker'",
    "retrieval.speaker_filter": "'off' or 'surname' (keep candidates whose speaker contains the statement's "
                                "speaker surname, plus unattributed quotes)",
    "verify.effort": "Haiku effort: low / medium / high",
    "verify.batch": "candidates per Haiku call (5-30)",
    "verify.prompt": "the verifier system prompt (200-4000 chars); must stay general, no benchmark content",
    "verify.show_date": "show each candidate's first-seen date to the verifier",
    "verify.speaker_hint": "annotate candidates with '(speaker matches claim)' / '(different speaker?)' / "
                           "'(speaker unknown)'",
}

IMPROVER_SYSTEM = """You are the improvement step of a self-improving provenance tracer.

The tracer takes a news statement (usually reported speech, e.g. "X said that ...") and a date, \
retrieves candidate quotations from a quote corpus with BM25, asks a small verifier model which \
candidates state the claim, then reports the earliest article that carried a matched quotation. \
It is scored on whether that earliest article is the true first appearance.

You get the current configuration, the knobs you may change, the history of earlier proposals \
and their outcomes, metrics, and categorised failure cases. Propose up to three alternative \
single changes (each may touch one or two closely related knobs), ranked best first.

Rules:
- Target a failure category that accounts for many failures, and say which.
- Fixes must generalise. Never put names, quotes, numbers or topics from the failure cases \
into the verifier prompt; prompts that do are rejected automatically.
- Don't repeat a change that was already tried and rejected unless you change it materially.
- Prefer changes that also reduce or keep verifier calls per claim."""

VALUE = {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}, {"type": "null"}]}
IMPROVER_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "proposals": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["not_retrieved", "rejected_by_verifier",
                                                        "false_positive_earlier", "other"]},
                "rationale": {"type": "string"},
                "changes": {"type": "array", "items": {
                    "type": "object", "properties": {"path": {"type": "string"}, "value": VALUE},
                    "required": ["path", "value"], "additionalProperties": False}},
            },
            "required": ["category", "rationale", "changes"], "additionalProperties": False}},
    },
    "required": ["analysis", "proposals"], "additionalProperties": False,
})


# ---------- improvement lines ----------

def configure(line=None, focus=None, workers=WORKERS):
    """Point the loop at one improvement line; a new line starts from the main baseline."""
    global EXP, LEDGER, STATE, FOCUS, evaluate
    EXP = ROOT if line in (None, "all") else ROOT / "lines" / line
    LEDGER, STATE, FOCUS = EXP / "ledger.jsonl", EXP / "state.json", focus
    evaluate = partial(_evaluate, workers=workers, mode=MODE)
    if EXP != ROOT and not STATE.exists():
        (EXP / "configs").mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "configs" / "v0.json", EXP / "configs" / "v0.json")
        base = json.loads((ROOT / "state.json").read_text())
        base_entry = next(json.loads(l) for l in open(ROOT / "ledger.jsonl") if json.loads(l)["type"] == "baseline")
        # baseline metrics are the starting point; reset the trajectory
        start = {k: base[k] for k in ("threshold", "noise", "baseline_test")}
        start |= {"best_version": 0, "best_config": str(EXP / "configs" / "v0.json"), "round": 0,
                  "train_metrics": base_entry["train"], "val_metrics": base_entry["val"], "focus": focus}
        STATE.write_text(json.dumps(start, indent=2))
        LEDGER.write_text(json.dumps(base_entry | {"line": line, "focus": focus}) + "\n")


def knobs():
    return {p: h for p, h in KNOB_HELP.items() if not FOCUS or p.startswith(FOCUS + ".")}


# ---------- state & ledger ----------

def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save_state(state):
    STATE.write_text(json.dumps(state, indent=2))


def log(entry):
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def ledger():
    return [json.loads(l) for l in open(LEDGER)] if LEDGER.exists() else []


def save_config(cfg, version):
    path = EXP / "configs" / f"v{version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return str(path)


# ---------- evaluation helpers ----------

def by_id(rows):
    return {r["quote_id"]: r for r in rows}


def paired(old_rows, new_rows, key="first_any"):
    old, new = by_id(old_rows), by_id(new_rows)
    fixed = sum(1 for q in new if new[q][key] and not old[q][key])
    broken = sum(1 for q in new if old[q][key] and not new[q][key])
    return fixed, broken


def category(r):
    if r["first_any"]:
        return "ok"
    if not r["retrieved"]:
        return "not_retrieved"
    if not r["gt_matched"]:
        return "rejected_by_verifier"
    if r["fp_earlier"]:
        return "false_positive_earlier"
    return "other"


def leak_tokens(items):
    """Proper nouns and multi-digit numbers that appear in train items (quotes, paraphrases, speakers)."""
    toks = set()
    for it in items:
        for text in (it["quotation"], it["paraphrase"], it.get("paraphrase_hard", ""), it["speaker"] or ""):
            for w in re.findall(r"\b[A-Z][a-z]{3,}\b|\b\d{2,}\b", text):
                toks.add(w)
    return toks


def prompt_leaks(new_prompt, base_prompt, toks):
    added = set(re.findall(r"\b[A-Z][a-z]{3,}\b|\b\d{2,}\b", new_prompt)) - \
        set(re.findall(r"\b[A-Z][a-z]{3,}\b|\b\d{2,}\b", base_prompt))
    return sorted(added & toks)


# ---------- improver ----------

def failure_report(items, rows, con, limit=16):
    its = by_id(items)
    cats = {}
    for r in rows:
        cats.setdefault(category(r), []).append(r)
    lines = ["FAILURE COUNTS: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(cats.items()) if k != "ok")]
    shown = 0
    for cat in ("rejected_by_verifier", "not_retrieved", "false_positive_earlier", "other"):
        for r in cats.get(cat, [])[: max(3, limit // 3)]:
            if shown >= limit:
                break
            it = its[r["quote_id"]]
            lines += ["", f"[{cat}] STATEMENT: {it[MODE]}",
                      f"  TRUE QUOTE ({it['speaker']}): \"{it['quotation']}\""]
            if r["matched"]:
                wrong = [q for q in r["matched"] if q not in it["variant_ids"]]
                for (q, s, sp) in con.execute(
                        "SELECT quote_id, quotation, coalesce(speaker,'None') FROM quotes WHERE quote_id IN (SELECT unnest(?))",
                        [wrong[:3]]).fetchall():
                    lines.append(f"  WRONGLY MATCHED ({sp}): \"{s[:200]}\"")
            if r["pred_first_any"]:
                lines.append(f"  predicted first {r['pred_first_any']['date'][:16]} vs true {it['gt_first_any']['date'][:16]}")
            shown += 1
    return "\n".join(lines)


def ask_improver(cfg, metrics, report, history):
    knob_lines = "\n".join(f"- {p} = {json.dumps(qb_config.get(cfg, p)) if p != 'verify.prompt' else '(below)'}: {h}"
                           for p, h in knobs().items())
    hist = "\n".join(
        f"- round {h['round']}: {h.get('category')} {json.dumps(h.get('changes'))[:300]} -> {h['decision']}"
        f" ({h.get('reason', '')})" for h in history if h.get("type") == "proposal") or "(none yet)"
    prompt = (f"KNOBS (current values):\n{knob_lines}\n\nCURRENT VERIFIER PROMPT:\n{cfg['verify']['prompt']}\n\n"
              f"TRAIN METRICS: {json.dumps(metrics)}\n\nHISTORY:\n{hist}\n\n{report}")
    out, stats = claude_json(prompt, IMPROVER_SYSTEM, IMPROVER_SCHEMA, IMPROVER_MODEL, IMPROVER_EFFORT)
    return out, stats


# ---------- commands ----------

def cmd_baseline(con):
    EXP.mkdir(exist_ok=True)
    cfg = qb_config.load()
    train = load_items("train", TRAIN_N)
    m_tr, rows_tr = evaluate(cfg, train, con=con)
    _, rows_noise = evaluate(cfg, train, nonce="noise-1", con=con)
    fixed, broken = paired(rows_tr, rows_noise)
    flips = fixed + broken
    # under pure noise, net = fixed - broken has sd ~ sqrt(flips); require a one-sided 95% margin
    threshold = max(2, math.ceil(1.64 * math.sqrt(flips)))
    m_val, rows_val = evaluate(cfg, load_items("val"), con=con)
    m_test, _ = evaluate(cfg, load_items("test"), con=con)
    path = save_config(cfg, 0)
    state = {"best_version": 0, "best_config": path, "round": 0, "threshold": threshold,
             "noise": {"flips": flips, "fixed": fixed, "broken": broken, "n": len(train)},
             "train_metrics": m_tr, "val_metrics": m_val, "baseline_test": m_test}
    save_state(state)
    log({"type": "baseline", "round": 0, "version": 0, "train": m_tr, "val": m_val, "test": m_test,
         "noise": state["noise"], "threshold": threshold})
    print(json.dumps(state, indent=2))


def cmd_run(con, rounds, budget):
    state = load_state()
    if not state:
        sys.exit("run `improve.py baseline` first")
    train, val = load_items("train", TRAIN_N), load_items("val")
    toks = leak_tokens(load_items("train") + load_items("val"))
    base_prompt = qb_config.load()["verify"]["prompt"]
    spent = sum(e.get("new_calls", 0) for e in ledger())
    for _ in range(rounds):
        if spent >= budget:
            print(f"budget reached ({spent} new Haiku calls)")
            break
        state["round"] += 1
        rnd = state["round"]
        cfg = qb_config.load(state["best_config"])
        m_cur, rows_cur = evaluate(cfg, train, con=con)  # cached: free for the incumbent
        report = failure_report(train, rows_cur, con)
        out, stats = ask_improver(cfg, m_cur, report, ledger())
        log({"type": "improver", "round": rnd, "analysis": out["analysis"], "n_proposals": len(out["proposals"]),
             "latency_s": stats["latency_s"]})
        print(f"\n=== round {rnd}: {out['analysis'][:300]}")

        # validate and screen all proposals in parallel; fully evaluate only the most promising one
        fails = [r["quote_id"] for r in rows_cur if not r["first_any"]]
        passes = [r["quote_id"] for r in rows_cur if r["first_any"]][:SENTINELS]
        probe_ids = set(fails + passes)
        probe = [it for it in train if it["quote_id"] in probe_ids]
        probe_cur = [r for r in rows_cur if r["quote_id"] in probe_ids]

        def screen(ip):
            i, p = ip
            entry = {"type": "proposal", "round": rnd, "rank": i, "category": p["category"],
                     "rationale": p["rationale"], "changes": p["changes"]}
            outside = [c["path"] for c in p["changes"] if c["path"] not in knobs()]
            if outside:
                return entry | {"decision": "invalid", "reason": f"outside this line's knobs: {outside}"}, None, None
            try:
                cand = qb_config.apply(cfg, p["changes"])
            except ValueError as e:
                return entry | {"decision": "invalid", "reason": str(e)}, None, None
            leaks = prompt_leaks(cand["verify"]["prompt"], base_prompt, toks)
            if leaks:
                return entry | {"decision": "rejected", "reason": f"prompt leaks benchmark tokens {leaks[:8]}"}, None, None
            touched = {c["path"].split(".")[0] for c in p["changes"]}
            if "retrieval" in touched:
                r_new, _ = evaluate(cand, train, retrieval_only=True, con=con)
                if r_new["recall_at_k"] < m_cur["recall_at_k"]:
                    return entry | {"decision": "rejected", "reason": f"recall@k {m_cur['recall_at_k']:.2f} -> "
                                                                       f"{r_new['recall_at_k']:.2f} (free screen)"}, None, None
            # pre-screen on the failing items plus passing sentinels
            m_s, rows_s = evaluate(cand, probe, con=con)
            fx, br = paired(probe_cur, rows_s)
            entry["new_calls"] = m_s["new_calls"]
            if fx - br <= 0:
                return entry | {"decision": "rejected", "reason": f"pre-screen fixed {fx}, broke {br}"}, None, None
            return entry, cand, fx - br

        with ThreadPoolExecutor(MAX_PROPOSALS) as ex:
            screened = list(ex.map(screen, enumerate(out["proposals"][:MAX_PROPOSALS])))
        spent += sum(e.get("new_calls", 0) for e, _, _ in screened)
        passed = sorted([s for s in screened if s[1] is not None], key=lambda s: (-s[2], s[0]["rank"]))
        for e, _, _ in screened:
            if e.get("decision"):
                log(e)
        for e, _, score in passed[FULL_EVALS:]:
            log(e | {"decision": "not_evaluated", "reason": f"pre-screen net {score:+d}, better proposals were evaluated"})
        if not passed:
            log({"type": "round_end", "round": rnd, "accepted": False, "reason": "no proposal passed screening"})
            save_state(state)
            continue

        def full_eval(entry, cand):
            # train and val run at the same time (val is wasted if train fails; Haiku budget is ample)
            with ThreadPoolExecutor(2) as ex:
                f_tr, f_val = ex.submit(evaluate, cand, train, con=con), ex.submit(evaluate, cand, val, con=con)
                (m_new, rows_new), (m_val_new, _) = f_tr.result(), f_val.result()
            fixed, broken = paired(rows_cur, rows_new)
            net = fixed - broken
            reason, second, passes = f"train net {net:+d} < threshold {state['threshold']}", None, \
                net >= state["threshold"]
            if 0 < net < state["threshold"] and TRAIN_N < len(load_items("train")):
                # borderline: look again on the rest of the train split, against a threshold scaled to
                # the larger sample (noise grows ~ sqrt(n))
                seen = {it["quote_id"] for it in train}
                rest = [it for it in load_items("train") if it["quote_id"] not in seen]
                m_rc, rows_rc = evaluate(cfg, rest, con=con)
                m_rn, rows_rn = evaluate(cand, rest, con=con)
                f2, b2 = paired(rows_rc, rows_rn)
                n_all = len(train) + len(rest)
                full_threshold = max(state["threshold"], math.ceil(
                    1.64 * math.sqrt(state["noise"]["flips"] * n_all / len(train))))
                second = {"n": n_all, "net": net + f2 - b2, "threshold": full_threshold,
                          "new_calls": m_rc["new_calls"] + m_rn["new_calls"]}
                passes = second["net"] >= full_threshold
                reason = f"borderline train net {net:+d}; second look on {n_all} items net {second['net']:+d} " \
                         f"{'>=' if passes else '<'} {full_threshold}"
            decision = "rejected"
            if passes:
                train_note = f"train net {net:+d}" + (f" (second look {second['net']:+d}/{second['n']})" if second else "")
                if m_val_new["first_any"] >= state["val_metrics"]["first_any"]:
                    decision, reason = "accepted", f"{train_note}, val {state['val_metrics']['first_any']:.3f}" \
                                                   f" -> {m_val_new['first_any']:.3f}"
                else:
                    reason = f"{train_note} but val dropped {state['val_metrics']['first_any']:.3f} -> " \
                             f"{m_val_new['first_any']:.3f}"
            calls = entry.get("new_calls", 0) + m_new["new_calls"] + m_val_new["new_calls"] + \
                (second or {}).get("new_calls", 0)
            return {"entry": entry, "cand": cand, "decision": decision, "reason": reason, "net": net,
                    "fixed": fixed, "broken": broken, "train": m_new, "val": m_val_new, "second": second,
                    "calls": calls}

        # fully evaluate the top proposals side by side; accept at most one (best net, then val)
        with ThreadPoolExecutor(FULL_EVALS) as ex:
            results = list(ex.map(lambda pc: full_eval(pc[0], pc[1]), [(e, c) for e, c, _ in passed[:FULL_EVALS]]))
        winners = sorted([r for r in results if r["decision"] == "accepted"],
                         key=lambda r: (-r["net"], -r["val"]["first_any"]))
        for r in results:
            if winners and r is not winners[0] and r["decision"] == "accepted":
                r["decision"], r["reason"] = "runner_up", r["reason"] + "; another proposal scored higher"
            spent += r["calls"]
            log(r["entry"] | {"decision": r["decision"], "reason": r["reason"], "train": r["train"], "val": r["val"],
                              "fixed": r["fixed"], "broken": r["broken"], "diff": qb_config.diff(cfg, r["cand"]),
                              "second_look": r["second"], "new_calls": r["calls"]})
            print(f"  -> {r['decision']}: {r['reason']}")
        decision = "accepted" if winners else "rejected"
        if winners:
            w = winners[0]
            state["best_version"] += 1
            state["best_config"] = save_config(w["cand"], state["best_version"])
            state["train_metrics"], state["val_metrics"] = w["train"], w["val"]
        log({"type": "round_end", "round": rnd, "accepted": decision == "accepted"})
        save_state(state)
    print(f"\nbest: v{state['best_version']}  train {state['train_metrics']['first_any']:.3f}  "
          f"val {state['val_metrics']['first_any']:.3f}  ({spent} new Haiku calls in total)")


def cmd_rebaseline(con):
    """Re-score after a ground-truth change: v0 (train/val/test + noise) and this line's best (train/val).
    Keeps the line's versions; the ledger records old vs new numbers."""
    state = load_state()
    train, val, test = load_items("train", TRAIN_N), load_items("val"), load_items("test")
    v0 = qb_config.load(EXP / "configs" / "v0.json")
    m0_tr, rows0 = evaluate(v0, train, con=con)
    _, rows_noise = evaluate(v0, train, nonce="noise-1", con=con)
    fixed, broken = paired(rows0, rows_noise)
    flips = fixed + broken
    threshold = max(2, math.ceil(1.64 * math.sqrt(flips)))
    m0_val, _ = evaluate(v0, val, con=con)
    m0_test, _ = evaluate(v0, test, con=con)
    best = qb_config.load(state["best_config"])
    mb_tr, _ = evaluate(best, train, con=con)
    mb_val, _ = evaluate(best, val, con=con)
    old = {k: state.get(k) for k in ("train_metrics", "val_metrics", "baseline_test", "threshold", "noise")}
    state.update(threshold=threshold, noise={"flips": flips, "fixed": fixed, "broken": broken, "n": len(train)},
                 baseline_test=m0_test, train_metrics=mb_tr, val_metrics=mb_val, gt_version=2)
    save_state(state)
    log({"type": "rebaseline", "round": state["round"], "gt_version": 2, "threshold": threshold,
         "noise": state["noise"], "v0": {"train": m0_tr, "val": m0_val, "test": m0_test},
         "best": {"version": state["best_version"], "train": mb_tr, "val": mb_val}, "old": old})
    print(f"{EXP}: v0 train {m0_tr['first_any']:.3f} val {m0_val['first_any']:.3f} test {m0_test['first_any']:.3f} | "
          f"best v{state['best_version']} train {mb_tr['first_any']:.3f} val {mb_val['first_any']:.3f} | "
          f"was val {old['val_metrics']['first_any']:.3f} | threshold {threshold} (flips {flips})")


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def cmd_merge(con, lines):
    """Combine the lines' best configs (retrieval from one, verifier from another) and keep the best."""
    best = {}
    for line in lines:
        st = json.loads(((ROOT if line == "all" else ROOT / "lines" / line) / "state.json").read_text())
        best[line] = qb_config.load(st["best_config"])
    cands = dict(best)
    if "retrieval" in best and "verify" in best:
        combo = json.loads(json.dumps(best["verify"]))
        combo["retrieval"] = best["retrieval"]["retrieval"]
        cands["retrieval+verify"] = combo
    train, val = load_items("train", TRAIN_N), load_items("val")
    scores = {}
    for name, cfg in cands.items():
        m_tr, _ = evaluate(cfg, train, con=con)
        m_v, _ = evaluate(cfg, val, con=con)
        scores[name] = {"train": m_tr["first_any"], "val": m_v["first_any"],
                        "combined": (m_tr["first_any"] * len(train) + m_v["first_any"] * len(val)) / (len(train) + len(val))}
        print(f"{name:<18} train {m_tr['first_any']:.3f}  val {m_v['first_any']:.3f}")
    winner = max(scores, key=lambda k: scores[k]["combined"])
    path = ROOT / "configs" / "merged.json"
    path.write_text(json.dumps(cands[winner], indent=2))
    with open(ROOT / "merge.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "scores": scores, "winner": winner}) + "\n")
    print(f"winner: {winner} -> {path}")


def cmd_final(con, config=None):
    state = load_state()
    test = load_items("test")
    m_best, rows_best = evaluate(qb_config.load(config or state["best_config"]), test, con=con)
    _, rows_base = evaluate(qb_config.load(ROOT / "configs" / "v0.json"), test, con=con)
    fixed, broken = paired(rows_base, rows_best)
    res = {"base": state["baseline_test"]["first_any"], "best": m_best["first_any"],
           "best_ci95": wilson(m_best["first_any"], len(test)), "fixed": fixed, "broken": broken,
           "n": len(test), "best_version": state["best_version"], "best_metrics": m_best}
    log({"type": "final", **res})
    print(json.dumps(res, indent=2))


def main():
    global TRAIN_N, MODE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["baseline", "rebaseline", "run", "merge", "final"])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=2500, help="max new Haiku calls across all rounds")
    ap.add_argument("--line", help="improvement line name (default: the main line, experiments/)")
    ap.add_argument("--focus", choices=["retrieval", "verify"], help="restrict the line to one knob group")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--mode", default=MODE, help="query form to trace: paraphrase | paraphrase_hard | verbatim")
    ap.add_argument("--train-n", type=int, default=TRAIN_N,
                    help="train items scored per round (the full split is 145); rebaseline after changing it")
    ap.add_argument("--lines", default="all,retrieval,verify", help="for merge")
    ap.add_argument("--config", help="for final: config to score (default: this line's best)")
    a = ap.parse_args()
    TRAIN_N, MODE = a.train_n, a.mode
    configure(a.line, a.focus, a.workers)
    con = duckdb.connect("data/news.duckdb", read_only=True)
    # many traces run at once and each BM25 query would otherwise use every core; several loop
    # processes share the machine, so keep each DuckDB instance small
    con.execute("SET threads TO 3")
    {"baseline": lambda: cmd_baseline(con), "rebaseline": lambda: cmd_rebaseline(con),
     "final": lambda: cmd_final(con, a.config),
     "merge": lambda: cmd_merge(con, a.lines.split(",")),
     "run": lambda: cmd_run(con, a.rounds, a.budget)}[a.command]()


if __name__ == "__main__":
    main()
