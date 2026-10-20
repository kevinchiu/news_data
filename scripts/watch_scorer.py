"""Score every config version of the improvement lines on the held-out watch set (display only).

Every --interval seconds, discovers experiments/configs/v*.json (line "all") and
experiments/lines/<line>/configs/v*.json, hashes each config, and evaluates each config hash
not yet scored on bench/qb_watch.jsonl (paraphrase mode). Results are appended to
experiments/watch.jsonl, one line per scored hash:
  {ts, config_hash, metrics, versions: [{line, version, config_path}], ...}
and a small alias line {ts, config_hash, alias: true, versions: [...]} when a version turns up
whose config was already scored. Order: v0 first, then versions in ascending order.
Resumable: on start it re-reads watch.jsonl. The improvement loop never reads any of this.

Usage:
  python -I scripts/watch_scorer.py                 # long-running, polls every 60 s
  python -I scripts/watch_scorer.py --once          # one pass, then exit
  python -I scripts/watch_scorer.py --once --only-best   # v0 + each line's current best only
  python -I scripts/watch_scorer.py --task event    # event tracer: experiments_event/, bench/ev_watch.jsonl
"""
import argparse
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev_config  # noqa: E402
import evaluate_event  # noqa: E402
import qb_config  # noqa: E402
from evaluate import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
WATCH = ROOT / "bench" / "qb_watch.jsonl"
OUT = EXP / "watch.jsonl"
LOCK = EXP / ".watch_scorer.lock"
# per task: experiments dir, watch set, config module, evaluate, query mode, headline metric, error metric
TASKS = {"quote": ("experiments", "qb_watch.jsonl", qb_config, evaluate, "paraphrase", "first_any", "fp_earlier"),
         "event": ("experiments_event", "ev_watch.jsonl", ev_config, evaluate_event.evaluate, "query",
                   "first_report", "early")}
CONFIG, MODE, METRIC, ERROR = qb_config, "paraphrase", "first_any", "fp_earlier"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def config_hash(path):
    """Hash of the effective config (prompt file references resolved, keys sorted)."""
    cfg = CONFIG.load(path)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16], cfg


def discover():
    """[(line, version, path)] for every saved config version, v0s first, then ascending version."""
    dirs = [("all", EXP / "configs")]
    if (EXP / "lines").is_dir():
        dirs += [(d.name, d / "configs") for d in sorted((EXP / "lines").iterdir()) if d.is_dir()]
    out = []
    for i, (line, d) in enumerate(dirs):
        for p in d.glob("v*.json"):
            m = re.fullmatch(r"v(\d+)\.json", p.name)
            if m:
                out.append((int(m.group(1)), i, line, p))
    return [(line, v, p) for v, _, line, p in sorted(out)]


def current_bests():
    """{line: best_version} from each line's state.json."""
    out = {}
    for line, d in [("all", EXP)] + [(d.name, d) for d in sorted((EXP / "lines").glob("*")) if d.is_dir()]:
        try:
            out[line] = json.loads((d / "state.json").read_text())["best_version"]
        except (OSError, ValueError, KeyError):
            pass
    return out


def read_done():
    """(scored hashes, (line, version) pairs already recorded) from watch.jsonl."""
    scored, seen = set(), set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # torn line from an interrupted write
            if not rec.get("alias") and rec.get("metrics"):
                scored.add(rec["config_hash"])
            seen.update((v["line"], v["version"]) for v in rec.get("versions", []))
    return scored, seen


def append(rec):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")} | rec
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()


def run_eval(cfg, items, workers):
    """evaluate() in a daemon thread so Ctrl-C / SIGTERM can stop the process promptly
    (finished Haiku calls are already in the LLM cache, so an interrupted score resumes cheaply)."""
    box = {}

    def target():
        try:
            box["res"] = evaluate(cfg, items, mode=MODE, workers=workers)
        except BaseException as e:  # surfaced in the main thread
            box["err"] = e
    t = threading.Thread(target=target, daemon=True)
    t.start()
    while t.is_alive():
        t.join(0.5)
    if "err" in box:
        raise box["err"]
    return box["res"]


def one_pass(items, workers, only_best=False):
    scored, seen = read_done()
    found, bad = [], []
    for line, v, p in discover():
        if (line, v) in seen:
            continue
        try:
            h, cfg = config_hash(p)
        except (OSError, ValueError, KeyError):
            bad.append(str(p.relative_to(ROOT)))  # probably being written right now; retry next pass
            continue
        found.append({"line": line, "version": v, "config_path": str(p.relative_to(ROOT)), "hash": h, "cfg": cfg})
    if bad:
        log(f"skipping unreadable config(s) this pass: {', '.join(bad)}")
    bests = current_bests() if only_best else None
    n_scored = 0
    for f in found:
        if (f["line"], f["version"]) in seen:
            continue  # recorded earlier in this pass, together with an identical config
        same = [g for g in found if g["hash"] == f["hash"] and (g["line"], g["version"]) not in seen]
        versions = [{k: g[k] for k in ("line", "version", "config_path")} for g in same]
        if f["hash"] in scored:
            append({"config_hash": f["hash"], "alias": True, "versions": versions})
            log(f"{f['hash']} already scored; alias for {', '.join(_name(g) for g in versions)}")
        else:
            if only_best and not any(g["version"] == 0 or bests.get(g["line"]) == g["version"] for g in same):
                continue
            log(f"scoring {f['hash']} ({', '.join(_name(g) for g in versions)}) on {len(items)} watch items")
            t0 = time.time()
            m, rows = run_eval(f["cfg"], items, workers)
            append({"config_hash": f["hash"], "metrics": m, "versions": versions,
                    "n_items": len(items), "elapsed_s": round(time.time() - t0, 1),
                    "correct": "".join("1" if r[METRIC] else "0" for r in rows)})
            scored.add(f["hash"])
            n_scored += 1
            log(f"  {METRIC} {m[METRIC]:.3f}  recall@k {m['recall_at_k']:.3f}  {ERROR} "
                f"{m[ERROR]:.3f}  new Haiku calls {m['new_calls']}  ({time.time() - t0:.0f}s)")
        seen.update((g["line"], g["version"]) for g in same)
    return n_scored


def _name(g):
    return f"{g['line']}:v{g['version']}"


def main():
    global EXP, WATCH, OUT, LOCK, CONFIG, evaluate, MODE, METRIC, ERROR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--only-best", action="store_true",
                    help="score only v0 and each line's current best (others are left for later passes)")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--task", choices=list(TASKS), default="quote")
    a = ap.parse_args()
    exp, watch, CONFIG, evaluate, MODE, METRIC, ERROR = TASKS[a.task]
    EXP, WATCH = ROOT / exp, ROOT / "bench" / watch
    OUT, LOCK = EXP / "watch.jsonl", EXP / ".watch_scorer.lock"
    os.chdir(ROOT)  # evaluate() and the LLM cache use project-relative paths

    import fcntl
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another watch_scorer is already running")

    def stop(signum, _frame):
        log(f"stopping (signal {signum}); finished results are in {OUT.relative_to(ROOT)}, "
            "partial Haiku work is in the LLM cache")
        sys.stdout.flush()
        os._exit(0)  # don't wait for evaluate()'s worker threads to drain their queue
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    items = [json.loads(l) for l in open(WATCH)]
    log(f"{len(items)} watch items; writing {OUT.relative_to(ROOT)}")
    while True:
        one_pass(items, a.workers, a.only_best)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
