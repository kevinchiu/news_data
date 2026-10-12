"""Search-space views for the dashboard (read-only): the true retrieval landscape from
experiments/retrieval_sweep.json, the explored fitness landscape of every fully evaluated config,
and per-line search progress / search trees. Used by scripts/dashboard.py at /api/landscape.

Everything is recomputed only when a ledger, state file, config directory or the sweep changes.
"""
import hashlib
import json
import re
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
SWEEP = EXP / "retrieval_sweep.json"
sys.path.insert(0, str(ROOT / "scripts"))
import qb_config  # noqa: E402

INDEXES = ["quotation", "quotation_speaker"]
TEMPLATES = ["raw", "content", "content_plus_speaker"]
FILTERS = ["off", "surname"]
COMBO_LABEL = {"quotation": "q", "quotation_speaker": "q+spk", "raw": "raw", "content": "content",
               "content_plus_speaker": "content+spk", "off": "", "surname": " · surname"}
EFFORT = {"low": 0.0, "medium": 0.5, "high": 1.0}
N_PROMPT_DIMS = 3


def _read_jsonl(path):
    out = []
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _read_json(path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def discover_lines():
    lines = [("all", EXP)]
    base = EXP / "lines"
    if base.is_dir():
        lines += [(d.name, d) for d in sorted(p for p in base.iterdir() if p.is_dir())]
    return lines


def _fingerprint():
    parts = [str(_mtime(SWEEP))]
    for name, d in discover_lines():
        parts += [name, str(_mtime(d / "ledger.jsonl")), str(_mtime(d / "state.json")), str(_mtime(d / "configs"))]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _fa(m):
    return m.get("first_any") if isinstance(m, dict) and isinstance(m.get("first_any"), (int, float)) else None


def _line_mode(name, state):
    return state.get("mode") or ("paraphrase_hard" if name == "hard" else "paraphrase")


def _load_cfg(d, version, cache):
    if version not in cache:
        try:
            cache[version] = qb_config.load(d / "configs" / f"v{version}.json")
        except (OSError, ValueError, KeyError):
            cache[version] = None
    return cache[version]


def _rtuple(cfg):
    r = cfg["retrieval"]
    return r["index"], r["query_template"], r["speaker_filter"], int(r["k"])


# ---------- per-line history ----------

def line_history(name, d, prompts):
    """Rebuild a line's search from its ledger: every proposal with its full config, version
    nodes, rebaseline re-scores. `prompts` maps prompt text -> short id (shared across lines)."""
    entries = _read_jsonl(d / "ledger.jsonl")
    state = _read_json(d / "state.json")
    mode = _line_mode(name, state)
    copied_baseline = bool(state.get("started_from"))  # baseline entry is the main line's, not this line's
    cfgs = {}
    props = [e for e in entries if e.get("type") == "proposal"]
    acc_rounds = sorted(p.get("round") for p in props if p.get("decision") == "accepted")

    def base_version(rnd):
        return sum(1 for r in acc_rounds if r < rnd)

    def pid(text):
        if text not in prompts:
            prompts[text] = f"P{len(prompts)}"
        return prompts[text]

    v0 = _load_cfg(d, 0, cfgs)
    if v0:
        pid(v0["verify"]["prompt"])
    gt, out, rescores, baseline = 1, [], [], None
    for i, e in enumerate(entries):
        t = e.get("type")
        if t == "baseline" and not copied_baseline and _fa(e.get("train")) is not None:
            baseline = {"kind": "v0", "version": 0, "round": 0, "gt": gt, "n": e["train"].get("n"),
                        "train": e["train"], "val": e.get("val"), "cfg": v0, "decision": "baseline", "ts": e.get("ts")}
        elif t == "rebaseline":
            gt = e.get("gt_version") or gt + 1
            for which, ver in (("v0", 0), ("best", (e.get("best") or {}).get("version"))):
                blk = e.get(which) if which == "v0" else e.get("best")
                if not isinstance(blk, dict) or _fa(blk.get("train")) is None or ver is None:
                    continue
                rescores.append({"kind": "rescore", "which": which, "version": ver, "round": e.get("round") or 0,
                                 "gt": gt, "n": blk["train"].get("n"), "train": blk["train"], "val": blk.get("val"),
                                 "cfg": _load_cfg(d, ver, cfgs), "decision": "rescore", "ts": e.get("ts"), "seq": i})
        elif t == "proposal":
            rnd = e.get("round") or 0
            bv = base_version(rnd)
            base = _load_cfg(d, bv, cfgs)
            cfg = None
            if base is not None:
                try:
                    cfg = qb_config.apply(base, e.get("changes") or [])
                except (ValueError, KeyError, TypeError):
                    cfg = None
            ch = []
            for c in e.get("changes") or []:
                v = c.get("value")
                if c.get("path") == "verify.prompt" and isinstance(v, str):
                    v = f"{pid(v)} ({len(v)} chars)"
                ch.append(f"{c.get('path')}={json.dumps(v) if not isinstance(v, str) else v}")
            if cfg:
                pid(cfg["verify"]["prompt"])
            out.append({"kind": "proposal", "round": rnd, "rank": e.get("rank"), "base_version": bv,
                        "decision": e.get("decision"), "reason": e.get("reason"), "category": e.get("category"),
                        "changes": ch, "raw_changes": e.get("changes") or [], "gt": gt,
                        "n": (e.get("train") or {}).get("n"), "train": e.get("train"), "val": e.get("val"),
                        "fixed": e.get("fixed"), "broken": e.get("broken"), "cfg": cfg, "ts": e.get("ts"), "seq": i})
    # accepted proposals become versions 1..N in ledger order
    version = 0
    for p in sorted(out, key=lambda p: (p["round"], p["seq"])):
        if p["decision"] == "accepted":
            version += 1
            p["version"] = version
    rounds = [e.get("round") for e in entries if isinstance(e.get("round"), int)]
    return {"name": name, "mode": mode, "state": state, "baseline": baseline, "proposals": out,
            "rescores": rescores, "cfgs": cfgs, "copied_baseline": copied_baseline,
            "round": max(rounds + [state.get("round") or 0]), "best_version": state.get("best_version", 0),
            "round_ends": [e for e in entries if e.get("type") == "round_end"]}


# ---------- (a) retrieval landscape ----------

def _sweep_recall(cells, mode, index, template, filt, k):
    c = cells.get(f"{mode}|{index}|{template}|{filt}")
    if not c:
        return None
    ranks = c["ranks"]
    return sum(1 for r in ranks if r is not None and r < k) / len(ranks)


def retrieval_landscape(hist):
    sweep = _read_json(SWEEP)
    cells = sweep.get("cells") or {}
    meta = sweep.get("meta") or {}
    if not cells:
        return {"available": False}
    combos = [(i, t, f) for i in INDEXES for t in TEMPLATES for f in FILTERS]
    labels = [f"{COMBO_LABEL[i]} / {COMBO_LABEL[t]}{COMBO_LABEL[f]}" for i, t, f in combos]
    ks = list(range(5, 61))
    modes = {}
    for mode in meta.get("modes") or ["paraphrase", "paraphrase_hard"]:
        z = [[_sweep_recall(cells, mode, i, t, f, k) for k in ks] for i, t, f in combos]
        grid = [{"index": i, "query_template": t, "speaker_filter": f, "k": k,
                 "recall": _sweep_recall(cells, mode, i, t, f, k), "y": yi}
                for yi, (i, t, f) in enumerate(combos) for k in meta.get("k_grid", [])]
        grid = [g for g in grid if g["recall"] is not None]
        top = sorted(grid, key=lambda g: (-g["recall"], g["k"]))[:6]
        modes[mode] = {"z": z, "grid": grid, "top": top,
                       "done": sum(1 for c in combos if f"{mode}|{c[0]}|{c[1]}|{c[2]}" in cells)}
    # retrieval settings the loop tried (proposal configs and versions), and each line's current best
    tried, bests = [], []
    for h in hist:
        seen = {}
        nodes = [(0, "v0", "baseline", h["cfgs"].get(0))] + \
                [(p["round"], f"r{p['round']}#{p['rank']}", p["decision"], p["cfg"]) for p in h["proposals"]
                 if p["cfg"] and p["decision"] != "invalid"
                 and any(str(c.get("path", "")).startswith("retrieval.") for c in p["raw_changes"])]
        for rnd, lab, dec, cfg in nodes:
            if not cfg:
                continue
            rt = _rtuple(cfg)
            s = seen.setdefault(rt, {"line": h["name"], "index": rt[0], "query_template": rt[1],
                                     "speaker_filter": rt[2], "k": rt[3], "y": combos.index(rt[:3]),
                                     "labels": [], "decisions": []})
            s["labels"].append(lab)
            s["decisions"].append(dec)
        for s in seen.values():
            s["recall"] = {m: _sweep_recall(cells, m, s["index"], s["query_template"], s["speaker_filter"], s["k"])
                           for m in modes}
            tried.append(s)
        best = h["cfgs"].get(h["best_version"])
        if best:
            rt = _rtuple(best)
            bests.append({"line": h["name"], "line_mode": h["mode"], "version": h["best_version"], "index": rt[0],
                          "query_template": rt[1], "speaker_filter": rt[2], "k": rt[3], "y": combos.index(rt[:3]),
                          "recall": {m: _sweep_recall(cells, m, *rt) for m in modes}})
    return {"available": True, "ks": ks, "combos": [list(c) for c in combos], "labels": labels, "modes": modes,
            "tried": tried, "bests": bests, "n_items": meta.get("n_items"), "k_grid": meta.get("k_grid"),
            "updated": meta.get("updated"), "elapsed_s": meta.get("total_elapsed_s")}


# ---------- (b) explored fitness landscape ----------

def _tfidf_svd(texts, dims):
    toks = [re.findall(r"[a-z]{3,}", t.lower()) for t in texts]
    vocab = sorted({w for ts in toks for w in ts})
    if not vocab or len(texts) < 2:
        return np.zeros((len(texts), dims))
    ix = {w: i for i, w in enumerate(vocab)}
    tf = np.zeros((len(texts), len(vocab)))
    for r, ts in enumerate(toks):
        for w in ts:
            tf[r, ix[w]] += 1
    tf = np.log1p(tf)
    idf = np.log((1 + len(texts)) / (1 + (tf > 0).sum(0))) + 1
    x = tf * idf
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x -= x.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    comp = u[:, :dims] * s[:dims]
    out = np.zeros((len(texts), dims))
    out[:, :comp.shape[1]] = comp
    scale = np.abs(out).max(0)
    scale[scale == 0] = 1
    return out / scale  # each prompt dimension spans [-1, 1], comparable to the one-hot knobs


FEATURES = (["k", "index=q+spk", "tmpl=raw", "tmpl=content", "tmpl=content+spk", "filter=surname",
             "effort", "batch", "show_date", "speaker_hint"] + [f"prompt-svd{i + 1}" for i in range(N_PROMPT_DIMS)])


def _knob_vec(cfg):
    r, v = cfg["retrieval"], cfg["verify"]
    return [(r["k"] - 5) / 55, float(r["index"] == "quotation_speaker"),
            float(r["query_template"] == "raw"), float(r["query_template"] == "content"),
            float(r["query_template"] == "content_plus_speaker"), float(r["speaker_filter"] == "surname"),
            EFFORT.get(v.get("effort"), 0.0), (v.get("batch", 15) - 5) / 25,
            float(bool(v.get("show_date"))), float(bool(v.get("speaker_hint")))]


def _surface(xy, z, bbox, n=40):
    """Linear interpolation of z over the convex hull of the (deduplicated) points."""
    from scipy.interpolate import griddata
    if len(z) < 3:
        return None
    keys = {}
    for (x, y), zz in zip(np.round(xy, 6), z):
        keys.setdefault((x, y), []).append(zz)
    pts = np.array(list(keys.keys()))
    vals = np.array([np.mean(v) for v in keys.values()])
    if len(pts) < 3 or np.linalg.matrix_rank(pts - pts.mean(0)) < 2:
        return None
    gx = np.linspace(bbox[0], bbox[1], n)
    gy = np.linspace(bbox[2], bbox[3], n)
    mx, my = np.meshgrid(gx, gy)
    try:
        gz = griddata(pts, vals, (mx, my), method="linear")
    except Exception:  # degenerate hull (qhull)
        return None
    return {"x": gx.round(4).tolist(), "y": gy.round(4).tolist(),
            "z": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in gz],
            "n_points": int(len(pts))}


def fitness_landscape(hist, prompt_ids):
    pts = []
    for h in hist:
        if h["baseline"] and h["baseline"]["cfg"]:
            pts.append(dict(h["baseline"], line=h["name"], mode=h["mode"], label="v0 baseline"))
        for p in h["proposals"]:
            if _fa(p["train"]) is not None and p["cfg"]:
                lab = f"r{p['round']}#{p['rank']} {p['decision']}" + (f" → v{p['version']}" if p.get("version") else "")
                pts.append(dict(p, line=h["name"], mode=h["mode"], label=lab))
        for r in h["rescores"]:
            if r["cfg"]:
                pts.append(dict(r, line=h["name"], mode=h["mode"],
                                label=f"v{r['version']} re-scored ({'v0' if r['which'] == 'v0' else 'incumbent'}, "
                                      f"GT v{r['gt']}, n={r['n']})"))
    if not pts:
        return {"available": False, "points": []}
    texts = [p["cfg"]["verify"]["prompt"] for p in pts]
    uniq = sorted(set(texts))
    pv = _tfidf_svd(uniq, N_PROMPT_DIMS)
    pmap = {t: pv[i] for i, t in enumerate(uniq)}
    X = np.array([_knob_vec(p["cfg"]) + list(pmap[t]) for p, t in zip(pts, texts)])
    Xc = X - X.mean(0)
    if len(pts) >= 2 and np.abs(Xc).sum() > 0:
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        comps = vt[:2] if vt.shape[0] >= 2 else np.vstack([vt, np.zeros_like(vt)])
        xy = Xc @ comps.T
        var = (s ** 2) / max((s ** 2).sum(), 1e-12)
        evr = [float(var[i]) if i < len(var) else 0.0 for i in range(2)]
    else:
        comps, xy, evr = np.zeros((2, X.shape[1])), np.zeros((len(pts), 2)), [0.0, 0.0]
    axes = []
    for i in range(2):
        order = np.argsort(-np.abs(comps[i]))[:3]
        axes.append({"evr": round(evr[i], 3),
                     "loadings": [{"feature": FEATURES[j], "w": round(float(comps[i][j]), 3)} for j in order]})
    pad = lambda a, b: ((b - a) * 0.04 or 0.5)  # noqa: E731
    x0, x1, y0, y1 = xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()
    bbox = (x0 - pad(x0, x1), x1 + pad(x0, x1), y0 - pad(y0, y1), y1 + pad(y0, y1))
    epochs = sorted({(p["gt"], p["n"]) for p in pts if p["n"]})
    current = epochs[-1] if epochs else None
    out_pts = []
    for p, (x, y), t in zip(pts, xy, texts):
        r = p["cfg"]["retrieval"]
        v = p["cfg"]["verify"]
        out_pts.append({
            "line": p["line"], "mode": p["mode"], "kind": p["kind"], "decision": p["decision"],
            "round": p["round"], "version": p.get("version"), "label": p["label"], "which": p.get("which"),
            "x": round(float(x), 4), "y": round(float(y), 4), "z": round(_fa(p["train"]), 4),
            "val": _fa(p.get("val")), "gt": p["gt"], "n": p["n"], "seq": p.get("seq", -1),
            "current": (p["gt"], p["n"]) == current,
            "cfg": f"k={r['k']} {r['index']}/{r['query_template']}/{r['speaker_filter']} · effort={v['effort']} "
                   f"batch={v['batch']} date={v['show_date']} hint={v['speaker_hint']} · prompt {prompt_ids.get(t, '?')}",
            "changes": p.get("changes") or [],
        })
    # accepted path per line: v0 → accepted versions, with re-scores inserted in ledger order
    paths = {}
    for h in hist:
        acc_seqs = sorted(q["seq"] for q in out_pts if q["line"] == h["name"] and q["decision"] == "accepted")
        path = []
        for q in sorted((q for q in out_pts if q["line"] == h["name"]), key=lambda q: q["seq"]):
            if q["kind"] == "rescore":
                # only the incumbent's re-score is on the path (and only once, via its "best" re-score)
                if q["which"] != "best" or q["version"] != sum(1 for s in acc_seqs if s < q["seq"]):
                    continue
            elif q["kind"] != "v0" and q["decision"] != "accepted":
                continue
            path.append(q)
        paths[h["name"]] = [{"x": q["x"], "y": q["y"], "z": q["z"], "label": q["label"], "current": q["current"]}
                            for q in path]
    para = [(q["x"], q["y"], q["z"], q["current"]) for q in out_pts if q["mode"] == "paraphrase"]
    surfaces = {}
    for key, sel in (("all", para), ("current", [p for p in para if p[3]])):
        surfaces[key] = _surface(np.array([[p[0], p[1]] for p in sel]) if sel else np.zeros((0, 2)),
                                 [p[2] for p in sel], bbox)
    return {"available": True, "points": out_pts, "paths": paths, "axes": axes, "surfaces": surfaces,
            "bbox": [round(float(b), 4) for b in bbox], "n_points": len(out_pts),
            "n_prompts": len(uniq), "features": FEATURES,
            "current_epoch": {"gt": current[0], "n": current[1]} if current else None,
            "epochs": [{"gt": g, "n": n} for g, n in epochs]}


# ---------- (c) search progress & tree ----------

def search_progress(hist):
    out = []
    for h in hist:
        cands = [{"round": p["round"], "rank": p["rank"], "decision": p["decision"], "reason": p["reason"],
                  "changes": p["changes"], "z": _fa(p["train"]), "val": _fa(p["val"]), "gt": p["gt"], "n": p["n"],
                  "version": p.get("version"), "base_version": p["base_version"], "category": p["category"],
                  "fixed": p["fixed"], "broken": p["broken"]}
                 for p in sorted(h["proposals"], key=lambda p: (p["round"], p["rank"] if p["rank"] is not None else 9))]
        # best-so-far train first_any: v0, step at each accepted version, reset to re-scored values
        events = []
        if h["baseline"]:
            events.append((-1, 0, _fa(h["baseline"]["train"]), "v0", h["baseline"]["gt"], h["baseline"]["n"]))
        for p in h["proposals"]:
            if p["decision"] == "accepted" and _fa(p["train"]) is not None:
                events.append((p["seq"], p["round"], _fa(p["train"]), f"v{p['version']}", p["gt"], p["n"]))
        ver_at = {}
        for p in h["proposals"]:
            if p.get("version"):
                ver_at[p["seq"]] = p["version"]
        for r in h["rescores"]:
            cur = sum(1 for s in ver_at if s < r["seq"])
            if r["which"] == "best" and r["version"] == cur:  # the incumbent re-scored
                events.append((r["seq"], r["round"], _fa(r["train"]), f"v{r['version']} re-scored (GT v{r['gt']}, n={r['n']})", r["gt"], r["n"]))
        events.sort(key=lambda e: e[0])
        best = [{"round": e[1], "z": e[2], "label": e[3], "gt": e[4], "n": e[5]} for e in events]
        if best and h["round"] > best[-1]["round"]:
            best.append(dict(best[-1], round=h["round"], label=best[-1]["label"] + " (carried)"))
        rescore_marks = sorted({(r["round"], r["gt"], r["n"]) for r in h["rescores"]})
        # tree: spine = version nodes; per round the candidates hang off the incumbent
        rounds = sorted({c["round"] for c in cands} | {e.get("round") for e in h["round_ends"]} - {None})
        tree = []
        for rnd in rounds:
            rc = [c for c in cands if c["round"] == rnd]
            acc = next((c for c in rc if c["decision"] == "accepted"), None)
            tree.append({"round": rnd, "base_version": rc[0]["base_version"] if rc else None,
                         "accepted_version": acc and acc["version"], "candidates": rc,
                         "ended": any(e.get("round") == rnd for e in h["round_ends"])})
        out.append({"line": h["name"], "mode": h["mode"], "round": h["round"], "best_version": h["best_version"],
                    "candidates": cands, "best": best,
                    "rescores": [{"round": r, "gt": g, "n": n} for r, g, n in rescore_marks],
                    "tree": tree, "copied_baseline": h["copied_baseline"],
                    "started_from": h["state"].get("started_from")})
    return out


# ---------- entry point with cache ----------

_lock = threading.Lock()
_cache = {"key": None, "data": None}


def landscape():
    key = _fingerprint()
    with _lock:
        if _cache["key"] == key:
            return _cache["data"]
        prompts = {}
        hist = [line_history(n, d, prompts) for n, d in discover_lines()]
        data = {"version": key, "lines": [h["name"] for h in hist],
                "line_modes": {h["name"]: h["mode"] for h in hist},
                "retrieval": retrieval_landscape(hist),
                "fitness": fitness_landscape(hist, prompts),
                "progress": search_progress(hist)}
        data = json.loads(json.dumps(data, default=_jsonable))
        _cache.update(key=key, data=data)
        return data


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    import time
    t = time.time()
    d = landscape()
    print(f"{time.time() - t:.2f}s", json.dumps({k: (v if k in ("version", "lines", "line_modes") else "...")
                                                 for k, v in d.items()}))
    f = d["fitness"]
    print("fitness points", f.get("n_points"), "prompts", f.get("n_prompts"), "axes", f.get("axes"))
    print("surfaces", {k: (v and v["n_points"]) for k, v in (f.get("surfaces") or {}).items()})
    print("paths", {k: [p["label"] for p in v] for k, v in (f.get("paths") or {}).items()})
    for p in d["progress"]:
        print(p["line"], "best", [(b["round"], round(b["z"], 3), b["label"]) for b in p["best"]])
    r = d["retrieval"]
    print("retrieval", r.get("available"), {m: v["done"] for m, v in (r.get("modes") or {}).items()})
    print("bests", r.get("bests"))
