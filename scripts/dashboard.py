"""Live dashboard for the self-improvement loop (read-only).

Run:  .venv/bin/python -I scripts/dashboard.py [--port 8765]
Then open http://127.0.0.1:8765/
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))  # for scripts/landscape.py (python -I drops the script dir)
EXP = ROOT / "experiments"
RUNNING_WINDOW_S = 90
LOG_LINES = 15
BUDGETS = {"all": 2500}
DEFAULT_BUDGET = 1500


def read_jsonl(path):
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue  # partial last line while being appended
        if isinstance(obj, dict):
            out.append(obj)
    return out


def read_json(path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def tail(path, n=LOG_LINES):
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = [l[:600] for l in data.splitlines()]
    return lines[-n:]


def mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def fa(metrics):
    if isinstance(metrics, dict) and isinstance(metrics.get("first_any"), (int, float)):
        return metrics["first_any"]
    return None


def discover_lines():
    lines = [("all", EXP, EXP / "run.log")]
    base = EXP / "lines"
    if base.is_dir():
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            lines.append((d.name, d, EXP / f"run_{d.name}.log"))
    return lines


def summarize(name, d, log, now):
    ledger_p, state_p = d / "ledger.jsonl", d / "state.json"
    entries = read_jsonl(ledger_p)
    state = read_json(state_p)
    baseline = next((e for e in entries if e.get("type") == "baseline"), {})
    proposals = [e for e in entries if e.get("type") == "proposal"]
    improvers = [e for e in entries if e.get("type") == "improver"]
    finals = [e for e in entries if e.get("type") == "final"]
    counts = {k: 0 for k in ("accepted", "rejected", "invalid", "not_evaluated")}
    for p in proposals:
        dec = p.get("decision") or "other"
        counts[dec] = counts.get(dec, 0) + 1
    spent = sum(e.get("new_calls") or 0 for e in entries
                if isinstance(e.get("new_calls"), (int, float)))
    # best-so-far val curve: baseline, then step at every round_end
    curve = []
    cur = fa(baseline.get("val"))
    if cur is not None:
        curve.append({"x": 0, "y": cur})
    for e in entries:
        if e.get("type") != "round_end":
            continue
        if e.get("accepted"):
            acc = [p for p in proposals if p.get("round") == e.get("round")
                   and p.get("decision") == "accepted" and fa(p.get("val")) is not None]
            if acc:
                cur = fa(acc[-1]["val"])
        if cur is not None:
            curve.append({"x": e.get("round"), "y": cur})
    rounds = [e.get("round") for e in entries if isinstance(e.get("round"), int)]
    log_m = mtime(log)
    recent = max((m for m in (log_m, mtime(ledger_p), mtime(state_p)) if m), default=None)
    if recent is not None and now - recent < RUNNING_WINDOW_S:
        status = "running"
    elif finals:
        status = "finished"
    else:
        status = "idle"
    bl_train, bl_val = fa(baseline.get("train")), fa(baseline.get("val"))
    # version n is created by the n-th accepted proposal (improve.py bumps best_version on accept)
    accepted = [p for p in proposals if p.get("decision") == "accepted"]
    versions = [{"version": 0, "round": 0, "ts": baseline.get("ts")}]
    versions += [{"version": i, "round": p.get("round"), "ts": p.get("ts")} for i, p in enumerate(accepted, 1)]
    return {
        "name": name,
        "focus": state.get("focus") or baseline.get("focus") or ("all knobs" if name == "all" else name),
        "status": status,
        "round": max(rounds + [state.get("round") or 0]),
        "best_version": state.get("best_version", 0),
        "threshold": state.get("threshold", baseline.get("threshold")),
        "noise": state.get("noise") or baseline.get("noise") or {},
        "baseline_train": bl_train, "baseline_val": bl_val,
        "train_now": fa(state.get("train_metrics")) if state.get("train_metrics") else bl_train,
        "val_now": fa(state.get("val_metrics")) if state.get("val_metrics") else bl_val,
        "baseline_test": fa(state.get("baseline_test") or baseline.get("test")),
        "spent": spent, "budget": BUDGETS.get(name, DEFAULT_BUDGET),
        "counts": counts, "curve": curve, "versions": versions,
        "proposals": [dict(p, line=name, id=f"{name}:{i}") for i, p in enumerate(proposals)],
        "analysis": improvers[-1] if improvers else None,
        "finals": finals,
        "log": tail(log), "log_path": str(log.relative_to(ROOT)),
        "mtimes": {"ledger": mtime(ledger_p), "state": mtime(state_p), "log": log_m},
    }


WATCH_KEYS = ("first_any", "recall_at_k", "fp_earlier", "calls_per_claim", "verifier_recall", "first_original", "n")


def attach_watch(lines):
    """Join experiments/watch.jsonl (written by scripts/watch_scorer.py) onto each line's versions.

    Adds per line: watch_versions [{version, round, ts, hash, metrics|None}], watch_curve (step
    points of the best version's watch first_any after each round) and watch_best. Returns a
    summary {available, scorer_mtime, v0, n_records}."""
    recs = read_jsonl(EXP / "watch.jsonl")
    by_hash, ver_hash = {}, {}
    for r in recs:
        h = r.get("config_hash")
        if not r.get("alias") and isinstance(r.get("metrics"), dict):
            by_hash[h] = {k: r["metrics"].get(k) for k in WATCH_KEYS} | {"ts": r.get("ts")}
        for v in r.get("versions") or []:
            ver_hash[(v.get("line"), v.get("version"))] = h
    v0 = None
    for l in lines:
        wv = []
        for v in l["versions"]:
            h = ver_hash.get((l["name"], v["version"]))
            wv.append(dict(v, hash=h, metrics=by_hash.get(h)))
        l["watch_versions"] = wv
        if v0 is None and wv and wv[0]["metrics"]:
            v0 = wv[0]["metrics"]
        best = next((v for v in wv if v["version"] == l["best_version"]), None)
        l["watch_best"] = best and {"version": best["version"], "metrics": best["metrics"]}
        curve, cur = [], None
        for v in wv:
            cur = v["metrics"] and v["metrics"].get("first_any")
            curve.append({"x": v["round"] or 0, "y": cur, "v": v["version"]})
        if curve and l["round"] > curve[-1]["x"] and curve[-1]["y"] is not None:
            curve.append({"x": l["round"], "y": curve[-1]["y"], "v": curve[-1]["v"]})
        l["watch_curve"] = curve
    return {"available": bool(recs), "n_records": len(recs), "v0": v0,
            "mtime": mtime(EXP / "watch.jsonl")}


def status():
    now = time.time()
    lines = [summarize(n, d, log, now) for n, d, log in discover_lines()]
    watch = attach_watch(lines)
    return {"server_time": now, "lines": lines, "merge": read_jsonl(EXP / "merge.jsonl"), "watch": watch}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Provenance tracer loop</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--border:rgba(11,11,11,.10);--good:#006300;--bad:#d03b3b;--warn:#b07800;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
--s4:#eda100;--s5:#e87ba4;--s6:#008300;--wash:rgba(11,11,11,.04);--seq0:#86b6ef;--seq1:#2a78d6;--seq2:#0d366b}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--border:rgba(255,255,255,.10);--good:#0ca30c;--bad:#e66767;--warn:#fab219;--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--wash:rgba(255,255,255,.05);--seq0:#184f95;--seq1:#5598e7;--seq2:#cde2fb}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
main{max-width:1280px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:24px 0 8px}
.sub{color:var(--ink2)}.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.row{display:flex;justify-content:space-between;gap:8px;margin:3px 0}.pill{border-radius:99px;padding:1px 9px;font-size:12px;border:1px solid var(--border)}
.pill.running{color:var(--good)}.pill.idle{color:var(--warn)}.pill.finished{color:var(--ink2)}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.bar{height:8px;background:var(--grid);border-radius:4px;overflow:hidden;margin-top:4px}.bar>div{height:100%;background:var(--s1);border-radius:4px}
.up{color:var(--good)}.down{color:var(--bad)}.num{font-variant-numeric:tabular-nums}
.chartbox{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px;height:340px}
.tablewrap{overflow-x:auto;background:var(--card);border:1px solid var(--border);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;text-align:left;vertical-align:top;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600;white-space:nowrap}tr.p{cursor:pointer}tr.p:hover{background:var(--wash)}
.d-accepted{color:var(--good);font-weight:600}.d-rejected{color:var(--bad)}.d-invalid{color:var(--warn)}.d-not_evaluated{color:var(--muted)}
pre{white-space:pre-wrap;word-break:break-word;margin:4px 0;font:12px/1.4 ui-monospace,Menlo,monospace}
.log{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:8px;max-height:280px;overflow:auto}
details{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:8px 12px;margin:6px 0}summary{cursor:pointer;font-weight:600}
.detail td{background:var(--wash)}select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--border);border-radius:6px;padding:2px 6px}
.tag{font-size:11px;border:1px solid var(--border);border-radius:99px;padding:0 6px;margin-left:4px;color:var(--ink2)}
code{font:12px ui-monospace,Menlo,monospace}
.tabs{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:8px}
.tabs button,.seg button{font:inherit;font-size:13px;color:var(--ink2);background:var(--card);border:1px solid var(--border);border-radius:99px;padding:3px 12px;cursor:pointer}
.tabs button.on,.seg button.on{color:var(--ink);background:var(--wash);border-color:var(--ink2);font-weight:600}
.seg{display:inline-flex;gap:4px;align-items:center;flex-wrap:wrap}.seg .lab{font-size:12px;color:var(--muted);margin-right:2px}
.ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:6px 0}
.plot3d{background:var(--card);border:1px solid var(--border);border-radius:10px;height:600px;min-width:0}
.cap{font-size:12px;color:var(--ink2);margin:6px 0;max-width:980px}.cap b{color:var(--ink)}
.lsgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:12px}
.lsgrid .card{min-width:0}.prog{height:230px}.tree{overflow-x:auto;margin-top:6px}.tree svg{display:block}
.tree text{font:11px system-ui,sans-serif;fill:var(--ink2)}.legend{font-size:12px;color:var(--ink2);display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:4px}
.mk{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px;border:2px solid}
@media (max-width:640px){.plot3d{height:440px}}
</style></head><body><main>
<h1>Provenance tracer — self-improvement</h1>
<div id="hdr" class="sub">loading…</div>
<div class="cards" id="cards" style="margin-top:12px"></div>
<h2>Validation first_any over rounds</h2>
<div class="chartbox"><canvas id="chart"></canvas></div>
<div class="muted" style="font-size:12px;margin-top:4px">Step line = best-so-far val first_any per line. Points = train first_any of fully evaluated candidates: filled triangle = accepted, hollow circle = rejected.</div>
<h2>Held-out watch set (never seen by the loop)</h2>
<div class="muted" style="font-size:12px;margin-bottom:6px" id="watchhdr">Watch set: 100 paraphrased items disjoint from the benchmark (bench/qb_watch.jsonl), scored by <code>scripts/watch_scorer.py</code>. Scores are computed only after a version is accepted, so the latest best may show "pending" until the scorer catches up.</div>
<div class="chartbox" style="height:280px"><canvas id="wchart"></canvas></div>
<div class="muted" style="font-size:12px;margin-top:4px">Step line = watch first_any of each line's best version, starting at v0 and stepping at the round a new version was accepted; gaps = pending.</div>
<div class="tablewrap" style="margin-top:8px"><table><thead><tr><th>line</th><th>version</th><th>accepted</th><th>watch first_any</th><th>recall@k</th><th>fp_earlier</th><th>calls/claim</th></tr></thead><tbody id="watch"></tbody></table></div>
<h2 id="ls-h">Search space &amp; search progress</h2>
<div id="ls">
<div class="tabs" role="tablist"><button data-t="ret" class="on">Retrieval landscape <span class="tag">exact</span></button><button data-t="fit">Explored fitness landscape <span class="tag">interpolated</span></button><button data-t="prog">Search progress &amp; tree</button><span id="lsstatus" class="muted" style="font-size:12px"></span></div>
<div class="lstab" data-t="ret">
<div class="ctl"><span class="seg" id="ret-mode"><span class="lab">query form</span><button data-v="paraphrase" class="on">paraphrase</button><button data-v="paraphrase_hard">paraphrase_hard</button></span></div>
<div class="plot3d" id="p-ret"></div>
<div class="cap"><b>Exact, exhaustive.</b> Every combination of the retrieval knobs, scored by BM25 alone (no LLM) on all train items (<code>scripts/retrieval_sweep.py</code> → <code>experiments/retrieval_sweep.json</code>). <b>x</b> = <code>retrieval.k</code> (top-k is a prefix of the top-60 ranking, so recall is exact at every k from 5 to 60); <b>y</b> = the 12 combinations of index / query template / speaker filter (q = quote text only, q+spk = quote text + speaker; content = statement with “X said that” stripped); <b>z</b> = recall@k, the share of items whose true quotation is among the k candidates. This caps what the verifier can find. Small dots = retrieval settings the loop tried (any line); diamonds = each line’s current best. Drag to rotate.</div>
<div class="tablewrap" style="margin-top:6px"><table><thead><tr><th>rank</th><th>index</th><th>query template</th><th>speaker filter</th><th>k</th><th>recall@k</th></tr></thead><tbody id="ret-top"></tbody></table></div>
</div>
<div class="lstab" data-t="fit" hidden>
<div class="ctl"><span class="seg" id="fit-ep"><span class="lab">scores</span><button data-v="all" class="on">all evaluations</button><button data-v="current">current scoring only</button></span>
<span class="seg" id="fit-surf"><span class="lab">surface</span><button data-v="on" class="on">show</button><button data-v="off">hide</button></span></div>
<div class="plot3d" id="p-fit"></div>
<div class="cap" id="fit-cap"></div>
</div>
<div class="lstab" data-t="prog" hidden>
<div class="cap">Per line: <b>top</b> = train first_any of each fully evaluated candidate by round (status colour = decision), with the best-so-far as a step line; dashed verticals mark re-scoring after a ground-truth change, where the train set and scores changed. <b>Bottom</b> = search tree: the top row is the incumbent (v0 → accepted versions); each round’s candidates hang below it. Filled = fully evaluated on train + val; hollow = stopped at the cheap screen. Hover any node for its changes and reason.</div>
<div class="legend"><span><i class="mk" style="background:var(--good);border-color:var(--good)"></i>accepted</span><span><i class="mk" style="border-color:var(--warn)"></i>runner-up</span><span><i class="mk" style="border-color:var(--bad)"></i>rejected</span><span><i class="mk" style="border-color:var(--muted)"></i>not evaluated</span><span><i class="mk" style="border-color:var(--warn);border-radius:2px"></i>invalid</span></div>
<div class="lsgrid" id="prog"></div>
</div>
</div>
<h2>Proposals <select id="filter"><option value="">all lines</option></select></h2>
<div class="tablewrap"><table><thead><tr><th>time</th><th>line</th><th>rnd</th><th>rank</th><th>category</th><th>changes</th><th>decision</th><th>reason</th><th>fix/brk</th><th>calls</th></tr></thead><tbody id="props"></tbody></table></div>
<h2>Latest improver analysis</h2><div id="analysis"></div>
<h2>Log tails</h2><div id="logs"></div>
<div id="results"></div>
</main><script>
const COL=['--s1','--s2','--s3','--s4','--s5','--s6'];
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=v=>v==null?'–':(100*v).toFixed(1)+'%';
const delta=(a,b)=>{if(a==null||b==null)return'';const d=100*(a-b);return `<span class="${d>0?'up':d<0?'down':'muted'}">${d>=0?'+':''}${d.toFixed(1)}pp</span>`};
const deltaInv=(a,b,pp=true)=>{if(a==null||b==null)return'';const d=(pp?100:1)*(a-b);return `<span class="${d<0?'up':d>0?'down':'muted'}">${d>=0?'+':''}${d.toFixed(pp?1:2)}${pp?'pp':''}</span>`};
const tm=ts=>ts?String(ts).replace('T',' ').slice(5,16):'';
const short=(v,n=60)=>{let s=typeof v==='string'?v:JSON.stringify(v);return s.length>n?s.slice(0,n)+`… (${s.length} chars)`:s};
const full=v=>typeof v==='string'?v:JSON.stringify(v,null,2);
const expanded=new Set();let last={},chart=null,data=null;
function setHTML(id,html){if(last[id]===html)return;const el=document.getElementById(id);
  const open=new Set([...el.querySelectorAll('details[open]')].map(d=>d.dataset.k));
  el.innerHTML=html;last[id]=html;el.querySelectorAll('details').forEach(d=>{if(open.has(d.dataset.k))d.open=true});
  el.querySelectorAll('.log').forEach(p=>p.scrollTop=p.scrollHeight);}
function color(i){return css(COL[i%COL.length])}
function renderCards(L){setHTML('cards',L.map((l,i)=>{const c=l.counts,sp=Math.min(100,100*l.spent/l.budget);
  return `<div class="card"><div class="row"><b><span class="sw" style="background:${color(i)}"></span>${esc(l.name)}</b><span class="pill ${l.status}">${l.status}</span></div>
  <div class="muted">focus: ${esc(l.focus)}</div>
  <div class="row"><span>round</span><span class="num">${l.round}</span></div>
  <div class="row"><span>best version</span><span class="num">v${l.best_version}</span></div>
  <div class="row"><span>train first_any</span><span class="num">${pct(l.train_now)} ${delta(l.train_now,l.baseline_train)}</span></div>
  <div class="row"><span>val first_any</span><span class="num">${pct(l.val_now)} ${delta(l.val_now,l.baseline_val)}</span></div>
  <div class="row"><span>watch first_any${l.watch_best?' (v'+l.watch_best.version+')':''}</span><span class="num">${l.watch_best&&l.watch_best.metrics?pct(l.watch_best.metrics.first_any)+' '+delta(l.watch_best.metrics.first_any,data.watch?.v0?.first_any):'<span class="muted">pending</span>'}</span></div>
  <div class="row muted"><span>baseline train / val</span><span class="num">${pct(l.baseline_train)} / ${pct(l.baseline_val)}</span></div>
  <div class="row"><span>threshold · noise flips</span><span class="num">${l.threshold??'–'} · ${l.noise.flips??'–'}/${l.noise.n??'–'}</span></div>
  <div class="row"><span>budget</span><span class="num">${l.spent} / ${l.budget}</span></div><div class="bar"><div style="width:${sp}%"></div></div>
  <div class="row" style="margin-top:6px"><span class="d-accepted">${c.accepted} acc</span><span class="d-rejected">${c.rejected} rej</span><span class="d-invalid">${c.invalid} inv</span><span class="d-not_evaluated">${c.not_evaluated} n/e</span></div></div>`}).join(''));}
function renderChart(L){const ds=[];L.forEach((l,i)=>{const col=color(i),j=(i-(L.length-1)/2)*0.08;
  ds.push({label:l.name,data:l.curve,borderColor:col,backgroundColor:col,stepped:'before',borderWidth:2,pointRadius:3,kind:'line'});
  const ev=l.proposals.filter(p=>p.train&&p.train.first_any!=null);
  ds.push({label:l.name+' accepted',data:ev.filter(p=>p.decision==='accepted').map(p=>({x:p.round+j,y:p.train.first_any,p})),showLine:false,
    pointStyle:'triangle',pointRadius:6,borderColor:col,backgroundColor:col,kind:'pt'});
  ds.push({label:l.name+' rejected',data:ev.filter(p=>p.decision!=='accepted').map(p=>({x:p.round+j,y:p.train.first_any,p})),showLine:false,
    pointStyle:'circle',pointRadius:5,borderColor:col,backgroundColor:'transparent',borderWidth:2,kind:'pt'});});
  const maxR=Math.max(1,...L.map(l=>l.round));
  if(!chart){Chart.defaults.color=css('--ink2');Chart.defaults.borderColor=css('--grid');
    chart=new Chart(document.getElementById('chart'),{type:'line',data:{datasets:ds},options:{animation:false,maintainAspectRatio:false,parsing:true,
      interaction:{mode:'nearest',intersect:false},
      scales:{x:{type:'linear',min:0,suggestedMax:maxR,ticks:{stepSize:1,precision:0},title:{display:true,text:'round'}},
        y:{min:0,max:1,ticks:{callback:v=>Math.round(v*100)+'%'},title:{display:true,text:'first_any'}}},
      plugins:{legend:{labels:{filter:(it,d)=>d.datasets[it.datasetIndex].kind==='line'}},
        tooltip:{callbacks:{label:c=>{const r=c.raw;return r.p?`${c.dataset.label} r${r.p.round} #${r.p.rank} ${r.p.category}: train ${pct(r.y)}${r.p.val?' val '+pct(r.p.val.first_any):''}`:`${c.dataset.label} best val after r${r.x}: ${pct(r.y)}`}}}}}});}
  else{chart.data.datasets=ds;chart.options.scales.x.suggestedMax=maxR;chart.update('none');}}
let wchart=null;
function renderWatch(L,W){const ds=L.map((l,i)=>{const col=color(i);return{label:l.name,data:l.watch_curve||[],borderColor:col,backgroundColor:col,
    stepped:'after',borderWidth:2,pointRadius:3,spanGaps:false}});
  const maxR=Math.max(1,...L.map(l=>l.round));
  if(!wchart){wchart=new Chart(document.getElementById('wchart'),{type:'line',data:{datasets:ds},options:{animation:false,maintainAspectRatio:false,
      interaction:{mode:'nearest',intersect:false},
      scales:{x:{type:'linear',min:0,suggestedMax:maxR,ticks:{stepSize:1,precision:0},title:{display:true,text:'round'}},
        y:{suggestedMin:0.4,max:1,ticks:{callback:v=>Math.round(v*100)+'%'},title:{display:true,text:'watch first_any'}}},
      plugins:{tooltip:{callbacks:{label:c=>`${c.dataset.label} best v${c.raw.v} (round ${c.raw.x}): watch ${pct(c.raw.y)}`}}}}});}
  else{wchart.data.datasets=ds;wchart.options.scales.x.suggestedMax=maxR;wchart.update('none');}
  const v0=W.v0||{};const cell=(m,k,base,inv,pp=true)=>`${k==='calls_per_claim'?(m[k]??0).toFixed(2):pct(m[k])} ${base?'':inv?deltaInv(m[k],v0[k],pp):delta(m[k],v0[k])}`;
  const row=(name,v,best)=>{const m=v.metrics;return `<tr><td>${esc(name)}</td><td class="num">v${v.version}${best?'<span class="tag">best</span>':''}</td>
    <td class="num">${v.version?'r'+esc(v.round)+' · '+tm(v.ts):'baseline'}</td>
    ${m?`<td class="num"><b>${pct(m.first_any)}</b> ${v.version?delta(m.first_any,v0.first_any):''}</td><td class="num">${cell(m,'recall_at_k',!v.version)}</td><td class="num">${cell(m,'fp_earlier',!v.version,true)}</td><td class="num">${cell(m,'calls_per_claim',!v.version,true,false)}</td>`
      :'<td colspan="4" class="muted">pending — not scored yet</td>'}</tr>`};
  const first=L[0]?.watch_versions?.[0];const bestV0=L.filter(l=>l.best_version===0).map(l=>l.name);
  let h=first?row('v0 (all lines)'+(bestV0.length?' — best for '+bestV0.join(', '):''),first,false):'';
  L.forEach(l=>(l.watch_versions||[]).filter(v=>v.version>0).forEach(v=>{h+=row(l.name,v,v.version===l.best_version)}));
  setHTML('watch',(W.available?'':'<tr><td colspan="7" class="muted">experiments/watch.jsonl not found yet — start scripts/watch_scorer.py</td></tr>')+h);}
function detailRow(p){const diff=(p.diff||[]).map(d=>`<div><code>${esc(d.path)}</code></div><pre class="muted">old: ${esc(full(d.old))}</pre><pre>new: ${esc(full(d.new))}</pre>`).join('');
  const ch=(p.changes||[]).map(c=>`<div><code>${esc(c.path)}</code></div><pre>${esc(full(c.value))}</pre>`).join('');
  const m=k=>p[k]?`${k}: first_any ${pct(p[k].first_any)}, recall@k ${pct(p[k].recall_at_k)}, fp_earlier ${pct(p[k].fp_earlier)}`:'';
  return `<tr class="detail"><td colspan="10"><b>Rationale</b><pre>${esc(p.rationale)}</pre>
  ${m('train')?`<div>${m('train')}</div>`:''}${m('val')?`<div>${m('val')}</div>`:''}
  ${diff?`<b>Diff</b>${diff}`:`<b>Changes</b>${ch}`}</td></tr>`}
function renderProps(L){const f=document.getElementById('filter').value;
  const P=L.flatMap(l=>l.proposals).filter(p=>!f||p.line===f).sort((a,b)=>String(b.ts).localeCompare(String(a.ts))||(b.round-a.round)||(a.rank-b.rank));
  setHTML('props',P.map(p=>`<tr class="p" data-id="${esc(p.id)}"><td class="num">${tm(p.ts)}</td><td>${esc(p.line)}</td><td class="num">${p.round}</td><td class="num">${p.rank}</td>
  <td>${esc(p.category)}</td><td>${(p.changes||[]).map(c=>`<div><code>${esc(c.path)}=${esc(short(c.value))}</code></div>`).join('')}</td>
  <td class="d-${esc(p.decision)}">${esc(p.decision)}</td><td>${esc(p.reason)}</td><td class="num">${p.fixed??'–'}/${p.broken??'–'}</td><td class="num">${p.new_calls??''}</td></tr>`
  +(expanded.has(p.id)?detailRow(p):'')).join('')||'<tr><td colspan="10" class="muted">no proposals yet</td></tr>');}
function renderRest(L,merge){setHTML('analysis',L.map(l=>l.analysis?`<details data-k="a-${esc(l.name)}"><summary>${esc(l.name)} — round ${l.analysis.round}, ${l.analysis.n_proposals} proposals, ${l.analysis.latency_s}s <span class="muted">${tm(l.analysis.ts)}</span></summary><pre>${esc(l.analysis.analysis)}</pre></details>`:`<details data-k="a-${esc(l.name)}"><summary>${esc(l.name)} — none yet</summary></details>`).join(''));
  setHTML('logs',L.map(l=>`<div style="margin:8px 0"><b>${esc(l.name)}</b> <span class="muted">${esc(l.log_path)}${l.mtimes.log?' · modified '+Math.round(data.server_time-l.mtimes.log)+'s ago':' · missing'}</span><pre class="log">${esc(l.log.join('\n'))}</pre></div>`).join(''));
  let h='';const finals=L.flatMap(l=>l.finals.map(f=>({...f,line:l.name})));
  if(merge.length)h+=`<h2>Merge</h2><div class="tablewrap"><table><tr><th>time</th><th>winner</th><th>candidate</th><th>train</th><th>val</th><th>combined</th></tr>`+merge.map(m=>Object.entries(m.scores||{}).map(([k,s])=>`<tr><td>${tm(m.ts)}</td><td>${k===m.winner?'<b class="up">winner</b>':''}</td><td>${esc(k)}</td><td>${pct(s.train)}</td><td>${pct(s.val)}</td><td>${pct(s.combined)}</td></tr>`).join('')).join('')+'</table></div>';
  if(finals.length)h+=`<h2>Final</h2>`+finals.map(f=>`<div class="card" style="margin:6px 0"><b>${esc(f.line)}</b> v${esc(f.best_version)} — test first_any ${pct(f.base)} → ${pct(f.best)} ${delta(f.best,f.base)} ${f.best_ci95?'(95% CI '+esc(JSON.stringify(f.best_ci95))+')':''}, fixed ${f.fixed??'–'}, broken ${f.broken??'–'}, n ${f.n??'–'}</div>`).join('');
  setHTML('results',h);}
async function poll(){try{const r=await fetch('/api/status',{cache:'no-store'});data=await r.json();const L=data.lines;
  const sel=document.getElementById('filter');L.forEach(l=>{if(![...sel.options].some(o=>o.value===l.name))sel.add(new Option(l.name,l.name))});
  const base=L[0]?.baseline_val;const best=L.filter(l=>l.val_now!=null).sort((a,b)=>b.val_now-a.val_now)[0];
  document.getElementById('hdr').innerHTML=`updated ${new Date(data.server_time*1000).toLocaleTimeString()} · best val first_any <b class="num">${pct(best?.val_now)}</b> (${esc(best?.name)}) vs baseline ${pct(base)} ${delta(best?.val_now,base)}`;
  renderCards(L);renderChart(L);renderWatch(L,data.watch||{});renderProps(L);renderRest(L,data.merge);}
  catch(e){document.getElementById('hdr').textContent='update failed: '+e;}}
document.getElementById('props').addEventListener('click',e=>{const tr=e.target.closest('tr.p');if(!tr)return;const id=tr.dataset.id;
  expanded.has(id)?expanded.delete(id):expanded.add(id);if(data)renderProps(data.lines);});
document.getElementById('filter').addEventListener('change',()=>data&&renderProps(data.lines));
poll();setInterval(poll,5000);
</script>
<script>
// ---- Search space & search progress (lazy: Plotly + /api/landscape load when the section is near view) ----
const LS={data:null,ver:null,tab:'ret',mode:'paraphrase',ep:'all',surf:'on',rendered:{},plotly:null,started:false};
const PLOTLY_SRC='https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js';
const DEC={accepted:'--good',runner_up:'--warn',rejected:'--bad',not_evaluated:'--muted',invalid:'--warn'};
function lineColor(name){const i=(LS.data?.lines||[]).indexOf(name);return color(i<0?0:i)}
function loadPlotly(){if(LS.plotly)return LS.plotly;LS.plotly=new Promise((ok,no)=>{const s=document.createElement('script');
  s.src=PLOTLY_SRC;s.onload=()=>ok(window.Plotly);s.onerror=()=>{LS.plotly=null;no(new Error('could not load plotly.js'))};document.head.appendChild(s)});return LS.plotly}
function baseLayout(extra){const ink=css('--ink2'),grid=css('--grid');
  return Object.assign({paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'system-ui,sans-serif',size:12,color:ink},
    margin:{l:0,r:0,t:10,b:0},legend:{orientation:'h',y:1.02,x:0,font:{size:12}},hoverlabel:{font:{family:'system-ui,sans-serif',size:12}}},extra||{})}
function axis3d(title,extra){const ink=css('--ink2'),grid=css('--grid');
  return Object.assign({title:{text:title,font:{size:12}},gridcolor:grid,zerolinecolor:grid,color:ink,backgroundcolor:'rgba(0,0,0,0)',showbackground:false,tickfont:{size:11}},extra||{})}
function seqScale(){return[[0,css('--seq0')],[0.5,css('--seq1')],[1,css('--seq2')]]}
function msg3d(el,html){Plotly.purge(el);el.innerHTML=`<div class="muted" style="padding:16px">${html}</div>`}
function react3d(el,traces,layout){if(!el.classList.contains('js-plotly-plot'))el.innerHTML='';Plotly.react(el,traces,layout,{responsive:true,displaylogo:false})}
const pct1=v=>v==null?'–':(100*v).toFixed(1)+'%';

function renderRet(){const R=LS.data.retrieval,el=document.getElementById('p-ret');
  if(!R||!R.available){msg3d(el,'No sweep yet — run <code>scripts/retrieval_sweep.py</code>.');return}
  const M=R.modes[LS.mode];if(!M||!M.done){msg3d(el,'This query form is not in the sweep yet.');return}
  const cd=R.labels.map(l=>R.ks.map(()=>l)),traces=[];
  traces.push({type:'surface',name:'recall@k',x:R.ks,y:R.labels.map((_,i)=>i),z:M.z,customdata:cd,colorscale:seqScale(),opacity:0.95,
    colorbar:{title:{text:'recall@k',side:'right'},tickformat:'.0%',len:0.6,thickness:12,outlinewidth:0},
    hovertemplate:'k=%{x}<br>%{customdata}<br>recall@k <b>%{z:.1%}</b><extra>exact (sweep)</extra>'});
  const byLine={};(R.tried||[]).forEach(t=>{(byLine[t.line]=byLine[t.line]||[]).push(t)});
  Object.entries(byLine).forEach(([ln,arr])=>{const pts=arr.filter(t=>t.recall[LS.mode]!=null);if(!pts.length)return;
    traces.push({type:'scatter3d',mode:'markers',name:ln+' tried',legendgroup:ln,x:pts.map(t=>t.k),y:pts.map(t=>t.y),z:pts.map(t=>t.recall[LS.mode]+0.006),
      marker:{size:5,color:lineColor(ln),line:{color:css('--card'),width:1}},
      text:pts.map(t=>`${ln}: ${t.labels.map((l,i)=>l+' '+t.decisions[i]).join(', ')}<br>${t.index}/${t.query_template}/${t.speaker_filter} k=${t.k}<br>recall@k ${pct1(t.recall[LS.mode])}`),
      hovertemplate:'%{text}<extra></extra>'})});
  const BG={};(R.bests||[]).filter(b=>b.recall[LS.mode]!=null).forEach(b=>{(BG[b.k+'|'+b.y]=BG[b.k+'|'+b.y]||[]).push(b)});
  const B=Object.values(BG).map(g=>Object.assign({},g[0],{line:g[0].line,label:g.map(b=>`${b.line} v${b.version}`).join(', '),
    hover:g.map(b=>`${b.line} best v${b.version}${b.line_mode!==LS.mode?' (line scored on '+b.line_mode+')':''}`).join('<br>')}));
  if(B.length)traces.push({type:'scatter3d',mode:'markers+text',name:'current best per line',x:B.map(b=>b.k),y:B.map(b=>b.y),z:B.map(b=>b.recall[LS.mode]+0.015),
    marker:{size:8,symbol:'diamond',color:B.map(b=>lineColor(b.line)),line:{color:css('--ink'),width:2}},
    text:B.map(b=>b.label),textposition:'top center',textfont:{size:11,color:css('--ink')},
    customdata:B.map(b=>`${b.hover}<br>${b.index}/${b.query_template}/${b.speaker_filter} k=${b.k}<br>recall@k ${pct1(b.recall[LS.mode])}`),
    hovertemplate:'%{customdata}<extra></extra>'});
  const layout=baseLayout({uirevision:'ret',scene:{uirevision:'ret',
    xaxis:axis3d('k (candidates retrieved)',{range:[5,60]}),
    yaxis:axis3d('index / template · filter',{tickvals:R.labels.map((_,i)=>i),ticktext:R.labels,tickfont:{size:10}}),
    zaxis:axis3d('recall@k',{tickformat:'.0%'}),aspectmode:'manual',aspectratio:{x:1.2,y:1.4,z:0.7},
    camera:{eye:{x:1.25,y:-1.3,z:0.7}}}});
  react3d(el,traces,layout);
  setHTML('ret-top',(M.top||[]).map((g,i)=>`<tr><td class="num">${i+1}</td><td>${esc(g.index)}</td><td>${esc(g.query_template)}</td><td>${esc(g.speaker_filter)}</td><td class="num">${g.k}</td><td class="num"><b>${pct1(g.recall)}</b></td></tr>`).join('')
    +`<tr><td colspan="6" class="muted">Top grid cells (k ∈ ${esc((R.k_grid||[]).join(', '))}) for ${esc(LS.mode)} on ${R.n_items} train items; ${M.done}/12 combinations swept${R.updated?' · updated '+esc(tm(R.updated)):''}.</td></tr>`);}

const SYM={v0:'square',rescore:'cross',accepted:'diamond',rejected:'circle-open',runner_up:'square-open'};
function renderFit(){const F=LS.data.fitness,el=document.getElementById('p-fit');
  if(!F||!F.available){msg3d(el,'No fully evaluated configs yet.');return}
  const cur=LS.ep==='current',traces=[],S=F.surfaces[cur?'current':'all'];
  if(LS.surf==='on'&&S)traces.push({type:'surface',name:'interpolated (illustrative)',x:S.x,y:S.y,z:S.z,colorscale:seqScale(),cmin:0.4,cmax:1,opacity:0.5,showscale:false,
    hovertemplate:'interpolated %{z:.1%}<extra>illustrative only</extra>',showlegend:true});
  (LS.data.lines||[]).forEach(ln=>{const P=F.points.filter(p=>p.line===ln&&(!cur||p.current));if(!P.length)return;const col=lineColor(ln);
    traces.push({type:'scatter3d',mode:'markers',name:ln+(LS.data.line_modes[ln]!=='paraphrase'?' ('+LS.data.line_modes[ln]+')':''),legendgroup:ln,
      x:P.map(p=>p.x),y:P.map(p=>p.y),z:P.map(p=>p.z),
      marker:{size:P.map(p=>p.decision==='accepted'||p.kind==='v0'?9:6),symbol:P.map(p=>SYM[p.kind==='proposal'?p.decision:p.kind]||'circle'),color:col,line:{color:col,width:2}},
      text:P.map(p=>`<b>${esc(ln)}</b> ${esc(p.label)}<br>train first_any <b>${pct1(p.z)}</b>${p.val!=null?' · val '+pct1(p.val):''} (GT v${p.gt}, n=${p.n})<br>${esc(p.cfg)}${p.changes.length?'<br>changes: '+esc(p.changes.join('; ')).slice(0,300):''}`),
      hovertemplate:'%{text}<extra></extra>'});
    const path=(F.paths[ln]||[]).filter(q=>!cur||q.current);
    if(path.length>1)traces.push({type:'scatter3d',mode:'lines',name:ln+' accepted path',legendgroup:ln,showlegend:false,x:path.map(q=>q.x),y:path.map(q=>q.y),z:path.map(q=>q.z),
      line:{color:col,width:5},text:path.map(q=>`${ln} path: ${q.label} ${pct1(q.z)}`),hovertemplate:'%{text}<extra></extra>'});});
  const ax=F.axes.map((a,i)=>`PC${i+1} (${Math.round(100*a.evr)}%)`);
  const zs=F.points.filter(p=>!cur||p.current).map(p=>p.z);if(!zs.length)zs.push(0.5,0.9);
  const layout=baseLayout({uirevision:'fit',scene:{uirevision:'fit',xaxis:axis3d(ax[0]),yaxis:axis3d(ax[1]),
    zaxis:axis3d('train first_any',{tickformat:'.0%',range:[Math.max(0,Math.min(...zs,0.9)-0.05),Math.min(1,Math.max(...zs,0.5)+0.03)]}),
    aspectmode:'manual',aspectratio:{x:1.2,y:1.2,z:0.7},camera:{eye:{x:1.2,y:-1.35,z:0.7}}}});
  react3d(el,traces,layout);
  const load=a=>a.loadings.map(l=>`${esc(l.feature)} ${l.w>=0?'+':'−'}${Math.abs(l.w).toFixed(2)}`).join(', ');
  const ce=F.current_epoch;
  document.getElementById('fit-cap').innerHTML=`<b>Positions are a projection; the surface is interpolated, illustrative only.</b> One point per fully evaluated config (${F.n_points} in all, ${F.points.filter(p=>p.current).length} on the current scoring): proposals that reached the full train/val evaluation, each line’s v0 and its re-scored incumbents. Each config is a vector of its knobs (k, one-hot index / template / filter, effort, batch, show_date, speaker_hint) plus a TF-IDF of the verifier prompt reduced by SVD (${F.n_prompts} distinct prompts), projected to 2D by PCA. <b>x</b> = ${ax[0]}: ${load(F.axes[0])}; <b>y</b> = ${ax[1]}: ${load(F.axes[1]||{loadings:[]})}. Nearby points are similar configs, not similar scores. <b>z</b> = train first_any. Square = v0, diamond = accepted, hollow circle = rejected, hollow square = runner-up, cross = incumbent re-scored after the ground-truth change; lines join each line’s accepted path. The surface is a linear interpolation (scipy griddata) over paraphrase-line points only; between points it is a guess.
  <br><b>Caveats:</b> scores mix train sizes and ground-truth versions (${F.epochs.map(e=>'GT v'+e.gt+' n='+e.n).join(', ')}); use “current scoring only” (GT v${ce?.gt}, n=${ce?.n}) to compare like with like. The <b>hard</b> line traces <code>paraphrase_hard</code>, so its points are drawn in its own colour and left out of the surface${F.points.some(p=>p.line==='hard')?'':' (none yet — its re-baseline has not been logged)'}. Train n=60 scores move ±2–3 items by chance.`;}

function treeSVG(L){const T=L.tree;if(!T.length)return '<div class="muted" style="font-size:12px">no rounds yet</div>';
  const colW=58,rowH=22,x0=34,y0=28,maxC=Math.max(1,...T.map(t=>t.candidates.filter(c=>c.decision!=='accepted').length));
  const W=x0+colW*(T.length)+30,H=y0+rowH*(maxC+1)+16,lc=lineColor(L.line);
  const sx=i=>x0+colW*i;let h='';
  // spine
  h+=`<line x1="${sx(0)}" y1="${y0}" x2="${sx(T.length)}" y2="${y0}" style="stroke:${lc};stroke-width:2"/>`;
  h+=`<circle cx="${sx(0)}" cy="${y0}" r="8" style="fill:${lc}"><title>v0 (start${L.started_from?' = '+esc(L.started_from):''})</title></circle><text x="${sx(0)}" y="${y0-12}" text-anchor="middle">v0</text>`;
  T.forEach((t,i)=>{const cx=sx(i+1);h+=`<text x="${cx}" y="${H-4}" text-anchor="middle">r${t.round}</text>`;
    const tip=c=>`<title>${esc(`r${c.round} #${c.rank} ${c.decision}${c.z!=null?' · train '+pct1(c.z):''}${c.val!=null?' val '+pct1(c.val):''}${c.fixed!=null?' · fixed '+c.fixed+' broke '+c.broken:''}\n${c.changes.join('\n')}\n${c.reason||''}`)}</title>`;
    let j=0;t.candidates.forEach(c=>{const full=c.z!=null,col=css(DEC[c.decision]||'--muted');
      if(c.decision==='accepted'){h+=`<circle cx="${cx}" cy="${y0}" r="9" style="fill:${lc};stroke:${css('--good')};stroke-width:3">${tip(c)}</circle><text x="${cx}" y="${y0-12}" text-anchor="middle" style="fill:${css('--ink')};font-weight:600">v${c.version}</text>`;return}
      j++;const cy=y0+rowH*j;
      h+=`<line x1="${sx(i)}" y1="${y0}" x2="${cx}" y2="${cy}" style="stroke:${css('--grid')};stroke-width:1"/>`;
      h+=c.decision==='invalid'?`<rect x="${cx-5}" y="${cy-5}" width="10" height="10" rx="2" style="fill:none;stroke:${col};stroke-width:2">${tip(c)}</rect>`
        :`<circle cx="${cx}" cy="${cy}" r="6" style="fill:${full?col:css('--card')};stroke:${col};stroke-width:2">${tip(c)}</circle>`;});
    if(!t.accepted_version)h+=`<circle cx="${cx}" cy="${y0}" r="3.5" style="fill:${css('--card')};stroke:${lc};stroke-width:2"><title>r${t.round}: no change accepted, incumbent v${t.base_version??'?'} kept${t.ended?'':' (round in progress)'}</title></circle>`;});
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="search tree for ${esc(L.line)}">${h}</svg>`;}

function renderProg(){const P=LS.data.progress||[],box=document.getElementById('prog');
  if(box.dataset.v!==LS.ver){box.innerHTML=P.map((L,i)=>`<div class="card"><div class="row"><b><span class="sw" style="background:${lineColor(L.line)}"></span>${esc(L.line)}</b><span class="muted">${esc(L.mode)} · round ${L.round} · best v${L.best_version}</span></div>
    ${L.copied_baseline&&!L.best.length?`<div class="muted" style="font-size:12px">started from ${esc(L.started_from)}; waiting for its re-baseline on ${esc(L.mode)}</div>`:''}
    <div class="prog" id="prog-${i}"></div><div class="tree">${treeSVG(L)}</div></div>`).join('');box.dataset.v=LS.ver;
    box.querySelectorAll('.tree').forEach(t=>t.scrollLeft=t.scrollWidth);}  // latest rounds in view
  P.forEach((L,i)=>{const el=document.getElementById('prog-'+i),C=L.candidates.filter(c=>c.z!=null),tr=[];
    if(!C.length&&!L.best.length){el.style.height='auto';el.innerHTML='<div class="muted" style="font-size:12px;padding:8px 0">no evaluated configs yet</div>';return}
    const groups=[['accepted','triangle-up',true],['runner_up','square-open',false],['rejected','circle-open',false],['not_evaluated','circle-open',false]];
    groups.forEach(([d,sym])=>{const pts=C.filter(c=>c.decision===d);if(!pts.length)return;
      tr.push({type:'scatter',mode:'markers',name:d.replace('_',' '),x:pts.map(c=>c.round+(c.rank-1)*0.12),y:pts.map(c=>c.z),
        marker:{symbol:sym,size:9,color:css(DEC[d]),line:{color:css(DEC[d]),width:2}},
        text:pts.map(c=>`r${c.round} #${c.rank} ${c.decision}: train ${pct1(c.z)}${c.val!=null?' · val '+pct1(c.val):''} (GT v${c.gt}, n=${c.n})<br>${esc(c.changes.join('; ')).slice(0,200)}`),hovertemplate:'%{text}<extra></extra>'})});
    if(L.best.length)tr.unshift({type:'scatter',mode:'lines+markers',name:'best so far',x:L.best.map(b=>b.round),y:L.best.map(b=>b.z),line:{shape:'hv',color:lineColor(L.line),width:2},marker:{size:5,color:lineColor(L.line)},
      text:L.best.map(b=>`${b.label}: ${pct1(b.z)} (GT v${b.gt}, n=${b.n})`),hovertemplate:'%{text}<extra></extra>'});
    const shapes=(L.rescores||[]).map(r=>({type:'line',xref:'x',yref:'paper',x0:r.round+0.5,x1:r.round+0.5,y0:0,y1:1,line:{color:css('--muted'),width:1,dash:'dash'}}));
    const ann=(L.rescores||[]).slice(-1).map(r=>({xref:'x',yref:'paper',x:r.round+0.5,y:1,text:'re-scored',showarrow:false,xanchor:'right',yanchor:'top',font:{size:10,color:css('--muted')}}));
    const grid=css('--grid');
    Plotly.react(el,tr,baseLayout({uirevision:'p'+L.line,margin:{l:44,r:8,t:26,b:34},showlegend:true,legend:{orientation:'h',y:1.0,yanchor:'bottom',x:0,font:{size:11}},
      xaxis:{title:{text:'round',font:{size:11}},dtick:1,range:[-0.5,Math.max(1,L.round)+0.6],gridcolor:grid,zeroline:false},
      yaxis:{tickformat:'.0%',gridcolor:grid,zeroline:false,title:{text:'train first_any',font:{size:11}}},shapes,annotations:ann}),{displayModeBar:false,responsive:true});});}

function renderLS(force){if(!LS.data||!window.Plotly)return;const key=`${LS.ver}|${LS.mode}|${LS.ep}|${LS.surf}`;
  if(!force&&LS.rendered[LS.tab]===key)return;LS.rendered[LS.tab]=key;
  try{({ret:renderRet,fit:renderFit,prog:renderProg})[LS.tab]()}catch(e){document.getElementById('lsstatus').textContent='render failed: '+e;console.error(e)}}
async function pollLS(){try{const r=await fetch('/api/landscape',{cache:'no-store'});const d=await r.json();if(d.error)throw new Error(d.error);
    if(d.version!==LS.ver){LS.data=d;LS.ver=d.version;renderLS();}
    document.getElementById('lsstatus').textContent=`data ${d.version} · checked ${new Date().toLocaleTimeString()}`;}
  catch(e){document.getElementById('lsstatus').textContent='landscape update failed: '+e.message}}
async function startLS(){if(LS.started)return;LS.started=true;document.getElementById('lsstatus').textContent='loading plotly…';
  try{await loadPlotly();}catch(e){LS.started=false;document.getElementById('lsstatus').textContent=e.message;return}
  await pollLS();setInterval(pollLS,30000);}
document.querySelectorAll('#ls .tabs button').forEach(b=>b.addEventListener('click',()=>{LS.tab=b.dataset.t;
  document.querySelectorAll('#ls .tabs button').forEach(x=>x.classList.toggle('on',x===b));
  document.querySelectorAll('#ls .lstab').forEach(t=>t.hidden=t.dataset.t!==LS.tab);startLS();renderLS();}));
function seg(id,prop){document.querySelectorAll('#'+id+' button').forEach(b=>b.addEventListener('click',()=>{LS[prop]=b.dataset.v;
  document.querySelectorAll('#'+id+' button').forEach(x=>x.classList.toggle('on',x===b));renderLS();}))}
seg('ret-mode','mode');seg('fit-ep','ep');seg('fit-surf','surf');
const lsHash=location.hash.match(/^#ls-(ret|fit|prog)$/);  // deep link, e.g. #ls-fit
if(lsHash){document.querySelector(`#ls .tabs button[data-t="${lsHash[1]}"]`).click();document.getElementById('ls-h').scrollIntoView()}
else if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){io.disconnect();startLS()}},{rootMargin:'300px'});io.observe(document.getElementById('ls'))}
else startLS();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            try:
                body = json.dumps(status(), default=str).encode("utf-8")
                self._send(200, body, "application/json")
            except Exception as e:  # never crash the server on odd data
                self._send(500, json.dumps({"error": repr(e)}).encode(), "application/json")
        elif path == "/api/landscape":
            try:
                import landscape  # numpy/scipy only load when the section is first opened
                body = json.dumps(landscape.landscape(), default=str).encode("utf-8")
                self._send(200, body, "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": repr(e)}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--budget-all", type=int, default=2500, help="budget for line 'all'")
    ap.add_argument("--budget-default", type=int, default=1500, help="budget for other lines")
    ap.add_argument("--budget", action="append", default=[], metavar="LINE=N", help="per-line override")
    args = ap.parse_args()
    global DEFAULT_BUDGET
    BUDGETS["all"] = args.budget_all
    DEFAULT_BUDGET = args.budget_default
    for item in args.budget:
        k, _, v = item.partition("=")
        BUDGETS[k] = int(v)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard on http://127.0.0.1:{args.port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
