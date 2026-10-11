"""Live dashboard for the self-improvement loop (read-only).

Run:  .venv/bin/python -I scripts/dashboard.py [--port 8765]
Then open http://127.0.0.1:8765/
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
--s4:#eda100;--s5:#e87ba4;--s6:#008300;--wash:rgba(11,11,11,.04)}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--border:rgba(255,255,255,.10);--good:#0ca30c;--bad:#e66767;--warn:#fab219;--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--wash:rgba(255,255,255,.05)}}
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
