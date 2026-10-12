"""Firstline: web app for the news provenance tracer (demo front end).

  .venv/bin/python -I scripts/app.py [--port 8780] [--dashboard http://127.0.0.1:8765]
      ->  http://127.0.0.1:8780/        (tabs: #trace = the tracer, #loop = the self-improvement dashboard)

The page has two tabs. "Self-improvement loop" frames the loop dashboard (scripts/dashboard.py), which
this server reverse-proxies so it is same-origin: GET /dash/<path> -> <dashboard>/<path>, and any GET
/api/* this app does not handle itself (e.g. /api/status, /api/landscape) -> <dashboard>/api/*, since
the dashboard's JS fetches absolute /api/... URLs.

Every trace uses the improvement loop's current best config (experiments/state.json, re-read per
request), so the app gets better while the loop runs. Traces run in background threads (3 at a
time, extras queue); results are kept in an in-memory LRU keyed by (statement, as_of, config hash).
No LLM calls happen at startup; each new trace makes ~2 Haiku calls (cached on disk by verify.py).
"""
import argparse
import hashlib
import html
import http.client
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import duckdb

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # state.json config paths and the LLM cache path are relative to the project
sys.path.insert(0, str(ROOT / "scripts"))
import qb_config  # noqa: E402
import trace_quote as tq  # noqa: E402
from verify import LLMCache  # noqa: E402

STATIC = ROOT / "scripts" / "app_static"
STATE = ROOT / "experiments" / "state.json"
LEDGER = ROOT / "experiments" / "ledger.jsonl"
WATCH = ROOT / "bench" / "qb_watch.jsonl"
DASHBOARD = "http://127.0.0.1:8765"  # upstream for the proxied dashboard tab (--dashboard)
DEFAULT_AS_OF = "2020-04-18T00:00:00"
MAX_CHARS = 500
WORKERS = 3

# Held-out watch-set items shown as clickable examples (index into bench/qb_watch.jsonl).
EXAMPLES = [
    (42, "Politics", "Putin on Russia's weapons"),
    (13, "Politics", "Macron on Brexit"),
    (27, "Politics", "Pelosi on the ACA lawsuit"),
    (22, "COVID-19", "Sunak's reassurance"),
    (65, "COVID-19", "Lifting lockdowns too early"),
    (99, "COVID-19", "Zhong Nanshan on reinfection"),
    (24, "Business", "Economy losing ground"),
    (11, "Sport", "Fred Couples on Augusta"),
]

# --- pipeline stage tracking: wrap the tracer's steps (in this process only) so a job can report
# which one it is in; trace() looks these up as module globals at call time.
_tls = threading.local()


def _staged(stage, fn):
    def wrapper(*a, **k):
        job = getattr(_tls, "job", None)
        if job is not None:
            job["stage"] = stage
        return fn(*a, **k)
    return wrapper


tq.candidates = _staged("search", tq.candidates)
tq.verify = _staged("verify", tq.verify)
tq.timeline = _staged("timeline", tq.timeline)


# --- loop state / config -------------------------------------------------------------------------
_state_lock = threading.Lock()
_last_state = {}


def read_json_retry(path, tries=5):
    for i in range(tries):
        try:
            return json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            if i == tries - 1:
                raise
            time.sleep(0.05)


def current_config():
    """(cfg, info) for the loop's current best config; falls back to the last good read."""
    global _last_state
    try:
        st = read_json_retry(STATE)
        cfg = qb_config.load(ROOT / st["best_config"])
        blob = json.dumps(cfg, sort_keys=True)
        info = {"version": st.get("best_version"), "path": st["best_config"],
                "hash": hashlib.sha256(blob.encode()).hexdigest()[:12],
                "val_first_any": (st.get("val_metrics") or {}).get("first_any"),
                "val_n": (st.get("val_metrics") or {}).get("n"),
                "gt_version": st.get("gt_version"), "round": st.get("round")}
        with _state_lock:
            _last_state = {"cfg": cfg, "info": info}
        return cfg, info
    except Exception:
        with _state_lock:
            if _last_state:
                return _last_state["cfg"], _last_state["info"]
        raise


def baseline_val(gt_version=None):
    """Baseline (v0) val first_any: the rebaseline under the current ground truth if there is one,
    else the first rebaseline, else the original baseline."""
    base, rebases = None, []
    try:
        for line in open(LEDGER):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "baseline" and base is None:
                base = e.get("val", {}).get("first_any")
            elif e.get("type") == "rebaseline":
                rebases.append(e)
    except OSError:
        return None
    for e in rebases:
        if gt_version is not None and e.get("gt_version") == gt_version:
            return e.get("v0", {}).get("val", {}).get("first_any")
    if rebases:
        return rebases[0].get("v0", {}).get("val", {}).get("first_any")
    return base


def load_examples():
    items = [json.loads(l) for l in open(WATCH)]
    out = []
    for idx, topic, label in EXAMPLES:
        it = items[idx]
        gt = it.get("gt_first_any") or {}
        out.append({"id": it["quote_id"], "topic": topic, "label": label, "speaker": it["speaker"],
                    "statement": it["paraphrase"], "as_of": it["as_of"],
                    "benchmark": {"date": gt.get("date"), "domain": gt.get("domain"), "url": gt.get("url")}})
    return out


# --- database ------------------------------------------------------------------------------------
class DB:
    def __init__(self, path):
        self.con = duckdb.connect(path, read_only=True)
        self.con.execute("LOAD fts")
        self._stats = None
        self._lock = threading.Lock()

    def cursor(self):
        return self.con.cursor()

    def stats(self):
        with self._lock:
            if self._stats is None:
                cur = self.cursor()
                lo, hi, n = cur.execute("SELECT min(date), max(date), count(*) FROM quote_mentions").fetchone()
                nq = cur.execute("SELECT count(*) FROM quotes").fetchone()[0]
                self._stats = {"start": lo.isoformat()[:10], "end": hi.isoformat()[:10],
                               "n_mentions": n, "n_quotes": nq}
            return self._stats


# --- jobs ----------------------------------------------------------------------------------------
class Jobs:
    def __init__(self, db, cache):
        self.db, self.cache = db, cache
        self.lock = threading.Lock()
        self.jobs = OrderedDict()      # id -> job
        self.inflight = {}             # key -> job id (dedupe identical concurrent requests)
        self.lru = OrderedDict()       # key -> result
        self.pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="trace")

    def submit(self, statement, as_of):
        cfg, info = current_config()
        key = (statement, as_of, info["hash"])
        now = time.time()
        with self.lock:
            if key in self.lru:
                self.lru.move_to_end(key)
                res = dict(self.lru[key], app_cache=True)
                job = self._new(key, now, status="done", result=res, finished=now, started=now)
                return job["id"]
            if key in self.inflight and self.inflight[key] in self.jobs:
                return self.inflight[key]
            job = self._new(key, now, status="queued")
            self.inflight[key] = job["id"]
        self.pool.submit(self._run, job, cfg, info)
        return job["id"]

    def _new(self, key, now, **kw):
        job = {"id": uuid.uuid4().hex[:16], "key": key, "created": now, "started": None, "finished": None,
               "stage": None, "result": None, "error": None, **kw}
        self.jobs[job["id"]] = job
        while len(self.jobs) > 500:
            self.jobs.popitem(last=False)
        return job

    def _run(self, job, cfg, info):
        job["status"], job["started"], job["stage"] = "running", time.time(), "search"
        _tls.job = job
        statement, as_of, _ = job["key"]
        try:
            res = tq.trace(self.db.cursor(), self.cache, statement, as_of, cfg)
            out = shape(res, info, time.time() - job["started"])
            with self.lock:
                self.lru[job["key"]] = out
                while len(self.lru) > 256:
                    self.lru.popitem(last=False)
            job["result"], job["status"] = out, "done"
        except Exception as e:  # report, never crash the worker
            traceback.print_exc()
            job["error"], job["status"] = f"{type(e).__name__}: {e}"[:500], "error"
        finally:
            _tls.job = None
            job["finished"] = time.time()
            with self.lock:
                self.inflight.pop(job["key"], None)

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            queued_ahead = sum(1 for j in self.jobs.values()
                               if j["status"] == "queued" and j["created"] < job["created"])
        end = job["finished"] or time.time()
        out = {"id": job["id"], "status": job["status"], "stage": job["stage"],
               "elapsed": round(end - job["created"], 1)}
        if job["status"] == "queued":
            out["queue_position"] = queued_ahead + 1
        if job["status"] == "done":
            out["result"] = job["result"]
        if job["status"] == "error":
            out["error"] = job["error"]
        return out


def shape(res, info, elapsed):
    def mention(t):
        if not t:
            return None
        return {k: t.get(k) for k in ("date", "has_time", "domain", "kind", "grp", "url", "title",
                                      "quote_id", "via")}
    return {
        "statement": res["statement"], "as_of": res["as_of"],
        "n_candidates": len(res["candidates"]),
        "matched_quotes": res["matched_quotes"],
        "first_any": mention(res["first_any"]), "first_original": mention(res["first_original"]),
        "spread": [mention(t) for t in res["spread"]],
        "n_mentions": res["n_mentions"], "n_outlets": len(res["spread"]),
        "llm_calls": res["llm_calls"],
        "config": {"version": info["version"], "hash": info["hash"]},
        "trace_seconds": round(elapsed, 1),
    }


# --- input validation ----------------------------------------------------------------------------
def parse_as_of(value):
    if value in (None, ""):
        return DEFAULT_AS_OF
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?", value.strip()):
        raise ValueError("as_of must look like 2020-03-15 or 2020-03-15T12:00")
    try:
        dt = datetime.fromisoformat(value.strip().replace(" ", "T"))
    except ValueError:
        raise ValueError("as_of is not a valid date/time")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def clean_statement(value):
    if not isinstance(value, str):
        raise ValueError("statement must be text")
    s = re.sub(r"\s+", " ", value).strip()
    if not s:
        raise ValueError("Please enter a statement to trace.")
    if len(s) > MAX_CHARS:
        raise ValueError(f"Statement is too long ({len(s)} characters; the limit is {MAX_CHARS}).")
    return s


# --- HTTP ----------------------------------------------------------------------------------------
# The shell may frame same-origin pages (the proxied dashboard) and be framed only by itself.
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
       "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
       "frame-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'self'")
# Proxied dashboard HTML: inline scripts/styles, Chart.js + Plotly from jsDelivr (Plotly's WebGL
# traces compile code at runtime, hence 'unsafe-eval'; blob: workers/images), same-origin fetches.
DASH_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; "
            "connect-src 'self'; worker-src 'self' blob:; frame-src 'none'; base-uri 'none'; "
            "form-action 'self'; frame-ancestors 'self'")
DASH_HINT = "dashboard not running \u2014 start it with `.venv/bin/python -I scripts/dashboard.py`"
DASH_DOWN_PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Dashboard unavailable</title><style>
:root{color-scheme:light;--bg:#f4f3ef;--surface:#fcfcfb;--border:#dedcd5;--text:#0f0f0e;--text-2:#4a4946;--link:#1d5fae}
@media (prefers-color-scheme: dark){:root{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--border:#36352f;
--text:#f0efec;--text-2:#c9c8be;--link:#8bb8f2}}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:620px;margin:48px auto;padding:22px;background:var(--surface);border:1px solid var(--border);border-radius:14px}
h1{font-size:18px;margin:0 0 8px}p{color:var(--text-2);margin:8px 0}a{color:var(--link)}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);padding:2px 6px;border-radius:6px}
</style></head><body><main><h1>The self-improvement dashboard isn\u2019t running</h1>
<p>This tab shows the loop dashboard from <code>{upstream}</code>, which did not answer ({why}).</p>
<p>Start it with <code>.venv/bin/python -I scripts/dashboard.py</code>, then <a href="/dash/">retry</a>.</p>
</main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "Firstline/1.0"
    app = None  # set in main

    def log_message(self, fmt, *args):
        if "/api/job/" not in (args[0] if args else ""):
            sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if ctype.startswith("text/html"):
            self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        self.wfile.write(data)

    # --- reverse proxy to the loop dashboard (GET only; streamed) ---
    def _proxy(self, upstream_path):
        up = self.app.dashboard
        is_api = upstream_path.startswith("/api/")
        conn = http.client.HTTPConnection(up["host"], up["port"], timeout=up["connect_timeout"])
        try:
            try:
                conn.connect()
                conn.sock.settimeout(up["read_timeout"])
                conn.request("GET", up["prefix"] + upstream_path,
                             headers={"Accept": self.headers.get("Accept", "*/*"), "Accept-Encoding": "identity"})
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException, ValueError) as e:
                why = "timed out" if isinstance(e, TimeoutError) else type(e).__name__
                if is_api:
                    return self._send(502, {"error": DASH_HINT, "upstream": up["url"], "detail": why})
                page = DASH_DOWN_PAGE.replace("{upstream}", html.escape(up["url"])).replace("{why}", html.escape(why))
                return self._send(502, page.encode(), "text/html; charset=utf-8")
            ctype = resp.getheader("Content-Type") or "application/octet-stream"
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype)
            if resp.getheader("Content-Length") is not None:
                self.send_header("Content-Length", resp.getheader("Content-Length"))
            self.send_header("Cache-Control", resp.getheader("Cache-Control") or "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy", DASH_CSP)
            self.end_headers()
            self.close_connection = True  # length may be unknown: end the body by closing
            while True:
                try:
                    chunk = resp.read(65536)
                except (OSError, http.client.HTTPException):
                    break  # upstream died mid-body; headers are already out, so just close
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except OSError:
                    break  # client went away
        finally:
            conn.close()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/meta":
                return self._send(200, self.app.meta())
            if path.startswith("/api/job/"):
                job = self.app.jobs.get(path[len("/api/job/"):])
                return self._send(200, job) if job else self._send(404, {"error": "unknown job"})
            if path == "/article":
                url = parse_qs(urlsplit(self.path).query).get("url", [""])[0]
                if not url.startswith(("http://", "https://")) or len(url) > 2000:
                    return self._send(400, {"error": "expected ?url=http(s)://..."})
                page = self.app.article(url)
                return self._send(200, page, "text/html; charset=utf-8") if page else \
                    self._send(404, {"error": "article not in corpus"})
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            query = ("?" + self.path.split("?", 1)[1]) if "?" in self.path else ""
            if path == "/dash":
                self.send_response(301)
                self.send_header("Location", "/dash/" + query)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path.startswith("/dash/"):
                return self._proxy(path[len("/dash"):] + query)
            if path == "/api/trace":
                return self._send(405, {"error": "POST a JSON body to /api/trace"})
            if path.startswith("/api/"):  # the dashboard's JS fetches absolute /api/... URLs
                return self._proxy(path + query)
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/trace":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20000:
                return self._send(413, {"error": "request too large"})
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
            statement = clean_statement(body.get("statement"))
            as_of = parse_as_of(body.get("as_of"))
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"error": str(e)})
        try:
            return self._send(202, {"job_id": self.app.jobs.submit(statement, as_of)})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


# --- article excerpts ---------------------------------------------------------------------------
# Quotebank keeps each quote with ~500 characters of surrounding text, not the full article, and
# many 2020 URLs no longer resolve. /article?url=... serves what the corpus holds for that article
# (title, outlet, time, every quoted passage in context) plus the original and an archived link.
ARTICLE_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f22;--muted:#5f6670;--card:#fff;--line:#e3e5e8;--accent:#2a78d6;--hl:#fff1b8}
@media (prefers-color-scheme: dark){:root{--bg:#121417;--fg:#e8eaed;--muted:#9aa1ab;--card:#1b1e22;--line:#2c3036;--accent:#6aa8ff;--hl:#4a3f12}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 48px}
.kicker{color:var(--muted);font-size:14px}h1{font-size:24px;line-height:1.3;margin:6px 0 12px}
.links a{color:var(--accent);margin-right:16px;font-size:14px}.note{color:var(--muted);font-size:13px;margin:12px 0 20px}
.ex{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.ex .who{font-size:13px;color:var(--muted);margin-bottom:6px}mark{background:var(--hl);color:inherit;padding:0 2px;border-radius:3px}
"""


def wayback(url, date):
    stamp = re.sub(r"[^0-9]", "", str(date))[:14]
    return f"https://web.archive.org/web/{stamp}/{url}"


def render_article(db, url):
    # the corpus stores URLs as crawled (often with raw non-ASCII, e.g. curly quotes), while browsers
    # percent-encode them; match either form
    forms = list(dict.fromkeys([url, unquote(url)]))
    rows = db.cursor().execute(f"""
        SELECT date, domain, title, speaker, quotation, left_context, right_context, has_time, url
        FROM quote_mentions WHERE url IN ({','.join('?' * len(forms))}) ORDER BY quote_id""", forms).fetchall()
    if not rows:
        return None
    date, domain, title = rows[0][0], rows[0][1], rows[0][2] or url
    when = date.strftime("%Y-%m-%d %H:%M:%S") if rows[0][7] else date.strftime("%Y-%m-%d") + " (date only)"
    e = html.escape
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
             f"<meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>{e(title[:80])} · Firstline</title><style>{ARTICLE_CSS}</style></head><body><main>",
             f"<div class=kicker>{e(domain)} · {e(when)}</div><h1>{e(title)}</h1>",
             f"<div class=links><a href='{e(url)}' target=_blank rel='noopener noreferrer nofollow'>Original article ↗</a>"
             f"<a href='{e(wayback(url, date))}' target=_blank rel='noopener noreferrer nofollow'>Archived copy ↗</a></div>",
             "<p class=note>Excerpts as stored in Quotebank: each quotation with the text around it. "
             "The full article is not part of the corpus, and many 2020 links no longer resolve.</p>"]
    for _, _, _, speaker, quote, left, right, _, _ in rows:
        parts.append(f"<div class=ex><div class=who>{e(speaker or 'Unattributed')}</div>"
                     f"…{e(left or '')} <mark>{e(quote)}</mark> {e(right or '')}…</div>")
    parts.append("</main></body></html>")
    return "".join(parts).encode()


def parse_upstream(url, timeout):
    u = urlsplit(url if "://" in url else "http://" + url)
    if u.scheme != "http" or not u.hostname:
        raise SystemExit(f"--dashboard must be an http:// URL, got {url!r}")
    return {"url": f"http://{u.netloc}{u.path.rstrip('/')}", "host": u.hostname, "port": u.port or 80,
            "prefix": u.path.rstrip("/"), "connect_timeout": 3.0, "read_timeout": timeout}


class App:
    def __init__(self, db_path, dashboard=None):
        self.dashboard = dashboard or parse_upstream(DASHBOARD, 30.0)
        self.db = DB(db_path)
        self.jobs = Jobs(self.db, LLMCache())
        self.examples = load_examples()
        self.articles = OrderedDict()  # url -> rendered page (small LRU; lookups scan the mentions table)

    def article(self, url):
        if url in self.articles:
            self.articles.move_to_end(url)
            return self.articles[url]
        page = render_article(self.db, url)
        if page:
            self.articles[url] = page
            while len(self.articles) > 256:
                self.articles.popitem(last=False)
        return page

    def meta(self):
        _, info = current_config()
        return {"config_version": info["version"], "config_hash": info["hash"],
                "val_first_any": info["val_first_any"], "val_n": info["val_n"],
                "baseline_val_first_any": baseline_val(info["gt_version"]),
                "loop_round": info["round"], "dashboard": "/#loop", "dashboard_upstream": self.dashboard["url"],
                "corpus": self.db.stats(), "default_as_of": DEFAULT_AS_OF,
                "max_chars": MAX_CHARS, "examples": self.examples}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--db", default="data/news.duckdb")
    ap.add_argument("--dashboard", default=DASHBOARD, help="loop dashboard to proxy under /dash/ and /api/*")
    ap.add_argument("--dashboard-timeout", type=float, default=30.0,
                    help="seconds to wait on a dashboard response (connecting is capped at 3 s)")
    a = ap.parse_args()
    Handler.app = App(a.db, parse_upstream(a.dashboard, a.dashboard_timeout))
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Firstline on http://{a.host}:{a.port}/  (config: {current_config()[1]['path']}; "
          f"dashboard tab proxies {Handler.app.dashboard['url']})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
