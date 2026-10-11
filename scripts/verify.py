"""Same-fact verifier: asks Claude (via headless Claude Code, i.e. the subscription) which
candidate passages report the same fact as a claim.

Usage (bench): python -I scripts/verify.py bench/queries.jsonl <label> <model> <effort> [workers]
Writes bench/runs/<label>.jsonl; already-finished queries are skipped, so reruns are free.
"""
import hashlib
import json
import os
import sqlite3
import subprocess
import threading
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLAUDE = os.environ.get("CLAUDE_BIN") or os.environ.get("CLAUDE_CODE_EXECPATH") or "claude"

SYSTEM = """You check whether earlier news passages report the same fact as a claim.

A candidate MATCHES if it reports the claim's central fact: the same event, action or \
statement about the same entities, with no conflicting key details (figures, dates, \
amounts, direction of change). Different wording is fine. A candidate that gives less detail \
than the claim still matches if it states the central fact; one that adds detail still matches.

A candidate does NOT match if it is only on the same topic, concerns a different company, \
period, figure or event, reports an earlier stage that the claim updates (e.g. a proposal vs. \
its approval, an estimate vs. the final number), or contradicts the claim. Use the dates: a \
candidate about an earlier, separate occurrence of a similar event (a previous crisis, quarter, \
auction or offering) does NOT match, and relative times ("tomorrow", "last week") must line up.

Return the ids of all matching candidates (an empty list if none)."""

QUOTE_SYSTEM = """You check whether quotations from news coverage state the same fact as a claim.

Each candidate is a quotation with its attributed speaker and the date it was first seen. A \
candidate MATCHES if the speaker, in that quotation, asserts the claim's central fact or \
statement: same event, action, figure or position, with no conflicting key details. The claim \
may paraphrase or summarise the quotation, or cover only part of it. Speaker attributions are \
automatic and sometimes wrong or missing ("None"); judge mainly by content, but a clearly \
different speaker making a similar statement does NOT match.

A candidate does NOT match if it is only on the same topic, makes a different or opposite \
point, concerns a different event, period or figure, or is a generic remark the claim does not \
report.

Return the ids of all matching candidates (an empty list if none)."""

SCHEMA = json.dumps({
    "type": "object",
    "properties": {"matches": {"type": "array", "items": {"type": "integer"}}},
    "required": ["matches"],
    "additionalProperties": False,
})


def build_prompt(claim, candidates, headline=None, date=None):
    """Headlines supply the subject a bare sentence often omits ("the company"); dates let the
    model tell a repeat of a similar event from the same one."""
    lines = [f"CLAIM: {claim}"]
    if headline or date:
        lines.append(f"(from story{' headlined: ' + headline if headline else ''}{', ' + date[:16] if date else ''})")
    lines += ["", "CANDIDATES:"]
    for i, c in enumerate(candidates):
        meta = [c["date"][:16]] if c.get("date") else []
        if c.get("headline") and c["headline"] != c["text"]:
            meta.append(f"story: {c['headline']}")
        lines.append(f"[{i}] {c['text']}" + (f" ({'; '.join(meta)})" if meta else ""))
    return "\n".join(lines)


def claude_json(prompt, system, schema, model, effort, retries=3):
    """One headless Claude Code call with structured output; returns (parsed output, usage stats)."""
    cmd = [CLAUDE, "-p", prompt, "--model", model, "--effort", effort, "--tools", "",
           "--system-prompt", system, "--output-format", "json", "--json-schema", schema,
           "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config"]
    for attempt in range(retries):
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        try:
            out = json.loads(p.stdout)
        except json.JSONDecodeError:
            out = {"is_error": True, "result": (p.stdout + p.stderr)[-500:]}
        if not out.get("is_error") and out.get("structured_output") is not None:
            u = out["usage"]
            return out["structured_output"], {
                "latency_s": round(time.time() - t0, 2),
                "output_tokens": u["output_tokens"],
                "thinking_tokens": u.get("output_tokens_details", {}).get("thinking_tokens", 0),
                "input_tokens": u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"],
                "cost_usd": out["total_cost_usd"],
            }
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"claude call failed: {out.get('result') or out.get('api_error_status')}")


def call_claude(prompt, model, effort, system=SYSTEM):
    out, stats = claude_json(prompt, system, SCHEMA, model, effort)
    return {"matches": sorted(set(out["matches"])), **stats}


class LLMCache:
    """On-disk cache of LLM results so re-running a trace costs nothing."""

    def __init__(self, path="data/cache/llm.sqlite"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # several loop processes share this cache: WAL + a generous lock timeout
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, result TEXT)")
        self.lock = threading.Lock()

    def cached(self, parts, compute):
        """Return compute()'s result for these key parts, calling it only on a cache miss."""
        key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
        with self.lock:
            row = self.db.execute("SELECT result FROM calls WHERE key=?", (key,)).fetchone()
        if row:
            return dict(json.loads(row[0]), cached=True)
        r = compute()
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)", (key, json.dumps(r)))
            self.db.commit()
        return dict(r, cached=False)

    def verify(self, claim, candidates, headline, model, effort, date=None, system=SYSTEM, nonce=""):
        """nonce != "" forces a fresh draw (used to measure run-to-run noise)."""
        prompt = build_prompt(claim, candidates, headline, date)

        def compute():
            r = call_claude(prompt, model, effort, system)
            r["matches"] = [i for i in r["matches"] if 0 <= i < len(candidates)]
            return r
        return self.cached([model, effort, system, prompt] + ([nonce] if nonce else []), compute)


def run(queries_path, label, model, effort, workers=6):
    out_path = Path("bench/runs") / f"{label}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["qid"] for l in open(out_path)} if out_path.exists() else set()
    todo = [q for q in map(json.loads, open(queries_path)) if q["qid"] not in done]
    print(f"{label}: {len(done)} done, {len(todo)} to run ({model}, effort={effort})", flush=True)

    def one(q):
        n = len(q["candidates"])
        r = call_claude(build_prompt(q["text"], q["candidates"], q.get("headline"), q.get("date")), model, effort)
        r["matches"] = [i for i in r["matches"] if 0 <= i < n]
        return {"qid": q["qid"], "model": model, "effort": effort, **r}

    with ThreadPoolExecutor(workers) as ex, open(out_path, "a") as fh:
        for i, rec in enumerate(ex.map(one, todo), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"  {label}: {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], a[1], a[2], a[3], *(int(x) for x in a[4:]))
