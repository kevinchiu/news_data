"""Build the Quotebank provenance benchmark, where ground truth is structural, not model labels.

Each item is a quotation carried by several outlets. The query is a later outlet's report of
it (verbatim, and a Haiku-written paraphrase in reported speech), dated at that later mention;
the answer is the quote's earliest mention (anywhere, and in an original outlet) before then.

Items are split train / val / test (default 150 / 75 / 150) by a hash of the quote id.

Usage: python -I scripts/make_qb_bench.py bench/qb_bench.jsonl [n_train n_val n_test] [min_domains]
"""
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import LLMCache, claude_json  # noqa: E402

PARA_SYSTEM = """Rewrite a quotation as one sentence of reported speech, as a journalist would \
write it in a news story. Name the speaker. Keep the factual content and key figures. Do not \
copy more than three consecutive words from the quotation and do not use quotation marks."""
PARA_SCHEMA = json.dumps({"type": "object", "properties": {"sentence": {"type": "string"}},
                          "required": ["sentence"], "additionalProperties": False})


def paraphrase(cache, speaker, quote):
    prompt = f"SPEAKER: {speaker}\nQUOTATION: {quote}"

    def compute():
        out, stats = claude_json(prompt, PARA_SYSTEM, PARA_SCHEMA, "claude-haiku-5-5", "low")
        return {"sentence": out["sentence"], **stats}
    return cache.cached(["claude-haiku-5-5", "low", PARA_SYSTEM, prompt], compute)["sentence"]


def main(dst, n_train=150, n_val=75, n_test=150, min_domains=5):
    n = n_train + n_val + n_test
    con = duckdb.connect("data/news.duckdb", read_only=True)
    items = con.execute("""
      WITH m AS (
        SELECT m.*, coalesce(d.kind, 'outlet') AS kind FROM quote_mentions m LEFT JOIN domains d USING (domain)),
      per_quote AS (
        SELECT quote_id, count(DISTINCT domain) AS n_domains, min(date) AS first_date FROM m GROUP BY 1)
      SELECT q.quote_id, q.quotation, q.speaker, p.n_domains
      FROM per_quote p JOIN quotes q USING (quote_id)
      WHERE p.n_domains >= ? AND q.speaker IS NOT NULL AND q.speaker_proba >= 0.7
        AND len(string_split(q.quotation, ' ')) BETWEEN 12 AND 60
        -- a later report at least 6 hours after the first, so there is something to trace back
        AND EXISTS (SELECT 1 FROM m WHERE m.quote_id = p.quote_id AND m.date >= p.first_date + INTERVAL 6 HOUR
                    AND m.has_time)
      ORDER BY hash(q.quote_id) LIMIT ?""", [min_domains, n]).fetchall()
    print(f"{len(items)} items")

    def build(row):
        i, (qid, quote, speaker, n_domains) = row
        cur = con.cursor()
        ids = [qid] + [r[0] for r in cur.execute(
            "SELECT b FROM quote_links WHERE a = ? UNION SELECT a FROM quote_links WHERE b = ?", [qid, qid]).fetchall()]
        ms = cur.execute("""
          SELECT m.date, m.domain, m.url, coalesce(d.kind, 'outlet'), m.has_time FROM quote_mentions m
          LEFT JOIN domains d USING (domain) WHERE m.quote_id IN (SELECT unnest(?)) ORDER BY m.date, m.url""",
                         [ids]).fetchall()
        # date-only stamps (00:00:00) would sort ahead of every timed article that day, so
        # "first" is ranked among mentions with a real time of day
        ms = [r for r in ms if r[4]]
        later = next((r for r in ms if r[0] >= ms[0][0] + dt.timedelta(hours=6)), None)
        if later is None:
            return None
        before = [r for r in ms if r[0] < later[0]]
        orig = [r for r in before if r[3] == "outlet"]
        split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        return {
            "quote_id": qid, "split": split, "variant_ids": ids, "speaker": speaker, "quotation": quote,
            "n_domains": n_domains, "as_of": later[0].isoformat(), "query_url": later[2],
            "verbatim": f'{speaker} said: "{quote}"', "paraphrase": paraphrase(cache, speaker, quote),
            "gt_first_any": {"date": before[0][0].isoformat(), "domain": before[0][1], "url": before[0][2],
                             "has_time": before[0][4]},
            "gt_first_original": ({"date": orig[0][0].isoformat(), "domain": orig[0][1], "url": orig[0][2]}
                                  if orig else None),
        }

    cache = LLMCache()
    with ThreadPoolExecutor(6) as ex:
        out = [it for it in ex.map(build, enumerate(items)) if it]
    with open(dst, "w") as fh:
        fh.writelines(json.dumps(it) + "\n" for it in out)
    print({s: sum(it["split"] == s for it in out) for s in ("train", "val", "test")}, f"-> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
