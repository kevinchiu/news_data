"""Parallel version of parse_quotebank.py: bzip2 decompresses in one process while worker
processes parse JSON and each write their own Parquet parts (same schema, one row per mention).

Usage: python -I scripts/parse_quotebank_parallel.py data/raw/quotebank/quotebank-2020.json.bz2 data/processed/quotebank-2020-mentions [workers]
"""
import datetime as dt
import json
import subprocess
import sys
import threading
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_quotebank import SCHEMA, domain  # noqa: E402

CHUNK = 20_000  # articles per task


def convert(args):
    idx, lines, out_dir = args
    rows = []
    for line in lines:
        a = json.loads(line)
        dom = domain(a["url"])
        date = dt.datetime.strptime(a["date"], "%Y-%m-%d %H:%M:%S")
        for q in a.get("quotations") or []:
            probas = q.get("localProbas") or []
            top = probas[0] if probas else ["None", "0"]
            rows.append({
                "quote_id": q["quoteID"], "article_id": a["articleID"], "date": date, "domain": dom,
                "url": a["url"], "title": a.get("title"),
                "speaker": None if q["localTopSpeaker"] == "None" else q["localTopSpeaker"],
                "speaker_proba": float(top[1]), "quotation": q["quotation"],
                "left_context": q.get("leftContext"), "right_context": q.get("rightContext"),
                "quote_occurrences": q.get("numOccurrences"), "article_length": a.get("articleLength"),
                "phase": a.get("phase"),
            })
    pq.write_table(pa.Table.from_pylist(rows, SCHEMA), f"{out_dir}/part-{idx:05d}.parquet", compression="zstd")
    return len(rows)


def chunks(src, out_dir, slots):
    """Yield article chunks; `slots` bounds how many are in flight (Pool would otherwise read ahead
    the whole decompressed file into memory)."""
    p = subprocess.Popen(["bzip2", "-dc", src], stdout=subprocess.PIPE, bufsize=1 << 24)
    buf, idx = [], 0
    for line in p.stdout:
        buf.append(line)
        if len(buf) >= CHUNK:
            slots.acquire()
            yield idx, buf, out_dir
            buf, idx = [], idx + 1
    if buf:
        slots.acquire()
        yield idx, buf, out_dir
    if p.wait() != 0:
        raise RuntimeError("bzip2 failed")


def main(src, out_dir, workers=9):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n, slots = 0, threading.BoundedSemaphore(workers * 2)
    with Pool(workers) as pool:
        for i, k in enumerate(pool.imap_unordered(convert, chunks(src, out_dir, slots)), 1):
            slots.release()
            n += k
            if i % 50 == 0:
                print(f"{i * CHUNK:,} articles, {n:,} mentions", flush=True)
    print(f"done: {n:,} mentions -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
