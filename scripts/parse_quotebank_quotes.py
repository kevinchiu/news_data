"""Flatten Quotebank's quotation-centric file into two Parquet tables:
quotes (one row per unique quote) and quote_urls (one row per quote x article URL).

Usage: python -I scripts/parse_quotebank_quotes.py data/raw/quotebank/quotes-2020.json.bz2 data/processed/quotes-2020
"""
import bz2
import datetime as dt
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from parse_quotebank import domain  # noqa: E402

QUOTES = pa.schema([
    ("quote_id", pa.string()), ("quotation", pa.string()), ("speaker", pa.string()),
    ("speaker_proba", pa.float32()), ("qids", pa.list_(pa.string())), ("first_date", pa.timestamp("s")),
    ("num_occurrences", pa.int32()), ("num_urls", pa.int32()), ("phase", pa.string()),
])
URLS = pa.schema([("quote_id", pa.string()), ("url", pa.string()), ("domain", pa.string())])
BATCH = 500_000


def main(src, prefix):
    wq = pq.ParquetWriter(f"{prefix}-quotes.parquet", QUOTES, compression="zstd")
    wu = pq.ParquetWriter(f"{prefix}-urls.parquet", URLS, compression="zstd")
    qbuf, ubuf, n = [], [], 0
    with bz2.open(src, "rt") as f:
        for line in f:
            q = json.loads(line)
            probas = q.get("probas") or [["None", "0"]]
            qbuf.append({
                "quote_id": q["quoteID"], "quotation": q["quotation"],
                "speaker": None if q["speaker"] == "None" else q["speaker"],
                "speaker_proba": float(probas[0][1]), "qids": q.get("qids") or [],
                "first_date": dt.datetime.strptime(q["date"], "%Y-%m-%d %H:%M:%S"),
                "num_occurrences": q["numOccurrences"], "num_urls": len(q["urls"]), "phase": q.get("phase"),
            })
            ubuf.extend({"quote_id": q["quoteID"], "url": u, "domain": domain(u)} for u in q["urls"])
            if len(qbuf) >= BATCH:
                wq.write_table(pa.Table.from_pylist(qbuf, QUOTES)); wu.write_table(pa.Table.from_pylist(ubuf, URLS))
                n += len(qbuf); qbuf, ubuf = [], []
                print(f"{n:,} quotes", flush=True)
    wq.write_table(pa.Table.from_pylist(qbuf, QUOTES)); wu.write_table(pa.Table.from_pylist(ubuf, URLS))
    wq.close(); wu.close()
    print(f"done: {n + len(qbuf):,} quotes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
