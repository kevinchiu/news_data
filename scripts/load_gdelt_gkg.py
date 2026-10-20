"""Parse the raw full GDELT 2.1 GKG files (all outlets) into Parquet and load them into DuckDB.

Keeps per article: id, time GDELT saw it, outlet, URL, page title, themes, locations, persons,
organizations, tone, all names, amounts and quotations. GCAM, counts and image fields are
dropped (they are most of the bytes). One Parquet part per 15-minute file; re-running skips
finished parts.

Table written to data/news.duckdb:
  gdelt_gkg(gkg_id, date, source, url, title, themes, locations, persons, organizations,
            tone, all_names, amounts, quotations)

Usage: python -I scripts/load_gdelt_gkg.py data/raw/gdelt/gkg data/processed/gdelt_gkg_full [workers]
"""
import datetime as dt
import io
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_gdelt_wire import parse_quotes, title  # noqa: E402

SCHEMA = pa.schema([
    ("gkg_id", pa.string()), ("date", pa.timestamp("s")), ("source", pa.string()), ("url", pa.string()),
    ("title", pa.string()), ("themes", pa.string()), ("locations", pa.string()), ("persons", pa.string()),
    ("organizations", pa.string()), ("tone", pa.float32()), ("all_names", pa.string()), ("amounts", pa.string()),
    ("quotations", pa.list_(pa.struct([("offset", pa.int32()), ("verb", pa.string()), ("quote", pa.string())]))),
])


def tone(field):
    try:
        return float(field.split(",", 1)[0])
    except ValueError:
        return None


def convert(args):
    src, out_dir = args
    part = Path(out_dir) / (src.name.split(".")[0] + ".parquet")
    if part.exists():
        return 0
    rows = []
    with zipfile.ZipFile(src) as z:
        for raw in io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8", errors="replace"):
            f = raw.rstrip("\n").split("\t")
            if len(f) < 27:
                continue
            rows.append({
                "gkg_id": f[0], "date": dt.datetime.strptime(f[1], "%Y%m%d%H%M%S"), "source": f[3], "url": f[4],
                "title": title(f[26]), "themes": f[7], "locations": f[10], "persons": f[12],
                "organizations": f[14], "tone": tone(f[15]), "all_names": f[23], "amounts": f[24],
                "quotations": parse_quotes(f[22]),
            })
    tmp = part.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, SCHEMA), tmp)
    tmp.rename(part)
    return len(rows)


def main(raw_dir, out_dir, workers=4):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zips = sorted(Path(raw_dir).glob("*.zip"))
    t0, n_rows = time.time(), 0
    with ProcessPoolExecutor(workers) as ex:
        for i, n in enumerate(ex.map(convert, [(z, out) for z in zips], chunksize=8), 1):
            n_rows += n
            if i % 500 == 0:
                print(f"{i}/{len(zips)} files, {n_rows:,} rows this run, {time.time() - t0:.0f} s", flush=True)
    con = duckdb.connect("data/news.duckdb")
    con.execute("SET threads = 3")
    con.execute(f"CREATE OR REPLACE TABLE gdelt_gkg AS SELECT * FROM '{out}/*.parquet'")
    n, lo, hi = con.execute("SELECT count(*), min(date), max(date) FROM gdelt_gkg").fetchone()
    print(f"GKG_DONE gdelt_gkg: {n:,} rows ({lo} .. {hi}), {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
