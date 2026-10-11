"""Stream GDELT 2.1 GKG files and keep only rows from wire / press-release domains.

Each 15-minute file is downloaded into memory, checked against the manifest's MD5, filtered,
and written as one small Parquet part (also when empty, which marks it done). Re-running
skips finished parts, so the job can be resumed.

Usage: python -I scripts/fetch_gdelt_wire.py data/raw/gdelt/gkg_2020_jan_apr_manifest.txt data/processed/gdelt_gkg_wire [workers]
"""
import datetime as dt
import hashlib
import io
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

WIRE_DOMAINS = {
    "apnews.com", "reuters.com", "afp.com", "upi.com", "aninews.in", "pa.media",
    "prnewswire.com", "businesswire.com", "globenewswire.com", "xinhuanet.com", "tass.com",
    "efe.com", "dpa.com", "kyodonews.net", "aa.com.tr", "ians.in",
}
SCHEMA = pa.schema([
    ("gkg_id", pa.string()), ("date", pa.timestamp("s")), ("source", pa.string()), ("url", pa.string()),
    ("title", pa.string()), ("persons", pa.string()), ("organizations", pa.string()),
    ("quotations", pa.list_(pa.struct([("offset", pa.int32()), ("verb", pa.string()), ("quote", pa.string())]))),
])


def is_wire(source):
    return any(source == d or source.endswith("." + d) for d in WIRE_DOMAINS)


def parse_quotes(field):
    out = []
    for block in field.split("#"):
        parts = block.split("|", 3)
        if len(parts) == 4 and parts[3].strip():
            out.append({"offset": int(parts[0]) if parts[0].isdigit() else None, "verb": parts[2], "quote": parts[3]})
    return out


def title(extras):
    i = extras.find("<PAGE_TITLE>")
    return extras[i + 12:extras.find("</PAGE_TITLE>", i)] if i >= 0 else None


def fetch(size, md5, url, out_dir):
    part = out_dir / (url.rsplit("/", 1)[1].split(".")[0] + ".parquet")
    if part.exists():
        return 0, True
    url = url.replace("http://", "https://", 1)
    for attempt in range(5):
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            if hashlib.md5(data).hexdigest() != md5:
                raise IOError("md5 mismatch")
            break
        except Exception as e:  # noqa: BLE001 - retry any network/IO error
            if attempt == 4:
                print(f"FAILED {url}: {e}", flush=True)
                return 0, False
            time.sleep(10 * (attempt + 1))
    rows = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for raw in z.open(z.namelist()[0]):
            f = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
            if len(f) < 27 or not is_wire(f[3]):
                continue
            rows.append({
                "gkg_id": f[0], "date": dt.datetime.strptime(f[1], "%Y%m%d%H%M%S"),
                "source": f[3], "url": f[4], "title": title(f[26]), "persons": f[12], "organizations": f[14],
                "quotations": parse_quotes(f[22]),
            })
    tmp = part.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, SCHEMA), tmp)
    tmp.rename(part)
    return len(rows), True


def main(manifest, out_dir, workers=8):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [line.split() for line in open(manifest) if line.strip()]
    n_rows = n_done = n_fail = 0
    with ThreadPoolExecutor(workers) as ex:
        for rows, ok in ex.map(lambda j: fetch(int(j[0]), j[1], j[2], out), jobs):
            n_rows += rows; n_done += ok; n_fail += not ok
            if (n_done + n_fail) % 500 == 0:
                print(f"{n_done + n_fail}/{len(jobs)} files, {n_rows:,} wire rows, {n_fail} failed", flush=True)
    print(f"GDELT_DONE: {n_done}/{len(jobs)} files ok, {n_fail} failed, {n_rows:,} wire rows this run")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
