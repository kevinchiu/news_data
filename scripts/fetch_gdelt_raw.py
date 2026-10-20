"""Download GDELT 2.0 files listed in a manifest (size md5 url per line) and keep them as-is.

Used for the full Jan-Apr 2020 GKG and the Events (export) and Mentions tables, so later
parsing choices don't need another download. Each file is checked against the manifest's MD5
and written via a .tmp rename; re-running skips finished files, so the job can be resumed.

Usage: python -I scripts/fetch_gdelt_raw.py MANIFEST OUT_DIR [workers]
       python -I scripts/fetch_gdelt_raw.py --manifest KIND OUT_MANIFEST   # KIND: gkg, export, mentions
"""
import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MASTER = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
MONTHS = ("202001", "202002", "202003", "202004")


def make_manifest(kind, out):
    lines = urllib.request.urlopen(MASTER, timeout=300).read().decode().splitlines()
    keep = [l for l in lines if len(l.split()) == 3 and l.lower().endswith(f".{kind}.csv.zip")
            and l.split()[2].rsplit("/", 1)[1][:6] in MONTHS]
    Path(out).write_text("\n".join(keep) + "\n")
    print(f"{kind}: {len(keep)} files, {sum(int(l.split()[0]) for l in keep) / 1e9:.1f} GB -> {out}")


def fetch(size, md5, url, out_dir):
    path = out_dir / url.rsplit("/", 1)[1]
    if path.exists():
        return size, True
    url = url.replace("http://", "https://", 1)
    for attempt in range(5):
        try:
            data = urllib.request.urlopen(url, timeout=300).read()
            if hashlib.md5(data).hexdigest() != md5:
                raise IOError("md5 mismatch")
            break
        except Exception as e:  # noqa: BLE001 - retry any network/IO error
            if attempt == 4:
                print(f"FAILED {url}: {e}", flush=True)
                return 0, False
            time.sleep(10 * (attempt + 1))
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)
    return size, True


def main(manifest, out_dir, workers=4):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [line.split() for line in open(manifest) if line.strip()]
    n_bytes = n_done = n_fail = 0
    with ThreadPoolExecutor(workers) as ex:
        for size, ok in ex.map(lambda j: fetch(int(j[0]), j[1], j[2], out), jobs):
            n_bytes += size; n_done += ok; n_fail += not ok
            if (n_done + n_fail) % 500 == 0:
                print(f"{n_done + n_fail}/{len(jobs)} files, {n_bytes / 1e9:.1f} GB, {n_fail} failed", flush=True)
    print(f"GDELT_RAW_DONE {out}: {n_done}/{len(jobs)} files ok, {n_fail} failed, {n_bytes / 1e9:.1f} GB")


if __name__ == "__main__":
    if sys.argv[1] == "--manifest":
        make_manifest(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
