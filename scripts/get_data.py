"""Download the public datasets and rebuild data/ (nothing under data/ is committed except
data/domain_labels.csv).

  python -I scripts/get_data.py                 # Quotebank 2020 + Reuters-21578, then build data/news.duckdb
  python -I scripts/get_data.py --download-only # just fetch the raw files
  python -I scripts/get_data.py --gdelt         # also fetch GDELT GKG wire rows for Jan-Apr 2020 (~82 GB streamed)

Sources (all public):
  Quotebank (Vaucher et al., WSDM 2021), CC BY 4.0, Zenodo record 4277311:
    quotebank-2020.json.bz2 (article-centric, 5.3 GB) and quotes-2020.json.bz2 (quote-centric, 0.8 GB)
  Reuters-21578 (dev set only), UCI ML repository dataset 137
  GDELT 2.0 Global Knowledge Graph, data.gdeltproject.org (optional)
Downloads resume after interruptions and are checked against the published MD5s.
Building needs ~25 GB free disk and takes roughly 20-30 minutes; it makes no LLM calls.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
PY = [sys.executable, "-I"]
ZENODO = "https://zenodo.org/api/records/4277311"
QUOTEBANK_FILES = ["quotes-2020.json.bz2", "quotebank-2020.json.bz2"]
REUTERS_ZIP = "https://archive.ics.uci.edu/static/public/137/reuters+21578+text+categorization+collection.zip"
GDELT_MASTER = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url, dest, expected_md5=None, attempts=40):
    """curl with resume, retried until the checksum matches (Zenodo drops long transfers)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected_md5 and dest.exists() and md5(dest) == expected_md5:
        print(f"  {dest.name}: already downloaded")
        return
    for i in range(attempts):
        subprocess.run(["curl", "-sS", "-L", "-C", "-", "--retry", "5", "-o", str(dest), url])
        if not expected_md5 or (dest.exists() and md5(dest) == expected_md5):
            print(f"  {dest.name}: ok")
            return
        print(f"  {dest.name}: incomplete ({dest.stat().st_size if dest.exists() else 0:,} bytes), resuming…")
        time.sleep(10)
    sys.exit(f"could not download {url}")


def get_quotebank():
    print("Quotebank 2020 (Zenodo 4277311)")
    files = {f["key"]: f for f in json.load(urllib.request.urlopen(ZENODO))["files"]}
    manifest = []
    for name in QUOTEBANK_FILES:
        f = files[name]
        download(f["links"]["self"], RAW / "quotebank" / name, f["checksum"].split(":", 1)[1])
        manifest.append(f"{name} {f['checksum']} {f['links']['self']}")
    (RAW / "quotebank" / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")


def get_reuters():
    print("Reuters-21578 (UCI)")
    out = RAW / "reuters21578"
    if (out / "reut2-021.sgm").exists():
        print("  already extracted")
        return
    zpath = out / "r.zip"
    download(REUTERS_ZIP, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    subprocess.run(["tar", "-xzf", str(out / "reuters21578.tar.gz"), "-C", str(out)], check=True)
    print("  extracted")


def get_gdelt():
    print("GDELT GKG wire rows, Jan-Apr 2020")
    manifest = RAW / "gdelt" / "gkg_2020_jan_apr_manifest.txt"
    if not manifest.exists():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        lines = urllib.request.urlopen(GDELT_MASTER).read().decode().splitlines()
        keep = [l for l in lines if "/gdeltv2/2020" in l and l.split()[-1].split("/")[-1][:6] in
                ("202001", "202002", "202003", "202004") and l.endswith(".gkg.csv.zip")]
        manifest.write_text("\n".join(keep) + "\n")
    run(PY + ["scripts/fetch_gdelt_wire.py", str(manifest), str(PROC / "gdelt_gkg_wire"), "4"])


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def build(with_gdelt):
    print("Building data/processed and data/news.duckdb")
    PROC.mkdir(parents=True, exist_ok=True)
    qb = RAW / "quotebank"
    if not (PROC / "quotes-2020-quotes.parquet").exists():
        run(PY + ["scripts/parse_quotebank_quotes.py", str(qb / "quotes-2020.json.bz2"), str(PROC / "quotes-2020")])
    if not (PROC / "quotebank-2020-mentions").exists():
        run(PY + ["scripts/parse_quotebank_parallel.py", str(qb / "quotebank-2020.json.bz2"),
                  str(PROC / "quotebank-2020-mentions")])
    if not (PROC / "reuters21578.jsonl").exists():
        run(PY + ["scripts/parse_reuters21578.py", str(RAW / "reuters21578"), str(PROC / "reuters21578.jsonl")])
    if not (PROC / "sentences.jsonl").exists():
        run(PY + ["scripts/build_sentences.py", str(PROC / "reuters21578.jsonl"), str(PROC / "sentences.jsonl")])
    run(PY + ["scripts/build_db.py"])
    run(PY + ["scripts/quote_variants.py"])
    if with_gdelt:
        run(PY + ["scripts/match_wire_quotes.py", str(PROC / "gdelt_gkg_wire")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--gdelt", action="store_true", help="also fetch and match GDELT wire rows (optional)")
    a = ap.parse_args()
    get_quotebank()
    get_reuters()
    if a.gdelt:
        get_gdelt()
    if not a.download_only:
        build(a.gdelt)
    print("done")


if __name__ == "__main__":
    main()
