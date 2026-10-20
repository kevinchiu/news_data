"""Offline Firstline page: the app's look (scripts/app_static/index.html) with the demo's precomputed traces baked in.

  python -I scripts/build_firstline_offline.py [--source PAGE]   # -> docs/demo/demo.html

PAGE (default docs/demo/index.html) is a scripts/build_demo.py page whose embedded traces are used; build
one with the post-loop fixed config, e.g. build_demo.py --event-best experiments_event/configs/fix_retrieval.json
--out <tmp dir>, so the examples match the "with fixes" score shown.

One self-contained file to open in a browser; no server, no model calls.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "docs/demo/index.html"))
    page = Path(ap.parse_args().source).read_text()
    data = json.loads(re.search(r"^var DEMO_DATA = (.*);$", page, re.M).group(1))
    keep = {t: {k: data[t][k] for k in ("best_label", "base_label", "results", "examples")} for t in ("quotes", "events")}
    # a post-loop fix (ledger "fix" entry) supersedes the loop's final test score for events
    fixes = [json.loads(l) for l in open(ROOT / "experiments_event/ledger.jsonl") if '"type": "fix"' in l]
    if fixes:
        t = fixes[-1]["results"]["test"]
        keep["events"]["best_label"] = "with fixes"
        keep["events"]["results"]["test"] = {"base": {"score": t["v0"], "n": t["n"]}, "best": {"score": t["fix"], "n": t["n"]}}
    payload = json.dumps(keep, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    out = (ROOT / "scripts/firstline_offline_template.html").read_text() \
        .replace("__DATA__", payload).replace("__BUILT__", data["built"][:10])
    dest = ROOT / "docs/demo/demo.html"
    dest.write_text(out)
    print(f"{dest}: {len(out) // 1024} KB")


if __name__ == "__main__":
    main()
