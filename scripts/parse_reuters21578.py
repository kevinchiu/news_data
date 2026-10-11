"""Parse Reuters-21578 SGML files into a JSONL file of articles.

Usage: python -I scripts/parse_reuters21578.py <raw_dir> <out.jsonl>
"""
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

DOC_RE = re.compile(r"<REUTERS(.*?)>(.*?)</REUTERS>", re.S)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
LIST_FIELDS = ["TOPICS", "PLACES", "PEOPLE", "ORGS", "EXCHANGES", "COMPANIES"]
SLUG_RE = re.compile(r"\b(BC-[^\s]+)")


def tag(name, s):
    m = re.search(rf"<{name}(?:\s[^>]*)?>(.*?)</{name}>", s, re.S)
    return m.group(1) if m else None


def clean(s):
    if s is None:
        return None
    s = html.unescape(s)
    s = re.sub(r"[\x00-\x08\x0b-\x1f]", "", s)  # control chars
    s = re.sub(r"\s*Reuter\s*$", "", s.strip())   # trailing sign-off
    return s.strip()


def parse_date(s):
    s = re.match(r"[\d\-A-Za-z]+ +[\d:.]+", html.unescape(s).strip()).group(0)
    for fmt in ("%d-%b-%Y %H:%M:%S.%f", "%d-%b-%Y %H:%M:%S"):
        try:
            return datetime.strptime(" ".join(s.split()), fmt).isoformat()
        except ValueError:
            pass
    return None


def parse_file(path):
    raw = path.read_bytes().decode("latin-1")
    for attrs, body in DOC_RE.findall(raw):
        a = dict(ATTR_RE.findall(attrs))
        text = tag("TEXT", body) or ""
        unknown = tag("UNKNOWN", body) or ""
        slug = SLUG_RE.search(unknown)
        date_raw = (tag("DATE", body) or "").strip()
        rec = {
            "id": int(a["NEWID"]),
            "old_id": int(a["OLDID"]),
            "date": parse_date(date_raw),
            "date_raw": date_raw,
            "slug": slug.group(1) if slug else None,
            "title": clean(tag("TITLE", text)),
            "dateline": clean(tag("DATELINE", text)),
            "body": clean(tag("BODY", text)),
            "lewis_split": a.get("LEWISSPLIT"),
        }
        if rec["body"] is None and rec["title"] is None:
            # TYPE="BRIEF"/"UNPROC" docs: keep raw text
            rec["body"] = clean(re.sub(r"<[^>]+>", " ", text))
        for f in LIST_FIELDS:
            rec[f.lower()] = re.findall(r"<D>(.*?)</D>", tag(f, body) or "")
        yield rec


def main(raw_dir, out):
    files = sorted(Path(raw_dir).glob("reut2-*.sgm"))
    recs = [r for f in files for r in parse_file(f)]
    recs.sort(key=lambda r: (r["date"] or "", r["id"]))
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(recs)} docs from {len(files)} files -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
