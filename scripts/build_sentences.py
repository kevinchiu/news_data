"""Split parsed Reuters stories into sentences (plus headlines) for retrieval.

Usage: python -I scripts/build_sentences.py data/processed/reuters21578.jsonl data/processed/sentences.jsonl
"""
import json
import re
import sys

ABBREV = re.compile(
    r"(?:\b[A-Z]\.|\b(?:Corp|Inc|Co|Ltd|Mr|Mrs|Ms|Dr|St|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec|No|vs|Gen|Sen|Rep|Gov|Bros|Ave)\.)$"
)
SPLIT = re.compile(r"(?<=[.!?])[\"']?\s+(?=[\"'(]?[A-Z0-9])")


def paragraphs(body):
    for p in re.split(r"\n\s{3,}", body):
        p = " ".join(p.split())
        if p:
            yield p


def sentences(par):
    out, buf = [], ""
    pos = 0
    for m in SPLIT.finditer(par):
        piece = par[pos:m.start() + 1 if par[m.start()] in ".!?" else m.start()]
        buf = f"{buf} {piece}".strip() if buf else piece.strip()
        pos = m.end()
        if ABBREV.search(buf):
            continue
        out.append(buf)
        buf = ""
    tail = par[pos:].strip()
    buf = f"{buf} {tail}".strip() if buf else tail
    if buf:
        out.append(buf)
    return out


def main(src, dst):
    seen_bodies = set()
    n_docs = n_sents = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            r = json.loads(line)
            if not r["date"]:
                continue
            key = (r["title"], r["body"])
            if key in seen_bodies:  # repeated transmissions: keep first copy only
                continue
            seen_bodies.add(key)
            n_docs += 1
            items = []
            if r["title"]:
                items.append(("headline", r["title"]))
            for par in paragraphs(r["body"] or ""):
                items.extend(("sentence", s) for s in sentences(par))
            for i, (kind, text) in enumerate(items):
                fout.write(json.dumps({
                    "sid": f"{r['id']}:{i}", "doc_id": r["id"], "date": r["date"],
                    "slug": r["slug"], "topics": r["topics"], "kind": kind, "text": text,
                }) + "\n")
                n_sents += 1
    print(f"{n_sents} sentences from {n_docs} docs -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
