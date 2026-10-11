"""Split a news sentence into atomic, self-contained facts to trace one by one.

Usage (test): python -I scripts/split_claims.py <sid> [<sid> ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import LLMCache, claude_json  # noqa: E402

MODEL, EFFORT = "claude-haiku-5-5", "low"

SYSTEM = """You split a news sentence into atomic factual claims so each can be traced back \
to its first report.

Rules:
- One checkable fact per item: an event, action, decision, figure, or a statement made by a \
named party. Split "X happened after Y", "rose to A from B", "X, while Y" and similar \
combinations into separate facts. A fact must not embed another one: drop "following X", \
"after Y", "in response to Z" clauses from it and make X, Y, Z facts of their own.
- Make every fact stand alone. Replace "the company", "the central bank", "he", "they" with the \
named entity, using the headline to work out who or what is meant. Turn relative times ("yesterday", \
"tomorrow", "last week", "this year") into explicit dates using the story date. Keep the \
attribution ("..., X said") on each fact it applies to.
- kind = "new" for what this sentence reports as news, including the action it announces even \
if phrased in the past tense ("raised the prime to six pct on February 28"); "background" only \
for context: an earlier event cited as cause or history, or a prior figure given for comparison.
- Leave out opinions and characterisations with nothing checkable, and boilerplate.
- A sentence with a single fact yields one item, rewritten to stand alone. Don't restate the \
same fact twice in different words.
- Extract facts from the SENTENCE only. The headline and date are for resolving references; \
never turn a headline into a fact of its own, and never add facts."""

SCHEMA = json.dumps({
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "kind": {"type": "string", "enum": ["new", "background"]}},
        "required": ["text", "kind"], "additionalProperties": False}}},
    "required": ["facts"],
    "additionalProperties": False,
})


def build_prompt(sentence, headline=None, date=None):
    lines = [f"SENTENCE: {sentence}"]
    if headline:
        lines.append(f"HEADLINE: {headline}")
    if date:
        lines.append(f"STORY DATE: {date[:16]}")
    return "\n".join(lines)


def split(cache, sentence, headline=None, date=None, model=MODEL, effort=EFFORT):
    prompt = build_prompt(sentence, headline, date)

    def compute():
        out, stats = claude_json(prompt, SYSTEM, SCHEMA, model, effort)
        return {"facts": [f for f in out["facts"] if f["text"].strip()], **stats}
    return cache.cached([model, effort, SYSTEM, prompt], compute)


if __name__ == "__main__":

    rows = {}
    want = set(sys.argv[1:])
    for l in open("data/processed/sentences.jsonl"):
        r = json.loads(l)
        if r["sid"] in want:
            rows[r["sid"]] = r
    heads = {}
    for l in open("data/processed/reuters21578.jsonl"):
        a = json.loads(l)
        heads[a["id"]] = a["title"] or ""
    cache = LLMCache()
    for sid in sys.argv[1:]:
        r = rows[sid]
        res = split(cache, r["text"], heads[r["doc_id"]], r["date"])
        print(f"\n{sid} | {r['text']}")
        for f in res["facts"]:
            print(f"   {f['kind']:<10} {f['text']}")
