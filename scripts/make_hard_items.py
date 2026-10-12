"""Add a harder query form to every benchmark item: `paraphrase_hard`.

The default paraphrase names the speaker and keeps the quote's content, which the loop has
largely learned to exploit (speaker-aware search). The hard form describes the speaker by role
instead of name and summarises the gist in fresh wording, which is closer to how facts travel
in secondary coverage. Ground truth is unchanged (same quote, same as_of).

Usage: python -I scripts/make_hard_items.py bench/qb_bench.jsonl [workers]
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import LLMCache, claude_json  # noqa: E402

HARD_SYSTEM = """Rewrite a quotation as one sentence a journalist might write when summarising \
what was said, for a story that mentions it in passing.

- Do not name the speaker. Refer to them by role or description (e.g. "Ireland's prime \
minister", "a Democratic presidential candidate", "the Bruins captain"); if you cannot tell \
their role, use a neutral description such as "an official" or "the player".
- Summarise the gist in your own words; keep key figures and named things the claim depends on.
- Do not copy more than two consecutive words from the quotation. No quotation marks."""
SCHEMA = json.dumps({"type": "object", "properties": {"sentence": {"type": "string"}},
                     "required": ["sentence"], "additionalProperties": False})


def hard(cache, speaker, quote):
    prompt = f"SPEAKER: {speaker}\nQUOTATION: {quote}"

    def compute():
        out, stats = claude_json(prompt, HARD_SYSTEM, SCHEMA, "claude-haiku-5-5", "low")
        return {"sentence": out["sentence"], **stats}
    return cache.cached(["claude-haiku-5-5", "low", HARD_SYSTEM, prompt], compute)["sentence"]


def main(path, workers=8):
    items = [json.loads(l) for l in open(path)]
    cache = LLMCache()
    with ThreadPoolExecutor(workers) as ex:
        sentences = list(ex.map(lambda it: hard(cache, it["speaker"], it["quotation"]), items))
    for it, s in zip(items, sentences):
        it["paraphrase_hard"] = s
    tmp = Path(path + ".tmp")
    tmp.write_text("".join(json.dumps(it) + "\n" for it in items))
    tmp.replace(path)
    named = sum(1 for it in items if it["speaker"].split()[-1].lower() in it["paraphrase_hard"].lower())
    print(f"{len(items)} items; speaker surname still present in {named}")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
