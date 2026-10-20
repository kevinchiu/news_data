"""Tracer configuration: load, validate and apply changes by dotted path.

The improvement loop may only change the paths in ALLOWED, within their limits, so every
proposal is a small, reviewable diff of this structure.
"""
import copy
import fnmatch
import json
import re
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
KINDS = ["outlet", "wire", "press_release", "aggregator", "feed"]

# What the improvement loop may change. Deliberately excluded (see the plan review):
# retrieval.min_urls / window_days leak how the benchmark is built (every item is a
# multi-outlet quote from Jan-Apr 2020); timeline.variants is fixed by the ground-truth
# definition; domain labels feed the ground truth, so letting the loop edit them is circular;
# verify.show_num_urls hints at popularity, another benchmark leak.
ALLOWED = {
    "retrieval.k": lambda v: isinstance(v, int) and 5 <= v <= 60,
    "retrieval.index": lambda v: v in ("quotation", "quotation_speaker"),
    "retrieval.query_template": lambda v: v in ("raw", "content", "content_plus_speaker"),
    "retrieval.speaker_filter": lambda v: v in ("off", "surname"),
    "verify.effort": lambda v: v in ("low", "medium", "high"),
    "verify.batch": lambda v: isinstance(v, int) and 5 <= v <= 30,
    "verify.prompt": lambda v: isinstance(v, str) and 200 <= len(v) <= 4000,
    "verify.show_date": lambda v: isinstance(v, bool),
    "verify.speaker_hint": lambda v: isinstance(v, bool),
}


def load(path=CONFIG_DIR / "base.json"):
    cfg = json.loads(Path(path).read_text())
    p = cfg["verify"]["prompt"]
    if p.startswith("@"):
        cfg["verify"]["prompt"] = (CONFIG_DIR / p[1:]).read_text().strip()
    return cfg


def get(cfg, path):
    for part in path.split("."):
        cfg = cfg[part]
    return cfg


def apply(cfg, changes, allowed=None):
    """Return a new config with changes [{"path", "value"}] applied; raises ValueError if any is not allowed."""
    allowed = allowed or ALLOWED
    new = copy.deepcopy(cfg)
    for ch in changes:
        path, value = ch["path"], ch["value"]
        if path not in allowed or not allowed[path](value):
            raise ValueError(f"change not allowed: {path} = {json.dumps(value)[:200]}")
        node = new
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return new


def diff(old, new, prefix=""):
    out = []
    for key in sorted(set(old) | set(new)):
        a, b, p = old.get(key), new.get(key), f"{prefix}{key}"
        if isinstance(a, dict) and isinstance(b, dict):
            out += diff(a, b, p + ".")
        elif a != b:
            out.append({"path": p, "old": a, "new": b})
    return out


def domain_kind(cfg, domain, kind, grp):
    """Apply the loop's domain_overrides (glob patterns) on top of the curated labels."""
    for o in cfg.get("domain_overrides", []):
        if fnmatch.fnmatch(domain, o["pattern"]):
            return o["kind"], o.get("grp") or grp
    return kind or "outlet", grp


REPORTING = re.compile(
    r"^(?P<speaker>.{2,80}?)\s+(?:has |had |have )?(?:said|says|told [\w ]{1,30}?|stated|added|noted|argued|"
    r"warned|claimed|insisted|explained|stressed|declared|announced|admitted|acknowledged|suggested|"
    r"maintained|contended|asserted|remarked|observed|predicted|urged|vowed|pledged)\b,?\s*(?:that\s+)?",
    re.IGNORECASE)


def split_statement(statement):
    """Split reported speech into (speaker phrase, content): 'Jay Sekulow said that X' -> ('Jay Sekulow', 'X')."""
    m = REPORTING.match(statement)
    if not m or len(statement) - m.end() < 15:
        return None, statement
    return m.group("speaker").strip(" ,"), statement[m.end():]


def surname(speaker_phrase):
    """Last capitalised word of the speaker phrase, e.g. 'U.S. Senator Elizabeth Warren' -> 'Warren'."""
    if not speaker_phrase:
        return None
    caps = [w.strip(".,'\u2019s") for w in speaker_phrase.split() if w[:1].isupper()]
    caps = [w for w in caps if len(w) >= 3]
    return caps[-1] if caps else None
