"""Event-tracer configuration (scripts/trace_event.py): load, validate and apply changes.

Same shape as qb_config: the improvement loop may only change the paths in ALLOWED. Deliberately
excluded: the search window (fixed in trace_event.WINDOW_DAYS; it would leak how the benchmark is
built) and the verifier model.
"""
import json
from pathlib import Path

from qb_config import CONFIG_DIR, apply as _apply, diff, get  # noqa: F401

ALLOWED = {
    "retrieval.k": lambda v: isinstance(v, int) and 5 <= v <= 60,
    "retrieval.pool": lambda v: isinstance(v, int) and 2 <= v <= 8,
    "retrieval.query_template": lambda v: v in ("raw", "content", "headline", "raw+headline"),
    "verify.effort": lambda v: v in ("low", "medium", "high"),
    "verify.batch": lambda v: isinstance(v, int) and 5 <= v <= 30,
    "verify.prompt": lambda v: isinstance(v, str) and 200 <= len(v) <= 4000,
    "verify.show_date": lambda v: isinstance(v, bool),
    "verify.show_claim_date": lambda v: isinstance(v, bool),
    "verify.show_source": lambda v: isinstance(v, bool),
    "verify.show_copies": lambda v: isinstance(v, bool),
    "verify.order": lambda v: v in ("score", "time"),
}


def load(path=CONFIG_DIR / "event_base.json"):
    cfg = json.loads(Path(path).read_text())
    p = cfg["verify"]["prompt"]
    if p.startswith("@"):
        cfg["verify"]["prompt"] = (CONFIG_DIR / p[1:]).read_text().strip()
    cfg["verify"].setdefault("show_claim_date", False)  # older configs: the claim's own time is not shown
    rp = cfg.get("recheck", {}).get("prompt", "")
    if rp.startswith("@"):
        cfg["recheck"]["prompt"] = (CONFIG_DIR / rp[1:]).read_text().strip()
    return cfg


def apply(cfg, changes):
    """Return a new config with changes [{"path", "value"}] applied; raises ValueError if any is not allowed."""
    return _apply(cfg, changes, allowed=ALLOWED)
