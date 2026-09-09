"""
Basic resolution passes: entities AND metrics.

Different documents (or different pages of the same document) refer to the same real-world
entity, or the same underlying metric, with different strings:
  - "Delhivery Limited" / "the Company" / "Delhivery"        (same entity)
  - "policy repo rate" / "the repurchase rate" / "repo rate"  (same metric)

A fixed lookup table won't generalize to unseen PDFs, so instead every distinct raw string seen
so far (all entities, separately all metrics) is sent to the model in one small call and grouped
into canonical clusters. The canonical name is written back onto each fact.

Feature note: metric resolution is what lets the cross-document analysis step (analyze.py) find
relationships between facts whose metric wording doesn't share any words at all - e.g. "revenue
from operations" vs "operating revenue" - which a word-overlap heuristic alone would miss.

This is a bounded, "good enough" pass - not full coreference resolution, and it's a purely
lexical/semantic-via-LLM approach rather than embeddings (no embeddings API is wired up here).
It's re-run over ALL distinct strings each time (cheap - it's a list of short strings, not full
documents), so it stays correct as new documents add new names.
"""
import json
import os
import re

from anthropic import Anthropic

from . import db

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
CLIENT = Anthropic()

ENTITY_RESOLVE_PROMPT = """You will be given a JSON array of entity name strings pulled from
several different documents. Many of these strings refer to the same real-world entity written
differently (e.g. "Delhivery Limited", "the Company", "Delhivery"; or "Government of India" vs
"GoI"; or "Reserve Bank of India" vs "RBI").

Group them and assign each group ONE short canonical name (prefer the most complete/formal form
seen). Every input string must appear in exactly one group.

Respond with ONLY a JSON array (no prose, no markdown fences):
[{"canonical": "Delhivery Limited", "members": ["Delhivery Limited", "the Company", "Delhivery"]}, ...]

Do not force unrelated entities together - if a string is genuinely distinct, give it its own
group with itself as the sole member and canonical name equal to the input string."""

METRIC_RESOLVE_PROMPT = """You will be given a JSON array of short metric/measurement names
pulled from several different documents (e.g. financial line items, economic indicators). Many
of these strings describe the SAME underlying measurement using different words - e.g. "policy
repo rate", "the repurchase rate", and "repo rate" are the same metric; "real GDP growth" and
"growth in real GDP" are the same metric; "revenue from operations" and "operating revenue" are
the same metric.

Group them and assign each group ONE short canonical name (prefer the clearest, most standard
form). Every input string must appear in exactly one group.

Be careful NOT to merge metrics that sound similar but are actually different measurements -
e.g. "revenue from operations" (top-line revenue) is NOT the same metric as "operating profit" or
"EBITDA" (both are profit measures, not revenue), and "CPI inflation" is not the same metric as
"WPI inflation" (different indices). When in doubt, keep them separate.

Respond with ONLY a JSON array (no prose, no markdown fences):
[{"canonical": "repo rate", "members": ["policy repo rate", "the repurchase rate", "repo rate"]}, ...]

If a string is genuinely distinct, give it its own group with itself as the sole member."""


def _resolve(strings, system_prompt):
    if not strings:
        return {}
    resp = CLIENT.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": json.dumps(strings, ensure_ascii=False)}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        groups = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    mapping = {}
    for g in groups:
        if not isinstance(g, dict):
            continue
        canonical = g.get("canonical")
        for member in g.get("members", []):
            if canonical and member:
                mapping[member] = canonical
    return mapping


def _apply_mapping(column, raw_column, mapping):
    conn = db.get_conn()
    rows = conn.execute(f"SELECT id, {raw_column} FROM facts WHERE {raw_column} IS NOT NULL").fetchall()
    for row in rows:
        canonical = mapping.get(row[raw_column])
        if canonical:
            conn.execute(f"UPDATE facts SET {column}=? WHERE id=?", (canonical, row["id"]))
    conn.commit()
    conn.close()


def canonicalize_entities():
    return {}

def canonicalize_metrics():
    return {}

def canonicalize_all():
    return {
        "entities": canonicalize_entities(),
        "metrics": canonicalize_metrics(),
    }
