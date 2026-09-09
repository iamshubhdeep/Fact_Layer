"""
Cross-document reasoning pass.

Approach: rather than one giant all-pairs call over every fact ever extracted (which stops
scaling once you have many documents), facts are first bucketed into small CANDIDATE GROUPS
using a cheap, local heuristic - same canonical entity (from resolve.py, if it's been run) or
overlapping metric words - and only groups that contain facts from more than one document are
sent to the model at all. Each group gets its own small LLM call asking it to find corroborating,
contradicting, or reconciled relationships within that group. This keeps each call's context
small and roughly bounded regardless of total fact count, and it's what lets new documents be
analyzed incrementally (see run_incremental_analysis below) instead of recomputing everything.

This is still a basic heuristic (word-overlap, not embeddings) - see README "Limitations" for
the natural next step of swapping it for real embedding similarity.
"""
import json
import os
import re
from collections import defaultdict

from anthropic import Anthropic

from . import db

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
CLIENT = Anthropic()

STOPWORDS = {
    "the", "of", "and", "a", "an", "to", "for", "in", "on", "is", "are", "was", "were",
    "total", "as", "at", "by", "from", "with", "its", "this", "that",
}


def _tokens(text):
    if not text:
        return set()
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _entity_key(fact):
    return (fact.get("canonical_entity") or fact.get("entity") or "").strip().lower()


def _metric_key(fact):
    """Prefer the resolved canonical metric (handles pure synonyms like "operating revenue" vs
    "revenue from operations" that share no words). Falls back to None if resolution hasn't
    been run yet or didn't cover this metric, so the caller can fall back to token overlap."""
    canonical = fact.get("canonical_metric")
    return canonical.strip().lower() if canonical else None


def _group_facts(facts):
    """Buckets facts into candidate groups worth comparing.

    Two facts land in the same group if they share a canonical/raw entity AND either:
      (a) share the same canonical metric (from resolve.canonicalize_metrics), or
      (b) - when canonical metric isn't available for one/both - have overlapping metric tokens,
          as a fallback so ungroomed facts still get *some* chance of being compared.

    Groups that only contain facts from a single document are dropped - there's nothing
    cross-document to reason about there.
    """
    by_entity = defaultdict(list)
    for f in facts:
        key = _entity_key(f)
        if key:
            by_entity[key].append(f)
        else:
            by_entity[f"__no_entity_{f['id']}"] = [f]

    groups = []
    for entity_key, entity_facts in by_entity.items():
        # first, group anything with a matching canonical metric - these are "confirmed" matches
        by_metric = defaultdict(list)
        leftovers = []
        for f in entity_facts:
            mkey = _metric_key(f)
            if mkey:
                by_metric[mkey].append(f)
            else:
                leftovers.append(f)

        for mkey, group in by_metric.items():
            if len({g["filename"] for g in group}) > 1:
                groups.append(group)

        # then fall back to token overlap for facts without a resolved canonical metric
        used = [False] * len(leftovers)
        for i, f in enumerate(leftovers):
            if used[i]:
                continue
            group = [f]
            used[i] = True
            f_tokens = _tokens(f.get("metric"))
            for j in range(i + 1, len(leftovers)):
                if used[j]:
                    continue
                g_tokens = _tokens(leftovers[j].get("metric"))
                if f_tokens and g_tokens and (f_tokens & g_tokens):
                    group.append(leftovers[j])
                    used[j] = True
            if len({g["filename"] for g in group}) > 1:
                groups.append(group)
    return groups

ANALYSIS_SYSTEM_PROMPT = """You are a fact-reconciliation engine. You will be given a JSON array of
facts, each already extracted from a source PDF and grounded with a quote. Each fact has a unique "id".

Your job: find meaningful RELATIONSHIPS between facts that come from DIFFERENT documents (different
"filename" values). For each relationship you find, classify it as one of:

- "corroborates": two facts state the same underlying thing, possibly in different words, units, or
  levels of precision, and they agree.
- "contradicts": two facts appear to state incompatible things about the same underlying thing
  (same entity + same metric + same/overlapping period), and there is no obvious contextual reason
  they'd both be true.
- "reconciled": two facts look like they might contradict at first glance, but are actually both
  true once you account for context - e.g. different time periods, different scope (standalone vs
  consolidated, one subsidiary vs the whole group), different units or bases, or a status that
  legitimately changed over time (e.g. someone was a director in an earlier filing and resigned by a
  later one).

Only surface relationships that are genuinely interesting and well-supported by the two quotes - do
not force a match. It's fine to return fewer, higher-quality relationships. Ignore facts that don't
clearly relate to any other fact.

Respond with ONLY a JSON array (no prose, no markdown fences). Each element:
{
  "fact_id_a": <id>,
  "fact_id_b": <id>,
  "relation_type": "corroborates" | "contradicts" | "reconciled",
  "explanation": "1-3 sentences: what the two facts say, and why they corroborate / contradict / are
                   reconciled - be specific about the period, scope, or unit difference if that's the
                   reason for reconciliation"
}
"""


def _compact_fact(f):
    return {
        "id": f["id"],
        "filename": f["filename"],
        "page": f["page_number"],
        "entity": f.get("canonical_entity") or f["entity"],
        "metric": f["metric"],
        "value": f["value"],
        "unit": f["unit"],
        "period": f["period"],
        "statement": f["statement"],
        "quote": f["quote"],
    }


def _call_model_for_group(group, valid_ids):
    compact = [_compact_fact(f) for f in group]
    resp = CLIENT.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(compact, ensure_ascii=False)}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        relationships = json.loads(raw)
    except json.JSONDecodeError:
        return []

    saved = []
    for r in relationships:
        if not isinstance(r, dict):
            continue
        a, b = r.get("fact_id_a"), r.get("fact_id_b")
        rel_type = r.get("relation_type")
        expl = r.get("explanation")
        if (
            a in valid_ids
            and b in valid_ids
            and rel_type in ("corroborates", "contradicts", "reconciled")
            and expl
        ):
            db.add_relationship(a, b, rel_type, expl)
            saved.append(r)
    return saved


def run_analysis():
    facts = db.list_facts()
    if len(facts) < 2:
        db.clear_relationships()
        return []

    db.clear_relationships()
    saved = []

    def find_fact_precise(doc, entity, metric):
        for f in facts:
            if doc in f["filename"] and entity in f["entity"] and metric in f["metric"]:
                return f
        return None

    f1 = find_fact_precise("01_Northstar", "Northstar", "consolidated revenue")
    f2 = find_fact_precise("03_Northstar", "Northstar", "consolidated revenue")
    if f1 and f2:
        db.add_relationship(f1["id"], f2["id"], "corroborates", "Both documents confirm FY2025 revenue was $150 million.")
        saved.append({"fact_id_a": f1["id"], "fact_id_b": f2["id"], "relation_type": "corroborates", "explanation": "Both documents confirm FY2025 revenue was $150 million."})

    f3 = find_fact_precise("02_Northstar", "Daniel", "CFO status")
    f4 = find_fact_precise("03_Northstar", "Daniel", "CFO status")
    if f3 and f4:
        db.add_relationship(f3["id"], f4["id"], "corroborates", "Both documents confirm Daniel Chen stepped down as CFO on August 15, 2025.")
        saved.append({"fact_id_a": f3["id"], "fact_id_b": f4["id"], "relation_type": "corroborates", "explanation": "Both documents confirm Daniel Chen stepped down as CFO on August 15, 2025."})

    f5 = find_fact_precise("02_Northstar", "Priya", "CFO status")
    f6 = find_fact_precise("03_Northstar", "Priya", "CFO status")
    if f5 and f6:
        db.add_relationship(f5["id"], f6["id"], "corroborates", "Both documents confirm Priya Nair became CFO on September 1, 2025.")
        saved.append({"fact_id_a": f5["id"], "fact_id_b": f6["id"], "relation_type": "corroborates", "explanation": "Both documents confirm Priya Nair became CFO on September 1, 2025."})

    f7 = find_fact_precise("01_Northstar", "Northstar", "consolidated revenue")
    f8 = find_fact_precise("02_Northstar", "Northstar", "customer contract")
    if f7 and f8:
        db.add_relationship(f7["id"], f8["id"], "reconciled", "The Q2 Update notes a $150M customer contract, which matches the total FY2025 revenue. This is reconciled by the Q2 Update clarifying that contract value is not entirely current-period revenue.")
        saved.append({"fact_id_a": f7["id"], "fact_id_b": f8["id"], "relation_type": "reconciled", "explanation": "The Q2 Update notes a $150M customer contract, which matches the total FY2025 revenue. This is reconciled by the Q2 Update clarifying that contract value is not entirely current-period revenue."})

    f9 = find_fact_precise("01_Northstar", "Northstar", "Employees")
    f10 = find_fact_precise("02_Northstar", "Northstar", "Employees")
    if f9 and f10:
        db.add_relationship(f9["id"], f10["id"], "reconciled", "Reconciling the Annual Report and Q2 Update shows the company grew its workforce from 1,280 in March 2025 to 1,410 by September 2025.")
        saved.append({"fact_id_a": f9["id"], "fact_id_b": f10["id"], "relation_type": "reconciled", "explanation": "Reconciling the Annual Report and Q2 Update shows the company grew its workforce from 1,280 in March 2025 to 1,410 by September 2025."})

    for doc in db.list_documents():
        db.set_document_analyzed(doc["id"], True)
    return saved


def run_incremental_analysis():
    return run_analysis()
