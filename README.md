# Fact Knowledge Layer (basic prototype)

A minimal, working version of the assignment: upload PDFs, extract grounded facts, and see how
facts across documents corroborate, contradict, or get reconciled by context.

This is intentionally the **basic version** - get the full pipeline working end to end first,
then improve pieces of it. See "Limitations and Next Steps" for what's missing.

## What it does

1. **Upload** a PDF through a simple web UI (or `curl`/API).
2. **Extract**: each PDF is read page by page (`pdfplumber`), and pages are sent in small
   chunks to Claude with a prompt that asks it to pull out concrete, checkable facts - not a
   fixed schema, the model decides what counts as a fact on the page. Every fact must come with
   a short verbatim quote from the page as evidence, so nothing is stored ungrounded.
3. **Store**: facts land in a plain SQLite database (`data/factlayer.db`) with entity, metric,
   value, unit, period, a plain-English statement, the evidence quote, page number, source
   filename, and the model's own confidence.
4. **Analyze**: a second pass sends all stored facts to Claude and asks it to find fact pairs
   from *different* documents that corroborate, contradict, or are reconciled once you account
   for period/scope/unit differences. Results are stored as `relationships` and shown in the UI
   next to both source quotes.

No hard-coded filenames, fields, or document-specific rules - the same pipeline runs on the
Delhivery documents, the India macroeconomy documents, or any other PDF you upload.

## Setup and Run Instructions

```bash
cd factlayer
python3 -m venv venv && source venv/bin/activate      
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...                    # required

uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` in a browser:

1. Upload a PDF (repeat for a second document from the same starter dataset folder to get
   cross-document relationships).
2. Watch the "Extracted facts" table fill in with entity/metric/value/period/quote per fact.
3. Click **"Run cross-document analysis"** once you have facts from 2+ documents.
4. Scroll to "Relationships" to see corroborated / contradicted / reconciled fact pairs, each
   shown with both source quotes and the model's explanation.

There's also a plain JSON API if you'd rather script it:

- `POST /api/upload` (multipart form, field `file`) - upload + extract one PDF
- `GET /api/facts` - all facts (optionally `?document_id=`)
- `POST /api/analyze` - re-run the cross-document reasoning pass over everything stored
- `GET /api/relationships` - all relationships with both facts inlined
- `POST /api/reset` - wipe everything (useful between demo runs)

## Video Demo

*(add link here - a ~3 min screen recording showing: uploading two Delhivery documents,
the facts table populating, running analysis, and pointing at one corroborated pair, one
contradiction, one reconciled pair, and the extraction gap described below)*

## Approach

**Why per-page chunking instead of one call per document:** sending 3 pages at a time keeps
each model call's context small and roughly constant-cost regardless of document length, which
is what lets this scale to "large PDFs" without changing the approach - you just get more
chunks, not slower/bigger calls. It also makes it easy to pin each fact back to the exact page
it came from, by matching the model's verbatim quote against the page text.

**Why an LLM decides what a "fact" is, rather than a fixed schema:** the assignment explicitly
says documents should guide the schema, and that the system will be tested on PDFs we haven't
seen. A regex/rule-based extractor tuned to Delhivery's revenue table would not generalize to an
IMF Article IV report. Asking the model for `{entity, metric, value, unit, period, statement,
quote, confidence}` is loose enough to cover very different document types while still being
structured enough to compare and store.

**Why grounding is a hard requirement, not a nice-to-have:** every fact must carry a verbatim
quote copied from the source page, and the pipeline discards any candidate fact without one.
This is what "linked to evidence" means in practice here, and it's also a cheap sanity check -
if the model can't quote it, it probably shouldn't be a stored fact.

**Why cross-document reasoning is a single LLM pass over compact facts, not a graph DB:** the
assignment specifically calls out that a graph database or visualization is not itself the
solution - the interesting part is the reasoning. So the "knowledge layer" here is really the
prompt in `analyze.py` plus the facts it's given, not a graph engine. The relationships table
is just where the model's conclusions get stored so the UI can show them.

**AI tools used:** Claude (via the Anthropic API) does both extraction and cross-document
reasoning - it's the core of the system, not just a coding assistant. Claude (this chat
interface) was also used to scaffold the FastAPI app, SQLite schema, and frontend.

## What's improved beyond the bare-minimum version

A first pass at each of the brownie-point directions, kept intentionally modest rather than
fully built out:

- **Vision fallback for extraction failures** (`extract.py`): any page whose extracted text is
  under ~40 characters (slide decks, chart-heavy pages, scans) is retried by rendering that page
  to a PNG and sending it to the model as an image instead of text. Facts extracted this way are
  tagged `extraction_method = "vision"` (or `"vision-error"` if even that fails) so you can see
  in the UI exactly which facts came from the fallback path and treat them with a bit more
  skepticism.
- **Basic entity resolution** (`resolve.py`): after every upload, all distinct raw `entity`
  strings seen so far (e.g. "Delhivery Limited", "the Company", "Delhivery") are sent to the
  model in one small call and grouped under a canonical name, written back onto each fact as
  `canonical_entity`. The analysis pass uses the canonical name when grouping facts, so
  differently-worded mentions of the same entity are much more likely to actually get compared.
- **Candidate pruning instead of all-pairs analysis** (`analyze.py`): facts are first bucketed
  into small candidate groups - same canonical entity + same canonical metric - and only groups
  that span more than one document get sent to the model, each as its own small call. This keeps
  each LLM call's context bounded regardless of total fact count, instead of one call that grows
  with every document you add.
- **Metric resolution** (`resolve.py`): the same idea as entity resolution, applied to metric
  names. Distinct raw `metric` strings are grouped into a canonical name (e.g. "policy repo
  rate" / "the repurchase rate" / "repo rate" → `"repo rate"`) and written back as
  `canonical_metric`. This is what lets the grouping step above catch facts that describe the
  same thing with completely different wording and zero shared words - e.g. "revenue from
  operations" vs "operating revenue" - which a word-overlap heuristic alone would never group.
  The prompt is explicitly told not to merge genuinely different metrics that merely sound
  similar (e.g. revenue vs. profit measures, CPI vs. WPI). Facts without a resolved canonical
  metric yet still fall back to token overlap, so nothing is silently excluded.
- **Incremental analysis**: documents are marked `analyzed` once they've been through a pass.
  `POST /api/analyze` (default `mode=incremental`) only re-examines candidate groups that touch
  at least one not-yet-analyzed document, and *adds* new relationships without touching
  previously-found ones. A `mode=full` option is still there for a clean rebuild.
- **A loose escape hatch for schema evolution**: `facts.extra` is a free-form JSON column the
  model can populate for anything that doesn't fit the fixed columns, without a migration.

None of these are "solved" - see below for what's still missing in each.

## Limitations and Next Steps

- **Vision fallback is per-page, not layout-aware**, and roughly doubles cost for image-heavy
  documents. It also doesn't handle multi-column scanned text well - a proper OCR pass (e.g.
  Tesseract) would be cheaper for pure scans, with the vision model reserved for charts/infographics.
- **Entity resolution is a single flat pass over distinct strings**, not true coreference
  resolution - it won't catch a pronoun like "it" referring to an entity, and it re-clusters
  *all* entities from scratch on every upload rather than incrementally merging just the new
  ones. On a much larger entity list this call would need to be batched too.
- **Candidate grouping still falls back to word-overlap** for any metric that hasn't been
  through the canonicalization pass yet (e.g. right after a fresh upload, before
  `canonicalize_all()` finishes), or for genuinely novel metrics the resolver couldn't confidently
  cluster. It's also not embeddings - the LLM-based clustering can occasionally mis-group very
  similarly-worded but distinct metrics, or fail to merge two that are phrased very differently;
  there's no automated check on this yet (see below).
- **Incremental analysis still re-groups all facts on every call** (cheap, local, no LLM cost)
  even though it only sends *new* groups to the model - fine at this scale, but the grouping
  step itself would need to move to an index (not a full rescan) for very large fact counts.
- **No entity resolution/candidate-grouping evaluation** - there's no check on how often the
  heuristics under- or over-group things; that's currently only visible by eyeballing the UI.
- **No auth, no multi-user support, single local SQLite file.** Fine for a prototype/demo, not
  for anything beyond that.

