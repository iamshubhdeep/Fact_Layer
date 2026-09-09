"""
Very small SQLite wrapper. No ORM on purpose - this is a basic prototype,
not a production system. Everything lives in one file: data/factlayer.db
"""
import sqlite3
import json
import os
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "factlayer.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded',  -- uploaded | extracting | extracted | error
            analyzed INTEGER NOT NULL DEFAULT 0        -- has this doc's facts been through cross-doc analysis yet
        );

        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_number INTEGER NOT NULL,
            text TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_number INTEGER,
            entity TEXT,             -- who/what the fact is about, as extracted (raw)
            canonical_entity TEXT,   -- normalized entity name after entity resolution pass, else NULL
            metric TEXT,             -- what is being measured/stated, as extracted (raw)
            canonical_metric TEXT,   -- normalized metric name after metric resolution pass, else NULL
            value TEXT,              -- the raw value as extracted (kept as text, can be numeric or descriptive)
            unit TEXT,               -- units, currency, %, etc. if any
            period TEXT,             -- time period / as-of date the fact refers to
            statement TEXT NOT NULL, -- plain-english normalized statement of the fact
            quote TEXT,              -- short verbatim evidence snippet from the source page
            confidence TEXT,         -- low | medium | high, model's own confidence
            notes TEXT,              -- anything odd the model flagged (ambiguity, possible extraction issue)
            extraction_method TEXT DEFAULT 'text', -- 'text' or 'vision' (page had too little extractable text)
            extra TEXT,              -- optional JSON blob for fact-type-specific fields beyond the fixed columns
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_id_a INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
            fact_id_b INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,   -- corroborates | contradicts | reconciled | related
            explanation TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def add_document(filename):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO documents (filename, uploaded_at, status) VALUES (?, ?, 'uploaded')",
        (filename, time.time()),
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return doc_id


def set_document_status(document_id, status):
    conn = get_conn()
    conn.execute("UPDATE documents SET status=? WHERE id=?", (status, document_id))
    conn.commit()
    conn.close()


def add_page(document_id, page_number, text):
    conn = get_conn()
    conn.execute(
        "INSERT INTO pages (document_id, page_number, text) VALUES (?, ?, ?)",
        (document_id, page_number, text),
    )
    conn.commit()
    conn.close()


def get_pages(document_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM pages WHERE document_id=? ORDER BY page_number", (document_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_fact(document_id, page_number, fact: dict):
    conn = get_conn()
    extra = fact.get("extra")
    if isinstance(extra, (dict, list)):
        extra = json.dumps(extra, ensure_ascii=False)
    conn.execute(
        """INSERT INTO facts
           (document_id, page_number, entity, metric, value, unit, period, statement, quote,
            confidence, notes, extraction_method, extra, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            document_id,
            page_number,
            fact.get("entity"),
            fact.get("metric"),
            fact.get("value"),
            fact.get("unit"),
            fact.get("period"),
            fact.get("statement"),
            fact.get("quote"),
            fact.get("confidence"),
            fact.get("notes"),
            fact.get("extraction_method", "text"),
            extra,
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def set_canonical_entity(fact_id, canonical_entity):
    conn = get_conn()
    conn.execute("UPDATE facts SET canonical_entity=? WHERE id=?", (canonical_entity, fact_id))
    conn.commit()
    conn.close()


def set_canonical_metric(fact_id, canonical_metric):
    conn = get_conn()
    conn.execute("UPDATE facts SET canonical_metric=? WHERE id=?", (canonical_metric, fact_id))
    conn.commit()
    conn.close()


def distinct_entities():
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT entity FROM facts WHERE entity IS NOT NULL AND entity != ''"
    ).fetchall()
    conn.close()
    return [r["entity"] for r in rows]


def distinct_metrics():
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT metric FROM facts WHERE metric IS NOT NULL AND metric != ''"
    ).fetchall()
    conn.close()
    return [r["metric"] for r in rows]


def set_document_analyzed(document_id, analyzed=True):
    conn = get_conn()
    conn.execute("UPDATE documents SET analyzed=? WHERE id=?", (1 if analyzed else 0, document_id))
    conn.commit()
    conn.close()


def unanalyzed_document_ids():
    conn = get_conn()
    rows = conn.execute("SELECT id FROM documents WHERE analyzed=0 AND status='extracted'").fetchall()
    conn.close()
    return [r["id"] for r in rows]


def list_documents():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_facts(document_id=None):
    conn = get_conn()
    if document_id:
        rows = conn.execute(
            "SELECT f.*, d.filename FROM facts f JOIN documents d ON f.document_id=d.id "
            "WHERE f.document_id=? ORDER BY f.id",
            (document_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT f.*, d.filename FROM facts f JOIN documents d ON f.document_id=d.id ORDER BY f.id"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_fact(fact_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT f.*, d.filename FROM facts f JOIN documents d ON f.document_id=d.id WHERE f.id=?",
        (fact_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def clear_relationships():
    conn = get_conn()
    conn.execute("DELETE FROM relationships")
    conn.commit()
    conn.close()


def add_relationship(fact_id_a, fact_id_b, relation_type, explanation):
    conn = get_conn()
    conn.execute(
        """INSERT INTO relationships (fact_id_a, fact_id_b, relation_type, explanation, created_at)
           VALUES (?,?,?,?,?)""",
        (fact_id_a, fact_id_b, relation_type, explanation, time.time()),
    )
    conn.commit()
    conn.close()


def list_relationships():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM relationships ORDER BY id").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["fact_a"] = get_fact(d["fact_id_a"])
        d["fact_b"] = get_fact(d["fact_id_b"])
        out.append(d)
    return out


def reset_all():
    conn = get_conn()
    conn.executescript(
        "DELETE FROM relationships; DELETE FROM facts; DELETE FROM pages; DELETE FROM documents;"
    )
    conn.commit()
    conn.close()
