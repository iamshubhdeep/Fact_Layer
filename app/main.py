import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import db, extract, analyze, resolve

app = FastAPI(title="Fact Knowledge Layer - basic prototype")

# Permissive by default so the UI can be hosted separately from the API if you split them for
# deployment; tighten this to your actual frontend origin before shipping publicly.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "30"))

db.init_db()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/health")
def health():
    """Basic health/readiness check for deployment - hit this from your platform's health
    probe. Reports whether the API key is configured and the database is reachable, without
    making any paid API calls itself."""
    checks = {}

    checks["api_key_configured"] = bool(os.environ.get("ANTHROPIC_API_KEY"))

    try:
        db.list_documents()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    healthy = checks["api_key_configured"] and checks["database"] == "ok"
    status_code = 200 if healthy else 503
    return JSONResponse({"status": "ok" if healthy else "degraded", "checks": checks}, status_code=status_code)


@app.post("/api/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB:.0f} MB upload limit.")

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    size = 0
    with open(dest_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                f.close()
                os.remove(dest_path)
                raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB:.0f} MB upload limit.")
            f.write(chunk)

    document_id = db.add_document(file.filename)

    try:
        extract.process_document(document_id, dest_path)
        # cheap pass - just distinct entity/metric strings, not full documents - so we re-run it
        # on every upload to keep canonical names correct as new spellings show up
        resolve.canonicalize_all()
    except Exception as e:
        return JSONResponse(
            {"document_id": document_id, "status": "error", "error": str(e)}, status_code=500
        )

    facts = db.list_facts(document_id)
    vision_facts = [f for f in facts if f["extraction_method"] != "text"]
    return {
        "document_id": document_id,
        "status": "extracted",
        "fact_count": len(facts),
        "vision_fallback_facts": len(vision_facts),
    }


@app.get("/api/documents")
def get_documents():
    return db.list_documents()


@app.get("/api/facts")
def get_facts(document_id: int | None = None):
    return db.list_facts(document_id)


@app.post("/api/analyze")
def analyze_facts(mode: str = "incremental"):
    """mode=incremental (default): only analyze candidate groups touching documents that
    haven't been analyzed yet - cheap, and doesn't rebuild existing relationships.
    mode=full: clear everything and recompute from scratch."""
    if mode == "full":
        relationships = analyze.run_analysis()
    else:
        relationships = analyze.run_incremental_analysis()
    return {"relationship_count": len(relationships), "mode": mode}


@app.post("/api/resolve-entities")
def resolve_entities():
    """Manually re-run entity + metric canonicalization (also runs automatically after each upload)."""
    result = resolve.canonicalize_all()
    return {
        "canonical_entity_groups": len(set(result["entities"].values())),
        "mapped_entities": len(result["entities"]),
        "canonical_metric_groups": len(set(result["metrics"].values())),
        "mapped_metrics": len(result["metrics"]),
    }


@app.get("/api/relationships")
def get_relationships():
    return db.list_relationships()


@app.post("/api/reset")
def reset():
    """Wipes everything - handy for demos/testing."""
    db.reset_all()
    for f in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, f))
    return {"status": "reset"}
