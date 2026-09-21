"""
main.py

FastAPI backend. Run with:
    uvicorn backend.main:app --reload --port 8000

Single endpoint design explained in Step 9's write-up: one request,
one Wav2Vec2 pass, every downstream model reused from the SAME
extracted features.
"""

import os
import shutil
import tempfile
import sys

sys.path.append(os.getcwd())
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from backend.inference_pipeline import InferencePipeline

pipeline_holder = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load every model exactly once, store on the app state.
    # This is the FastAPI-recommended pattern (replaces the older
    # @app.on_event("startup") decorator) for expensive one-time setup.
    print("Starting up -- loading inference pipeline (this may take a moment)...")
    pipeline_holder["pipeline"] = InferencePipeline()
    yield
    # Shutdown: nothing to clean up explicitly -- torch/models are
    # garbage collected with the process.
    print("Shutting down.")


app = FastAPI(title="Speech-Driven 3D Avatar API", lifespan=lifespan)

# CORS: allows the browser-based frontend, served from a different
# origin/port (locally OR once deployed to Netlify/Vercel/etc), to
# call this API. ALLOWED_ORIGINS can be set as an environment variable
# (comma-separated) when deploying -- e.g.
#   ALLOWED_ORIGINS=https://your-frontend.netlify.app,http://localhost:5500
# Falls back to "*" (allow everything) for local development, which is
# fine for an academic project but should be tightened for any real
# public deployment.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
allow_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm"}

# Serves assets/avatars/*.glb at http://localhost:8000/assets/avatars/...
# so the frontend's fetch of avatar_path (returned by /animate) resolves
# to an actual downloadable file rather than a server-side-only path.
if os.path.isdir("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/health")
def health_check():
    """Simple liveness check -- lets the frontend confirm the backend
    (and its models) finished loading before allowing an upload."""
    return {"status": "ok", "models_loaded": "pipeline" in pipeline_holder}


@app.post("/animate")
@app.post("/generate")  # alias -- matches the endpoint name requested in
                         # the spec; kept alongside /animate (Step 9's
                         # original name) so no existing client breaks.
async def animate(file: UploadFile = File(...)):
    """
    Accepts an audio file upload, runs the full pipeline, returns the
    animation payload described in Step 9's API design.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400,
                             detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    pipeline = pipeline_holder.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models still loading, try again shortly.")

    # Save the upload to a temp file -- librosa/soundfile need a real
    # file path (or file-like object with seek support); using a
    # NamedTemporaryFile keeps this safe under concurrent requests
    # (each gets its own uniquely-named file) and auto-cleans up.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = pipeline.run(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        os.remove(tmp_path)  # always clean up, even if pipeline.run() raised

    return result


# Allows `python backend/main.py` to run the server directly, reading
# the PORT from the environment -- cloud platforms like Render/Railway
# assign a dynamic port via the $PORT env var rather than a fixed one,
# so hardcoding 8000 would break those deployments. Local development
# still works fine with `uvicorn backend.main:app --reload --port 8000`
# as before; this block is just an alternate entry point for hosting
# platforms that invoke `python main.py` directly (see DEPLOYMENT.md).
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
