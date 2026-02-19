from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import sys

from predict import load_model, predict_image
from metadata import check_metadata

# ============================================================
# STARTUP — Load model once when server starts
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists("model.pth"):
        print("ERROR: model.pth not found!")
        print("Run: python train.py first")
        sys.exit(1)

    print("Loading VERITAS model...")
    load_model()
    print("ML Engine ready on port 8000")
    yield

app = FastAPI(
    title="Veritas ML Engine",
    description="Deepfake detection API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ROUTES
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": "loaded",
        "service": "veritas-ml-engine"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept image file, run deepfake detection + metadata analysis.
    Returns combined result.
    """
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPG, PNG, or WebP."
        )

    image_bytes = await file.read()

    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 20MB.")

    if len(image_bytes) < 1000:
        raise HTTPException(status_code=400, detail="File too small or corrupted.")

    try:
        prediction = predict_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    try:
        metadata = check_metadata(image_bytes)
    except Exception:
        metadata = {
            "status": "Error",
            "reason": "Could not analyze metadata",
            "details": {}
        }

    return {
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "scores": prediction["scores"],
        "metadata": metadata,
        "filename": file.filename,
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
