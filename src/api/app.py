import os
import glob
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.pipeline import MRIAnalysisEngine, CLASS_NAMES, CHECKPOINT_CONFIGS, RUNS_DIR
from src.inference.report_generator import generate_pdf_report

app = FastAPI(
    title="BrainX Multi-Model MRI Analysis API",
    description="Interactive 3-Model Brain MRI Classification & Grad-CAM Explainability System",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine and history store
engine: Optional[MRIAnalysisEngine] = None
history_store = []


@app.on_event("startup")
def startup_event():
    global engine
    print("[API] Starting BrainX Inference Service...")
    engine = MRIAnalysisEngine.get_instance()
    print("[API] Engine loaded and ready.")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "engine_ready": engine is not None,
        "device": str(engine.device) if engine else "uninitialized",
        "models_loaded": list(CHECKPOINT_CONFIGS.keys()) if engine else []
    }


@app.get("/api/metrics")
def get_research_metrics():
    """Returns official research evaluation metrics from held-out test cohort (N=960)."""
    return {
        "test_set_size": 960,
        "total_dataset_size": 6400,
        "classes": CLASS_NAMES,
        "models": [
            {
                "id": "baseline",
                "name": "Baseline ResNet18",
                "preprocessing": "Original Unenhanced MRI",
                "accuracy": 0.9438,
                "macro_precision": 0.9537,
                "macro_recall": 0.9523,
                "macro_f1": 0.9530,
                "weighted_f1": 0.9436,
                "per_class_f1": {
                    "Mild Demented": 0.9259,
                    "Moderate Demented": 1.0000,
                    "Non Demented": 0.9597,
                    "Very Mild Demented": 0.9263
                }
            },
            {
                "id": "he",
                "name": "Histogram Equalization (HE)",
                "preprocessing": "Global EqualizeHist",
                "accuracy": 0.7823,
                "macro_precision": 0.7863,
                "macro_recall": 0.8249,
                "macro_f1": 0.7976,
                "weighted_f1": 0.7840,
                "per_class_f1": {
                    "Mild Demented": 0.7284,
                    "Moderate Demented": 0.8889,
                    "Non Demented": 0.8180,
                    "Very Mild Demented": 0.7551
                }
            },
            {
                "id": "clahe",
                "name": "CLAHE ResNet18 (Top Accuracy)",
                "preprocessing": "Adaptive Local Contrast (clipLimit=2.0)",
                "accuracy": 0.9510,
                "macro_precision": 0.9668,
                "macro_recall": 0.9281,
                "macro_f1": 0.9461,
                "weighted_f1": 0.9511,
                "per_class_f1": {
                    "Mild Demented": 0.9421,
                    "Moderate Demented": 0.9412,
                    "Non Demented": 0.9634,
                    "Very Mild Demented": 0.9376
                }
            }
        ]
    }


@app.get("/api/sample-images")
def get_sample_images():
    """Provides representative test-ready sample images across all 4 disease categories."""
    dataset_dir = os.path.abspath("Dataset")
    class_folders = {
        "Mild Demented": "Mild_Demented",
        "Moderate Demented": "Moderate_Demented",
        "Non Demented": "Non_Demented",
        "Very Mild Demented": "Very_Mild_Demented"
    }

    samples = []
    for c_name, folder in class_folders.items():
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.exists(folder_path):
            files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))[:3]
            for idx, f in enumerate(files):
                fname = os.path.basename(f)
                samples.append({
                    "id": f"{folder}_{idx}",
                    "filename": fname,
                    "class_name": c_name,
                    "path": f
                })
    return samples


class SampleAnalysisRequest(BaseModel):
    sample_path: str


@app.post("/api/analyze-sample")
def analyze_sample(req: SampleAnalysisRequest):
    """Analyzes a selected sample image from disk."""
    if not os.path.exists(req.sample_path):
        raise HTTPException(status_code=404, detail="Sample image not found on disk.")

    with open(req.sample_path, "rb") as f:
        image_bytes = f.read()

    fname = os.path.basename(req.sample_path)
    result = engine.analyze(image_bytes, filename=fname)

    # Save to history
    history_store.insert(0, {
        "run_id": result["run_id"],
        "timestamp": result["timestamp"],
        "filename": result["filename"],
        "agreement": result["agreement"]["status"],
        "consensus_class": result["agreement"]["consensus_class"],
        "run_dir": result["run_dir"]
    })

    return result


@app.post("/api/analyze")
async def analyze_uploaded_mri(file: UploadFile = File(...)):
    """Receives an uploaded MRI image, validates it, and executes 3-model inference."""
    # 1. Validate file format
    allowed_exts = {".png", ".jpg", ".jpeg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Please upload a PNG, JPG, or JPEG brain MRI image."
        )

    # 2. Read bytes
    try:
        contents = await file.read()
        if len(contents) < 100:
            raise ValueError("Uploaded file is empty or corrupted.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # 3. Execute analysis
    try:
        result = engine.analyze(contents, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # 4. Save to session history
    history_store.insert(0, {
        "run_id": result["run_id"],
        "timestamp": result["timestamp"],
        "filename": result["filename"],
        "agreement": result["agreement"]["status"],
        "consensus_class": result["agreement"]["consensus_class"],
        "run_dir": result["run_dir"]
    })

    return result


@app.get("/api/report/{run_id}")
def download_pdf_report(run_id: str):
    """Generates and serves a comprehensive PDF analysis report."""
    run_dir = os.path.join(RUNS_DIR, run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Analysis run not found.")

    pdf_path = os.path.join(run_dir, f"{run_id}_report.pdf")

    # If PDF already generated, serve it
    if os.path.exists(pdf_path):
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"BrainX_Analysis_{run_id}.pdf"
        )

    # Find the run in history or reconstruct basic analysis data
    matched_run = next((item for item in history_store if item["run_id"] == run_id), None)
    
    # Check if raw image and heatmaps exist in run_dir to generate PDF
    raw_img = os.path.join(run_dir, "original_raw.png")
    if not os.path.exists(raw_img):
        raise HTTPException(status_code=404, detail="Run assets incomplete.")

    with open(raw_img, "rb") as f:
        img_bytes = f.read()

    fname = matched_run["filename"] if matched_run else "brain_mri.png"
    analysis_data = engine.analyze(img_bytes, filename=fname)
    analysis_data["run_dir"] = run_dir

    generate_pdf_report(analysis_data, pdf_path)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"BrainX_Analysis_{run_id}.pdf"
    )


@app.get("/api/history")
def get_analysis_history():
    """Returns recent analysis runs performed in this server session."""
    return history_store[:25]


# Mount static assets for run results and comparisons
if os.path.exists(RUNS_DIR):
    app.mount("/results/app_runs", StaticFiles(directory=RUNS_DIR), name="app_runs")

comparison_dir = os.path.abspath(os.path.join("results", "comparison"))
if os.path.exists(comparison_dir):
    app.mount("/results/comparison", StaticFiles(directory=comparison_dir), name="comparison")

# Mount frontend directory
frontend_dir = os.path.abspath("frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
