"""FastAPI application for Veo Analytics."""

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import uuid
import shutil
import logging

from src.application.use_cases.analyze_match import (
    AnalyzeMatchUseCase,
    AnalyzeMatchRequest
)
from src.infrastructure.ml.detector import PlayerDetector, YOLOPlayerDetector
from src.infrastructure.ml.tracker import PlayerTracker
from src.infrastructure.video.video_reader import VideoReader

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Veo Analytics API",
    description="Sports video analysis API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage (in production use Redis/DB)
jobs = {}


class AnalysisRequest(BaseModel):
    """Request model for video analysis."""
    start_time: Optional[float] = 0.0
    end_time: Optional[float] = -1.0
    save_video: bool = True
    generate_report: bool = True


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "Veo Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze",
            "status": "/status/{job_id}",
            "result": "/result/{job_id}",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/analyze", response_model=JobStatus)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: AnalysisRequest = AnalysisRequest()
):
    """Upload and analyze video."""
    # Validate file
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"{job_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create job
    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "file_path": str(file_path),
        "request": request.dict()
    }
    
    # Start background processing
    background_tasks.add_task(process_video, job_id, file_path, request)
    
    return JobStatus(
        job_id=job_id,
        status="pending",
        progress=0.0
    )


@app.get("/status/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        result=job.get("result"),
        error=job.get("error")
    )


@app.get("/result/{job_id}/video")
def get_result_video(job_id: str):
    """Download processed video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output video not found")
    
    return FileResponse(output_path, media_type="video/mp4")


@app.get("/result/{job_id}/report")
def get_result_report(job_id: str):
    """Get analysis report."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    return JSONResponse(content=job.get("result", {}))


def process_video(job_id: str, file_path: Path, request: AnalysisRequest):
    """Process video in background."""
    try:
        # Update status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1
        
        # Initialize components
        detector = PlayerDetector(YOLOPlayerDetector())
        tracker = PlayerTracker()
        video_reader = VideoReader()
        
        # Create use case
        use_case = AnalyzeMatchUseCase(
            detector=detector,
            tracker=tracker,
            video_reader=video_reader
        )
        
        # Prepare request
        output_path = Path("outputs/videos") / f"{job_id}_analyzed.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        analyze_request = AnalyzeMatchRequest(
            video_path=file_path,
            output_path=output_path,
            start_time=request.start_time,
            end_time=request.end_time,
            save_video=request.save_video,
            generate_report=request.generate_report
        )
        
        # Execute analysis
        response = use_case.execute(analyze_request)
        
        # Update job
        if response.success:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["output_path"] = str(response.output_path)
            jobs[job_id]["result"] = {
                "statistics": response.statistics,
                "players_detected": response.players_detected,
                "processing_time": response.processing_time
            }
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = response.message
            
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# Mount static files
if Path("outputs").exists():
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
