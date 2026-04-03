import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import AegisConfig
from utils.preprocessing import Preprocessor
from core.tools.registry import get_registry

app = FastAPI(title="Aegis-X Web Interface")

# Initialize config and registry once
config = AegisConfig()
preprocessor = Preprocessor(config)
registry = get_registry()
cpu_tools = ["check_c2pa", "run_dct", "run_geometry", "run_illumination"]

UPLOAD_DIR = Path("downloads/web_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

from fastapi.responses import JSONResponse, StreamingResponse
import json

@app.post("/api/analyze")
async def analyze_media(file: UploadFile = File(...)):
    try:
        # Save temp file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Preprocessing
        prep_result = preprocessor.process_media(file_path)
        if not prep_result.has_face:
            return JSONResponse(status_code=400, content={"error": "No faces detected in the media.", "success": False})
            
        # Run Tools with Streaming
        from core.agent import ForensicAgent
        
        async def event_generator():
            agent = ForensicAgent(config)
            
            # Send initial success event
            yield f"data: {json.dumps({'event_type': 'init', 'faces_detected': len(prep_result.tracked_faces)})}\n\n"
            
            for event in agent.analyze(prep_result, media_path=str(file_path)):
                # Clean up data payload for SSE
                payload = {
                    "event_type": event.event_type,
                    "tool_name": event.tool_name,
                    "data": event.data
                }
                yield f"data: {json.dumps(payload)}\n\n"
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "success": False})

@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    # Prevent browser caching of static files like HTML, CSS, JS during development
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Mount frontend
app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    # run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv_main/bin/python run_web.py`
    uvicorn.run(
        "run_web:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_includes=["*.html", "*.css", "*.js"]
    )
