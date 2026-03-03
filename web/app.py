#!/usr/bin/env python3
"""
LLMlab Benchmark Web UI
FastAPI + htmx interface for context ladder benchmarking
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
from pydantic import BaseModel

# Add parent directory to path to import the benchmark script
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

app = FastAPI(title="LLMlab Benchmark UI")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Configuration
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://192.168.0.27:8080")
RESULTS_DIR = Path(__file__).parent.parent / "scripts" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class BenchmarkRequest(BaseModel):
    """Request model for benchmark runs"""
    test_points: str = "9,3000,6000,10000,13000,19000,24000,32000,64000,95000,128000,192000,250000"
    output_formats: List[str] = ["json", "csv", "md"]
    monitor_resources: bool = True


class ServerSettings(BaseModel):
    """Server settings that can be configured"""
    ctx_size: Optional[int] = None
    n_parallel: Optional[int] = None
    # Add more settings as needed


async def get_llama_server_info() -> Dict[str, Any]:
    """Fetch server info from llama.cpp"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{LLAMA_SERVER_URL}/props")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        print(f"Error fetching server info: {e}")
    return {}


async def get_llama_slots() -> Dict[str, Any]:
    """Fetch current slot status from llama.cpp"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{LLAMA_SERVER_URL}/slots")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        print(f"Error fetching slots: {e}")
    return {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/models")
async def get_models():
    """Get available models and current server info"""
    server_info = await get_llama_server_info()
    slots = await get_llama_slots()
    
    # Extract model info
    model_name = "Unknown"
    if server_info:
        # Try to extract model name from server props
        model_name = server_info.get("model_alias", "Unknown")
        if model_name == "Unknown":
            # Fallback to model_path
            model_name = server_info.get("model_path", "Unknown")
    
    return {
        "current_model": model_name,
        "server_info": server_info,
        "slots": slots
    }


@app.get("/api/settings")
async def get_settings():
    """Get current server settings"""
    server_info = await get_llama_server_info()
    
    # Extract relevant settings
    settings = {}
    if server_info:
        default_gen = server_info.get("default_generation_settings", {})
        settings = {
            "ctx_size": default_gen.get("n_ctx", "Unknown"),
            "total_slots": server_info.get("total_slots", "Unknown"),
            "model": server_info.get("model_alias", "Unknown"),
            "build": server_info.get("build_info", "Unknown")
        }
    
    return settings


@app.post("/api/settings")
async def update_settings(settings: ServerSettings):
    """
    Update server settings (requires server restart)
    Note: This is a placeholder - actual implementation would need to
    modify server launch args and restart the service
    """
    return {
        "status": "info",
        "message": "Server setting changes require manual restart. Please update your server launch command."
    }


@app.get("/api/results")
async def get_results():
    """Get list of previous benchmark results"""
    results = []
    
    for result_file in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(result_file) as f:
                data = json.load(f)
                results.append({
                    "filename": result_file.name,
                    "timestamp": data.get("metadata", {}).get("timestamp", ""),
                    "model": data.get("metadata", {}).get("model", "Unknown"),
                    "test_points": len(data.get("results", []))
                })
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    return results


@app.get("/api/results/{filename}")
async def get_result(filename: str):
    """Get specific benchmark result"""
    result_file = RESULTS_DIR / filename
    if not result_file.exists():
        return JSONResponse({"error": "Result not found"}, status_code=404)
    
    try:
        with open(result_file) as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def run_benchmark_async(test_points: str, output_formats: List[str], monitor: bool):
    """Run the benchmark script and yield progress updates"""
    
    # Build command
    script_path = Path(__file__).parent.parent / "scripts" / "run_context_ladder.py"
    cmd = [
        "python3",
        str(script_path),
        "--test-points", test_points,
    ]
    
    # Add output formats
    for fmt in output_formats:
        cmd.append(f"--{fmt}")
    
    if monitor:
        cmd.append("--monitor")
    
    # Run the script
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=script_path.parent
    )
    
    # Stream output
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        
        line_text = line.decode().strip()
        if line_text:
            yield f"data: {json.dumps({'type': 'log', 'message': line_text})}\n\n"
    
    # Wait for completion
    await process.wait()
    
    # Send completion message
    if process.returncode == 0:
        yield f"data: {json.dumps({'type': 'complete', 'status': 'success'})}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'complete', 'status': 'error', 'code': process.returncode})}\n\n"


@app.post("/api/benchmark/run")
async def run_benchmark(request: BenchmarkRequest):
    """Start a benchmark run with SSE streaming"""
    
    async def event_generator():
        async for event in run_benchmark_async(
            request.test_points,
            request.output_formats,
            request.monitor_resources
        ):
            yield event
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    server_reachable = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{LLAMA_SERVER_URL}/health")
            server_reachable = response.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy",
        "llama_server": "reachable" if server_reachable else "unreachable",
        "llama_server_url": LLAMA_SERVER_URL
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting LLMlab Benchmark UI on {host}:{port}")
    print(f"Llama server: {LLAMA_SERVER_URL}")
    
    uvicorn.run(app, host=host, port=port)
