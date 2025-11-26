#!/usr/bin/env python3
"""
Simplified main entry point for initial deployment testing
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# GPU detection (simplified for initial deployment)
import os
GPU_AVAILABLE = os.getenv("CUDA_VISIBLE_DEVICES") is not None
print(f"GPU Available: {GPU_AVAILABLE}")

app = FastAPI(
    title="RuvTrade GPU Trading Platform",
    description="High-performance GPU-accelerated trading platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "RuvTrade GPU Trading Platform",
        "version": "1.0.0",
        "gpu_available": GPU_AVAILABLE,
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_available": GPU_AVAILABLE}

@app.get("/gpu-status")
async def gpu_status():
    return {
        "gpu_available": GPU_AVAILABLE,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "not_set"),
        "message": "GPU libraries will be available when deployed to GPU instance"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)