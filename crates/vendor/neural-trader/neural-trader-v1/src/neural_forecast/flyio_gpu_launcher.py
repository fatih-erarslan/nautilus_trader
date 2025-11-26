#!/usr/bin/env python3
"""
Fly.io GPU Launcher for AI News Trading Platform
NeuralForecast NHITS Implementation with GPU Optimization
"""

import os
import sys
import time
import json
import asyncio
import logging
import signal
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psutil
import GPUtil

# Add project root to path
sys.path.append('/app')

from src.neural_forecast.optimized_nhits_engine import OptimizedNHITSEngine
from src.neural_forecast.lightning_inference_engine import LightningInferenceEngine
from src.neural_forecast.advanced_memory_manager import AdvancedMemoryManager
from gpu_acceleration.flyio_optimizer import initialize_flyio_optimization
from gpu_acceleration.gpu_monitor import GPUMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/flyio_gpu.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FlyIOGPUConfig:
    """Configuration for Fly.io GPU deployment."""
    gpu_type: str = "a100-40gb"
    auto_scale: bool = True
    cost_optimization: bool = True
    batch_size: int = 32
    precision: str = "mixed"
    tensorrt_enabled: bool = True
    max_memory_gb: int = 40
    region: str = "ord"

class FlyIOGPULauncher:
    """Main launcher for Fly.io GPU-optimized neural forecasting."""
    
    def __init__(self):
        self.config = self._load_config()
        self.app = FastAPI(
            title="AI News Trading Platform - Neural Forecasting",
            description="GPU-Optimized NHITS Neural Forecasting on Fly.io",
            version="1.0.0"
        )
        self.setup_middleware()
        self.setup_routes()
        
        # Core components
        self.gpu_monitor: Optional[GPUMonitor] = None
        self.memory_manager: Optional[AdvancedMemoryManager] = None
        self.inference_engine: Optional[LightningInferenceEngine] = None
        self.nhits_engine: Optional[OptimizedNHITSEngine] = None
        
        # State tracking
        self.is_initialized = False
        self.last_prediction_time = time.time()
        self.prediction_count = 0
        self.error_count = 0
        
    def _load_config(self) -> FlyIOGPUConfig:
        """Load configuration from environment variables."""
        return FlyIOGPUConfig(
            gpu_type=os.getenv("FLYIO_GPU_TYPE", "a100-40gb"),
            auto_scale=os.getenv("FLYIO_AUTO_SCALE", "true").lower() == "true",
            cost_optimization=os.getenv("FLYIO_COST_OPTIMIZATION", "true").lower() == "true",
            batch_size=int(os.getenv("NEURAL_FORECAST_BATCH_SIZE", "32")),
            precision=os.getenv("NEURAL_FORECAST_PRECISION", "mixed"),
            tensorrt_enabled=os.getenv("TENSORRT_ENABLED", "true").lower() == "true",
            max_memory_gb=int(os.getenv("GPU_MAX_MEMORY_GB", "40")),
            region=os.getenv("FLY_REGION", "ord")
        )
    
    def setup_middleware(self):
        """Configure FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Set up API routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize GPU components on startup."""
            await self.initialize_gpu_components()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            await self.cleanup()
        
        @self.app.get("/health")
        async def health_check():
            """Basic health check."""
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.app.get("/health/gpu")
        async def gpu_health_check():
            """GPU-specific health check."""
            try:
                gpu_info = self._get_gpu_info()
                return {
                    "status": "healthy",
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_info": gpu_info,
                    "memory_usage": self._get_memory_usage(),
                    "initialized": self.is_initialized
                }
            except Exception as e:
                logger.error(f"GPU health check failed: {e}")
                raise HTTPException(status_code=503, detail=f"GPU health check failed: {str(e)}")
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus-compatible metrics."""
            try:
                gpu_metrics = await self._collect_gpu_metrics()
                return JSONResponse(content=gpu_metrics)
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.get("/metrics/gpu")
        async def gpu_metrics():
            """Detailed GPU metrics."""
            try:
                if not self.gpu_monitor:
                    raise HTTPException(status_code=503, detail="GPU monitor not initialized")
                
                metrics = await self.gpu_monitor.get_comprehensive_metrics()
                return JSONResponse(content=metrics)
            except Exception as e:
                logger.error(f"GPU metrics failed: {e}")
                raise HTTPException(status_code=503, detail=f"GPU metrics failed: {str(e)}")
        
        @self.app.post("/neural/predict")
        async def neural_predict(
            symbol: str,
            horizon: int = 24,
            confidence_level: float = 0.95,
            background_tasks: BackgroundTasks = None
        ):
            """Generate neural forecast prediction."""
            try:
                if not self.is_initialized:
                    raise HTTPException(status_code=503, detail="Neural engine not initialized")
                
                start_time = time.time()
                
                # Generate prediction using optimized engine
                prediction = await self.inference_engine.predict_symbol(
                    symbol=symbol,
                    horizon=horizon,
                    confidence_level=confidence_level
                )
                
                latency = (time.time() - start_time) * 1000  # Convert to ms
                
                # Update metrics
                self.prediction_count += 1
                self.last_prediction_time = time.time()
                
                # Log performance metrics
                if background_tasks:
                    background_tasks.add_task(self._log_prediction_metrics, symbol, latency)
                
                return {
                    "symbol": symbol,
                    "prediction": prediction,
                    "latency_ms": latency,
                    "timestamp": time.time(),
                    "gpu_utilized": torch.cuda.is_available(),
                    "model": "nhits_optimized"
                }
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Prediction failed for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @self.app.post("/neural/batch_predict")
        async def batch_predict(
            symbols: list[str],
            horizon: int = 24,
            confidence_level: float = 0.95
        ):
            """Generate batch predictions for multiple symbols."""
            try:
                if not self.is_initialized:
                    raise HTTPException(status_code=503, detail="Neural engine not initialized")
                
                start_time = time.time()
                
                # Use batch processing for efficiency
                predictions = await self.inference_engine.predict_batch(
                    symbols=symbols,
                    horizon=horizon,
                    confidence_level=confidence_level
                )
                
                total_latency = (time.time() - start_time) * 1000
                
                self.prediction_count += len(symbols)
                
                return {
                    "predictions": predictions,
                    "total_latency_ms": total_latency,
                    "avg_latency_ms": total_latency / len(symbols),
                    "symbols_count": len(symbols),
                    "timestamp": time.time()
                }
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Batch prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
        
        @self.app.get("/status")
        async def status():
            """Comprehensive system status."""
            return {
                "system": {
                    "initialized": self.is_initialized,
                    "uptime": time.time() - self.start_time,
                    "region": self.config.region,
                    "gpu_type": self.config.gpu_type
                },
                "performance": {
                    "prediction_count": self.prediction_count,
                    "error_count": self.error_count,
                    "error_rate": self.error_count / max(1, self.prediction_count),
                    "last_prediction": time.time() - self.last_prediction_time
                },
                "gpu": self._get_gpu_info(),
                "memory": self._get_memory_usage()
            }
    
    async def initialize_gpu_components(self):
        """Initialize all GPU-optimized components."""
        try:
            logger.info("Initializing Fly.io GPU components...")
            
            # Initialize fly.io optimization
            optimization_result = initialize_flyio_optimization("neural_trading")
            logger.info(f"Fly.io optimization: {optimization_result}")
            
            # Initialize GPU monitoring
            self.gpu_monitor = GPUMonitor()
            await self.gpu_monitor.start_monitoring()
            
            # Initialize memory manager
            self.memory_manager = AdvancedMemoryManager(
                gpu_memory_gb=self.config.max_memory_gb
            )
            await self.memory_manager.initialize()
            
            # Initialize inference engine with GPU optimization
            self.inference_engine = LightningInferenceEngine(
                batch_size=self.config.batch_size,
                enable_tensorrt=self.config.tensorrt_enabled,
                precision=self.config.precision,
                memory_manager=self.memory_manager
            )
            await self.inference_engine.initialize()
            
            # Initialize NHITS engine
            self.nhits_engine = OptimizedNHITSEngine(
                gpu_type=self.config.gpu_type,
                enable_mixed_precision=True,
                enable_tensorrt=self.config.tensorrt_enabled
            )
            await self.nhits_engine.initialize()
            
            self.is_initialized = True
            self.start_time = time.time()
            
            logger.info("GPU components initialized successfully")
            
            # Log initialization metrics
            gpu_info = self._get_gpu_info()
            logger.info(f"GPU Configuration: {gpu_info}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU components: {e}")
            self.is_initialized = False
            raise
    
    async def cleanup(self):
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up GPU resources...")
        
        try:
            if self.gpu_monitor:
                await self.gpu_monitor.stop_monitoring()
            
            if self.memory_manager:
                await self.memory_manager.cleanup()
            
            if self.inference_engine:
                await self.inference_engine.cleanup()
            
            if self.nhits_engine:
                await self.nhits_engine.cleanup()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("GPU cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get current GPU information."""
        try:
            if not torch.cuda.is_available():
                return {"available": False, "reason": "CUDA not available"}
            
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                return {"available": False, "reason": "No GPU devices found"}
            
            gpu_info = {}
            for i in range(gpu_count):
                gpu = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                
                gpu_info[f"gpu_{i}"] = {
                    "name": gpu.name,
                    "compute_capability": f"{gpu.major}.{gpu.minor}",
                    "total_memory_gb": gpu.total_memory / 1024**3,
                    "allocated_memory_gb": memory_allocated,
                    "cached_memory_gb": memory_cached,
                    "utilization": self._get_gpu_utilization(i)
                }
            
            return {"available": True, "devices": gpu_info}
            
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def _get_gpu_utilization(self, device_id: int) -> float:
        """Get GPU utilization percentage."""
        try:
            gpus = GPUtil.getGPUs()
            if device_id < len(gpus):
                return gpus[device_id].load * 100
            return 0.0
        except:
            return 0.0
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get system memory usage."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / 1024**3,
            "available_gb": memory.available / 1024**3,
            "used_gb": memory.used / 1024**3,
            "percentage": memory.percent
        }
    
    async def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics for monitoring."""
        try:
            metrics = {
                "timestamp": time.time(),
                "prediction_count": self.prediction_count,
                "error_count": self.error_count,
                "uptime": time.time() - getattr(self, 'start_time', time.time()),
                "gpu": self._get_gpu_info(),
                "memory": self._get_memory_usage(),
                "config": {
                    "gpu_type": self.config.gpu_type,
                    "batch_size": self.config.batch_size,
                    "precision": self.config.precision,
                    "tensorrt_enabled": self.config.tensorrt_enabled
                }
            }
            
            if self.gpu_monitor:
                gpu_metrics = await self.gpu_monitor.get_realtime_metrics()
                metrics["detailed_gpu"] = gpu_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    async def _log_prediction_metrics(self, symbol: str, latency: float):
        """Log prediction performance metrics."""
        try:
            metrics = {
                "symbol": symbol,
                "latency_ms": latency,
                "timestamp": time.time(),
                "gpu_utilization": self._get_gpu_utilization(0) if torch.cuda.is_available() else 0
            }
            
            # Log to structured logger for monitoring
            logger.info(f"PREDICTION_METRICS: {json.dumps(metrics)}")
            
        except Exception as e:
            logger.error(f"Failed to log prediction metrics: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main entry point for Fly.io deployment."""
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create logs directory
    os.makedirs('/app/logs', exist_ok=True)
    
    logger.info("Starting AI News Trading Platform - Neural Forecasting on Fly.io")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    logger.info(f"GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    # Initialize launcher
    launcher = FlyIOGPULauncher()
    
    # Configure uvicorn for production
    config = uvicorn.Config(
        app=launcher.app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        workers=1,  # Single worker for GPU usage
        loop="asyncio",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    
    try:
        logger.info("Starting server on port 8080...")
        await server.serve()
    except Exception as e:
        logger.error(f"Server failed: {e}")
        raise
    finally:
        await launcher.cleanup()

if __name__ == "__main__":
    asyncio.run(main())