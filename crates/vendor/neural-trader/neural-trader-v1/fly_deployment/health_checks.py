"""
Comprehensive Health Check System for GPU Trading Platform
Provides detailed health monitoring for all system components
"""

import asyncio
import json
import logging
import os
import psutil
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import aiohttp
import GPUtil
import torch
import redis
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: str
    duration_ms: float


class SystemHealthChecker:
    """Comprehensive system health monitoring"""
    
    def __init__(self):
        self.redis_client = None
        self.db_engine = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize external service clients"""
        try:
            # Redis connection
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Database connection
            db_url = os.getenv('DATABASE_URL', 'sqlite:///app/data/trading.db')
            self.db_engine = create_engine(db_url)
            
        except Exception as e:
            logger.error(f"Failed to initialize health check clients: {e}")
    
    async def check_gpu_health(self) -> HealthCheck:
        """Check GPU availability and utilization"""
        start_time = time.time()
        
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                return HealthCheck(
                    name="gpu",
                    status=HealthStatus.CRITICAL,
                    message="CUDA not available",
                    details={"cuda_available": False},
                    timestamp=datetime.utcnow().isoformat(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Get GPU information
            gpus = GPUtil.getGPUs()
            if not gpus:
                return HealthCheck(
                    name="gpu",
                    status=HealthStatus.CRITICAL,
                    message="No GPUs detected",
                    details={"gpu_count": 0},
                    timestamp=datetime.utcnow().isoformat(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            gpu = gpus[0]  # Primary GPU
            
            # Check GPU utilization and temperature
            gpu_details = {
                "gpu_count": len(gpus),
                "gpu_name": gpu.name,
                "gpu_utilization": gpu.load * 100,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_free": gpu.memoryFree,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                "temperature": gpu.temperature,
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__
            }
            
            # Determine status based on metrics
            status = HealthStatus.HEALTHY
            message = "GPU operating normally"
            
            if gpu.temperature > 85:
                status = HealthStatus.DEGRADED
                message = f"GPU temperature high: {gpu.temperature}°C"
            elif gpu.temperature > 90:
                status = HealthStatus.UNHEALTHY
                message = f"GPU temperature critical: {gpu.temperature}°C"
            
            if gpu.load > 0.95:
                status = HealthStatus.DEGRADED
                message = f"GPU utilization very high: {gpu.load * 100}%"
                
            if (gpu.memoryUsed / gpu.memoryTotal) > 0.9:
                status = HealthStatus.DEGRADED 
                message = f"GPU memory usage high: {gpu_details['memory_percent']:.1f}%"
            
            return HealthCheck(
                name="gpu",
                status=status,
                message=message,
                details=gpu_details,
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return HealthCheck(
                name="gpu",
                status=HealthStatus.CRITICAL,
                message=f"GPU health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def check_system_resources(self) -> HealthCheck:
        """Check system CPU, memory, and disk usage"""
        start_time = time.time()
        
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            # Memory information
            memory = psutil.virtual_memory()
            
            # Disk information
            disk = psutil.disk_usage('/')
            
            details = {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "load_average": {
                    "1min": load_avg[0],
                    "5min": load_avg[1],
                    "15min": load_avg[2]
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "percent": (disk.used / disk.total) * 100
                }
            }
            
            # Determine status
            status = HealthStatus.HEALTHY
            message = "System resources normal"
            
            if cpu_percent > 90:
                status = HealthStatus.DEGRADED
                message = f"High CPU usage: {cpu_percent}%"
            elif memory.percent > 90:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory.percent}%"
            elif details["disk"]["percent"] > 90:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {details['disk']['percent']:.1f}%"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"System resource check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def check_redis_connection(self) -> HealthCheck:
        """Check Redis connectivity and performance"""
        start_time = time.time()
        
        try:
            if not self.redis_client:
                return HealthCheck(
                    name="redis",
                    status=HealthStatus.CRITICAL,
                    message="Redis client not initialized",
                    details={"error": "Client not initialized"},
                    timestamp=datetime.utcnow().isoformat(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Test basic connectivity
            pong = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            
            if not pong:
                return HealthCheck(
                    name="redis",
                    status=HealthStatus.UNHEALTHY,
                    message="Redis ping failed",
                    details={"ping_response": pong},
                    timestamp=datetime.utcnow().isoformat(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Get Redis info
            info = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.info
            )
            
            details = {
                "connected": True,
                "version": info.get('redis_version', 'unknown'),
                "uptime_seconds": info.get('uptime_in_seconds', 0),
                "connected_clients": info.get('connected_clients', 0),
                "used_memory_human": info.get('used_memory_human', 'unknown'),
                "memory_usage_percent": info.get('used_memory_rss', 0) / info.get('maxmemory', 1) * 100 if info.get('maxmemory', 0) > 0 else 0
            }
            
            # Test write/read performance
            test_key = "health_check_test"
            test_value = f"test_{int(time.time())}"
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.set, test_key, test_value, 60
            )
            
            retrieved_value = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, test_key
            )
            
            if retrieved_value != test_value:
                return HealthCheck(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    message="Redis read/write test failed",
                    details={**details, "test_failed": True},
                    timestamp=datetime.utcnow().isoformat(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Clean up test key
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.delete, test_key
            )
            
            return HealthCheck(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection healthy",
                details=details,
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return HealthCheck(
                name="redis",
                status=HealthStatus.CRITICAL,
                message=f"Redis health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def check_database_connection(self) -> HealthCheck:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            if not self.db_engine:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.CRITICAL,
                    message="Database engine not initialized",
                    details={"error": "Engine not initialized"},
                    timestamp=datetime.utcnow().isoformat(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Test connection with a simple query
            with self.db_engine.connect() as connection:
                result = connection.execute(text("SELECT 1 as test"))
                test_result = result.fetchone()
                
                if test_result[0] != 1:
                    return HealthCheck(
                        name="database",
                        status=HealthStatus.UNHEALTHY,
                        message="Database test query failed",
                        details={"test_result": test_result[0]},
                        timestamp=datetime.utcnow().isoformat(),
                        duration_ms=(time.time() - start_time) * 1000
                    )
            
            # Get database information
            details = {
                "connected": True,
                "engine_url": str(self.db_engine.url).split('@')[0] + '@***',  # Hide credentials
                "pool_size": self.db_engine.pool.size() if hasattr(self.db_engine.pool, 'size') else 'unknown',
                "checked_in": self.db_engine.pool.checkedin() if hasattr(self.db_engine.pool, 'checkedin') else 'unknown'
            }
            
            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection healthy",
                details=details,
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.CRITICAL,
                message=f"Database health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def check_trading_services(self) -> HealthCheck:
        """Check trading-specific services and data feeds"""
        start_time = time.time()
        
        try:
            details = {
                "api_key_configured": bool(os.getenv('API_KEY')),
                "trading_mode": os.getenv('TRADING_MODE', 'unknown'),
                "max_concurrent_trades": os.getenv('MAX_CONCURRENT_TRADES', 'unknown'),
                "risk_multiplier": os.getenv('RISK_MULTIPLIER', 'unknown')
            }
            
            # Check if required environment variables are set
            required_vars = ['API_KEY', 'TRADING_MODE']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                return HealthCheck(
                    name="trading_services",
                    status=HealthStatus.CRITICAL,
                    message=f"Missing required environment variables: {', '.join(missing_vars)}",
                    details={**details, "missing_vars": missing_vars},
                    timestamp=datetime.utcnow().isoformat(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Test API connectivity (mock for now)
            # In production, this would test actual trading API endpoints
            api_status = "healthy"  # This would be a real API test
            
            details.update({
                "api_connectivity": api_status,
                "last_data_update": datetime.utcnow().isoformat(),  # Would be real timestamp
            })
            
            return HealthCheck(
                name="trading_services",
                status=HealthStatus.HEALTHY,
                message="Trading services operational",
                details=details,
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return HealthCheck(
                name="trading_services",
                status=HealthStatus.CRITICAL,
                message=f"Trading services check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report"""
        start_time = time.time()
        
        # Run all checks concurrently
        checks = await asyncio.gather(
            self.check_gpu_health(),
            self.check_system_resources(),
            self.check_redis_connection(),
            self.check_database_connection(),
            self.check_trading_services(),
            return_exceptions=True
        )
        
        # Process results
        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": HealthStatus.HEALTHY.value,
            "total_duration_ms": (time.time() - start_time) * 1000,
            "checks": {}
        }
        
        overall_status = HealthStatus.HEALTHY
        
        for check in checks:
            if isinstance(check, Exception):
                # Handle exceptions
                check_name = "unknown"
                health_report["checks"][check_name] = {
                    "status": HealthStatus.CRITICAL.value,
                    "message": f"Health check exception: {str(check)}",
                    "details": {"exception": str(check)},
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration_ms": 0
                }
                overall_status = HealthStatus.CRITICAL
            else:
                # Normal health check result
                health_report["checks"][check.name] = asdict(check)
                
                # Update overall status (take the worst status)
                if check.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif check.status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.UNHEALTHY
                elif check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        health_report["overall_status"] = overall_status.value
        
        # Add summary statistics
        health_report["summary"] = {
            "total_checks": len(health_report["checks"]),
            "healthy_checks": len([c for c in health_report["checks"].values() if c["status"] == HealthStatus.HEALTHY.value]),
            "degraded_checks": len([c for c in health_report["checks"].values() if c["status"] == HealthStatus.DEGRADED.value]),
            "unhealthy_checks": len([c for c in health_report["checks"].values() if c["status"] == HealthStatus.UNHEALTHY.value]),
            "critical_checks": len([c for c in health_report["checks"].values() if c["status"] == HealthStatus.CRITICAL.value]),
        }
        
        return health_report


# FastAPI/Flask endpoints for health checks
def create_health_endpoints(app_framework="fastapi"):
    """Create health check endpoints for web framework"""
    
    health_checker = SystemHealthChecker()
    
    if app_framework == "fastapi":
        from fastapi import FastAPI, Response
        from fastapi.responses import JSONResponse
        
        def add_endpoints(app: FastAPI):
            @app.get("/health")
            async def basic_health_check():
                """Basic health check endpoint"""
                return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
            
            @app.get("/health/detailed")
            async def detailed_health_check():
                """Detailed health check with all components"""
                report = await health_checker.run_all_checks()
                
                # Set HTTP status based on overall health
                status_code = 200
                if report["overall_status"] == HealthStatus.DEGRADED.value:
                    status_code = 200  # Still OK, but with warnings
                elif report["overall_status"] == HealthStatus.UNHEALTHY.value:
                    status_code = 503  # Service Unavailable
                elif report["overall_status"] == HealthStatus.CRITICAL.value:
                    status_code = 503  # Service Unavailable
                
                return JSONResponse(content=report, status_code=status_code)
            
            @app.get("/gpu-status")
            async def gpu_status():
                """GPU-specific health check for fly.io"""
                gpu_check = await health_checker.check_gpu_health()
                
                status_code = 200 if gpu_check.status == HealthStatus.HEALTHY else 503
                return JSONResponse(content=asdict(gpu_check), status_code=status_code)
            
            @app.get("/metrics")
            async def prometheus_metrics():
                """Prometheus-compatible metrics endpoint"""
                report = await health_checker.run_all_checks()
                
                # Convert health report to Prometheus format
                metrics = []
                
                # Overall health metric
                overall_status_value = {
                    HealthStatus.HEALTHY.value: 1,
                    HealthStatus.DEGRADED.value: 0.75,
                    HealthStatus.UNHEALTHY.value: 0.5,
                    HealthStatus.CRITICAL.value: 0
                }.get(report["overall_status"], 0)
                
                metrics.append(f'health_overall_status {overall_status_value}')
                
                # Individual check metrics
                for check_name, check_data in report["checks"].items():
                    status_value = {
                        HealthStatus.HEALTHY.value: 1,
                        HealthStatus.DEGRADED.value: 0.75,
                        HealthStatus.UNHEALTHY.value: 0.5,
                        HealthStatus.CRITICAL.value: 0
                    }.get(check_data["status"], 0)
                    
                    metrics.append(f'health_check_status{{check="{check_name}"}} {status_value}')
                    metrics.append(f'health_check_duration_ms{{check="{check_name}"}} {check_data["duration_ms"]}')
                
                # Add specific metrics from details
                if "gpu" in report["checks"] and "details" in report["checks"]["gpu"]:
                    gpu_details = report["checks"]["gpu"]["details"]
                    if "gpu_utilization" in gpu_details:
                        metrics.append(f'gpu_utilization_percent {gpu_details["gpu_utilization"]}')
                    if "memory_percent" in gpu_details:
                        metrics.append(f'gpu_memory_usage_percent {gpu_details["memory_percent"]}')
                    if "temperature" in gpu_details:
                        metrics.append(f'gpu_temperature_celsius {gpu_details["temperature"]}')
                
                return Response(content="\n".join(metrics), media_type="text/plain")
        
        return add_endpoints
    
    else:
        # Flask implementation
        from flask import Flask, jsonify, Response
        
        def add_endpoints(app: Flask):
            @app.route("/health")
            def basic_health_check():
                return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})
            
            @app.route("/health/detailed")
            async def detailed_health_check():
                report = await health_checker.run_all_checks()
                
                status_code = 200
                if report["overall_status"] in [HealthStatus.UNHEALTHY.value, HealthStatus.CRITICAL.value]:
                    status_code = 503
                
                return jsonify(report), status_code
            
            @app.route("/gpu-status")
            async def gpu_status():
                gpu_check = await health_checker.check_gpu_health()
                status_code = 200 if gpu_check.status == HealthStatus.HEALTHY else 503
                return jsonify(asdict(gpu_check)), status_code
        
        return add_endpoints


if __name__ == "__main__":
    # CLI tool for running health checks
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="GPU Trading Platform Health Checker")
        parser.add_argument("--check", choices=["all", "gpu", "system", "redis", "database", "trading"], 
                          default="all", help="Specific health check to run")
        parser.add_argument("--format", choices=["json", "pretty"], default="pretty", 
                          help="Output format")
        
        args = parser.parse_args()
        
        health_checker = SystemHealthChecker()
        
        if args.check == "all":
            result = await health_checker.run_all_checks()
        elif args.check == "gpu":
            result = asdict(await health_checker.check_gpu_health())
        elif args.check == "system":
            result = asdict(await health_checker.check_system_resources())
        elif args.check == "redis":
            result = asdict(await health_checker.check_redis_connection())
        elif args.check == "database":
            result = asdict(await health_checker.check_database_connection())
        elif args.check == "trading":
            result = asdict(await health_checker.check_trading_services())
        
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            # Pretty print
            if args.check == "all":
                print(f"Overall Status: {result['overall_status'].upper()}")
                print(f"Total Duration: {result['total_duration_ms']:.2f}ms")
                print("\nIndividual Checks:")
                for check_name, check_data in result["checks"].items():
                    status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌", "critical": "🚨"}.get(check_data["status"], "❓")
                    print(f"  {status_emoji} {check_name}: {check_data['status']} - {check_data['message']}")
            else:
                status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌", "critical": "🚨"}.get(result["status"], "❓")
                print(f"{status_emoji} {result['name']}: {result['status']} - {result['message']}")
                if result.get("details"):
                    print(f"Details: {json.dumps(result['details'], indent=2)}")
    
    asyncio.run(main())