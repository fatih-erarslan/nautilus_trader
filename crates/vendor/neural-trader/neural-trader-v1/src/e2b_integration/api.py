"""
FastAPI endpoints for E2B integration
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
import asyncio
import logging

from .sandbox_manager import SandboxManager
from .agent_runner import AgentRunner
from .process_executor import ProcessExecutor
from .models import (
    SandboxConfig,
    AgentConfig,
    ProcessConfig,
    SandboxInfo,
    AgentResult,
    ProcessResult,
    SandboxStatus
)
from ..e2b_templates import template_router

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/e2b", tags=["E2B Integration"])

# Initialize managers
sandbox_manager = SandboxManager()
agent_runner = AgentRunner(sandbox_manager)
process_executor = ProcessExecutor(sandbox_manager)


# Sandbox Management Endpoints

@router.post("/sandbox/create")
async def create_sandbox(config: SandboxConfig) -> Dict[str, Any]:
    """Create a new E2B sandbox"""
    try:
        sandbox_id = sandbox_manager.create_sandbox(config)
        return {
            "status": "success",
            "sandbox_id": sandbox_id,
            "message": f"Sandbox '{config.name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create sandbox: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sandbox/list")
async def list_sandboxes() -> List[SandboxInfo]:
    """List all active sandboxes"""
    return sandbox_manager.list_sandboxes()


@router.get("/sandbox/{sandbox_id}")
async def get_sandbox_info(sandbox_id: str) -> SandboxInfo:
    """Get information about a specific sandbox"""
    info = sandbox_manager.get_sandbox_info(sandbox_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Sandbox {sandbox_id} not found")
    return info


@router.delete("/sandbox/{sandbox_id}")
async def terminate_sandbox(sandbox_id: str) -> Dict[str, Any]:
    """Terminate a sandbox"""
    success = sandbox_manager.terminate_sandbox(sandbox_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Sandbox {sandbox_id} not found")
    
    return {
        "status": "success",
        "message": f"Sandbox {sandbox_id} terminated"
    }


@router.post("/sandbox/{sandbox_id}/execute")
async def execute_command(sandbox_id: str, command: str, timeout: int = 30) -> ProcessResult:
    """Execute a command in a sandbox"""
    try:
        result = sandbox_manager.execute_command(sandbox_id, command, timeout)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sandbox/{sandbox_id}/upload")
async def upload_file(sandbox_id: str, local_path: str, sandbox_path: str) -> Dict[str, Any]:
    """Upload a file to a sandbox"""
    try:
        success = sandbox_manager.upload_file(sandbox_id, local_path, sandbox_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to upload file")
        
        return {
            "status": "success",
            "message": f"File uploaded to {sandbox_path}"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sandbox/{sandbox_id}/download")
async def download_file(sandbox_id: str, sandbox_path: str) -> Dict[str, Any]:
    """Download a file from a sandbox"""
    try:
        content = sandbox_manager.download_file(sandbox_id, sandbox_path)
        if content is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {
            "status": "success",
            "content": content
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Agent Execution Endpoints

@router.post("/agent/run")
async def run_agent(config: AgentConfig) -> AgentResult:
    """Run a trading agent in a sandbox"""
    try:
        if not agent_runner.validate_agent_config(config):
            raise HTTPException(status_code=400, detail="Invalid agent configuration")
        
        result = await agent_runner.run_agent(config)
        return result
    except Exception as e:
        logger.error(f"Failed to run agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/run-batch")
async def run_multiple_agents(configs: List[AgentConfig]) -> List[AgentResult]:
    """Run multiple agents concurrently"""
    try:
        for config in configs:
            if not agent_runner.validate_agent_config(config):
                raise HTTPException(status_code=400, detail=f"Invalid config for {config.agent_type}")
        
        results = await agent_runner.run_multiple_agents(configs)
        return results
    except Exception as e:
        logger.error(f"Failed to run agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/types")
async def get_agent_types() -> List[str]:
    """Get available agent types"""
    return agent_runner.get_agent_types()


# Process Execution Endpoints

@router.post("/process/execute")
async def execute_process(config: ProcessConfig, sandbox_id: Optional[str] = None) -> ProcessResult:
    """Execute a process in a sandbox"""
    try:
        result = await process_executor.execute_process(config, sandbox_id)
        return result
    except Exception as e:
        logger.error(f"Failed to execute process: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/batch")
async def execute_batch_processes(configs: List[ProcessConfig], parallel: bool = True) -> List[ProcessResult]:
    """Execute multiple processes"""
    try:
        results = await process_executor.execute_batch(configs, parallel)
        return results
    except Exception as e:
        logger.error(f"Failed to execute batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/pipeline")
async def execute_pipeline(configs: List[ProcessConfig], sandbox_id: Optional[str] = None) -> List[ProcessResult]:
    """Execute a pipeline of processes"""
    try:
        results = await process_executor.execute_pipeline(configs, sandbox_id)
        return results
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/background")
async def run_background_process(config: ProcessConfig, sandbox_id: Optional[str] = None) -> Dict[str, Any]:
    """Start a background process"""
    try:
        process_id = await process_executor.run_background_process(config, sandbox_id)
        return {
            "status": "success",
            "process_id": process_id,
            "message": "Background process started"
        }
    except Exception as e:
        logger.error(f"Failed to start background process: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/process/{process_id}/status")
async def get_process_status(process_id: str) -> Dict[str, Any]:
    """Get status of a process"""
    status = process_executor.get_process_status(process_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found")
    return status


@router.delete("/process/{process_id}")
async def kill_process(process_id: str) -> Dict[str, Any]:
    """Kill a running process"""
    success = process_executor.kill_process(process_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found")
    
    return {
        "status": "success",
        "message": f"Process {process_id} killed"
    }


@router.get("/process/list")
async def list_active_processes() -> List[Dict[str, Any]]:
    """List all active processes"""
    return process_executor.list_active_processes()


from pydantic import BaseModel

class ScriptExecutionRequest(BaseModel):
    script_content: str
    language: str = "python"
    sandbox_id: Optional[str] = None

@router.post("/script/execute")
async def execute_script(request: ScriptExecutionRequest) -> ProcessResult:
    """Execute a script in a sandbox"""
    try:
        result = await process_executor.execute_script(request.script_content, request.language, request.sandbox_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute script: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility Endpoints

@router.delete("/cleanup")
async def cleanup_all_sandboxes() -> Dict[str, Any]:
    """Clean up all sandboxes"""
    sandbox_manager.cleanup_all()
    return {
        "status": "success",
        "message": "All sandboxes cleaned up"
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    active_sandboxes = len(sandbox_manager.sandboxes)
    active_processes = len(process_executor.active_processes)
    
    return {
        "status": "healthy",
        "active_sandboxes": active_sandboxes,
        "active_processes": active_processes,
        "max_sandboxes": sandbox_manager.max_sandboxes
    }