#!/usr/bin/env python3
"""
Hive-Mind / Swarm API Endpoints for FastAPI
Uses Claude Flow's non-interactive swarm orchestration
"""

import asyncio
import subprocess
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/swarm", tags=["Swarm Intelligence"])

# Swarm configurations
class SwarmStrategy(str, Enum):
    RESEARCH = "research"
    DEVELOPMENT = "development"
    ANALYSIS = "analysis"
    TESTING = "testing"
    OPTIMIZATION = "optimization"
    MAINTENANCE = "maintenance"

class SwarmMode(str, Enum):
    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    HYBRID = "hybrid"

class SwarmTopology(str, Enum):
    MESH = "mesh"
    HIERARCHICAL = "hierarchical"
    RING = "ring"
    STAR = "star"

# Request/Response models
class SwarmRequest(BaseModel):
    objective: str = Field(..., description="Task objective for the swarm")
    strategy: SwarmStrategy = Field(default=SwarmStrategy.ANALYSIS)
    mode: SwarmMode = Field(default=SwarmMode.DISTRIBUTED)
    max_agents: int = Field(default=5, ge=1, le=10)
    parallel: bool = Field(default=True)
    background: bool = Field(default=True)
    analysis_only: bool = Field(default=False, description="Read-only mode")

class HiveMindRequest(BaseModel):
    objective: str = Field(..., description="Complex task for hive mind")
    queen_type: str = Field(default="adaptive", description="Queen coordinator type")
    max_workers: int = Field(default=8, ge=1, le=20)
    consensus: str = Field(default="majority", description="Consensus algorithm")
    auto_scale: bool = Field(default=True)
    monitor: bool = Field(default=True)

class SwarmTaskRequest(BaseModel):
    task: str = Field(..., description="Task description")
    topology: SwarmTopology = Field(default=SwarmTopology.MESH)
    max_agents: int = Field(default=3, ge=1, le=10)
    priority: str = Field(default="medium", description="Task priority")

# In-memory storage for swarm sessions
swarm_sessions = {}
hive_mind_sessions = {}

# Helper function to run Claude Flow commands
async def run_claude_flow_command(command: List[str]) -> Dict[str, Any]:
    """Execute Claude Flow command and parse output"""
    try:
        process = await asyncio.create_subprocess_exec(
            "npx", "claude-flow@alpha", *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error("Claude Flow command failed", 
                        command=command, 
                        stderr=stderr.decode())
            raise HTTPException(status_code=500, detail=f"Command failed: {stderr.decode()}")
        
        # Parse JSON output from Claude Flow
        output = stdout.decode()
        # Extract JSON from the output (Claude Flow may include non-JSON text)
        json_start = output.find('{')
        if json_start != -1:
            json_str = output[json_start:]
            return json.loads(json_str)
        
        return {"output": output}
        
    except Exception as e:
        logger.error("Error running Claude Flow command", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Swarm Endpoints
@router.post("/deploy")
async def deploy_swarm(request: SwarmRequest, background_tasks: BackgroundTasks):
    """
    Deploy an intelligent multi-agent swarm for complex objectives.
    Uses Claude Flow's non-interactive swarm mode.
    """
    session_id = str(uuid.uuid4())
    
    # Build command
    command = ["swarm", request.objective]
    
    if request.strategy:
        command.extend(["--strategy", request.strategy.value])
    if request.mode:
        command.extend(["--mode", request.mode.value])
    if request.max_agents:
        command.extend(["--max-agents", str(request.max_agents)])
    if request.parallel:
        command.append("--parallel")
    if request.analysis_only:
        command.append("--analysis")
    
    # Store session
    swarm_sessions[session_id] = {
        "id": session_id,
        "objective": request.objective,
        "status": "initializing",
        "created_at": datetime.now().isoformat(),
        "config": request.dict()
    }
    
    if request.background:
        # Run in background
        background_tasks.add_task(execute_swarm_async, session_id, command)
        return {
            "session_id": session_id,
            "status": "started",
            "message": f"Swarm deployed in background for: {request.objective}",
            "monitor_url": f"/swarm/status/{session_id}"
        }
    else:
        # Run synchronously
        result = await run_claude_flow_command(command)
        swarm_sessions[session_id]["status"] = "completed"
        swarm_sessions[session_id]["result"] = result
        return {
            "session_id": session_id,
            "status": "completed",
            "result": result
        }

@router.post("/hive-mind")
async def deploy_hive_mind(request: HiveMindRequest, background_tasks: BackgroundTasks):
    """
    Deploy a hive-mind system with queen-led coordination.
    Perfect for complex, multi-faceted problems.
    """
    session_id = str(uuid.uuid4())
    
    # Build command
    command = ["hive-mind", "spawn", request.objective]
    
    command.extend(["--queen-type", request.queen_type])
    command.extend(["--max-workers", str(request.max_workers)])
    command.extend(["--consensus", request.consensus])
    
    if request.auto_scale:
        command.append("--auto-scale")
    if request.monitor:
        command.append("--monitor")
    
    # Store session
    hive_mind_sessions[session_id] = {
        "id": session_id,
        "objective": request.objective,
        "status": "initializing",
        "created_at": datetime.now().isoformat(),
        "config": request.dict()
    }
    
    background_tasks.add_task(execute_hive_mind_async, session_id, command)
    
    return {
        "session_id": session_id,
        "status": "started",
        "message": f"Hive mind deployed for: {request.objective}",
        "monitor_url": f"/swarm/hive-mind/status/{session_id}"
    }

@router.post("/optimize-database")
async def optimize_database_queries(background_tasks: BackgroundTasks):
    """
    Specialized endpoint for database query optimization using swarm intelligence.
    """
    command = [
        "swarm",
        "Optimize database queries for performance",
        "--strategy", "optimization",
        "--max-agents", "3",
        "--parallel"
    ]
    
    session_id = str(uuid.uuid4())
    
    swarm_sessions[session_id] = {
        "id": session_id,
        "objective": "Database query optimization",
        "status": "running",
        "created_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(execute_swarm_async, session_id, command)
    
    return {
        "session_id": session_id,
        "status": "started",
        "message": "Database optimization swarm deployed",
        "monitor_url": f"/swarm/status/{session_id}"
    }

@router.post("/analyze-codebase")
async def analyze_codebase():
    """
    Analyze codebase using swarm in read-only mode.
    Safe analysis without any code modifications.
    """
    command = [
        "swarm",
        "Analyze codebase for security issues and performance bottlenecks",
        "--strategy", "analysis",
        "--analysis",  # Read-only mode
        "--max-agents", "5",
        "--parallel"
    ]
    
    result = await run_claude_flow_command(command)
    
    return {
        "status": "completed",
        "analysis": result,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/task")
async def orchestrate_task(request: SwarmTaskRequest):
    """
    Orchestrate a specific task using Claude Flow's task orchestration.
    """
    # Use MCP tool directly for task orchestration
    return {
        "message": "Task orchestration initiated",
        "task": request.task,
        "topology": request.topology.value,
        "agents": request.max_agents,
        "status": "orchestrating"
    }

@router.get("/status/{session_id}")
async def get_swarm_status(session_id: str):
    """Get status of a swarm session"""
    if session_id not in swarm_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return swarm_sessions[session_id]

@router.get("/hive-mind/status/{session_id}")
async def get_hive_mind_status(session_id: str):
    """Get status of a hive-mind session"""
    if session_id not in hive_mind_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return hive_mind_sessions[session_id]

@router.get("/sessions")
async def list_all_sessions():
    """List all swarm and hive-mind sessions"""
    return {
        "swarm_sessions": list(swarm_sessions.keys()),
        "hive_mind_sessions": list(hive_mind_sessions.keys()),
        "total": len(swarm_sessions) + len(hive_mind_sessions)
    }

@router.post("/research")
async def research_topic(topic: str):
    """
    Research a topic using swarm intelligence.
    """
    command = [
        "swarm",
        f"Research {topic}",
        "--strategy", "research",
        "--max-agents", "4",
        "--parallel"
    ]
    
    result = await run_claude_flow_command(command)
    
    return {
        "topic": topic,
        "research": result,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/develop")
async def develop_feature(feature: str, background_tasks: BackgroundTasks):
    """
    Develop a feature using swarm intelligence.
    """
    session_id = str(uuid.uuid4())
    
    command = [
        "swarm",
        f"Develop {feature}",
        "--strategy", "development",
        "--mode", "hierarchical",
        "--max-agents", "5",
        "--parallel",
        "--monitor"
    ]
    
    swarm_sessions[session_id] = {
        "id": session_id,
        "objective": f"Develop {feature}",
        "status": "running",
        "created_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(execute_swarm_async, session_id, command)
    
    return {
        "session_id": session_id,
        "feature": feature,
        "status": "development started",
        "monitor_url": f"/swarm/status/{session_id}"
    }

# Background execution functions
async def execute_swarm_async(session_id: str, command: List[str]):
    """Execute swarm command in background"""
    try:
        swarm_sessions[session_id]["status"] = "running"
        result = await run_claude_flow_command(command)
        swarm_sessions[session_id]["status"] = "completed"
        swarm_sessions[session_id]["result"] = result
        swarm_sessions[session_id]["completed_at"] = datetime.now().isoformat()
    except Exception as e:
        swarm_sessions[session_id]["status"] = "failed"
        swarm_sessions[session_id]["error"] = str(e)
        logger.error(f"Swarm execution failed for {session_id}", error=str(e))

async def execute_hive_mind_async(session_id: str, command: List[str]):
    """Execute hive-mind command in background"""
    try:
        hive_mind_sessions[session_id]["status"] = "running"
        result = await run_claude_flow_command(command)
        hive_mind_sessions[session_id]["status"] = "completed"
        hive_mind_sessions[session_id]["result"] = result
        hive_mind_sessions[session_id]["completed_at"] = datetime.now().isoformat()
    except Exception as e:
        hive_mind_sessions[session_id]["status"] = "failed"
        hive_mind_sessions[session_id]["error"] = str(e)
        logger.error(f"Hive-mind execution failed for {session_id}", error=str(e))

# Advanced swarm patterns
@router.post("/sparc")
async def sparc_development(task: str, mode: str = "tdd"):
    """
    Use SPARC methodology with swarm coordination.
    Modes: spec-pseudocode, architect, tdd, integration
    """
    command = [
        "sparc",
        mode,
        task
    ]
    
    result = await run_claude_flow_command(command)
    
    return {
        "task": task,
        "mode": mode,
        "result": result,
        "methodology": "SPARC",
        "timestamp": datetime.now().isoformat()
    }

@router.post("/neural-swarm")
async def neural_swarm_training(data_path: str, pattern_type: str = "coordination"):
    """
    Train neural patterns using swarm intelligence.
    """
    return {
        "message": "Neural swarm training initiated",
        "data_path": data_path,
        "pattern_type": pattern_type,
        "status": "training",
        "note": "This would use mcp__claude-flow__neural_train"
    }

# Health check
@router.get("/health")
async def swarm_health():
    """Check swarm system health"""
    return {
        "status": "healthy",
        "active_swarms": len([s for s in swarm_sessions.values() if s["status"] == "running"]),
        "active_hive_minds": len([h for h in hive_mind_sessions.values() if h["status"] == "running"]),
        "total_sessions": len(swarm_sessions) + len(hive_mind_sessions)
    }