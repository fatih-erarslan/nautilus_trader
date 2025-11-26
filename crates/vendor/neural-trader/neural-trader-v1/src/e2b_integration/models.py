"""
Data models for E2B integration
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class SandboxStatus(str, Enum):
    """Sandbox lifecycle states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PROCESSING = "processing"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentType(str, Enum):
    """Types of agents that can run in sandboxes"""
    MOMENTUM_TRADER = "momentum_trader"
    MEAN_REVERSION_TRADER = "mean_reversion_trader"
    SWING_TRADER = "swing_trader"
    MIRROR_TRADER = "mirror_trader"
    NEURAL_FORECASTER = "neural_forecaster"
    NEWS_ANALYZER = "news_analyzer"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_OPTIMIZER = "portfolio_optimizer"
    CUSTOM = "custom"


class SandboxConfig(BaseModel):
    """Configuration for E2B sandbox"""
    name: str = Field(..., description="Sandbox name")
    template: str = Field(default="base", description="E2B template to use")
    timeout: int = Field(default=300, description="Timeout in seconds")
    memory_mb: int = Field(default=512, description="Memory allocation in MB")
    cpu_count: int = Field(default=1, description="Number of CPUs")
    allow_internet: bool = Field(default=True, description="Allow internet access")
    envs: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class AgentConfig(BaseModel):
    """Configuration for agent execution"""
    agent_type: AgentType
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    symbols: List[str] = Field(default_factory=list)
    data_source: Optional[str] = None
    risk_limits: Optional[Dict[str, float]] = None
    execution_mode: str = Field(default="simulation", description="simulation or live")
    use_gpu: bool = Field(default=False, description="Request GPU acceleration")


class ProcessConfig(BaseModel):
    """Configuration for process execution"""
    command: str = Field(..., description="Command to execute")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    working_dir: str = Field(default="/workspace", description="Working directory")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    capture_output: bool = Field(default=True, description="Capture stdout/stderr")
    timeout: Optional[int] = Field(default=60, description="Process timeout in seconds")


class ProcessResult(BaseModel):
    """Result from process execution"""
    sandbox_id: str
    process_id: Optional[str] = None
    command: str
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Result from agent execution"""
    sandbox_id: str
    agent_type: AgentType
    status: str
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    performance: Dict[str, float] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    started_at: datetime
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SandboxInfo(BaseModel):
    """Information about a sandbox instance"""
    sandbox_id: str
    name: str
    status: SandboxStatus
    created_at: datetime
    last_activity: datetime
    processes: List[str] = Field(default_factory=list)
    agents: List[str] = Field(default_factory=list)
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)