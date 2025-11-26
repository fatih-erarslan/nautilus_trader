"""
Data models for E2B template system
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class TemplateType(str, Enum):
    """Types of E2B templates"""
    # Trading Agent Templates
    TRADING_AGENT = "trading_agent"
    MOMENTUM_TRADER = "momentum_trader"
    MEAN_REVERSION = "mean_reversion"
    NEURAL_TRADER = "neural_trader"
    ARBITRAGE_BOT = "arbitrage_bot"
    PORTFOLIO_MANAGER = "portfolio_manager"
    
    # Claude-Flow Templates
    CLAUDE_FLOW_SWARM = "claude_flow_swarm"
    CLAUDE_FLOW_AGENT = "claude_flow_agent"
    CLAUDE_FLOW_ORCHESTRATOR = "claude_flow_orchestrator"
    CLAUDE_FLOW_MEMORY = "claude_flow_memory"
    
    # Claude Code Templates
    CLAUDE_CODE_DEVELOPER = "claude_code_developer"
    CLAUDE_CODE_REVIEWER = "claude_code_reviewer"
    CLAUDE_CODE_TESTER = "claude_code_tester"
    CLAUDE_CODE_SPARC = "claude_code_sparc"
    
    # Specialized Templates
    DATA_ANALYZER = "data_analyzer"
    ML_TRAINER = "ml_trainer"
    BACKTESTER = "backtester"
    RISK_ANALYZER = "risk_analyzer"
    NEWS_PROCESSOR = "news_processor"
    
    # Base Templates
    PYTHON_BASE = "python_base"
    NODE_BASE = "node_base"
    CUSTOM = "custom"


class RuntimeEnvironment(str, Enum):
    """Runtime environments for templates"""
    PYTHON_3_9 = "python3.9"
    PYTHON_3_10 = "python3.10"
    PYTHON_3_11 = "python3.11"
    NODE_18 = "node18"
    NODE_20 = "node20"
    DENO = "deno"
    RUST = "rust"


class TemplateRequirements(BaseModel):
    """Requirements for a template"""
    runtime: RuntimeEnvironment = Field(default=RuntimeEnvironment.PYTHON_3_10)
    cpu_cores: int = Field(default=1, ge=1, le=8)
    memory_mb: int = Field(default=512, ge=256, le=8192)
    storage_gb: int = Field(default=1, ge=1, le=100)
    gpu_enabled: bool = Field(default=False)
    network_access: bool = Field(default=True)
    persistent_storage: bool = Field(default=False)
    
    # Package dependencies
    python_packages: List[str] = Field(default_factory=list)
    node_packages: List[str] = Field(default_factory=list)
    system_packages: List[str] = Field(default_factory=list)
    
    # Special requirements
    api_keys: List[str] = Field(default_factory=list, description="Required API keys")
    env_vars: Dict[str, str] = Field(default_factory=dict)
    ports: List[int] = Field(default_factory=list)


class TemplateFiles(BaseModel):
    """Files to include in template"""
    main_script: str = Field(..., description="Main entry point script")
    modules: Dict[str, str] = Field(default_factory=dict, description="Additional modules")
    configs: Dict[str, str] = Field(default_factory=dict, description="Configuration files")
    data: Dict[str, str] = Field(default_factory=dict, description="Data files")
    scripts: Dict[str, str] = Field(default_factory=dict, description="Utility scripts")


class TemplateMetadata(BaseModel):
    """Metadata for a template"""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    version: str = Field(default="1.0.0")
    author: str = Field(default="AI News Trader")
    tags: List[str] = Field(default_factory=list)
    category: str = Field(default="general")
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    license: str = Field(default="MIT")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TemplateHooks(BaseModel):
    """Lifecycle hooks for template"""
    pre_install: Optional[str] = Field(None, description="Script to run before installation")
    post_install: Optional[str] = Field(None, description="Script to run after installation")
    pre_start: Optional[str] = Field(None, description="Script to run before starting")
    post_start: Optional[str] = Field(None, description="Script to run after starting")
    health_check: Optional[str] = Field(None, description="Health check script")
    cleanup: Optional[str] = Field(None, description="Cleanup script")


class ClaudeFlowConfig(BaseModel):
    """Configuration for Claude-Flow integration"""
    swarm_topology: str = Field(default="mesh", description="Swarm topology type")
    max_agents: int = Field(default=5)
    agent_types: List[str] = Field(default_factory=list)
    memory_namespace: str = Field(default="default")
    coordination_mode: str = Field(default="collaborative")
    enable_neural: bool = Field(default=False)
    enable_memory: bool = Field(default=True)
    hooks: Dict[str, str] = Field(default_factory=dict)


class ClaudeCodeConfig(BaseModel):
    """Configuration for Claude Code integration"""
    sparc_enabled: bool = Field(default=True)
    tdd_mode: bool = Field(default=True)
    parallel_execution: bool = Field(default=True)
    max_todos: int = Field(default=10)
    file_organization: Dict[str, str] = Field(default_factory=dict)
    agent_spawning: bool = Field(default=True)
    memory_persistence: bool = Field(default=True)
    github_integration: bool = Field(default=False)


class TradingAgentConfig(BaseModel):
    """Configuration for trading agents"""
    strategy_type: str = Field(..., description="Trading strategy type")
    symbols: List[str] = Field(default_factory=list)
    risk_params: Dict[str, float] = Field(default_factory=dict)
    execution_mode: str = Field(default="simulation")
    data_sources: List[str] = Field(default_factory=list)
    indicators: List[str] = Field(default_factory=list)
    backtest_enabled: bool = Field(default=True)
    paper_trading: bool = Field(default=True)
    live_trading: bool = Field(default=False)


class TemplateConfig(BaseModel):
    """Complete template configuration"""
    template_type: TemplateType
    metadata: TemplateMetadata
    requirements: TemplateRequirements
    files: TemplateFiles
    hooks: Optional[TemplateHooks] = None
    
    # Specialized configurations
    claude_flow: Optional[ClaudeFlowConfig] = None
    claude_code: Optional[ClaudeCodeConfig] = None
    trading_agent: Optional[TradingAgentConfig] = None
    
    # Custom configuration
    custom_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Deployment settings
    auto_start: bool = Field(default=True)
    restart_policy: str = Field(default="on-failure")
    max_retries: int = Field(default=3)
    timeout_seconds: int = Field(default=300)


class TemplateInstance(BaseModel):
    """Running template instance"""
    instance_id: str
    template_id: str
    template_type: TemplateType
    sandbox_id: str
    status: str = Field(default="initializing")
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)