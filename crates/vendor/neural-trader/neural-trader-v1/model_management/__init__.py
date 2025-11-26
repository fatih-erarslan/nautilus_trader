"""
AI News Trading Platform - Model Management System

This package provides comprehensive model management capabilities including:
- Model storage with versioning and compression
- Metadata management and search
- Model Context Protocol (MCP) server for real-time inference
- REST API and WebSocket interfaces
- Deployment orchestration to multiple targets
- Health monitoring and alerting
- Version control with Git-like functionality

Example Usage:
    # Basic model management
    from model_management import ModelManager, ManagerConfig
    
    config = ManagerConfig(storage_path="models")
    manager = ModelManager(config)
    await manager.start()
    
    # Create a model
    model_id = await manager.create_model(
        name="Mean Reversion Strategy",
        version="1.0.0",
        strategy_name="mean_reversion",
        model_type="parameter_set",
        parameters={"z_score_threshold": 2.0},
        performance_metrics={"sharpe_ratio": 2.5}
    )
    
    # Deploy the model
    from model_management.deployment import DeploymentConfig, DeploymentTarget
    
    config = DeploymentConfig(
        target=DeploymentTarget.PRODUCTION,
        strategy=DeploymentStrategy.BLUE_GREEN,
        resource_requirements={"cpu": "1", "memory": "2Gi"}
    )
    
    deployment_id = await manager.deploy_model(model_id, config)
"""

from .model_manager import ModelManager, ManagerConfig, DeploymentTarget
from .storage.model_storage import ModelStorage, ModelFormat, CompressionLevel
from .storage.metadata_manager import MetadataManager, ModelStatus
from .storage.version_control import ModelVersionControl
from .mcp_integration.trading_mcp_server import TradingMCPServer
from .mcp_integration.model_api import ModelAPI
from .mcp_integration.websocket_server import ModelWebSocketServer
from .deployment.deploy_orchestrator import DeploymentOrchestrator, DeploymentConfig, DeploymentStrategy
from .deployment.health_monitor import HealthMonitor

__version__ = "1.0.0"
__author__ = "AI Trading Platform"

__all__ = [
    "ModelManager",
    "ManagerConfig", 
    "DeploymentTarget",
    "ModelStorage",
    "ModelFormat",
    "CompressionLevel",
    "MetadataManager",
    "ModelStatus",
    "ModelVersionControl",
    "TradingMCPServer",
    "ModelAPI",
    "ModelWebSocketServer",
    "DeploymentOrchestrator",
    "DeploymentConfig",
    "DeploymentStrategy",
    "HealthMonitor"
]