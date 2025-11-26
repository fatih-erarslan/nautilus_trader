"""Central Model Manager for AI Trading Platform."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import traceback

# Import storage and MCP components
from .storage.model_storage import ModelStorage, ModelMetadata as StorageMetadata
from .storage.metadata_manager import MetadataManager, ModelMetadata, ModelStatus
from .storage.version_control import ModelVersionControl
from .mcp_integration.trading_mcp_server import TradingMCPServer
from .mcp_integration.model_api import ModelAPI
from .mcp_integration.websocket_server import ModelWebSocketServer

logger = logging.getLogger(__name__)


class ManagerStatus(Enum):
    """Model manager status."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class DeploymentTarget(Enum):
    """Deployment targets."""
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"
    CLOUD = "cloud"


@dataclass
class ModelRegistryEntry:
    """Model registry entry with deployment information."""
    model_id: str
    metadata: ModelMetadata
    deployment_status: Dict[str, Any]
    last_used: datetime
    usage_count: int
    performance_history: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]


@dataclass
class ManagerConfig:
    """Configuration for the model manager."""
    storage_path: str = "model_management"
    mcp_server_port: int = 8000
    api_server_port: int = 8001
    websocket_server_port: int = 8002
    enable_auto_cleanup: bool = True
    max_cached_models: int = 10
    cache_ttl_minutes: int = 30
    health_check_interval: int = 60
    performance_monitoring_interval: int = 300
    auto_deploy_validated: bool = False
    backup_interval_hours: int = 24


class ModelManager:
    """Central coordinator for all model management operations."""
    
    def __init__(self, config: ManagerConfig = None):
        """
        Initialize the model manager.
        
        Args:
            config: Manager configuration
        """
        self.config = config or ManagerConfig()
        self.status = ManagerStatus.INITIALIZING
        
        # Initialize storage components
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.model_storage = ModelStorage(str(self.storage_path / "models"))
        self.metadata_manager = MetadataManager(str(self.storage_path / "storage"))
        self.version_control = ModelVersionControl(str(self.storage_path / "models" / "versions"))
        
        # Initialize servers
        self.mcp_server = None
        self.api_server = None
        self.websocket_server = None
        
        # Model registry
        self.model_registry: Dict[str, ModelRegistryEntry] = {}
        self.registry_lock = threading.Lock()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics and monitoring
        self.stats = {
            'start_time': datetime.now(),
            'models_loaded': 0,
            'models_deployed': 0,
            'predictions_made': 0,
            'errors_count': 0,
            'last_health_check': None,
            'last_backup': None
        }
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'model_created': [],
            'model_updated': [],
            'model_deployed': [],
            'model_error': [],
            'performance_alert': []
        }
        
        logger.info("Model Manager initialized")
    
    async def start(self):
        """Start the model manager and all services."""
        try:
            self.status = ManagerStatus.INITIALIZING
            
            # Load existing models into registry
            await self._load_model_registry()
            
            # Start servers
            await self._start_servers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.status = ManagerStatus.RUNNING
            logger.info("Model Manager started successfully")
            
        except Exception as e:
            self.status = ManagerStatus.ERROR
            logger.error(f"Failed to start Model Manager: {e}")
            raise
    
    async def stop(self):
        """Stop the model manager and all services."""
        try:
            self.status = ManagerStatus.STOPPING
            
            # Stop background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Stop servers
            await self._stop_servers()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.status = ManagerStatus.STOPPED
            logger.info("Model Manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Model Manager: {e}")
            self.status = ManagerStatus.ERROR
    
    async def _start_servers(self):
        """Start MCP, API, and WebSocket servers."""
        try:
            # Start MCP server
            self.mcp_server = TradingMCPServer(
                host="0.0.0.0",
                port=self.config.mcp_server_port,
                model_storage_path=str(self.storage_path / "models")
            )
            
            # Start API server
            self.api_server = ModelAPI(str(self.storage_path))
            
            # Start WebSocket server
            self.websocket_server = ModelWebSocketServer(
                host="0.0.0.0",
                port=self.config.websocket_server_port,
                storage_path=str(self.storage_path)
            )
            
            # Start servers in background
            asyncio.create_task(self.mcp_server.start_async())
            asyncio.create_task(self.websocket_server.start_server())
            
            logger.info("All servers started")
            
        except Exception as e:
            logger.error(f"Failed to start servers: {e}")
            raise
    
    async def _stop_servers(self):
        """Stop all servers."""
        try:
            if self.websocket_server:
                await self.websocket_server.stop_server()
            
            # MCP and API servers will be stopped when tasks are cancelled
            
        except Exception as e:
            logger.error(f"Error stopping servers: {e}")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks."""
        self.background_tasks = [
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._auto_backup()),
            asyncio.create_task(self._cleanup_monitor())
        ]
    
    async def _load_model_registry(self):
        """Load existing models into the registry."""
        try:
            models = self.metadata_manager.search_models(limit=1000)
            
            with self.registry_lock:
                for metadata in models:
                    entry = ModelRegistryEntry(
                        model_id=metadata.model_id,
                        metadata=metadata,
                        deployment_status={},
                        last_used=metadata.updated_at,
                        usage_count=0,
                        performance_history=[],
                        alerts=[]
                    )
                    self.model_registry[metadata.model_id] = entry
            
            logger.info(f"Loaded {len(models)} models into registry")
            
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register callback for model events."""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    async def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger event callbacks."""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
    
    async def create_model(self, name: str, version: str, strategy_name: str,
                          model_type: str, parameters: Dict[str, Any],
                          performance_metrics: Dict[str, float],
                          description: str = "", tags: List[str] = None,
                          author: str = "Model Manager") -> str:
        """
        Create a new model.
        
        Args:
            name: Model name
            version: Model version
            strategy_name: Trading strategy name
            model_type: Type of model
            parameters: Model parameters
            performance_metrics: Performance metrics
            description: Model description
            tags: Model tags
            author: Model author
            
        Returns:
            Model ID
        """
        try:
            # Create metadata
            metadata = ModelMetadata(
                model_id="",  # Will be generated
                name=name,
                version=version,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model_type=model_type,
                strategy_name=strategy_name,
                status=ModelStatus.DEVELOPMENT,
                performance_metrics=performance_metrics,
                parameters=parameters,
                tags=set(tags or []),
                description=description,
                author=author
            )
            
            # Create storage metadata
            storage_metadata = StorageMetadata(
                model_id="",
                name=name,
                version=version,
                created_at=datetime.now(),
                model_type=model_type,
                strategy_name=strategy_name,
                performance_metrics=performance_metrics,
                parameters=parameters,
                tags=tags or [],
                description=description,
                author=author
            )
            
            # Save model
            model_id = self.model_storage.save_model(parameters, storage_metadata)
            metadata.model_id = model_id
            
            # Save metadata
            self.metadata_manager.save_metadata(metadata)
            
            # Create initial version
            self.version_control.commit_version(
                model_id=model_id,
                model_data=parameters,
                parameters=parameters,
                performance_metrics=performance_metrics,
                commit_message=f"Initial version of {name}",
                author=author
            )
            
            # Add to registry
            with self.registry_lock:
                entry = ModelRegistryEntry(
                    model_id=model_id,
                    metadata=metadata,
                    deployment_status={},
                    last_used=datetime.now(),
                    usage_count=0,
                    performance_history=[],
                    alerts=[]
                )
                self.model_registry[model_id] = entry
            
            # Update stats
            self.stats['models_loaded'] += 1
            
            # Trigger event
            await self._trigger_event('model_created', {
                'model_id': model_id,
                'metadata': metadata.to_dict()
            })
            
            # Broadcast to WebSocket clients
            if self.websocket_server:
                await self.websocket_server.broadcast_model_update(
                    model_id, {'action': 'created', 'model_data': metadata.to_dict()}
                )
            
            logger.info(f"Model created: {model_id}")
            return model_id
            
        except Exception as e:
            self.stats['errors_count'] += 1
            logger.error(f"Failed to create model: {e}")
            raise
    
    async def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update model metadata and parameters.
        
        Args:
            model_id: Model identifier
            updates: Updates to apply
            
        Returns:
            True if successful
        """
        try:
            # Get current metadata
            metadata = self.metadata_manager.load_metadata(model_id)
            if not metadata:
                raise ValueError(f"Model {model_id} not found")
            
            # Apply updates
            if 'name' in updates:
                metadata.name = updates['name']
            if 'description' in updates:
                metadata.description = updates['description']
            if 'status' in updates:
                metadata.status = ModelStatus(updates['status'])
            if 'parameters' in updates:
                metadata.parameters.update(updates['parameters'])
            if 'performance_metrics' in updates:
                metadata.performance_metrics.update(updates['performance_metrics'])
            if 'tags' in updates:
                metadata.tags = set(updates['tags'])
            
            metadata.updated_at = datetime.now()
            
            # Save updated metadata
            self.metadata_manager.save_metadata(metadata)
            
            # Update registry
            with self.registry_lock:
                if model_id in self.model_registry:
                    self.model_registry[model_id].metadata = metadata
                    self.model_registry[model_id].last_used = datetime.now()
            
            # Create new version if significant changes
            if 'parameters' in updates or 'performance_metrics' in updates:
                self.version_control.commit_version(
                    model_id=model_id,
                    model_data=metadata.parameters,
                    parameters=metadata.parameters,
                    performance_metrics=metadata.performance_metrics,
                    commit_message="Model update via Model Manager",
                    author="Model Manager"
                )
            
            # Trigger event
            await self._trigger_event('model_updated', {
                'model_id': model_id,
                'updates': updates,
                'metadata': metadata.to_dict()
            })
            
            # Broadcast to WebSocket clients
            if self.websocket_server:
                await self.websocket_server.broadcast_model_update(
                    model_id, {'action': 'updated', 'updates': updates}
                )
            
            logger.info(f"Model updated: {model_id}")
            return True
            
        except Exception as e:
            self.stats['errors_count'] += 1
            logger.error(f"Failed to update model {model_id}: {e}")
            raise
    
    async def deploy_model(self, model_id: str, target: DeploymentTarget,
                          config: Dict[str, Any] = None) -> bool:
        """
        Deploy model to target environment.
        
        Args:
            model_id: Model identifier
            target: Deployment target
            config: Deployment configuration
            
        Returns:
            True if successful
        """
        try:
            # Get model metadata
            metadata = self.metadata_manager.load_metadata(model_id)
            if not metadata:
                raise ValueError(f"Model {model_id} not found")
            
            # Validate model is ready for deployment
            if target == DeploymentTarget.PRODUCTION:
                if metadata.status != ModelStatus.VALIDATED:
                    raise ValueError("Model must be validated before production deployment")
            
            # Update deployment status
            deployment_info = {
                'target': target.value,
                'deployed_at': datetime.now().isoformat(),
                'config': config or {},
                'status': 'deployed'
            }
            
            # Update metadata
            metadata.deployment_info[target.value] = deployment_info
            if target == DeploymentTarget.PRODUCTION:
                metadata.status = ModelStatus.PRODUCTION
            
            metadata.updated_at = datetime.now()
            self.metadata_manager.save_metadata(metadata)
            
            # Update registry
            with self.registry_lock:
                if model_id in self.model_registry:
                    entry = self.model_registry[model_id]
                    entry.deployment_status[target.value] = deployment_info
                    entry.metadata = metadata
            
            # Update stats
            self.stats['models_deployed'] += 1
            
            # Trigger event
            await self._trigger_event('model_deployed', {
                'model_id': model_id,
                'target': target.value,
                'deployment_info': deployment_info
            })
            
            logger.info(f"Model deployed: {model_id} to {target.value}")
            return True
            
        except Exception as e:
            self.stats['errors_count'] += 1
            logger.error(f"Failed to deploy model {model_id}: {e}")
            
            # Trigger error event
            await self._trigger_event('model_error', {
                'model_id': model_id,
                'error_type': 'deployment_failed',
                'error_message': str(e)
            })
            
            raise
    
    async def get_model_prediction(self, model_id: str, input_data: Dict[str, Any],
                                 strategy_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get prediction from model.
        
        Args:
            model_id: Model identifier
            input_data: Input data for prediction
            strategy_context: Additional strategy context
            
        Returns:
            Prediction result
        """
        try:
            # Load model
            model, metadata = self.model_storage.load_model(model_id)
            
            # Update usage stats
            with self.registry_lock:
                if model_id in self.model_registry:
                    entry = self.model_registry[model_id]
                    entry.usage_count += 1
                    entry.last_used = datetime.now()
            
            # Make prediction (simplified)
            if isinstance(model, dict):
                # Parameter-based prediction
                prediction = {
                    'action': 'hold',
                    'confidence': 0.5,
                    'position_size': model.get('base_position_size', 0.05),
                    'model_id': model_id,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # ML model prediction
                if hasattr(model, 'predict'):
                    result = model.predict(input_data)
                    prediction = {
                        'prediction': result,
                        'model_id': model_id,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    prediction = {
                        'action': 'hold',
                        'confidence': 0.5,
                        'model_id': model_id,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Update stats
            self.stats['predictions_made'] += 1
            
            return prediction
            
        except Exception as e:
            self.stats['errors_count'] += 1
            logger.error(f"Prediction failed for model {model_id}: {e}")
            raise
    
    def get_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get current model registry."""
        with self.registry_lock:
            registry_data = {}
            for model_id, entry in self.model_registry.items():
                registry_data[model_id] = {
                    'metadata': entry.metadata.to_dict(),
                    'deployment_status': entry.deployment_status,
                    'last_used': entry.last_used.isoformat(),
                    'usage_count': entry.usage_count,
                    'performance_history': entry.performance_history[-10:],  # Last 10
                    'alerts': entry.alerts[-5:]  # Last 5
                }
            return registry_data
    
    def get_strategy_models(self, strategy_name: str) -> List[Dict[str, Any]]:
        """Get all models for a specific strategy."""
        models = []
        with self.registry_lock:
            for entry in self.model_registry.values():
                if entry.metadata.strategy_name == strategy_name:
                    models.append({
                        'model_id': entry.model_id,
                        'metadata': entry.metadata.to_dict(),
                        'deployment_status': entry.deployment_status,
                        'usage_count': entry.usage_count,
                        'last_used': entry.last_used.isoformat()
                    })
        
        # Sort by performance (example: Sharpe ratio)
        models.sort(
            key=lambda x: x['metadata']['performance_metrics'].get('sharpe_ratio', 0),
            reverse=True
        )
        
        return models
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = datetime.now() - self.stats['start_time']
        
        # Server status
        server_status = {}
        if self.mcp_server:
            server_status['mcp_server'] = {
                'status': 'running',
                'port': self.config.mcp_server_port
            }
        
        if self.websocket_server:
            server_status['websocket_server'] = {
                'status': 'running',
                'port': self.config.websocket_server_port,
                'stats': self.websocket_server.get_server_stats()
            }
        
        # Storage stats
        storage_stats = self.model_storage.get_storage_stats()
        version_stats = self.version_control.get_version_statistics()
        
        return {
            'manager_status': self.status.value,
            'uptime_seconds': uptime.total_seconds(),
            'statistics': self.stats,
            'model_registry_size': len(self.model_registry),
            'server_status': server_status,
            'storage_stats': storage_stats,
            'version_control_stats': version_stats,
            'config': asdict(self.config)
        }
    
    async def _health_monitor(self):
        """Monitor system health."""
        while self.status == ManagerStatus.RUNNING:
            try:
                # Check component health
                health_issues = []
                
                # Check storage health
                try:
                    storage_stats = self.model_storage.get_storage_stats()
                    if storage_stats['total_size_mb'] > 10000:  # 10GB limit
                        health_issues.append("Storage size approaching limit")
                except Exception as e:
                    health_issues.append(f"Storage health check failed: {e}")
                
                # Check server health
                if self.websocket_server:
                    ws_stats = self.websocket_server.get_server_stats()
                    if ws_stats['errors_count'] > 100:
                        health_issues.append("High error rate in WebSocket server")
                
                # Update health status
                self.stats['last_health_check'] = datetime.now().isoformat()
                
                if health_issues:
                    logger.warning(f"Health issues detected: {health_issues}")
                    # Could trigger alerts here
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _performance_monitor(self):
        """Monitor model performance and trigger alerts."""
        while self.status == ManagerStatus.RUNNING:
            try:
                # Check model performance
                with self.registry_lock:
                    for model_id, entry in self.model_registry.items():
                        # Example: Check if Sharpe ratio dropped
                        current_sharpe = entry.metadata.performance_metrics.get('sharpe_ratio', 0)
                        
                        # Add to performance history
                        performance_point = {
                            'timestamp': datetime.now().isoformat(),
                            'sharpe_ratio': current_sharpe,
                            'usage_count': entry.usage_count
                        }
                        entry.performance_history.append(performance_point)
                        
                        # Keep only recent history
                        if len(entry.performance_history) > 100:
                            entry.performance_history = entry.performance_history[-100:]
                        
                        # Check for performance degradation
                        if len(entry.performance_history) >= 10:
                            recent_avg = sum(
                                p.get('sharpe_ratio', 0) for p in entry.performance_history[-5:]
                            ) / 5
                            older_avg = sum(
                                p.get('sharpe_ratio', 0) for p in entry.performance_history[-10:-5]
                            ) / 5
                            
                            if recent_avg < older_avg * 0.8:  # 20% degradation
                                alert = {
                                    'type': 'performance_degradation',
                                    'timestamp': datetime.now().isoformat(),
                                    'message': f"Model performance degraded: {recent_avg:.2f} vs {older_avg:.2f}",
                                    'severity': 'warning'
                                }
                                entry.alerts.append(alert)
                                
                                # Trigger alert event
                                await self._trigger_event('performance_alert', {
                                    'model_id': model_id,
                                    'alert': alert
                                })
                
                await asyncio.sleep(self.config.performance_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(self.config.performance_monitoring_interval)
    
    async def _auto_backup(self):
        """Automatic backup of models and metadata."""
        while self.status == ManagerStatus.RUNNING:
            try:
                # Create backup
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = self.storage_path / "backups" / f"auto_backup_{timestamp}"
                backup_path.mkdir(parents=True, exist_ok=True)
                
                # Export metadata
                export_success = self.metadata_manager.export_metadata(
                    str(backup_path / "metadata_backup.json")
                )
                
                if export_success:
                    self.stats['last_backup'] = datetime.now().isoformat()
                    logger.info(f"Automatic backup created: {backup_path}")
                
                # Wait for next backup
                await asyncio.sleep(self.config.backup_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Auto backup error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _cleanup_monitor(self):
        """Monitor and cleanup old data."""
        while self.status == ManagerStatus.RUNNING:
            try:
                if self.config.enable_auto_cleanup:
                    # Clean up old metadata
                    deleted_count = self.metadata_manager.cleanup_metadata(
                        days_old=30, keep_production=True
                    )
                    
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} old models")
                
                # Wait 24 hours between cleanup runs
                await asyncio.sleep(24 * 3600)
                
            except Exception as e:
                logger.error(f"Cleanup monitor error: {e}")
                await asyncio.sleep(3600)


# Factory function
def create_model_manager(config: ManagerConfig = None) -> ModelManager:
    """Create and configure model manager."""
    return ModelManager(config)


# Example usage
async def main():
    """Example usage of Model Manager."""
    # Create configuration
    config = ManagerConfig(
        storage_path="model_management",
        mcp_server_port=8000,
        api_server_port=8001,
        websocket_server_port=8002
    )
    
    # Create and start manager
    manager = ModelManager(config)
    
    try:
        await manager.start()
        
        # Example: Create a model
        model_id = await manager.create_model(
            name="Test Strategy",
            version="1.0.0",
            strategy_name="mean_reversion",
            model_type="parameter_set",
            parameters={
                "z_score_threshold": 2.0,
                "base_position_size": 0.05,
                "stop_loss_multiplier": 1.5
            },
            performance_metrics={
                "sharpe_ratio": 2.5,
                "total_return": 0.18,
                "max_drawdown": 0.08
            },
            description="Test mean reversion strategy"
        )
        
        print(f"Created model: {model_id}")
        
        # Keep running
        await asyncio.sleep(3600)  # Run for 1 hour
        
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())