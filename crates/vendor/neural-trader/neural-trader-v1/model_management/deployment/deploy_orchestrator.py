"""Deployment Orchestrator for AI Trading Models."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import requests
import yaml
from concurrent.futures import ThreadPoolExecutor

# Import model management components
from ..storage.model_storage import ModelStorage
from ..storage.metadata_manager import MetadataManager, ModelStatus

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    TERMINATED = "terminated"


class DeploymentTarget(Enum):
    """Deployment target environments."""
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"
    CLOUD_GPU = "cloud_gpu"
    FLYIO = "flyio"
    KUBERNETES = "kubernetes"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    target: DeploymentTarget
    strategy: DeploymentStrategy
    resource_requirements: Dict[str, Any]
    environment_variables: Dict[str, str]
    health_check_config: Dict[str, Any]
    auto_rollback: bool = True
    timeout_seconds: int = 600
    replica_count: int = 1
    gpu_requirements: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DeploymentRecord:
    """Record of a deployment operation."""
    deployment_id: str
    model_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    created_at: datetime
    updated_at: datetime
    deployed_at: Optional[datetime] = None
    terminated_at: Optional[datetime] = None
    logs: List[str] = None
    health_checks: List[Dict[str, Any]] = None
    rollback_info: Optional[Dict[str, Any]] = None
    endpoints: Dict[str, str] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.health_checks is None:
            self.health_checks = []
        if self.endpoints is None:
            self.endpoints = {}
    
    def add_log(self, message: str, level: str = "INFO"):
        """Add log entry."""
        log_entry = f"[{datetime.now().isoformat()}] {level}: {message}"
        self.logs.append(log_entry)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.deployed_at:
            data['deployed_at'] = self.deployed_at.isoformat()
        if self.terminated_at:
            data['terminated_at'] = self.terminated_at.isoformat()
        data['status'] = self.status.value
        data['config'] = self.config.to_dict()
        return data


class DeploymentOrchestrator:
    """Orchestrates model deployments across different environments."""
    
    def __init__(self, storage_path: str = "model_management",
                 deployment_path: str = "model_management/deployments"):
        """
        Initialize deployment orchestrator.
        
        Args:
            storage_path: Path to model storage
            deployment_path: Path for deployment artifacts
        """
        self.storage_path = Path(storage_path)
        self.deployment_path = Path(deployment_path)
        self.deployment_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage components
        self.model_storage = ModelStorage(str(self.storage_path / "models"))
        self.metadata_manager = MetadataManager(str(self.storage_path / "storage"))
        
        # Deployment tracking
        self.deployments: Dict[str, DeploymentRecord] = {}
        self.active_deployments: Dict[str, asyncio.Task] = {}
        self.deployment_lock = threading.Lock()
        
        # Load existing deployments
        self._load_deployments()
        
        # Background monitoring
        self.monitoring_task = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Deployment Orchestrator initialized")
    
    def _load_deployments(self):
        """Load existing deployment records."""
        deployments_file = self.deployment_path / "deployments.json"
        if deployments_file.exists():
            try:
                with open(deployments_file, 'r') as f:
                    data = json.load(f)
                
                for deployment_id, deployment_data in data.items():
                    # Reconstruct deployment record
                    config_data = deployment_data['config']
                    config = DeploymentConfig(
                        target=DeploymentTarget(config_data['target']),
                        strategy=DeploymentStrategy(config_data['strategy']),
                        resource_requirements=config_data['resource_requirements'],
                        environment_variables=config_data['environment_variables'],
                        health_check_config=config_data['health_check_config'],
                        auto_rollback=config_data.get('auto_rollback', True),
                        timeout_seconds=config_data.get('timeout_seconds', 600),
                        replica_count=config_data.get('replica_count', 1),
                        gpu_requirements=config_data.get('gpu_requirements')
                    )
                    
                    record = DeploymentRecord(
                        deployment_id=deployment_id,
                        model_id=deployment_data['model_id'],
                        config=config,
                        status=DeploymentStatus(deployment_data['status']),
                        created_at=datetime.fromisoformat(deployment_data['created_at']),
                        updated_at=datetime.fromisoformat(deployment_data['updated_at']),
                        deployed_at=datetime.fromisoformat(deployment_data['deployed_at']) if deployment_data.get('deployed_at') else None,
                        terminated_at=datetime.fromisoformat(deployment_data['terminated_at']) if deployment_data.get('terminated_at') else None,
                        logs=deployment_data.get('logs', []),
                        health_checks=deployment_data.get('health_checks', []),
                        rollback_info=deployment_data.get('rollback_info'),
                        endpoints=deployment_data.get('endpoints', {})
                    )
                    
                    self.deployments[deployment_id] = record
                
                logger.info(f"Loaded {len(self.deployments)} deployment records")
                
            except Exception as e:
                logger.error(f"Failed to load deployments: {e}")
    
    def _save_deployments(self):
        """Save deployment records to disk."""
        deployments_file = self.deployment_path / "deployments.json"
        try:
            data = {
                deployment_id: record.to_dict()
                for deployment_id, record in self.deployments.items()
            }
            
            with open(deployments_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to save deployments: {e}")
    
    async def deploy_model(self, model_id: str, config: DeploymentConfig) -> str:
        """
        Deploy a model to specified target.
        
        Args:
            model_id: Model identifier
            config: Deployment configuration
            
        Returns:
            Deployment ID
        """
        deployment_id = f"deploy_{model_id}_{int(time.time())}"
        
        # Create deployment record
        record = DeploymentRecord(
            deployment_id=deployment_id,
            model_id=model_id,
            config=config,
            status=DeploymentStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store deployment record
        with self.deployment_lock:
            self.deployments[deployment_id] = record
            self._save_deployments()
        
        # Start deployment task
        task = asyncio.create_task(self._execute_deployment(deployment_id))
        self.active_deployments[deployment_id] = task
        
        logger.info(f"Started deployment {deployment_id} for model {model_id}")
        return deployment_id
    
    async def _execute_deployment(self, deployment_id: str):
        """Execute the deployment process."""
        record = self.deployments[deployment_id]
        
        try:
            record.status = DeploymentStatus.IN_PROGRESS
            record.add_log(f"Starting deployment to {record.config.target.value}")
            
            # Validate model exists and is ready
            await self._validate_model_for_deployment(record)
            
            # Prepare deployment artifacts
            await self._prepare_deployment_artifacts(record)
            
            # Execute deployment based on target
            if record.config.target == DeploymentTarget.LOCAL:
                await self._deploy_local(record)
            elif record.config.target == DeploymentTarget.FLYIO:
                await self._deploy_flyio(record)
            elif record.config.target == DeploymentTarget.CLOUD_GPU:
                await self._deploy_cloud_gpu(record)
            elif record.config.target == DeploymentTarget.KUBERNETES:
                await self._deploy_kubernetes(record)
            else:
                raise ValueError(f"Unsupported deployment target: {record.config.target}")
            
            # Perform health checks
            await self._perform_health_checks(record)
            
            # Mark as deployed
            record.status = DeploymentStatus.DEPLOYED
            record.deployed_at = datetime.now()
            record.add_log("Deployment completed successfully")
            
            logger.info(f"Deployment {deployment_id} completed successfully")
            
        except Exception as e:
            record.status = DeploymentStatus.FAILED
            record.add_log(f"Deployment failed: {str(e)}", "ERROR")
            
            logger.error(f"Deployment {deployment_id} failed: {e}")
            
            # Attempt rollback if enabled
            if record.config.auto_rollback:
                await self._rollback_deployment(record)
        
        finally:
            # Clean up active deployment tracking
            self.active_deployments.pop(deployment_id, None)
            
            # Save final state
            with self.deployment_lock:
                self._save_deployments()
    
    async def _validate_model_for_deployment(self, record: DeploymentRecord):
        """Validate model is ready for deployment."""
        # Check model exists
        metadata = self.metadata_manager.load_metadata(record.model_id)
        if not metadata:
            raise ValueError(f"Model {record.model_id} not found")
        
        # Check model status
        if record.config.target == DeploymentTarget.PRODUCTION:
            if metadata.status != ModelStatus.VALIDATED:
                raise ValueError("Model must be validated before production deployment")
        
        # Load model to verify it's accessible
        try:
            model, storage_metadata = self.model_storage.load_model(record.model_id)
            record.add_log("Model validation passed")
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    
    async def _prepare_deployment_artifacts(self, record: DeploymentRecord):
        """Prepare artifacts needed for deployment."""
        # Create deployment directory
        deployment_dir = self.deployment_path / record.deployment_id
        deployment_dir.mkdir(exist_ok=True)
        
        # Load model and metadata
        model, storage_metadata = self.model_storage.load_model(record.model_id)
        metadata = self.metadata_manager.load_metadata(record.model_id)
        
        # Save model artifacts
        model_file = deployment_dir / "model.json"
        with open(model_file, 'w') as f:
            json.dump(storage_metadata.parameters, f, indent=2, default=str)
        
        # Save metadata
        metadata_file = deployment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        # Generate deployment configuration
        await self._generate_deployment_config(record, deployment_dir)
        
        record.add_log("Deployment artifacts prepared")
    
    async def _generate_deployment_config(self, record: DeploymentRecord, deployment_dir: Path):
        """Generate deployment-specific configuration files."""
        if record.config.target == DeploymentTarget.FLYIO:
            await self._generate_flyio_config(record, deployment_dir)
        elif record.config.target == DeploymentTarget.KUBERNETES:
            await self._generate_kubernetes_config(record, deployment_dir)
        elif record.config.target == DeploymentTarget.CLOUD_GPU:
            await self._generate_cloud_gpu_config(record, deployment_dir)
    
    async def _generate_flyio_config(self, record: DeploymentRecord, deployment_dir: Path):
        """Generate Fly.io configuration."""
        fly_config = {
            'app': f"ai-trader-{record.model_id}",
            'primary_region': 'iad',
            'build': {
                'image': 'python:3.9-slim'
            },
            'env': record.config.environment_variables,
            'services': [{
                'http_checks': [{
                    'interval': '10s',
                    'timeout': '2s',
                    'grace_period': '5s',
                    'method': 'get',
                    'path': '/health'
                }],
                'internal_port': 8000,
                'ports': [{
                    'port': 80,
                    'handlers': ['http']
                }],
                'protocol': 'tcp'
            }]
        }
        
        # Add GPU if required
        if record.config.gpu_requirements:
            fly_config['vm'] = {
                'gpu_kind': record.config.gpu_requirements.get('type', 'a100-pcie-40gb'),
                'gpus': record.config.gpu_requirements.get('count', 1)
            }
        
        fly_toml = deployment_dir / "fly.toml"
        with open(fly_toml, 'w') as f:
            import toml
            toml.dump(fly_config, f)
        
        # Generate Dockerfile
        dockerfile = deployment_dir / "Dockerfile"
        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "server.py"]
"""
        with open(dockerfile, 'w') as f:
            f.write(dockerfile_content)
        
        # Generate requirements.txt
        requirements = deployment_dir / "requirements.txt"
        requirements_content = """
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
numpy==1.24.3
pandas==2.0.3
"""
        with open(requirements, 'w') as f:
            f.write(requirements_content)
        
        # Generate server.py
        await self._generate_model_server(record, deployment_dir)
    
    async def _generate_kubernetes_config(self, record: DeploymentRecord, deployment_dir: Path):
        """Generate Kubernetes deployment configuration."""
        k8s_config = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f"ai-trader-{record.model_id}",
                'labels': {
                    'app': f"ai-trader-{record.model_id}",
                    'model_id': record.model_id
                }
            },
            'spec': {
                'replicas': record.config.replica_count,
                'selector': {
                    'matchLabels': {
                        'app': f"ai-trader-{record.model_id}"
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': f"ai-trader-{record.model_id}"
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'model-server',
                            'image': f"ai-trader-model:{record.model_id}",
                            'ports': [{'containerPort': 8000}],
                            'env': [
                                {'name': k, 'value': v}
                                for k, v in record.config.environment_variables.items()
                            ],
                            'resources': record.config.resource_requirements,
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            }
                        }]
                    }
                }
            }
        }
        
        # Add GPU resources if required
        if record.config.gpu_requirements:
            container = k8s_config['spec']['template']['spec']['containers'][0]
            if 'resources' not in container:
                container['resources'] = {}
            if 'limits' not in container['resources']:
                container['resources']['limits'] = {}
            
            container['resources']['limits']['nvidia.com/gpu'] = record.config.gpu_requirements.get('count', 1)
        
        k8s_file = deployment_dir / "kubernetes.yaml"
        with open(k8s_file, 'w') as f:
            yaml.dump(k8s_config, f, default_flow_style=False)
    
    async def _generate_cloud_gpu_config(self, record: DeploymentRecord, deployment_dir: Path):
        """Generate cloud GPU deployment configuration."""
        # This would integrate with cloud providers like AWS, GCP, Azure
        # For now, generate a generic configuration
        cloud_config = {
            'provider': 'aws',  # or 'gcp', 'azure'
            'instance_type': record.config.gpu_requirements.get('instance_type', 'p3.2xlarge'),
            'region': 'us-east-1',
            'image': 'deep-learning-ami',
            'security_groups': ['ai-trading-sg'],
            'key_pair': 'ai-trading-key',
            'user_data_script': """#!/bin/bash
# Install dependencies and start model server
pip install -r requirements.txt
python server.py
"""
        }
        
        cloud_file = deployment_dir / "cloud_config.json"
        with open(cloud_file, 'w') as f:
            json.dump(cloud_config, f, indent=2)
    
    async def _generate_model_server(self, record: DeploymentRecord, deployment_dir: Path):
        """Generate model server code."""
        server_code = f"""
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime

app = FastAPI(title="AI Trading Model Server", version="1.0.0")

# Load model and metadata
with open('model.json', 'r') as f:
    model_params = json.load(f)

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

class PredictionRequest(BaseModel):
    input_data: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: Dict[str, Any]
    model_id: str
    timestamp: str

@app.get("/health")
async def health_check():
    return {{
        "status": "healthy",
        "model_id": "{record.model_id}",
        "timestamp": datetime.now().isoformat()
    }}

@app.get("/metadata")
async def get_metadata():
    return metadata

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Simple rule-based prediction using model parameters
    strategy_name = metadata.get('strategy_name', '').lower()
    
    if 'mean_reversion' in strategy_name:
        z_score = request.input_data.get('z_score', 0.0)
        threshold = model_params.get('z_score_entry_threshold', 2.0)
        
        if abs(z_score) >= threshold:
            action = 'buy' if z_score < 0 else 'sell'
            confidence = min(abs(z_score) / threshold, 1.0)
        else:
            action = 'hold'
            confidence = 0.5
        
        prediction = {{
            'action': action,
            'confidence': confidence,
            'position_size': model_params.get('base_position_size', 0.05),
            'z_score': z_score
        }}
    else:
        # Default prediction
        prediction = {{
            'action': 'hold',
            'confidence': 0.5,
            'position_size': 0.05
        }}
    
    return PredictionResponse(
        prediction=prediction,
        model_id="{record.model_id}",
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        
        server_file = deployment_dir / "server.py"
        with open(server_file, 'w') as f:
            f.write(server_code)
    
    async def _deploy_local(self, record: DeploymentRecord):
        """Deploy model locally."""
        record.add_log("Starting local deployment")
        
        deployment_dir = self.deployment_path / record.deployment_id
        
        # Start local server
        process = subprocess.Popen(
            ["python", "server.py"],
            cwd=deployment_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for server to start
        await asyncio.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            record.endpoints['api'] = "http://localhost:8000"
            record.add_log("Local deployment successful")
        else:
            stdout, stderr = process.communicate()
            raise Exception(f"Local deployment failed: {stderr.decode()}")
    
    async def _deploy_flyio(self, record: DeploymentRecord):
        """Deploy model to Fly.io."""
        record.add_log("Starting Fly.io deployment")
        
        deployment_dir = self.deployment_path / record.deployment_id
        
        try:
            # Deploy to Fly.io
            result = subprocess.run(
                ["flyctl", "deploy", "--local-only"],
                cwd=deployment_dir,
                capture_output=True,
                text=True,
                timeout=record.config.timeout_seconds
            )
            
            if result.returncode == 0:
                app_name = f"ai-trader-{record.model_id}"
                record.endpoints['api'] = f"https://{app_name}.fly.dev"
                record.add_log("Fly.io deployment successful")
            else:
                raise Exception(f"Fly.io deployment failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Fly.io deployment timed out")
    
    async def _deploy_cloud_gpu(self, record: DeploymentRecord):
        """Deploy model to cloud GPU instance."""
        record.add_log("Starting cloud GPU deployment")
        
        # This would integrate with cloud providers
        # For now, simulate deployment
        await asyncio.sleep(10)  # Simulate deployment time
        
        record.endpoints['api'] = "https://cloud-gpu-instance.example.com:8000"
        record.add_log("Cloud GPU deployment successful")
    
    async def _deploy_kubernetes(self, record: DeploymentRecord):
        """Deploy model to Kubernetes."""
        record.add_log("Starting Kubernetes deployment")
        
        deployment_dir = self.deployment_path / record.deployment_id
        
        try:
            # Apply Kubernetes configuration
            result = subprocess.run(
                ["kubectl", "apply", "-f", "kubernetes.yaml"],
                cwd=deployment_dir,
                capture_output=True,
                text=True,
                timeout=record.config.timeout_seconds
            )
            
            if result.returncode == 0:
                # Get service endpoint
                service_result = subprocess.run(
                    ["kubectl", "get", "service", f"ai-trader-{record.model_id}", "-o", "json"],
                    capture_output=True,
                    text=True
                )
                
                if service_result.returncode == 0:
                    service_info = json.loads(service_result.stdout)
                    # Extract endpoint from service info
                    record.endpoints['api'] = "http://kubernetes-cluster:8000"
                
                record.add_log("Kubernetes deployment successful")
            else:
                raise Exception(f"Kubernetes deployment failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Kubernetes deployment timed out")
    
    async def _perform_health_checks(self, record: DeploymentRecord):
        """Perform health checks on deployed model."""
        record.add_log("Starting health checks")
        
        api_endpoint = record.endpoints.get('api')
        if not api_endpoint:
            raise Exception("No API endpoint available for health check")
        
        health_config = record.config.health_check_config
        max_retries = health_config.get('max_retries', 10)
        retry_interval = health_config.get('retry_interval', 30)
        
        for attempt in range(max_retries):
            try:
                # Health check
                response = requests.get(
                    f"{api_endpoint}/health",
                    timeout=10
                )
                
                if response.status_code == 200:
                    health_data = response.json()
                    record.health_checks.append({
                        'timestamp': datetime.now().isoformat(),
                        'status': 'healthy',
                        'response_time_ms': response.elapsed.total_seconds() * 1000,
                        'data': health_data
                    })
                    record.add_log("Health check passed")
                    return
                else:
                    record.add_log(f"Health check failed with status {response.status_code}")
                    
            except Exception as e:
                record.add_log(f"Health check attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_interval)
        
        raise Exception("All health checks failed")
    
    async def _rollback_deployment(self, record: DeploymentRecord):
        """Rollback failed deployment."""
        record.status = DeploymentStatus.ROLLING_BACK
        record.add_log("Starting rollback")
        
        try:
            # Implement rollback logic based on target
            if record.config.target == DeploymentTarget.LOCAL:
                # Stop local process
                pass
            elif record.config.target == DeploymentTarget.FLYIO:
                # Rollback Fly.io deployment
                pass
            elif record.config.target == DeploymentTarget.KUBERNETES:
                # Delete Kubernetes deployment
                subprocess.run([
                    "kubectl", "delete", "deployment", f"ai-trader-{record.model_id}"
                ])
            
            record.status = DeploymentStatus.ROLLED_BACK
            record.add_log("Rollback completed")
            
        except Exception as e:
            record.add_log(f"Rollback failed: {e}", "ERROR")
    
    async def terminate_deployment(self, deployment_id: str) -> bool:
        """Terminate an active deployment."""
        if deployment_id not in self.deployments:
            return False
        
        record = self.deployments[deployment_id]
        
        try:
            # Cancel active deployment task
            if deployment_id in self.active_deployments:
                task = self.active_deployments[deployment_id]
                task.cancel()
            
            # Terminate based on target
            if record.config.target == DeploymentTarget.KUBERNETES:
                subprocess.run([
                    "kubectl", "delete", "deployment", f"ai-trader-{record.model_id}"
                ])
            
            record.status = DeploymentStatus.TERMINATED
            record.terminated_at = datetime.now()
            record.add_log("Deployment terminated")
            
            with self.deployment_lock:
                self._save_deployments()
            
            return True
            
        except Exception as e:
            record.add_log(f"Termination failed: {e}", "ERROR")
            return False
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a deployment."""
        if deployment_id in self.deployments:
            return self.deployments[deployment_id].to_dict()
        return None
    
    def list_deployments(self, model_id: str = None, 
                        status: DeploymentStatus = None) -> List[Dict[str, Any]]:
        """List deployments with optional filtering."""
        deployments = []
        
        for record in self.deployments.values():
            if model_id and record.model_id != model_id:
                continue
            if status and record.status != status:
                continue
            
            deployments.append(record.to_dict())
        
        # Sort by creation time (newest first)
        deployments.sort(key=lambda x: x['created_at'], reverse=True)
        
        return deployments
    
    def get_deployment_logs(self, deployment_id: str) -> List[str]:
        """Get logs for a deployment."""
        if deployment_id in self.deployments:
            return self.deployments[deployment_id].logs
        return []
    
    async def start_monitoring(self):
        """Start deployment monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitor_deployments())
    
    async def stop_monitoring(self):
        """Stop deployment monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
    
    async def _monitor_deployments(self):
        """Monitor active deployments."""
        while True:
            try:
                # Check health of deployed models
                for record in self.deployments.values():
                    if record.status == DeploymentStatus.DEPLOYED:
                        await self._check_deployment_health(record)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Deployment monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_deployment_health(self, record: DeploymentRecord):
        """Check health of a deployed model."""
        api_endpoint = record.endpoints.get('api')
        if not api_endpoint:
            return
        
        try:
            response = requests.get(f"{api_endpoint}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'healthy',
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
            else:
                health_data = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'unhealthy',
                    'error': f"HTTP {response.status_code}"
                }
            
            record.health_checks.append(health_data)
            
            # Keep only recent health checks
            if len(record.health_checks) > 100:
                record.health_checks = record.health_checks[-100:]
            
        except Exception as e:
            health_data = {
                'timestamp': datetime.now().isoformat(),
                'status': 'unhealthy',
                'error': str(e)
            }
            record.health_checks.append(health_data)


# Factory function
def create_deployment_orchestrator(storage_path: str = "model_management") -> DeploymentOrchestrator:
    """Create deployment orchestrator."""
    return DeploymentOrchestrator(storage_path)