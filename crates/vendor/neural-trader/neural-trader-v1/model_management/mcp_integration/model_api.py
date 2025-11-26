"""REST API for Model Management Operations."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import asyncio
from enum import Enum
import tempfile
import shutil

# Import storage components
from ..storage.model_storage import ModelStorage, ModelMetadata as StorageMetadata, ModelFormat, CompressionLevel
from ..storage.metadata_manager import MetadataManager, ModelMetadata, ModelStatus
from ..storage.version_control import ModelVersionControl

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Pydantic models for API
class ModelStatus(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    VALIDATED = "validated"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class ModelFormat(str, Enum):
    PICKLE = "pickle"
    JOBLIB = "joblib"
    JSON = "json"
    COMPRESSED_PICKLE = "cpickle"
    COMPRESSED_JOBLIB = "cjoblib"

class ModelCreateRequest(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    strategy_name: str = Field(..., description="Trading strategy name")
    model_type: str = Field(..., description="Type of model")
    description: str = Field("", description="Model description")
    parameters: Dict[str, Any] = Field(..., description="Model parameters")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    author: str = Field("API User", description="Model author")

class ModelUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ModelStatus] = None
    parameters: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None

class ModelSearchRequest(BaseModel):
    query: Optional[str] = None
    strategy_name: Optional[str] = None
    status: Optional[ModelStatus] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    min_performance: Optional[Dict[str, float]] = None
    limit: int = Field(100, ge=1, le=1000)

class ModelVersionRequest(BaseModel):
    commit_message: str = Field(..., description="Version commit message")
    author: str = Field("API User", description="Commit author")
    branch: str = Field("main", description="Branch name")

class ModelDeployRequest(BaseModel):
    target_environment: str = Field(..., description="Target deployment environment")
    deployment_config: Dict[str, Any] = Field(default_factory=dict, description="Deployment configuration")
    auto_rollback: bool = Field(True, description="Enable automatic rollback on failure")

class ModelAPIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ModelAPI:
    """REST API for comprehensive model management."""
    
    def __init__(self, storage_path: str = "model_management"):
        """
        Initialize Model API.
        
        Args:
            storage_path: Base path for model storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage components
        self.model_storage = ModelStorage(str(self.storage_path / "models"))
        self.metadata_manager = MetadataManager(str(self.storage_path / "storage"))
        self.version_control = ModelVersionControl(str(self.storage_path / "models" / "versions"))
        
        # FastAPI app
        self.app = FastAPI(
            title="AI Trading Model Management API",
            description="Comprehensive API for managing AI trading models",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Model API initialized")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # Health and status endpoints
        @self.app.get("/health", response_model=ModelAPIResponse)
        async def health_check():
            """API health check."""
            storage_stats = self.model_storage.get_storage_stats()
            
            return ModelAPIResponse(
                success=True,
                message="API is healthy",
                data={
                    "api_status": "healthy",
                    "storage_stats": storage_stats,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        @self.app.get("/status", response_model=ModelAPIResponse)
        async def get_system_status():
            """Get comprehensive system status."""
            try:
                storage_stats = self.model_storage.get_storage_stats()
                version_stats = self.version_control.get_version_statistics()
                
                return ModelAPIResponse(
                    success=True,
                    message="System status retrieved",
                    data={
                        "storage": storage_stats,
                        "version_control": version_stats,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Model CRUD operations
        @self.app.post("/models", response_model=ModelAPIResponse)
        async def create_model(request: ModelCreateRequest):
            """Create a new model."""
            try:
                # Create metadata
                metadata = ModelMetadata(
                    model_id="",  # Will be generated
                    name=request.name,
                    version=request.version,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    model_type=request.model_type,
                    strategy_name=request.strategy_name,
                    status=ModelStatus.DEVELOPMENT,
                    performance_metrics=request.performance_metrics,
                    parameters=request.parameters,
                    tags=set(request.tags),
                    description=request.description,
                    author=request.author
                )
                
                # Create storage metadata
                storage_metadata = StorageMetadata(
                    model_id="",
                    name=request.name,
                    version=request.version,
                    created_at=datetime.now(),
                    model_type=request.model_type,
                    strategy_name=request.strategy_name,
                    performance_metrics=request.performance_metrics,
                    parameters=request.parameters,
                    tags=request.tags,
                    description=request.description,
                    author=request.author
                )
                
                # Save model (parameters as model data)
                model_id = self.model_storage.save_model(
                    request.parameters,
                    storage_metadata,
                    ModelFormat.JSON
                )
                
                # Update metadata with generated ID
                metadata.model_id = model_id
                
                # Save metadata
                self.metadata_manager.save_metadata(metadata)
                
                # Create initial version
                self.version_control.commit_version(
                    model_id=model_id,
                    model_data=request.parameters,
                    parameters=request.parameters,
                    performance_metrics=request.performance_metrics,
                    commit_message=f"Initial version of {request.name}",
                    author=request.author
                )
                
                return ModelAPIResponse(
                    success=True,
                    message="Model created successfully",
                    data={"model_id": model_id}
                )
                
            except Exception as e:
                logger.error(f"Failed to create model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models", response_model=ModelAPIResponse)
        async def list_models(request: ModelSearchRequest = Depends()):
            """List models with optional filtering."""
            try:
                models = self.metadata_manager.search_models(
                    query=request.query,
                    strategy_name=request.strategy_name,
                    status=ModelStatus(request.status) if request.status else None,
                    tags=request.tags,
                    created_after=request.created_after,
                    created_before=request.created_before,
                    min_performance=request.min_performance,
                    limit=request.limit
                )
                
                model_list = []
                for model in models:
                    model_dict = model.to_dict()
                    model_dict['tags'] = list(model_dict['tags'])  # Convert set to list
                    model_list.append(model_dict)
                
                return ModelAPIResponse(
                    success=True,
                    message=f"Found {len(model_list)} models",
                    data={
                        "models": model_list,
                        "total": len(model_list)
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_id}", response_model=ModelAPIResponse)
        async def get_model(model_id: str):
            """Get model details by ID."""
            try:
                metadata = self.metadata_manager.load_metadata(model_id)
                if not metadata:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Get model data
                try:
                    model_data, storage_metadata = self.model_storage.load_model(model_id)
                except:
                    model_data = None
                
                # Get version history
                version_history = self.version_control.get_version_history(model_id)
                
                # Get performance evaluation
                evaluation = self.metadata_manager.evaluate_model_performance(model_id)
                
                model_dict = metadata.to_dict()
                model_dict['tags'] = list(model_dict['tags'])  # Convert set to list
                
                return ModelAPIResponse(
                    success=True,
                    message="Model retrieved successfully",
                    data={
                        "metadata": model_dict,
                        "model_data": model_data,
                        "version_history": [v.to_dict() for v in version_history[-5:]],  # Last 5 versions
                        "evaluation": evaluation
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get model {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/models/{model_id}", response_model=ModelAPIResponse)
        async def update_model(model_id: str, request: ModelUpdateRequest):
            """Update model metadata."""
            try:
                metadata = self.metadata_manager.load_metadata(model_id)
                if not metadata:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Update fields
                if request.name:
                    metadata.name = request.name
                if request.description:
                    metadata.description = request.description
                if request.status:
                    metadata.status = ModelStatus(request.status)
                if request.parameters:
                    metadata.parameters.update(request.parameters)
                if request.performance_metrics:
                    metadata.performance_metrics.update(request.performance_metrics)
                if request.tags is not None:
                    metadata.tags = set(request.tags)
                
                metadata.updated_at = datetime.now()
                
                # Save updated metadata
                self.metadata_manager.save_metadata(metadata)
                
                return ModelAPIResponse(
                    success=True,
                    message="Model updated successfully",
                    data={"model_id": model_id}
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to update model {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/models/{model_id}", response_model=ModelAPIResponse)
        async def delete_model(model_id: str):
            """Delete a model."""
            try:
                # Check if model exists
                metadata = self.metadata_manager.load_metadata(model_id)
                if not metadata:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Create backup before deletion
                backup_path = self.model_storage.backup_model(model_id)
                
                # Delete from storage
                self.model_storage.delete_model(model_id)
                
                return ModelAPIResponse(
                    success=True,
                    message="Model deleted successfully",
                    data={
                        "model_id": model_id,
                        "backup_path": backup_path
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to delete model {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Version control endpoints
        @self.app.post("/models/{model_id}/versions", response_model=ModelAPIResponse)
        async def create_version(model_id: str, request: ModelVersionRequest):
            """Create a new version of a model."""
            try:
                # Get current model data
                model_data, metadata = self.model_storage.load_model(model_id)
                
                # Create new version
                version_id = self.version_control.commit_version(
                    model_id=model_id,
                    model_data=model_data,
                    parameters=metadata.parameters,
                    performance_metrics=metadata.performance_metrics,
                    commit_message=request.commit_message,
                    author=request.author,
                    branch=request.branch
                )
                
                return ModelAPIResponse(
                    success=True,
                    message="Version created successfully",
                    data={"version_id": version_id}
                )
                
            except Exception as e:
                logger.error(f"Failed to create version for model {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_id}/versions", response_model=ModelAPIResponse)
        async def get_version_history(model_id: str, branch: str = None):
            """Get version history for a model."""
            try:
                versions = self.version_control.get_version_history(model_id, branch)
                
                return ModelAPIResponse(
                    success=True,
                    message="Version history retrieved",
                    data={
                        "versions": [v.to_dict() for v in versions],
                        "total": len(versions)
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to get version history for model {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/{model_id}/rollback/{version_id}", response_model=ModelAPIResponse)
        async def rollback_model(model_id: str, version_id: str, branch: str = "main"):
            """Rollback model to a specific version."""
            try:
                success = self.version_control.rollback_to_version(model_id, version_id, branch)
                
                if not success:
                    raise HTTPException(status_code=400, detail="Rollback failed")
                
                return ModelAPIResponse(
                    success=True,
                    message="Model rolled back successfully",
                    data={"model_id": model_id, "version_id": version_id}
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to rollback model {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Analytics endpoints
        @self.app.get("/analytics/strategies/{strategy_name}", response_model=ModelAPIResponse)
        async def get_strategy_analytics(strategy_name: str):
            """Get analytics for a specific strategy."""
            try:
                analytics = self.metadata_manager.get_strategy_analytics(strategy_name)
                
                return ModelAPIResponse(
                    success=True,
                    message="Strategy analytics retrieved",
                    data=analytics
                )
                
            except Exception as e:
                logger.error(f"Failed to get analytics for strategy {strategy_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/leaderboard/{metric}", response_model=ModelAPIResponse)
        async def get_performance_leaderboard(metric: str, strategy_name: str = None, limit: int = 10):
            """Get performance leaderboard for a metric."""
            try:
                leaderboard = self.metadata_manager.get_performance_leaderboard(
                    metric, strategy_name, limit
                )
                
                return ModelAPIResponse(
                    success=True,
                    message="Leaderboard retrieved",
                    data={
                        "metric": metric,
                        "leaderboard": leaderboard,
                        "strategy_filter": strategy_name
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to get leaderboard for metric {metric}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/performance/{model_id}", response_model=ModelAPIResponse)
        async def evaluate_model_performance(model_id: str):
            """Evaluate model performance against benchmarks."""
            try:
                evaluation = self.metadata_manager.evaluate_model_performance(model_id)
                
                return ModelAPIResponse(
                    success=True,
                    message="Performance evaluation completed",
                    data=evaluation
                )
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # File upload endpoints
        @self.app.post("/models/{model_id}/upload", response_model=ModelAPIResponse)
        async def upload_model_file(model_id: str, file: UploadFile = File(...),
                                  format_type: ModelFormat = ModelFormat.PICKLE):
            """Upload model file."""
            try:
                # Check if model exists
                metadata = self.metadata_manager.load_metadata(model_id)
                if not metadata:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    shutil.copyfileobj(file.file, temp_file)
                    temp_path = temp_file.name
                
                try:
                    # Load the model from uploaded file
                    if format_type == ModelFormat.PICKLE:
                        import pickle
                        with open(temp_path, 'rb') as f:
                            model_data = pickle.load(f)
                    elif format_type == ModelFormat.JOBLIB:
                        import joblib
                        model_data = joblib.load(temp_path)
                    elif format_type == ModelFormat.JSON:
                        with open(temp_path, 'r') as f:
                            model_data = json.load(f)
                    else:
                        raise ValueError(f"Unsupported format: {format_type}")
                    
                    # Update model storage
                    storage_metadata = StorageMetadata(
                        model_id=model_id,
                        name=metadata.name,
                        version=metadata.version,
                        created_at=metadata.created_at,
                        model_type=metadata.model_type,
                        strategy_name=metadata.strategy_name,
                        performance_metrics=metadata.performance_metrics,
                        parameters=metadata.parameters,
                        tags=list(metadata.tags),
                        description=metadata.description,
                        author=metadata.author
                    )
                    
                    # Save updated model
                    new_model_id = self.model_storage.save_model(
                        model_data, storage_metadata, ModelFormat(format_type)
                    )
                    
                    return ModelAPIResponse(
                        success=True,
                        message="Model file uploaded successfully",
                        data={
                            "model_id": model_id,
                            "new_model_id": new_model_id,
                            "file_name": file.filename,
                            "format": format_type
                        }
                    )
                    
                finally:
                    # Clean up temporary file
                    Path(temp_path).unlink(missing_ok=True)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to upload model file: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Export endpoints
        @self.app.get("/export/models", response_model=ModelAPIResponse)
        async def export_models(strategy_name: str = None, background_tasks: BackgroundTasks = None):
            """Export models to JSON file."""
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"models_export_{timestamp}.json"
                output_path = self.storage_path / "exports" / filename
                output_path.parent.mkdir(exist_ok=True)
                
                # Export metadata
                success = self.metadata_manager.export_metadata(str(output_path), strategy_name)
                
                if not success:
                    raise HTTPException(status_code=500, detail="Export failed")
                
                return ModelAPIResponse(
                    success=True,
                    message="Models exported successfully",
                    data={
                        "export_file": str(output_path),
                        "strategy_filter": strategy_name,
                        "timestamp": timestamp
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to export models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Maintenance endpoints
        @self.app.post("/maintenance/cleanup", response_model=ModelAPIResponse)
        async def cleanup_old_data(days_old: int = 30, keep_production: bool = True):
            """Clean up old model data."""
            try:
                # Cleanup metadata
                deleted_count = self.metadata_manager.cleanup_metadata(days_old, keep_production)
                
                return ModelAPIResponse(
                    success=True,
                    message="Cleanup completed",
                    data={
                        "deleted_models": deleted_count,
                        "days_old": days_old,
                        "kept_production": keep_production
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to cleanup data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app


# Factory function
def create_model_api(storage_path: str = "model_management") -> FastAPI:
    """Create and configure the Model API."""
    api = ModelAPI(storage_path)
    return api.get_app()


# For running directly
if __name__ == "__main__":
    import uvicorn
    
    app = create_model_api()
    uvicorn.run(app, host="0.0.0.0", port=8001)