"""
Neural Models Client
===================

Python client for managing neural network models with Supabase persistence.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID
from decimal import Decimal

from ..client import AsyncSupabaseClient, SupabaseError
from ..models.database_models import (
    NeuralModel, TrainingRun, ModelPrediction,
    CreateModelRequest, StartTrainingRequest,
    ModelStatus
)

logger = logging.getLogger(__name__)

class NeuralModelsClient:
    """
    Client for managing neural network models and training workflows.
    """
    
    def __init__(self, supabase_client: AsyncSupabaseClient):
        """
        Initialize neural models client.
        
        Args:
            supabase_client: Async Supabase client instance
        """
        self.client = supabase_client
    
    async def create_model(
        self, 
        user_id: UUID, 
        model_data: CreateModelRequest
    ) -> Tuple[Optional[NeuralModel], Optional[str]]:
        """
        Create a new neural model.
        
        Args:
            user_id: ID of the user creating the model
            model_data: Model creation request data
            
        Returns:
            Tuple of (created model, error message if any)
        """
        try:
            data = {
                "user_id": str(user_id),
                "name": model_data.name,
                "model_type": model_data.model_type,
                "architecture": model_data.architecture,
                "parameters": model_data.parameters,
                "training_data_hash": model_data.training_data_hash,
                "status": ModelStatus.TRAINING.value,
                "version": 1
            }
            
            result = await self.client.insert("neural_models", data)
            
            if result:
                model = NeuralModel.from_db(result[0])
                logger.info(f"Created neural model: {model.id}")
                return model, None
            else:
                return None, "Failed to create model"
                
        except Exception as e:
            logger.error(f"Error creating neural model: {e}")
            return None, str(e)
    
    async def get_user_models(
        self,
        user_id: UUID,
        status: Optional[ModelStatus] = None,
        model_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[NeuralModel], Optional[str]]:
        """
        Get neural models for a user with optional filters.
        
        Args:
            user_id: ID of the user
            status: Optional status filter
            model_type: Optional model type filter
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of models, error message if any)
        """
        try:
            filters = {"user_id": str(user_id)}
            
            if status:
                filters["status"] = status.value
            
            if model_type:
                filters["model_type"] = model_type
            
            result = await self.client.select(
                "neural_models",
                "*",
                filter_dict=filters,
                order_by="-created_at",
                limit=limit,
                offset=offset
            )
            
            models = [NeuralModel.from_db(record) for record in result]
            logger.debug(f"Retrieved {len(models)} models for user {user_id}")
            return models, None
            
        except Exception as e:
            logger.error(f"Error retrieving models: {e}")
            return [], str(e)
    
    async def get_model_by_id(
        self, 
        model_id: UUID, 
        user_id: Optional[UUID] = None
    ) -> Tuple[Optional[NeuralModel], Optional[str]]:
        """
        Get a specific model by ID.
        
        Args:
            model_id: ID of the model
            user_id: Optional user ID for access control
            
        Returns:
            Tuple of (model, error message if any)
        """
        try:
            filters = {"id": str(model_id)}
            
            if user_id:
                filters["user_id"] = str(user_id)
            
            result = await self.client.select(
                "neural_models",
                "*",
                filter_dict=filters,
                limit=1
            )
            
            if result:
                model = NeuralModel.from_db(result[0])
                return model, None
            else:
                return None, "Model not found"
                
        except Exception as e:
            logger.error(f"Error retrieving model {model_id}: {e}")
            return None, str(e)
    
    async def start_training(
        self, 
        training_data: StartTrainingRequest
    ) -> Tuple[Optional[TrainingRun], Optional[str]]:
        """
        Start training a neural model.
        
        Args:
            training_data: Training configuration
            
        Returns:
            Tuple of (training run, error message if any)
        """
        try:
            # Verify model exists and get user access
            model, error = await self.get_model_by_id(training_data.model_id)
            if error or not model:
                return None, error or "Model not found"
            
            # Create training run record
            data = {
                "model_id": str(training_data.model_id),
                "status": "running",
                "hyperparameters": training_data.hyperparameters,
                "epoch": 0
            }
            
            result = await self.client.insert("training_runs", data)
            
            if result:
                # Update model status to training
                await self.client.update(
                    "neural_models",
                    {"status": ModelStatus.TRAINING.value},
                    {"id": str(training_data.model_id)}
                )
                
                training_run = TrainingRun.from_db(result[0])
                logger.info(f"Started training run: {training_run.id}")
                return training_run, None
            else:
                return None, "Failed to create training run"
                
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return None, str(e)
    
    async def update_training_progress(
        self,
        training_run_id: UUID,
        epoch: int,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        validation_loss: Optional[float] = None,
        validation_accuracy: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        logs: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Update training progress for a training run.
        
        Args:
            training_run_id: ID of the training run
            epoch: Current epoch number
            loss: Training loss
            accuracy: Training accuracy
            validation_loss: Validation loss
            validation_accuracy: Validation accuracy
            metrics: Additional metrics
            logs: Training logs
            
        Returns:
            Tuple of (success, error message if any)
        """
        try:
            update_data = {
                "epoch": epoch,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if loss is not None:
                update_data["loss"] = loss
            if accuracy is not None:
                update_data["accuracy"] = accuracy
            if validation_loss is not None:
                update_data["validation_loss"] = validation_loss
            if validation_accuracy is not None:
                update_data["validation_accuracy"] = validation_accuracy
            if metrics is not None:
                update_data["metrics"] = metrics
            if logs is not None:
                update_data["logs"] = logs
            
            result = await self.client.update(
                "training_runs",
                update_data,
                {"id": str(training_run_id)}
            )
            
            if result:
                logger.debug(f"Updated training progress for run {training_run_id}")
                return True, None
            else:
                return False, "Failed to update training progress"
                
        except Exception as e:
            logger.error(f"Error updating training progress: {e}")
            return False, str(e)
    
    async def complete_training(
        self,
        training_run_id: UUID,
        final_metrics: Dict[str, Any],
        model_path: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Mark training as completed and update model.
        
        Args:
            training_run_id: ID of the training run
            final_metrics: Final training metrics
            model_path: Path to saved model file
            
        Returns:
            Tuple of (success, error message if any)
        """
        try:
            # Update training run as completed
            training_update = {
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "metrics": final_metrics
            }
            
            training_result = await self.client.update(
                "training_runs",
                training_update,
                {"id": str(training_run_id)}
            )
            
            if not training_result:
                return False, "Failed to update training run"
            
            # Get the training run to find model ID
            training_run = TrainingRun.from_db(training_result[0])
            
            # Update model status and metrics
            model_update = {
                "status": ModelStatus.TRAINED.value,
                "performance_metrics": final_metrics
            }
            
            if model_path:
                model_update["model_path"] = model_path
            
            model_result = await self.client.update(
                "neural_models",
                model_update,
                {"id": str(training_run.model_id)}
            )
            
            if model_result:
                logger.info(f"Completed training for run {training_run_id}")
                return True, None
            else:
                return False, "Failed to update model after training"
                
        except Exception as e:
            logger.error(f"Error completing training: {e}")
            return False, str(e)
    
    async def store_prediction(
        self,
        model_id: UUID,
        symbol_id: UUID,
        prediction_value: float,
        confidence: Optional[float] = None,
        actual_value: Optional[float] = None,
        features: Optional[Dict[str, Any]] = None,
        prediction_timestamp: Optional[datetime] = None
    ) -> Tuple[Optional[ModelPrediction], Optional[str]]:
        """
        Store a model prediction.
        
        Args:
            model_id: ID of the model
            symbol_id: ID of the symbol
            prediction_value: Predicted value
            confidence: Prediction confidence
            actual_value: Actual value (for accuracy calculation)
            features: Input features used for prediction
            prediction_timestamp: Timestamp of prediction
            
        Returns:
            Tuple of (prediction record, error message if any)
        """
        try:
            data = {
                "model_id": str(model_id),
                "symbol_id": str(symbol_id),
                "prediction_value": prediction_value,
                "prediction_timestamp": (
                    prediction_timestamp or datetime.utcnow()
                ).isoformat(),
                "features": features or {}
            }
            
            if confidence is not None:
                data["confidence"] = confidence
            if actual_value is not None:
                data["actual_value"] = actual_value
            
            result = await self.client.insert("model_predictions", data)
            
            if result:
                prediction = ModelPrediction.from_db(result[0])
                logger.debug(f"Stored prediction: {prediction.id}")
                return prediction, None
            else:
                return None, "Failed to store prediction"
                
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            return None, str(e)
    
    async def get_model_predictions(
        self,
        model_id: UUID,
        symbol_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[ModelPrediction], Optional[str]]:
        """
        Get predictions for a model with optional filters.
        
        Args:
            model_id: ID of the model
            symbol_id: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of predictions, error message if any)
        """
        try:
            filters = {"model_id": str(model_id)}
            
            if symbol_id:
                filters["symbol_id"] = str(symbol_id)
            
            # Note: Date filtering would need custom SQL for proper range queries
            result = await self.client.select(
                "model_predictions",
                "*",
                filter_dict=filters,
                order_by="-prediction_timestamp",
                limit=limit,
                offset=offset
            )
            
            predictions = [ModelPrediction.from_db(record) for record in result]
            logger.debug(f"Retrieved {len(predictions)} predictions for model {model_id}")
            return predictions, None
            
        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}")
            return [], str(e)
    
    async def update_model_performance(
        self, 
        model_id: UUID, 
        predictions_count: int = 100
    ) -> Tuple[bool, Optional[str]]:
        """
        Update model performance metrics using database function.
        
        Args:
            model_id: ID of the model
            predictions_count: Number of recent predictions to analyze
            
        Returns:
            Tuple of (success, error message if any)
        """
        try:
            result = await self.client.rpc(
                "update_model_performance",
                {
                    "model_id_param": str(model_id),
                    "predictions_count": predictions_count
                }
            )
            
            if result:
                logger.info(f"Updated performance metrics for model {model_id}")
                return True, None
            else:
                return False, "Failed to update model performance"
                
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
            return False, str(e)
    
    async def deploy_model(self, model_id: UUID) -> Tuple[bool, Optional[str]]:
        """
        Mark model as deployed.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Tuple of (success, error message if any)
        """
        try:
            result = await self.client.update(
                "neural_models",
                {"status": ModelStatus.DEPLOYED.value},
                {"id": str(model_id)}
            )
            
            if result:
                logger.info(f"Deployed model {model_id}")
                return True, None
            else:
                return False, "Failed to deploy model"
                
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return False, str(e)
    
    async def get_training_history(
        self, 
        model_id: UUID
    ) -> Tuple[List[TrainingRun], Optional[str]]:
        """
        Get training history for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Tuple of (list of training runs, error message if any)
        """
        try:
            result = await self.client.select(
                "training_runs",
                "*",
                filter_dict={"model_id": str(model_id)},
                order_by="-started_at"
            )
            
            training_runs = [TrainingRun.from_db(record) for record in result]
            return training_runs, None
            
        except Exception as e:
            logger.error(f"Error retrieving training history: {e}")
            return [], str(e)
    
    async def update_prediction_actual(
        self,
        prediction_id: UUID,
        actual_value: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Update a prediction with its actual value.
        
        Args:
            prediction_id: ID of the prediction
            actual_value: Actual observed value
            
        Returns:
            Tuple of (success, error message if any)
        """
        try:
            result = await self.client.update(
                "model_predictions",
                {"actual_value": actual_value},
                {"id": str(prediction_id)}
            )
            
            if result:
                logger.debug(f"Updated prediction {prediction_id} with actual value")
                return True, None
            else:
                return False, "Failed to update prediction"
                
        except Exception as e:
            logger.error(f"Error updating prediction: {e}")
            return False, str(e)
    
    async def delete_model(self, model_id: UUID) -> Tuple[bool, Optional[str]]:
        """
        Delete a model and all associated data.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Tuple of (success, error message if any)
        """
        try:
            # Delete in order due to foreign key constraints
            await self.client.delete("model_predictions", {"model_id": str(model_id)})
            await self.client.delete("training_runs", {"model_id": str(model_id)})
            result = await self.client.delete("neural_models", {"id": str(model_id)})
            
            if result:
                logger.info(f"Deleted model {model_id}")
                return True, None
            else:
                return False, "Failed to delete model"
                
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False, str(e)
    
    async def compare_models(
        self, 
        model_ids: List[UUID]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Compare performance of multiple models.
        
        Args:
            model_ids: List of model IDs to compare
            
        Returns:
            Tuple of (comparison data, error message if any)
        """
        try:
            # Get models with their performance metrics
            model_data = []
            
            for model_id in model_ids:
                model, error = await self.get_model_by_id(model_id)
                if error or not model:
                    continue
                
                # Get recent predictions for accuracy calculation
                predictions, _ = await self.get_model_predictions(
                    model_id, limit=100
                )
                
                # Calculate accuracy from predictions with actual values
                accurate_predictions = 0
                total_with_actual = 0
                
                for pred in predictions:
                    if pred.actual_value is not None:
                        total_with_actual += 1
                        if pred.accuracy and pred.accuracy > 0.95:  # 95% accuracy threshold
                            accurate_predictions += 1
                
                recent_accuracy = (
                    accurate_predictions / total_with_actual 
                    if total_with_actual > 0 else 0
                )
                
                model_comparison = {
                    "id": str(model.id),
                    "name": model.name,
                    "model_type": model.model_type,
                    "status": model.status,
                    "performance_metrics": model.performance_metrics,
                    "recent_accuracy": recent_accuracy,
                    "prediction_count": total_with_actual,
                    "created_at": model.created_at.isoformat()
                }
                
                model_data.append(model_comparison)
            
            return model_data, None
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return [], str(e)
    
    async def get_model_statistics(
        self, 
        user_id: UUID
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Get statistics about user's models.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Tuple of (statistics, error message if any)
        """
        try:
            models, error = await self.get_user_models(user_id, limit=1000)
            if error:
                return {}, error
            
            # Calculate statistics
            total_models = len(models)
            by_status = {}
            by_type = {}
            
            for model in models:
                # Count by status
                status = model.status.value if isinstance(model.status, ModelStatus) else model.status
                by_status[status] = by_status.get(status, 0) + 1
                
                # Count by type
                by_type[model.model_type] = by_type.get(model.model_type, 0) + 1
            
            statistics = {
                "total_models": total_models,
                "by_status": by_status,
                "by_type": by_type,
                "deployed_models": by_status.get("deployed", 0),
                "training_models": by_status.get("training", 0)
            }
            
            return statistics, None
            
        except Exception as e:
            logger.error(f"Error getting model statistics: {e}")
            return {}, str(e)