"""Model Context Protocol (MCP) Server for Trading Models."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import websockets
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Import our storage components
from ..storage.model_storage import ModelStorage, ModelMetadata
from ..storage.metadata_manager import MetadataManager, ModelStatus
from ..storage.version_control import ModelVersionControl

logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class ModelRequestType(Enum):
    """Types of model requests."""
    PREDICT = "predict"
    LOAD_MODEL = "load_model"
    LIST_MODELS = "list_models"
    GET_METADATA = "get_metadata"
    UPDATE_MODEL = "update_model"
    DELETE_MODEL = "delete_model"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_METRICS = "performance_metrics"
    STRATEGY_RECOMMENDATION = "strategy_recommendation"


@dataclass
class MCPMessage:
    """MCP protocol message."""
    message_type: MCPMessageType
    request_id: str
    method: str
    params: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'message_type': self.message_type.value,
            'request_id': self.request_id,
            'method': self.method,
            'params': self.params,
            'timestamp': self.timestamp.isoformat()
        }


class MCPResponse(BaseModel):
    """MCP response model."""
    request_id: str
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    timestamp: str
    processing_time_ms: float


class ModelPredictionRequest(BaseModel):
    """Model prediction request."""
    model_id: str
    input_data: Dict[str, Any]
    strategy_context: Optional[Dict[str, Any]] = None
    return_confidence: bool = False
    timeout_seconds: int = 30


class ModelListRequest(BaseModel):
    """Model listing request."""
    strategy_name: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 100


class TradingMCPServer:
    """MCP server for trading model access and inference."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000,
                 model_storage_path: str = "model_management/models"):
        """
        Initialize MCP server.
        
        Args:
            host: Server host
            port: Server port
            model_storage_path: Path to model storage
        """
        self.host = host
        self.port = port
        self.model_storage_path = model_storage_path
        
        # Initialize storage components
        self.model_storage = ModelStorage(model_storage_path)
        self.metadata_manager = MetadataManager(f"{model_storage_path}/../storage")
        self.version_control = ModelVersionControl(f"{model_storage_path}/versions")
        
        # FastAPI app
        self.app = FastAPI(
            title="Trading Model MCP Server",
            description="Model Context Protocol server for AI trading models",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Model cache for loaded models
        self.model_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = timedelta(minutes=30)
        
        # Active WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Performance tracking
        self.request_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'last_reset': datetime.now()
        }
        
        # Thread pool for heavy operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"MCP server initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models_loaded": len(self.model_cache),
                "total_models": len(self.model_storage._model_registry),
                "server_uptime": time.time()
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get server performance metrics."""
            return self.request_metrics
        
        @self.app.post("/models/predict")
        async def predict_model(request: ModelPredictionRequest):
            """Make prediction using specified model."""
            start_time = time.time()
            
            try:
                # Load model if not cached
                model, metadata = await self._get_or_load_model(request.model_id)
                
                # Prepare input data
                input_data = self._prepare_input_data(request.input_data, metadata)
                
                # Make prediction
                prediction = await self._make_prediction(
                    model, input_data, metadata, request.strategy_context
                )
                
                # Add confidence if requested
                if request.return_confidence:
                    confidence = self._calculate_prediction_confidence(
                        prediction, metadata, input_data
                    )
                    prediction['confidence'] = confidence
                
                processing_time = (time.time() - start_time) * 1000
                self._update_metrics(True, processing_time)
                
                return MCPResponse(
                    request_id=f"pred_{int(time.time())}",
                    success=True,
                    data=prediction,
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=processing_time
                )
                
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                self._update_metrics(False, processing_time)
                
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models")
        async def list_models(request: ModelListRequest):
            """List available models."""
            try:
                models = self.metadata_manager.search_models(
                    strategy_name=request.strategy_name,
                    status=ModelStatus(request.status) if request.status else None,
                    tags=request.tags,
                    limit=request.limit
                )
                
                model_list = []
                for model in models:
                    model_info = {
                        'model_id': model.model_id,
                        'name': model.name,
                        'version': model.version,
                        'strategy_name': model.strategy_name,
                        'status': model.status.value,
                        'created_at': model.created_at.isoformat(),
                        'performance_metrics': model.performance_metrics,
                        'tags': list(model.tags),
                        'description': model.description
                    }
                    model_list.append(model_info)
                
                return MCPResponse(
                    request_id=f"list_{int(time.time())}",
                    success=True,
                    data={'models': model_list, 'total': len(model_list)},
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=0
                )
                
            except Exception as e:
                logger.error(f"Model listing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_id}/metadata")
        async def get_model_metadata(model_id: str):
            """Get detailed model metadata."""
            try:
                metadata = self.metadata_manager.load_metadata(model_id)
                if not metadata:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Get performance evaluation
                evaluation = self.metadata_manager.evaluate_model_performance(model_id)
                
                return MCPResponse(
                    request_id=f"meta_{int(time.time())}",
                    success=True,
                    data={
                        'metadata': metadata.to_dict(),
                        'evaluation': evaluation
                    },
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=0
                )
                
            except Exception as e:
                logger.error(f"Metadata retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/strategies/{strategy_name}/analytics")
        async def get_strategy_analytics(strategy_name: str):
            """Get analytics for a specific strategy."""
            try:
                analytics = self.metadata_manager.get_strategy_analytics(strategy_name)
                
                return MCPResponse(
                    request_id=f"analytics_{int(time.time())}",
                    success=True,
                    data=analytics,
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=0
                )
                
            except Exception as e:
                logger.error(f"Analytics retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/strategies/{strategy_name}/recommendation")
        async def get_strategy_recommendation(strategy_name: str):
            """Get trading recommendation for strategy."""
            try:
                # Get best performing model for strategy
                models = self.metadata_manager.search_models(
                    strategy_name=strategy_name,
                    status=ModelStatus.PRODUCTION,
                    limit=1
                )
                
                if not models:
                    raise HTTPException(status_code=404, detail="No production models found")
                
                best_model = models[0]
                model, metadata = await self._get_or_load_model(best_model.model_id)
                
                # Generate recommendation based on current market conditions
                recommendation = await self._generate_trading_recommendation(
                    model, metadata, strategy_name
                )
                
                return MCPResponse(
                    request_id=f"rec_{int(time.time())}",
                    success=True,
                    data=recommendation,
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=0
                )
                
            except Exception as e:
                logger.error(f"Recommendation generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time model updates."""
            await self._handle_websocket_connection(websocket)
    
    async def _get_or_load_model(self, model_id: str) -> tuple:
        """Get model from cache or load from storage."""
        with self.cache_lock:
            # Check cache
            if model_id in self.model_cache:
                cache_entry = self.model_cache[model_id]
                if datetime.now() - cache_entry['timestamp'] < self.cache_ttl:
                    return cache_entry['model'], cache_entry['metadata']
                else:
                    # Remove expired entry
                    del self.model_cache[model_id]
        
        # Load from storage
        try:
            model, metadata = self.model_storage.load_model(model_id)
            
            # Cache the model
            with self.cache_lock:
                self.model_cache[model_id] = {
                    'model': model,
                    'metadata': metadata,
                    'timestamp': datetime.now()
                }
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    def _prepare_input_data(self, input_data: Dict, metadata: ModelMetadata) -> Dict:
        """Prepare input data for model prediction."""
        # Strategy-specific input preparation
        strategy_name = metadata.strategy_name.lower()
        
        if 'mean_reversion' in strategy_name:
            return self._prepare_mean_reversion_input(input_data)
        elif 'momentum' in strategy_name:
            return self._prepare_momentum_input(input_data)
        elif 'mirror' in strategy_name:
            return self._prepare_mirror_trading_input(input_data)
        elif 'swing' in strategy_name:
            return self._prepare_swing_trading_input(input_data)
        else:
            # Generic preparation
            return input_data
    
    def _prepare_mean_reversion_input(self, input_data: Dict) -> Dict:
        """Prepare input for mean reversion strategy."""
        required_fields = [
            'z_score', 'price', 'moving_average', 'volatility',
            'volume_ratio', 'rsi', 'market_regime'
        ]
        
        prepared = {}
        for field in required_fields:
            if field in input_data:
                prepared[field] = float(input_data[field])
            else:
                # Set default values
                defaults = {
                    'z_score': 0.0,
                    'price': 100.0,
                    'moving_average': 100.0,
                    'volatility': 0.2,
                    'volume_ratio': 1.0,
                    'rsi': 50.0,
                    'market_regime': 0.5  # Neutral
                }
                prepared[field] = defaults.get(field, 0.0)
        
        return prepared
    
    def _prepare_momentum_input(self, input_data: Dict) -> Dict:
        """Prepare input for momentum strategy."""
        required_fields = [
            'price_change', 'volume_change', 'momentum_score',
            'trend_strength', 'volatility', 'market_sentiment'
        ]
        
        prepared = {}
        for field in required_fields:
            if field in input_data:
                prepared[field] = float(input_data[field])
            else:
                defaults = {
                    'price_change': 0.0,
                    'volume_change': 0.0,
                    'momentum_score': 0.0,
                    'trend_strength': 0.5,
                    'volatility': 0.2,
                    'market_sentiment': 0.5
                }
                prepared[field] = defaults.get(field, 0.0)
        
        return prepared
    
    def _prepare_mirror_trading_input(self, input_data: Dict) -> Dict:
        """Prepare input for mirror trading strategy."""
        required_fields = [
            'institutional_positions', 'position_changes', 'confidence_scores',
            'entry_timing', 'market_conditions'
        ]
        
        prepared = {}
        for field in required_fields:
            if field in input_data:
                prepared[field] = input_data[field]
            else:
                defaults = {
                    'institutional_positions': {},
                    'position_changes': {},
                    'confidence_scores': {},
                    'entry_timing': 'immediate',
                    'market_conditions': 'normal'
                }
                prepared[field] = defaults.get(field, {})
        
        return prepared
    
    def _prepare_swing_trading_input(self, input_data: Dict) -> Dict:
        """Prepare input for swing trading strategy."""
        required_fields = [
            'support_levels', 'resistance_levels', 'trend_direction',
            'volume_profile', 'time_in_cycle', 'risk_metrics'
        ]
        
        prepared = {}
        for field in required_fields:
            if field in input_data:
                prepared[field] = input_data[field]
            else:
                defaults = {
                    'support_levels': [],
                    'resistance_levels': [],
                    'trend_direction': 'neutral',
                    'volume_profile': 'normal',
                    'time_in_cycle': 0.5,
                    'risk_metrics': {}
                }
                prepared[field] = defaults.get(field, [])
        
        return prepared
    
    async def _make_prediction(self, model: Any, input_data: Dict, 
                             metadata: ModelMetadata, strategy_context: Dict = None) -> Dict:
        """Make prediction using the model."""
        try:
            # If model is a parameter set (dictionary), use rule-based prediction
            if isinstance(model, dict):
                return await self._rule_based_prediction(model, input_data, metadata)
            
            # If model has a predict method, use it
            if hasattr(model, 'predict'):
                prediction = model.predict(input_data)
                return self._format_prediction_output(prediction, metadata)
            
            # If model is callable, call it
            if callable(model):
                prediction = model(input_data)
                return self._format_prediction_output(prediction, metadata)
            
            # Fallback to rule-based prediction
            return await self._rule_based_prediction(model, input_data, metadata)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    async def _rule_based_prediction(self, model_params: Dict, input_data: Dict, 
                                   metadata: ModelMetadata) -> Dict:
        """Generate prediction using rule-based approach."""
        strategy_name = metadata.strategy_name.lower()
        
        if 'mean_reversion' in strategy_name:
            return self._mean_reversion_prediction(model_params, input_data)
        elif 'momentum' in strategy_name:
            return self._momentum_prediction(model_params, input_data)
        elif 'mirror' in strategy_name:
            return self._mirror_trading_prediction(model_params, input_data)
        elif 'swing' in strategy_name:
            return self._swing_trading_prediction(model_params, input_data)
        else:
            return {'action': 'hold', 'confidence': 0.5, 'position_size': 0.0}
    
    def _mean_reversion_prediction(self, params: Dict, input_data: Dict) -> Dict:
        """Mean reversion strategy prediction."""
        z_score = input_data.get('z_score', 0.0)
        entry_threshold = params.get('z_score_entry_threshold', 2.0)
        exit_threshold = params.get('z_score_exit_threshold', 0.5)
        base_position_size = params.get('base_position_size', 0.05)
        
        # Determine action
        if abs(z_score) >= entry_threshold:
            action = 'buy' if z_score < 0 else 'sell'
            confidence = min(abs(z_score) / entry_threshold, 1.0)
            position_size = base_position_size * confidence
        elif abs(z_score) <= exit_threshold:
            action = 'close'
            confidence = 1.0 - abs(z_score) / exit_threshold
            position_size = 0.0
        else:
            action = 'hold'
            confidence = 0.5
            position_size = 0.0
        
        return {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'z_score': z_score,
            'entry_threshold': entry_threshold,
            'reasoning': f"Z-score {z_score:.2f} vs threshold {entry_threshold:.2f}"
        }
    
    def _momentum_prediction(self, params: Dict, input_data: Dict) -> Dict:
        """Momentum strategy prediction."""
        momentum = input_data.get('momentum_score', 0.0)
        trend_strength = input_data.get('trend_strength', 0.5)
        threshold = params.get('momentum_threshold', 0.6)
        base_position_size = params.get('base_position_size', 0.05)
        
        # Calculate combined signal
        signal_strength = (momentum + trend_strength) / 2
        
        if signal_strength > threshold:
            action = 'buy'
            confidence = signal_strength
            position_size = base_position_size * confidence
        elif signal_strength < -threshold:
            action = 'sell'
            confidence = abs(signal_strength)
            position_size = base_position_size * confidence
        else:
            action = 'hold'
            confidence = 0.5
            position_size = 0.0
        
        return {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'signal_strength': signal_strength,
            'reasoning': f"Signal strength {signal_strength:.2f} vs threshold {threshold:.2f}"
        }
    
    def _mirror_trading_prediction(self, params: Dict, input_data: Dict) -> Dict:
        """Mirror trading strategy prediction."""
        positions = input_data.get('institutional_positions', {})
        confidence_scores = input_data.get('confidence_scores', {})
        
        # Calculate weighted institutional signal
        total_signal = 0.0
        total_weight = 0.0
        
        for institution, position in positions.items():
            confidence = confidence_scores.get(institution, 0.5)
            weight = params.get(f'{institution}_confidence', 0.5)
            
            total_signal += position * confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            signal = total_signal / total_weight
        else:
            signal = 0.0
        
        # Determine action
        min_signal = params.get('min_signal_threshold', 0.3)
        base_position_size = params.get('base_position_size', 0.05)
        
        if signal > min_signal:
            action = 'buy'
            confidence = signal
            position_size = min(base_position_size * signal, params.get('max_position_size', 0.15))
        elif signal < -min_signal:
            action = 'sell'
            confidence = abs(signal)
            position_size = min(base_position_size * abs(signal), params.get('max_position_size', 0.15))
        else:
            action = 'hold'
            confidence = 0.5
            position_size = 0.0
        
        return {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'institutional_signal': signal,
            'reasoning': f"Institutional signal {signal:.2f} vs threshold {min_signal:.2f}"
        }
    
    def _swing_trading_prediction(self, params: Dict, input_data: Dict) -> Dict:
        """Swing trading strategy prediction."""
        price = input_data.get('price', 100.0)
        support_levels = input_data.get('support_levels', [])
        resistance_levels = input_data.get('resistance_levels', [])
        
        # Find nearest support and resistance
        nearest_support = max([s for s in support_levels if s < price], default=price * 0.95)
        nearest_resistance = min([r for r in resistance_levels if r > price], default=price * 1.05)
        
        # Calculate position in range
        range_size = nearest_resistance - nearest_support
        position_in_range = (price - nearest_support) / range_size if range_size > 0 else 0.5
        
        base_position_size = params.get('base_position_size', 0.05)
        
        # Trading logic
        if position_in_range < 0.2:  # Near support
            action = 'buy'
            confidence = 1.0 - position_in_range / 0.2
            position_size = base_position_size * confidence
        elif position_in_range > 0.8:  # Near resistance
            action = 'sell'
            confidence = (position_in_range - 0.8) / 0.2
            position_size = base_position_size * confidence
        else:
            action = 'hold'
            confidence = 0.5
            position_size = 0.0
        
        return {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'position_in_range': position_in_range,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'reasoning': f"Price {price:.2f} at {position_in_range:.1%} of range [{nearest_support:.2f}, {nearest_resistance:.2f}]"
        }
    
    def _format_prediction_output(self, prediction: Any, metadata: ModelMetadata) -> Dict:
        """Format prediction output."""
        if isinstance(prediction, dict):
            return prediction
        
        # Convert various prediction formats to standard format
        if isinstance(prediction, (list, tuple)):
            if len(prediction) >= 3:
                return {
                    'action': prediction[0],
                    'confidence': float(prediction[1]),
                    'position_size': float(prediction[2])
                }
        
        if isinstance(prediction, (int, float)):
            # Single value prediction
            if prediction > 0.6:
                action = 'buy'
            elif prediction < 0.4:
                action = 'sell'
            else:
                action = 'hold'
            
            return {
                'action': action,
                'confidence': abs(prediction - 0.5) * 2,
                'position_size': metadata.parameters.get('base_position_size', 0.05) * abs(prediction - 0.5) * 2,
                'raw_prediction': float(prediction)
            }
        
        # Fallback
        return {'action': 'hold', 'confidence': 0.5, 'position_size': 0.0}
    
    def _calculate_prediction_confidence(self, prediction: Dict, metadata: ModelMetadata, 
                                       input_data: Dict) -> float:
        """Calculate prediction confidence score."""
        base_confidence = prediction.get('confidence', 0.5)
        
        # Adjust based on model performance
        sharpe_ratio = metadata.performance_metrics.get('sharpe_ratio', 1.0)
        win_rate = metadata.performance_metrics.get('win_rate', 0.5)
        
        # Performance-based adjustment
        performance_factor = min((sharpe_ratio / 2.0) * win_rate, 1.0)
        
        # Data quality adjustment
        data_quality = self._assess_input_data_quality(input_data)
        
        # Combined confidence
        adjusted_confidence = base_confidence * performance_factor * data_quality
        return max(0.1, min(0.95, adjusted_confidence))
    
    def _assess_input_data_quality(self, input_data: Dict) -> float:
        """Assess quality of input data."""
        quality_score = 1.0
        
        # Check for missing or invalid values
        for key, value in input_data.items():
            if value is None or (isinstance(value, (int, float)) and np.isnan(value)):
                quality_score *= 0.8
            elif isinstance(value, (list, dict)) and len(value) == 0:
                quality_score *= 0.9
        
        return max(0.3, quality_score)
    
    async def _generate_trading_recommendation(self, model: Any, metadata: ModelMetadata, 
                                             strategy_name: str) -> Dict:
        """Generate comprehensive trading recommendation."""
        # Simulate current market data for demonstration
        current_data = self._get_current_market_data(strategy_name)
        
        # Get prediction
        prediction = await self._make_prediction(model, current_data, metadata)
        
        # Add market context
        market_analysis = self._analyze_market_conditions()
        
        # Generate recommendation
        recommendation = {
            'strategy_name': strategy_name,
            'model_id': metadata.model_id,
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'market_analysis': market_analysis,
            'risk_assessment': self._assess_strategy_risk(metadata, market_analysis),
            'position_recommendations': self._generate_position_recommendations(
                prediction, metadata, market_analysis
            ),
            'confidence_level': prediction.get('confidence', 0.5),
            'expected_performance': {
                'sharpe_ratio': metadata.performance_metrics.get('sharpe_ratio', 1.0),
                'expected_return': metadata.performance_metrics.get('total_return', 0.15),
                'max_drawdown': metadata.performance_metrics.get('max_drawdown', 0.10)
            }
        }
        
        return recommendation
    
    def _get_current_market_data(self, strategy_name: str) -> Dict:
        """Get current market data for strategy (simulated)."""
        # In production, this would fetch real market data
        if 'mean_reversion' in strategy_name.lower():
            return {
                'z_score': np.random.normal(0, 1.5),
                'price': 100 + np.random.normal(0, 5),
                'moving_average': 100,
                'volatility': 0.15 + np.random.uniform(-0.05, 0.05),
                'volume_ratio': 1.0 + np.random.uniform(-0.3, 0.3),
                'rsi': 50 + np.random.uniform(-20, 20),
                'market_regime': np.random.uniform(0, 1)
            }
        elif 'momentum' in strategy_name.lower():
            return {
                'price_change': np.random.uniform(-0.05, 0.05),
                'volume_change': np.random.uniform(-0.3, 0.3),
                'momentum_score': np.random.uniform(-1, 1),
                'trend_strength': np.random.uniform(0, 1),
                'volatility': 0.2 + np.random.uniform(-0.05, 0.05),
                'market_sentiment': np.random.uniform(0, 1)
            }
        else:
            return {}
    
    def _analyze_market_conditions(self) -> Dict:
        """Analyze current market conditions."""
        # Simulated market analysis
        return {
            'market_regime': np.random.choice(['bull', 'bear', 'sideways']),
            'volatility_level': np.random.choice(['low', 'medium', 'high']),
            'trend_strength': np.random.uniform(0, 1),
            'volume_profile': np.random.choice(['normal', 'high', 'low']),
            'sector_rotation': np.random.choice(['growth', 'value', 'defensive']),
            'risk_sentiment': np.random.uniform(0, 1)
        }
    
    def _assess_strategy_risk(self, metadata: ModelMetadata, market_analysis: Dict) -> Dict:
        """Assess risk for strategy given current conditions."""
        base_risk = metadata.performance_metrics.get('max_drawdown', 0.10)
        
        # Adjust risk based on market conditions
        volatility_multiplier = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.3
        }.get(market_analysis.get('volatility_level', 'medium'), 1.0)
        
        adjusted_risk = base_risk * volatility_multiplier
        
        return {
            'base_risk': base_risk,
            'adjusted_risk': adjusted_risk,
            'risk_level': 'low' if adjusted_risk < 0.08 else 'medium' if adjusted_risk < 0.15 else 'high',
            'risk_factors': [
                f"Market volatility: {market_analysis.get('volatility_level', 'medium')}",
                f"Market regime: {market_analysis.get('market_regime', 'unknown')}"
            ]
        }
    
    def _generate_position_recommendations(self, prediction: Dict, metadata: ModelMetadata, 
                                         market_analysis: Dict) -> Dict:
        """Generate position sizing recommendations."""
        base_position = prediction.get('position_size', 0.05)
        confidence = prediction.get('confidence', 0.5)
        
        # Adjust for market conditions
        market_adjustment = 1.0
        if market_analysis.get('volatility_level') == 'high':
            market_adjustment *= 0.7
        elif market_analysis.get('volatility_level') == 'low':
            market_adjustment *= 1.2
        
        recommended_position = base_position * market_adjustment
        max_position = metadata.parameters.get('max_position_size', 0.15)
        
        return {
            'recommended_position_size': min(recommended_position, max_position),
            'max_position_size': max_position,
            'confidence_adjusted_size': recommended_position * confidence,
            'risk_adjusted_size': recommended_position * (1 - market_analysis.get('risk_sentiment', 0.5)),
            'stop_loss_level': metadata.parameters.get('stop_loss_multiplier', 1.5),
            'take_profit_level': metadata.parameters.get('profit_target_multiplier', 2.0)
        }
    
    def _update_metrics(self, success: bool, processing_time: float):
        """Update server performance metrics."""
        self.request_metrics['total_requests'] += 1
        
        if success:
            self.request_metrics['successful_requests'] += 1
        else:
            self.request_metrics['failed_requests'] += 1
        
        # Update average response time
        current_avg = self.request_metrics['average_response_time']
        total_requests = self.request_metrics['total_requests']
        
        self.request_metrics['average_response_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connection for real-time updates."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Wait for client messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get('type') == 'subscribe_strategy':
                    strategy_name = message.get('strategy_name')
                    await self._handle_strategy_subscription(websocket, strategy_name)
                
                elif message.get('type') == 'get_real_time_prediction':
                    model_id = message.get('model_id')
                    input_data = message.get('input_data', {})
                    
                    try:
                        model, metadata = await self._get_or_load_model(model_id)
                        prediction = await self._make_prediction(model, input_data, metadata)
                        
                        response = {
                            'type': 'prediction_response',
                            'model_id': model_id,
                            'prediction': prediction,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        await websocket.send_text(json.dumps(response))
                        
                    except Exception as e:
                        error_response = {
                            'type': 'error',
                            'message': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                        await websocket.send_text(json.dumps(error_response))
                
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
    
    async def _handle_strategy_subscription(self, websocket: WebSocket, strategy_name: str):
        """Handle strategy subscription for real-time updates."""
        # Send current strategy status
        try:
            analytics = self.metadata_manager.get_strategy_analytics(strategy_name)
            
            response = {
                'type': 'strategy_update',
                'strategy_name': strategy_name,
                'analytics': analytics,
                'timestamp': datetime.now().isoformat()
            }
            
            await websocket.send_text(json.dumps(response))
            
        except Exception as e:
            error_response = {
                'type': 'error',
                'message': f"Failed to get strategy analytics: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(error_response))
    
    async def broadcast_update(self, message: Dict):
        """Broadcast update to all connected WebSocket clients."""
        if self.active_connections:
            message['timestamp'] = datetime.now().isoformat()
            message_json = json.dumps(message)
            
            # Send to all connections (remove failed ones)
            active_connections = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_json)
                    active_connections.append(connection)
                except:
                    pass  # Connection failed, will be removed
            
            self.active_connections = active_connections
    
    def start_server(self):
        """Start the MCP server."""
        logger.info(f"Starting MCP server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
    
    async def start_async(self):
        """Start the server asynchronously."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


# Convenience function to start server
def start_mcp_server(host: str = "0.0.0.0", port: int = 8000):
    """Start MCP server with default configuration."""
    server = TradingMCPServer(host=host, port=port)
    server.start_server()


if __name__ == "__main__":
    start_mcp_server()