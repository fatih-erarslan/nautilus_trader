"""
Database Models
===============

Pydantic models for type-safe database operations and validation.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
import json

# Enums matching database enums
class BotStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused" 
    STOPPED = "stopped"
    ERROR = "error"
    TRAINING = "training"

class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIAL = "partial"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

class ModelStatus(str, Enum):
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DeploymentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    TERMINATED = "terminated"

class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"

class AccountType(str, Enum):
    PAPER = "paper"
    LIVE = "live"

# Base model with common functionality
class DatabaseModel(BaseModel):
    """Base model for all database entities."""
    
    class Config:
        # Enable ORM mode for SQLAlchemy compatibility
        from_attributes = True
        # Use enum values
        use_enum_values = True
        # Allow population by field name or alias
        populate_by_name = True
        # JSON serialization
        json_encoders = {
            UUID: str,
            datetime: lambda dt: dt.isoformat(),
            Decimal: float
        }
    
    def dict_for_db(self, exclude_none: bool = True) -> Dict[str, Any]:
        """Convert to dictionary suitable for database insertion."""
        return self.model_dump(
            exclude_none=exclude_none,
            by_alias=True,
            exclude={"id"} if not hasattr(self, "id") or not self.id else set()
        )
    
    @classmethod
    def from_db(cls, data: Dict[str, Any]):
        """Create instance from database row."""
        return cls(**data)

# User and profile models
class Profile(DatabaseModel):
    """User profile model."""
    
    id: UUID
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = Field(None, max_length=200)
    avatar_url: Optional[str] = None
    tier: str = Field(default="basic", pattern=r'^(basic|pro|enterprise)$')
    api_quota: int = Field(default=1000, ge=0)
    api_usage: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Market data models
class Symbol(DatabaseModel):
    """Financial symbol model."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    symbol: str = Field(..., min_length=1, max_length=20)
    name: str = Field(..., min_length=1, max_length=200)
    exchange: str = Field(..., min_length=1, max_length=50)
    asset_type: AssetType
    is_active: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MarketData(DatabaseModel):
    """Market data model."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    symbol_id: UUID
    timestamp: datetime
    open: Decimal = Field(..., gt=0)
    high: Decimal = Field(..., gt=0)
    low: Decimal = Field(..., gt=0)
    close: Decimal = Field(..., gt=0)
    volume: Decimal = Field(..., ge=0)
    timeframe: str = Field(..., pattern=r'^(1m|5m|15m|30m|1h|4h|1d)$')
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related symbol
    symbol: Optional[Symbol] = None
    
    # TODO: Fix validator for Pydantic v2
    # @field_validator('high', 'low', 'close')
    # def validate_prices(cls, v, values):
    #     """Validate price relationships."""
    #     if 'low' in values and v < values['low']:
    #         raise ValueError('High and close must be >= low')
    #     return v

class NewsData(DatabaseModel):
    """News data model."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    title: str = Field(..., min_length=1, max_length=500)
    content: Optional[str] = None
    source: str = Field(..., min_length=1, max_length=100)
    url: Optional[str] = None
    published_at: datetime
    symbols: List[str] = Field(default_factory=list)
    sentiment_score: Optional[Decimal] = Field(None, ge=-1.0, le=1.0)
    relevance_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Trading models
class TradingAccount(DatabaseModel):
    """Trading account model."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: UUID
    name: str = Field(..., min_length=1, max_length=100)
    broker: str = Field(..., min_length=1, max_length=50)
    account_type: AccountType = Field(default=AccountType.PAPER)
    balance: Decimal = Field(default=Decimal('0'), ge=0)
    equity: Decimal = Field(default=Decimal('0'), ge=0)
    margin_used: Decimal = Field(default=Decimal('0'), ge=0)
    margin_available: Decimal = Field(default=Decimal('0'), ge=0)
    api_credentials: Optional[Dict[str, Any]] = None
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related profile
    profile: Optional[Profile] = None

class Position(DatabaseModel):
    """Trading position model."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    account_id: UUID
    symbol_id: UUID
    side: PositionSide
    quantity: Decimal = Field(..., gt=0)
    entry_price: Decimal = Field(..., gt=0)
    current_price: Optional[Decimal] = Field(None, gt=0)
    unrealized_pnl: Decimal = Field(default=Decimal('0'))
    realized_pnl: Decimal = Field(default=Decimal('0'))
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related objects
    account: Optional[TradingAccount] = None
    symbol: Optional[Symbol] = None
    
    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.closed_at is None
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value."""
        price = self.current_price or self.entry_price
        return self.quantity * price

class Order(DatabaseModel):
    """Trading order model."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    account_id: UUID
    symbol_id: UUID
    order_type: OrderType
    side: PositionSide
    quantity: Decimal = Field(..., gt=0)
    price: Optional[Decimal] = Field(None, gt=0)
    stop_price: Optional[Decimal] = Field(None, gt=0)
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    filled_quantity: Decimal = Field(default=Decimal('0'), ge=0)
    average_fill_price: Optional[Decimal] = Field(None, gt=0)
    commission: Decimal = Field(default=Decimal('0'), ge=0)
    external_order_id: Optional[str] = None
    placed_at: datetime = Field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related objects
    account: Optional[TradingAccount] = None
    symbol: Optional[Symbol] = None
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to fill."""
        return self.quantity - self.filled_quantity

# Neural network models
class NeuralModel(DatabaseModel):
    """Neural network model."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: UUID
    name: str = Field(..., min_length=1, max_length=100)
    model_type: str = Field(..., pattern=r'^(lstm|transformer|cnn|ensemble)$')
    architecture: Dict[str, Any] = Field(...)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: ModelStatus = Field(default=ModelStatus.TRAINING)
    version: int = Field(default=1, ge=1)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    training_data_hash: Optional[str] = None
    model_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related profile
    profile: Optional[Profile] = None

class TrainingRun(DatabaseModel):
    """Neural model training run."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    model_id: UUID
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = Field(default="running", pattern=r'^(running|completed|failed|cancelled)$')
    epoch: int = Field(default=0, ge=0)
    loss: Optional[Decimal] = Field(None, ge=0)
    accuracy: Optional[Decimal] = Field(None, ge=0, le=1)
    validation_loss: Optional[Decimal] = Field(None, ge=0)
    validation_accuracy: Optional[Decimal] = Field(None, ge=0, le=1)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    logs: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related model
    model: Optional[NeuralModel] = None
    
    @property
    def is_running(self) -> bool:
        """Check if training is still running."""
        return self.status == "running"
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate training duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

class ModelPrediction(DatabaseModel):
    """Model prediction."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    model_id: UUID
    symbol_id: UUID
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    prediction_value: Decimal
    confidence: Optional[Decimal] = Field(None, ge=0, le=1)
    actual_value: Optional[Decimal] = None
    error: Optional[Decimal] = None
    features: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related objects
    model: Optional[NeuralModel] = None
    symbol: Optional[Symbol] = None
    
    @property
    def accuracy(self) -> Optional[Decimal]:
        """Calculate prediction accuracy if actual value is available."""
        if self.actual_value is not None and self.actual_value != 0:
            return 1 - abs(self.prediction_value - self.actual_value) / abs(self.actual_value)
        return None

# Trading bot models
class TradingBot(DatabaseModel):
    """Trading bot configuration."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: UUID
    account_id: UUID
    name: str = Field(..., min_length=1, max_length=100)
    strategy_type: str = Field(..., min_length=1, max_length=50)
    configuration: Dict[str, Any] = Field(...)
    model_ids: List[UUID] = Field(default_factory=list)
    symbols: List[str] = Field(..., min_items=1)
    status: BotStatus = Field(default=BotStatus.PAUSED)
    max_position_size: Decimal = Field(default=Decimal('1000'), gt=0)
    risk_limit: Decimal = Field(default=Decimal('0.05'), gt=0, le=1)
    is_active: bool = Field(default=True)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related objects
    profile: Optional[Profile] = None
    account: Optional[TradingAccount] = None
    models: List[NeuralModel] = Field(default_factory=list)

class BotExecution(DatabaseModel):
    """Bot execution record."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    bot_id: UUID
    symbol_id: UUID
    action: str = Field(..., pattern=r'^(buy|sell|hold)$')
    signal_strength: Optional[Decimal] = Field(None, ge=0, le=1)
    reasoning: Optional[str] = None
    order_id: Optional[UUID] = None
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related objects
    bot: Optional[TradingBot] = None
    symbol: Optional[Symbol] = None
    order: Optional[Order] = None

# Sandbox models
class SandboxDeployment(DatabaseModel):
    """E2B sandbox deployment."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: UUID
    bot_id: Optional[UUID] = None
    sandbox_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=100)
    template: str = Field(default="base")
    configuration: Dict[str, Any] = Field(default_factory=dict)
    status: DeploymentStatus = Field(default=DeploymentStatus.PENDING)
    cpu_count: int = Field(default=1, ge=1, le=8)
    memory_mb: int = Field(default=512, ge=256, le=8192)
    timeout_seconds: int = Field(default=300, ge=60, le=7200)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    logs: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related objects
    profile: Optional[Profile] = None
    bot: Optional[TradingBot] = None
    
    @property
    def is_running(self) -> bool:
        """Check if sandbox is currently running."""
        return self.status == DeploymentStatus.RUNNING
    
    @property
    def runtime_duration(self) -> Optional[float]:
        """Calculate runtime duration in seconds."""
        if self.started_at:
            end_time = self.stopped_at or datetime.utcnow()
            return (end_time - self.started_at).total_seconds()
        return None

# Monitoring models
class PerformanceMetric(DatabaseModel):
    """Performance monitoring metric."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    entity_type: str = Field(..., min_length=1)
    entity_id: UUID
    metric_type: str = Field(..., min_length=1)
    metric_value: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Alert(DatabaseModel):
    """System alert."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: UUID
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1)
    severity: AlertSeverity = Field(default=AlertSeverity.INFO)
    entity_type: Optional[str] = None
    entity_id: Optional[UUID] = None
    is_read: bool = Field(default=False)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related profile
    profile: Optional[Profile] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if alert has expired."""
        return self.expires_at is not None and datetime.utcnow() > self.expires_at

class AuditLog(DatabaseModel):
    """Audit log entry."""
    
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: Optional[UUID] = None
    action: str = Field(..., min_length=1)
    entity_type: Optional[str] = None
    entity_id: Optional[UUID] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Related profile
    profile: Optional[Profile] = None

# Request/Response models for API operations
class CreateModelRequest(BaseModel):
    """Request model for creating neural models."""
    
    name: str = Field(..., min_length=1, max_length=100)
    model_type: str = Field(..., pattern=r'^(lstm|transformer|cnn|ensemble)$')
    architecture: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    training_data_hash: Optional[str] = None

class StartTrainingRequest(BaseModel):
    """Request model for starting training."""
    
    model_id: UUID
    hyperparameters: Dict[str, Any]
    training_data_path: Optional[str] = None

class CreateBotRequest(BaseModel):
    """Request model for creating trading bots."""
    
    name: str = Field(..., min_length=1, max_length=100)
    account_id: UUID
    strategy_type: str = Field(..., min_length=1, max_length=50)
    configuration: Dict[str, Any]
    model_ids: List[UUID] = Field(default_factory=list)
    symbols: List[str] = Field(..., min_items=1)
    max_position_size: Decimal = Field(default=Decimal('1000'), gt=0)
    risk_limit: Decimal = Field(default=Decimal('0.05'), gt=0, le=1)

class UpdateBotRequest(BaseModel):
    """Request model for updating trading bots."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    configuration: Optional[Dict[str, Any]] = None
    model_ids: Optional[List[UUID]] = None
    symbols: Optional[List[str]] = Field(None, min_items=1)
    max_position_size: Optional[Decimal] = Field(None, gt=0)
    risk_limit: Optional[Decimal] = Field(None, gt=0, le=1)
    status: Optional[BotStatus] = None

# Response models
class PaginatedResponse(BaseModel):
    """Paginated response model."""
    
    data: List[Any]
    count: int
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool

class PerformanceSummary(BaseModel):
    """Performance summary model."""
    
    total_return: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    win_rate: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    period_start: datetime
    period_end: datetime

# Utility functions
def convert_db_record(record: Dict[str, Any], model_class: type) -> DatabaseModel:
    """Convert database record to model instance."""
    return model_class.from_db(record)

def serialize_for_json(obj: Any) -> Any:
    """Serialize objects for JSON response."""
    if isinstance(obj, BaseModel):
        return obj.dict()
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (UUID, datetime)):
        return str(obj)
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj