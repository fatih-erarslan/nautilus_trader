"""Advanced Metadata Management System for ML Models."""

import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import threading
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    TRADES_PER_MONTH = "trades_per_month"
    AVERAGE_HOLDING_DAYS = "avg_holding_days"


class ModelStatus(Enum):
    """Model lifecycle status."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    VALIDATED = "validated"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class MetricBenchmark:
    """Performance metric benchmark data."""
    metric_name: str
    target_value: float
    minimum_value: float
    maximum_value: Optional[float] = None
    unit: str = ""
    description: str = ""
    
    def evaluate(self, value: float) -> Dict[str, Any]:
        """Evaluate a metric value against benchmarks."""
        score = 0
        status = "poor"
        
        if value >= self.target_value:
            score = 100
            status = "excellent"
        elif value >= self.minimum_value:
            # Linear interpolation between minimum and target
            score = 50 + (50 * (value - self.minimum_value) / (self.target_value - self.minimum_value))
            status = "good" if score >= 75 else "acceptable"
        else:
            # Below minimum
            score = max(0, 50 * (value / self.minimum_value) if self.minimum_value > 0 else 0)
            status = "poor"
        
        return {
            'score': round(score, 1),
            'status': status,
            'meets_target': value >= self.target_value,
            'meets_minimum': value >= self.minimum_value,
            'value': value,
            'target': self.target_value,
            'minimum': self.minimum_value
        }


@dataclass
class ModelMetadata:
    """Comprehensive model metadata with search and tagging capabilities."""
    model_id: str
    name: str
    version: str
    created_at: datetime
    updated_at: datetime
    model_type: str
    strategy_name: str
    status: ModelStatus
    performance_metrics: Dict[str, float]
    parameters: Dict[str, Any]
    training_data_info: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    deployment_info: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    file_paths: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    author: str = "AI Trading System"
    description: str = ""
    notes: List[str] = field(default_factory=list)
    
    def add_tag(self, tag: str):
        """Add a tag to the model."""
        self.tags.add(tag.lower().strip())
        self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str):
        """Remove a tag from the model."""
        self.tags.discard(tag.lower().strip())
        self.updated_at = datetime.now()
    
    def add_note(self, note: str):
        """Add a note to the model."""
        self.notes.append({
            'timestamp': datetime.now().isoformat(),
            'note': note
        })
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['status'] = self.status.value
        data['tags'] = list(self.tags)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['status'] = ModelStatus(data['status'])
        data['tags'] = set(data.get('tags', []))
        return cls(**data)


class MetadataManager:
    """Advanced metadata management system with search and analytics."""
    
    def __init__(self, base_path: str = "model_management/storage"):
        """
        Initialize metadata manager.
        
        Args:
            base_path: Base directory for metadata storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Database for advanced queries
        self.db_path = self.base_path / "metadata.db"
        self._init_database()
        
        # JSON storage for compatibility
        self.json_path = self.base_path / "metadata.json"
        
        # Benchmarks for performance evaluation
        self.benchmarks = self._setup_default_benchmarks()
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        logger.info(f"Metadata manager initialized at {self.base_path}")
    
    def _init_database(self):
        """Initialize SQLite database for metadata storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    author TEXT,
                    description TEXT,
                    performance_metrics TEXT,
                    parameters TEXT,
                    tags TEXT,
                    custom_fields TEXT,
                    file_paths TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_tags (
                    model_id TEXT,
                    tag TEXT,
                    FOREIGN KEY (model_id) REFERENCES model_metadata (model_id),
                    PRIMARY KEY (model_id, tag)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    timestamp TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    FOREIGN KEY (model_id) REFERENCES model_metadata (model_id)
                )
            """)
            
            # Create indexes for faster queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy_name ON model_metadata(strategy_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON model_metadata(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON model_metadata(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tags ON model_tags(tag)")
    
    def _setup_default_benchmarks(self) -> Dict[str, MetricBenchmark]:
        """Setup default performance benchmarks."""
        return {
            MetricType.SHARPE_RATIO.value: MetricBenchmark(
                metric_name="Sharpe Ratio",
                target_value=3.0,
                minimum_value=1.5,
                unit="ratio",
                description="Risk-adjusted return measure"
            ),
            MetricType.TOTAL_RETURN.value: MetricBenchmark(
                metric_name="Total Return",
                target_value=0.25,
                minimum_value=0.10,
                unit="percentage",
                description="Total portfolio return"
            ),
            MetricType.MAX_DRAWDOWN.value: MetricBenchmark(
                metric_name="Maximum Drawdown",
                target_value=0.05,
                minimum_value=0.15,
                maximum_value=0.0,
                unit="percentage",
                description="Largest peak-to-trough decline"
            ),
            MetricType.WIN_RATE.value: MetricBenchmark(
                metric_name="Win Rate",
                target_value=0.65,
                minimum_value=0.55,
                unit="percentage",
                description="Percentage of profitable trades"
            ),
            MetricType.PROFIT_FACTOR.value: MetricBenchmark(
                metric_name="Profit Factor",
                target_value=2.0,
                minimum_value=1.3,
                unit="ratio",
                description="Gross profit divided by gross loss"
            )
        }
    
    def save_metadata(self, metadata: ModelMetadata) -> bool:
        """
        Save model metadata to storage.
        
        Args:
            metadata: Model metadata to save
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                # Save to SQLite database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO model_metadata 
                        (model_id, name, version, created_at, updated_at, model_type, 
                         strategy_name, status, author, description, performance_metrics, 
                         parameters, tags, custom_fields, file_paths)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metadata.model_id,
                        metadata.name,
                        metadata.version,
                        metadata.created_at.isoformat(),
                        metadata.updated_at.isoformat(),
                        metadata.model_type,
                        metadata.strategy_name,
                        metadata.status.value,
                        metadata.author,
                        metadata.description,
                        json.dumps(metadata.performance_metrics),
                        json.dumps(metadata.parameters),
                        json.dumps(list(metadata.tags)),
                        json.dumps(metadata.custom_fields),
                        json.dumps(metadata.file_paths)
                    ))
                    
                    # Update tags table
                    conn.execute("DELETE FROM model_tags WHERE model_id = ?", (metadata.model_id,))
                    for tag in metadata.tags:
                        conn.execute("INSERT INTO model_tags (model_id, tag) VALUES (?, ?)",
                                   (metadata.model_id, tag))
                    
                    # Add performance history
                    for metric_name, value in metadata.performance_metrics.items():
                        conn.execute("""
                            INSERT INTO performance_history (model_id, timestamp, metric_name, metric_value)
                            VALUES (?, ?, ?, ?)
                        """, (metadata.model_id, datetime.now().isoformat(), metric_name, value))
                
                logger.info(f"Metadata saved for model {metadata.model_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save metadata for {metadata.model_id}: {e}")
                return False
    
    def load_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Load model metadata by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metadata or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM model_metadata WHERE model_id = ?
                """, (model_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                columns = [description[0] for description in cursor.description]
                data = dict(zip(columns, row))
                
                # Parse JSON fields
                data['performance_metrics'] = json.loads(data['performance_metrics'])
                data['parameters'] = json.loads(data['parameters'])
                data['tags'] = set(json.loads(data['tags']))
                data['custom_fields'] = json.loads(data['custom_fields'])
                data['file_paths'] = json.loads(data['file_paths'])
                
                # Convert datetime strings
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                data['status'] = ModelStatus(data['status'])
                
                # Load additional fields with defaults
                data.setdefault('training_data_info', {})
                data.setdefault('validation_results', {})
                data.setdefault('deployment_info', {})
                data.setdefault('dependencies', [])
                data.setdefault('notes', [])
                
                return ModelMetadata(**data)
                
        except Exception as e:
            logger.error(f"Failed to load metadata for {model_id}: {e}")
            return None
    
    def search_models(self, query: str = None, strategy_name: str = None,
                     status: ModelStatus = None, tags: List[str] = None,
                     created_after: datetime = None, created_before: datetime = None,
                     min_performance: Dict[str, float] = None,
                     limit: int = 100) -> List[ModelMetadata]:
        """
        Advanced model search with multiple filters.
        
        Args:
            query: Text search in name and description
            strategy_name: Filter by strategy name
            status: Filter by model status
            tags: Filter by tags (any match)
            created_after: Filter by creation date
            created_before: Filter by creation date
            min_performance: Minimum performance requirements
            limit: Maximum results to return
            
        Returns:
            List of matching model metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                sql = "SELECT * FROM model_metadata WHERE 1=1"
                params = []
                
                # Text search
                if query:
                    sql += " AND (name LIKE ? OR description LIKE ?)"
                    search_term = f"%{query}%"
                    params.extend([search_term, search_term])
                
                # Strategy filter
                if strategy_name:
                    sql += " AND strategy_name = ?"
                    params.append(strategy_name)
                
                # Status filter
                if status:
                    sql += " AND status = ?"
                    params.append(status.value)
                
                # Date filters
                if created_after:
                    sql += " AND created_at >= ?"
                    params.append(created_after.isoformat())
                
                if created_before:
                    sql += " AND created_at <= ?"
                    params.append(created_before.isoformat())
                
                # Tag filter
                if tags:
                    tag_placeholders = ",".join("?" * len(tags))
                    sql += f" AND model_id IN (SELECT model_id FROM model_tags WHERE tag IN ({tag_placeholders}))"
                    params.extend(tags)
                
                sql += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                results = []
                
                for row in cursor.fetchall():
                    columns = [description[0] for description in cursor.description]
                    data = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    performance_metrics = json.loads(data['performance_metrics'])
                    
                    # Apply performance filters
                    if min_performance:
                        skip = False
                        for metric, min_val in min_performance.items():
                            if metric not in performance_metrics or performance_metrics[metric] < min_val:
                                skip = True
                                break
                        if skip:
                            continue
                    
                    # Convert to metadata object
                    data['performance_metrics'] = performance_metrics
                    data['parameters'] = json.loads(data['parameters'])
                    data['tags'] = set(json.loads(data['tags']))
                    data['custom_fields'] = json.loads(data['custom_fields'])
                    data['file_paths'] = json.loads(data['file_paths'])
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                    data['status'] = ModelStatus(data['status'])
                    
                    # Add default fields
                    data.setdefault('training_data_info', {})
                    data.setdefault('validation_results', {})
                    data.setdefault('deployment_info', {})
                    data.setdefault('dependencies', [])
                    data.setdefault('notes', [])
                    
                    results.append(ModelMetadata(**data))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []
    
    def get_performance_leaderboard(self, metric: str, strategy_name: str = None,
                                  limit: int = 10) -> List[Dict]:
        """
        Get top performing models for a specific metric.
        
        Args:
            metric: Performance metric to rank by
            strategy_name: Filter by strategy (optional)
            limit: Number of results
            
        Returns:
            List of top performing models
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                sql = """
                    SELECT model_id, name, strategy_name, performance_metrics, created_at
                    FROM model_metadata 
                    WHERE performance_metrics LIKE ?
                """
                params = [f'%"{metric}"%']
                
                if strategy_name:
                    sql += " AND strategy_name = ?"
                    params.append(strategy_name)
                
                sql += " ORDER BY created_at DESC"
                
                cursor = conn.execute(sql, params)
                results = []
                
                for row in cursor.fetchall():
                    performance_metrics = json.loads(row[4])
                    if metric in performance_metrics:
                        results.append({
                            'model_id': row[0],
                            'name': row[1],
                            'strategy_name': row[2],
                            'metric_value': performance_metrics[metric],
                            'created_at': row[5]
                        })
                
                # Sort by metric value (descending for most metrics, ascending for drawdown)
                reverse = metric != MetricType.MAX_DRAWDOWN.value
                results.sort(key=lambda x: x['metric_value'], reverse=reverse)
                
                return results[:limit]
                
        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")
            return []
    
    def evaluate_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Evaluate model performance against benchmarks.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Performance evaluation results
        """
        metadata = self.load_metadata(model_id)
        if not metadata:
            return {}
        
        evaluation = {
            'model_id': model_id,
            'evaluation_timestamp': datetime.now().isoformat(),
            'overall_score': 0,
            'metrics_evaluation': {},
            'recommendations': []
        }
        
        total_score = 0
        evaluated_metrics = 0
        
        for metric_name, value in metadata.performance_metrics.items():
            if metric_name in self.benchmarks:
                benchmark = self.benchmarks[metric_name]
                metric_eval = benchmark.evaluate(value)
                evaluation['metrics_evaluation'][metric_name] = metric_eval
                total_score += metric_eval['score']
                evaluated_metrics += 1
                
                # Generate recommendations
                if not metric_eval['meets_target']:
                    if metric_eval['meets_minimum']:
                        evaluation['recommendations'].append(
                            f"Improve {benchmark.metric_name} from {value:.3f} to reach target of {benchmark.target_value:.3f}"
                        )
                    else:
                        evaluation['recommendations'].append(
                            f"Critical: {benchmark.metric_name} ({value:.3f}) is below minimum threshold ({benchmark.minimum_value:.3f})"
                        )
        
        if evaluated_metrics > 0:
            evaluation['overall_score'] = round(total_score / evaluated_metrics, 1)
        
        # Overall status
        if evaluation['overall_score'] >= 90:
            evaluation['status'] = "excellent"
        elif evaluation['overall_score'] >= 75:
            evaluation['status'] = "good"
        elif evaluation['overall_score'] >= 60:
            evaluation['status'] = "acceptable"
        else:
            evaluation['status'] = "needs_improvement"
        
        return evaluation
    
    def get_strategy_analytics(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get analytics for all models of a specific strategy.
        
        Args:
            strategy_name: Strategy name to analyze
            
        Returns:
            Analytics data
        """
        models = self.search_models(strategy_name=strategy_name, limit=1000)
        
        if not models:
            return {}
        
        analytics = {
            'strategy_name': strategy_name,
            'total_models': len(models),
            'status_distribution': defaultdict(int),
            'performance_summary': {},
            'evolution_timeline': [],
            'top_performers': {},
            'recommendations': []
        }
        
        # Status distribution
        for model in models:
            analytics['status_distribution'][model.status.value] += 1
        
        # Performance analysis
        all_metrics = set()
        for model in models:
            all_metrics.update(model.performance_metrics.keys())
        
        for metric in all_metrics:
            values = [model.performance_metrics[metric] for model in models 
                     if metric in model.performance_metrics]
            
            if values:
                analytics['performance_summary'][metric] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
                }
        
        # Timeline evolution
        sorted_models = sorted(models, key=lambda m: m.created_at)
        for model in sorted_models[-10:]:  # Last 10 models
            analytics['evolution_timeline'].append({
                'model_id': model.model_id,
                'version': model.version,
                'created_at': model.created_at.isoformat(),
                'key_metrics': {k: v for k, v in model.performance_metrics.items() 
                              if k in [MetricType.SHARPE_RATIO.value, MetricType.TOTAL_RETURN.value]}
            })
        
        # Top performers per metric
        for metric in [MetricType.SHARPE_RATIO.value, MetricType.TOTAL_RETURN.value, 
                      MetricType.WIN_RATE.value]:
            if metric in all_metrics:
                top_models = sorted(
                    [m for m in models if metric in m.performance_metrics],
                    key=lambda m: m.performance_metrics[metric],
                    reverse=metric != MetricType.MAX_DRAWDOWN.value
                )[:3]
                
                analytics['top_performers'][metric] = [
                    {
                        'model_id': m.model_id,
                        'name': m.name,
                        'value': m.performance_metrics[metric],
                        'created_at': m.created_at.isoformat()
                    } for m in top_models
                ]
        
        return analytics
    
    def cleanup_metadata(self, days_old: int = 30, keep_production: bool = True) -> int:
        """
        Clean up old metadata entries.
        
        Args:
            days_old: Remove entries older than this many days
            keep_production: Keep production models regardless of age
            
        Returns:
            Number of entries removed
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Build deletion criteria
                sql = "DELETE FROM model_metadata WHERE created_at < ?"
                params = [cutoff_date.isoformat()]
                
                if keep_production:
                    sql += " AND status != ?"
                    params.append(ModelStatus.PRODUCTION.value)
                
                cursor = conn.execute(sql, params)
                deleted_count = cursor.rowcount
                
                # Clean up orphaned entries
                conn.execute("DELETE FROM model_tags WHERE model_id NOT IN (SELECT model_id FROM model_metadata)")
                conn.execute("DELETE FROM performance_history WHERE model_id NOT IN (SELECT model_id FROM model_metadata)")
                
                logger.info(f"Cleaned up {deleted_count} old metadata entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup metadata: {e}")
            return 0
    
    def export_metadata(self, output_path: str, strategy_name: str = None) -> bool:
        """
        Export metadata to JSON file.
        
        Args:
            output_path: Output file path
            strategy_name: Filter by strategy (optional)
            
        Returns:
            True if successful
        """
        try:
            models = self.search_models(strategy_name=strategy_name, limit=10000)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_models': len(models),
                'strategy_filter': strategy_name,
                'models': [model.to_dict() for model in models]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(models)} model metadata to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metadata: {e}")
            return False