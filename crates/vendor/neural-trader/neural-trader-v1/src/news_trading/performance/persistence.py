"""Database persistence for performance tracking."""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models import TradeResult, TradeStatus


class PerformanceDatabase:
    """SQLite database for performance data persistence."""
    
    def __init__(self, db_path: str = "performance.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                signal_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                entry_price REAL NOT NULL,
                exit_price REAL,
                position_size REAL NOT NULL,
                pnl REAL DEFAULT 0,
                pnl_percentage REAL DEFAULT 0,
                status TEXT NOT NULL,
                news_events TEXT,  -- JSON array
                sentiment_scores TEXT,  -- JSON array
                fees REAL DEFAULT 0,
                slippage REAL DEFAULT 0,
                metadata TEXT,  -- JSON object
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ML predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                prediction_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                predicted_value REAL NOT NULL,
                actual_value REAL NOT NULL,
                confidence REAL NOT NULL,
                error REAL NOT NULL,
                feature_importance TEXT,  -- JSON object
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sentiment predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_predictions (
                news_id TEXT PRIMARY KEY,
                model_name TEXT,
                predicted_sentiment REAL NOT NULL,
                predicted_impact TEXT NOT NULL,
                actual_price_change REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trade attributions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_attributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                source TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        """)
        
        # Performance snapshots table (for historical tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date DATE NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                total_pnl REAL NOT NULL,
                win_rate REAL NOT NULL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                metadata TEXT,  -- JSON object
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(snapshot_date)
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions(model_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_predictions_timestamp ON ml_predictions(timestamp)")
        
        self.conn.commit()
    
    def save_trade(self, trade: TradeResult) -> None:
        """Save a trade to the database.
        
        Args:
            trade: Trade result to save
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO trades (
                trade_id, signal_id, asset, entry_time, exit_time,
                entry_price, exit_price, position_size, pnl, pnl_percentage,
                status, news_events, sentiment_scores, fees, slippage, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.trade_id,
            trade.signal_id,
            trade.asset,
            trade.entry_time,
            trade.exit_time,
            trade.entry_price,
            trade.exit_price,
            trade.position_size,
            trade.pnl,
            trade.pnl_percentage,
            trade.status.value,
            json.dumps(trade.news_events),
            json.dumps(trade.sentiment_scores),
            trade.fees,
            trade.slippage,
            json.dumps(trade.metadata),
        ))
        
        self.conn.commit()
    
    def save_ml_prediction(self, prediction: Dict[str, Any]) -> None:
        """Save an ML prediction to the database.
        
        Args:
            prediction: Prediction data
        """
        cursor = self.conn.cursor()
        
        error = abs(prediction["predicted_value"] - prediction["actual_value"])
        
        cursor.execute("""
            INSERT OR REPLACE INTO ml_predictions (
                prediction_id, model_name, predicted_value, actual_value,
                confidence, error, feature_importance, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction["prediction_id"],
            prediction["model_name"],
            prediction["predicted_value"],
            prediction["actual_value"],
            prediction["confidence"],
            error,
            json.dumps(prediction.get("feature_importance")),
            prediction["timestamp"],
        ))
        
        self.conn.commit()
    
    def save_sentiment_prediction(
        self,
        news_id: str,
        predicted_sentiment: float,
        predicted_impact: str,
        actual_price_change: float,
        model_name: Optional[str] = None,
    ) -> None:
        """Save a sentiment prediction.
        
        Args:
            news_id: News event ID
            predicted_sentiment: Predicted sentiment
            predicted_impact: Predicted impact
            actual_price_change: Actual price change
            model_name: Optional model name
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO sentiment_predictions (
                news_id, model_name, predicted_sentiment, predicted_impact,
                actual_price_change, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            news_id,
            model_name,
            predicted_sentiment,
            predicted_impact,
            actual_price_change,
            datetime.now(),
        ))
        
        self.conn.commit()
    
    def save_trade_attribution(
        self,
        trade_id: str,
        attributions: Dict[str, float],
    ) -> None:
        """Save trade attribution data.
        
        Args:
            trade_id: Trade ID
            attributions: Source attribution weights
        """
        cursor = self.conn.cursor()
        
        # Delete existing attributions
        cursor.execute("DELETE FROM trade_attributions WHERE trade_id = ?", (trade_id,))
        
        # Insert new attributions
        for source, weight in attributions.items():
            cursor.execute("""
                INSERT INTO trade_attributions (trade_id, source, weight)
                VALUES (?, ?, ?)
            """, (trade_id, source, weight))
        
        self.conn.commit()
    
    def save_performance_snapshot(self, metrics: Dict[str, Any]) -> None:
        """Save daily performance snapshot.
        
        Args:
            metrics: Performance metrics to save
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO performance_snapshots (
                snapshot_date, total_trades, winning_trades, losing_trades,
                total_pnl, win_rate, sharpe_ratio, max_drawdown, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().date(),
            metrics.get("total_trades", 0),
            metrics.get("winning_trades", 0),
            metrics.get("losing_trades", 0),
            metrics.get("total_pnl", 0),
            metrics.get("win_rate", 0),
            metrics.get("sharpe_ratio"),
            metrics.get("max_drawdown"),
            json.dumps(metrics.get("metadata", {})),
        ))
        
        self.conn.commit()
    
    def load_trades(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Load trades from database.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            List of trade dictionaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if filters:
            if "asset" in filters:
                query += " AND asset = ?"
                params.append(filters["asset"])
            
            if "status" in filters:
                query += " AND status = ?"
                params.append(filters["status"])
            
            if "date_from" in filters:
                query += " AND entry_time >= ?"
                params.append(filters["date_from"])
            
            if "date_to" in filters:
                query += " AND entry_time <= ?"
                params.append(filters["date_to"])
        
        query += " ORDER BY entry_time DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        trades = []
        for row in rows:
            trade = dict(row)
            
            # Parse JSON fields
            trade["news_events"] = json.loads(trade["news_events"] or "[]")
            trade["sentiment_scores"] = json.loads(trade["sentiment_scores"] or "[]")
            trade["metadata"] = json.loads(trade["metadata"] or "{}")
            
            trades.append(trade)
        
        return trades
    
    def load_ml_predictions(
        self,
        model_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load ML predictions from database.
        
        Args:
            model_name: Optional model name filter
            limit: Maximum number of predictions to load
            
        Returns:
            List of prediction dictionaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM ml_predictions WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        predictions = []
        for row in rows:
            pred = dict(row)
            pred["feature_importance"] = json.loads(pred["feature_importance"] or "{}")
            predictions.append(pred)
        
        return predictions
    
    def get_performance_history(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get historical performance snapshots.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of performance snapshots
        """
        cursor = self.conn.cursor()
        
        cutoff_date = datetime.now().date() - timedelta(days=days)
        
        cursor.execute("""
            SELECT * FROM performance_snapshots
            WHERE snapshot_date >= ?
            ORDER BY snapshot_date DESC
        """, (cutoff_date,))
        
        rows = cursor.fetchall()
        
        snapshots = []
        for row in rows:
            snapshot = dict(row)
            snapshot["metadata"] = json.loads(snapshot["metadata"] or "{}")
            snapshots.append(snapshot)
        
        return snapshots
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> None:
        """Clean up old data from the database.
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        cursor = self.conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up old trades
        cursor.execute("""
            DELETE FROM trades
            WHERE entry_time < ? AND status = 'CLOSED'
        """, (cutoff_date,))
        
        # Clean up old predictions
        cursor.execute("""
            DELETE FROM ml_predictions
            WHERE timestamp < ?
        """, (cutoff_date,))
        
        cursor.execute("""
            DELETE FROM sentiment_predictions
            WHERE timestamp < ?
        """, (cutoff_date,))
        
        # Vacuum to reclaim space
        cursor.execute("VACUUM")
        
        self.conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()