"""Database Index Management and Optimization

Manages index creation, analysis, and optimization for improved query performance.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class IndexDefinition:
    """Index definition with metadata."""
    name: str
    table: str
    columns: List[str]
    unique: bool = False
    partial: Optional[str] = None  # WHERE clause for partial index
    include: Optional[List[str]] = None  # INCLUDE columns for covering index
    description: str = ""
    priority: str = "medium"  # low, medium, high, critical
    
    def to_sql(self) -> str:
        """Generate SQL CREATE INDEX statement."""
        unique_clause = "UNIQUE " if self.unique else ""
        columns_clause = ", ".join(self.columns)
        
        sql = f"CREATE {unique_clause}INDEX IF NOT EXISTS {self.name} ON {self.table} ({columns_clause})"
        
        if self.include:
            # SQLite doesn't support INCLUDE directly, use covering index pattern
            all_columns = self.columns + self.include
            columns_clause = ", ".join(all_columns)
            sql = f"CREATE {unique_clause}INDEX IF NOT EXISTS {self.name} ON {self.table} ({columns_clause})"
        
        if self.partial:
            sql += f" WHERE {self.partial}"
        
        return sql


class IndexManager:
    """Manages database indexes for optimal performance."""
    
    # Critical indexes for trading platform
    CRITICAL_INDEXES = [
        IndexDefinition(
            name="idx_trades_timestamp_symbol",
            table="trades",
            columns=["timestamp", "symbol"],
            description="Speed up time-series queries for trades",
            priority="critical"
        ),
        IndexDefinition(
            name="idx_trades_user_status",
            table="trades",
            columns=["user_id", "status"],
            include=["symbol", "quantity", "price"],
            description="Covering index for user trade queries",
            priority="critical"
        ),
        IndexDefinition(
            name="idx_orders_symbol_status",
            table="orders",
            columns=["symbol", "status", "created_at"],
            description="Active orders lookup",
            priority="critical"
        ),
        IndexDefinition(
            name="idx_positions_user_symbol",
            table="positions",
            columns=["user_id", "symbol"],
            unique=True,
            description="Unique constraint on user positions",
            priority="critical"
        ),
        IndexDefinition(
            name="idx_news_symbol_sentiment",
            table="news_items",
            columns=["symbol", "sentiment_score", "published_at"],
            description="News sentiment analysis queries",
            priority="high"
        ),
        IndexDefinition(
            name="idx_news_timestamp",
            table="news_items",
            columns=["published_at"],
            description="Time-based news queries",
            priority="high"
        ),
        IndexDefinition(
            name="idx_portfolios_user_active",
            table="portfolios",
            columns=["user_id", "is_active"],
            partial="is_active = 1",
            description="Active portfolio lookup",
            priority="high"
        ),
        IndexDefinition(
            name="idx_market_data_symbol_time",
            table="market_data",
            columns=["symbol", "timestamp"],
            include=["open", "high", "low", "close", "volume"],
            description="Market data time series",
            priority="critical"
        ),
        IndexDefinition(
            name="idx_transactions_user_date",
            table="transactions",
            columns=["user_id", "transaction_date"],
            description="User transaction history",
            priority="medium"
        ),
        IndexDefinition(
            name="idx_alerts_user_active",
            table="alerts",
            columns=["user_id", "is_active", "trigger_price"],
            partial="is_active = 1",
            description="Active price alerts",
            priority="medium"
        )
    ]
    
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.existing_indexes = {}
        self.index_stats = {}
        self._load_existing_indexes()
    
    def _load_existing_indexes(self):
        """Load information about existing indexes."""
        cursor = self.connection.cursor()
        
        # Get all user-created indexes
        cursor.execute("""
            SELECT name, tbl_name, sql 
            FROM sqlite_master 
            WHERE type = 'index' 
            AND sql IS NOT NULL
        """)
        
        for row in cursor.fetchall():
            self.existing_indexes[row[0]] = {
                'table': row[1],
                'sql': row[2]
            }
        
        cursor.close()
    
    def create_critical_indexes(self) -> Dict[str, bool]:
        """Create all critical indexes for the trading platform."""
        results = {}
        
        for index_def in self.CRITICAL_INDEXES:
            if index_def.priority in ['critical', 'high']:
                success = self.create_index(index_def)
                results[index_def.name] = success
        
        return results
    
    def create_index(self, index_def: IndexDefinition) -> bool:
        """Create a single index."""
        try:
            # Check if index already exists
            if index_def.name in self.existing_indexes:
                logger.info(f"Index {index_def.name} already exists")
                return True
            
            # Create index
            sql = index_def.to_sql()
            logger.info(f"Creating index: {index_def.name}")
            logger.debug(f"SQL: {sql}")
            
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            cursor.close()
            
            # Add to existing indexes
            self.existing_indexes[index_def.name] = {
                'table': index_def.table,
                'sql': sql
            }
            
            logger.info(f"Successfully created index: {index_def.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index {index_def.name}: {e}")
            return False
    
    def analyze_index_usage(self) -> Dict[str, Any]:
        """Analyze index usage and effectiveness."""
        cursor = self.connection.cursor()
        analysis = {
            'total_indexes': len(self.existing_indexes),
            'index_details': [],
            'unused_indexes': [],
            'missing_indexes': [],
            'recommendations': []
        }
        
        # Check each index
        for index_name, index_info in self.existing_indexes.items():
            try:
                # Get index statistics (SQLite doesn't provide detailed stats)
                # We'll use EXPLAIN QUERY PLAN to check if indexes are used
                
                index_detail = {
                    'name': index_name,
                    'table': index_info['table'],
                    'status': 'active'
                }
                
                analysis['index_details'].append(index_detail)
                
            except Exception as e:
                logger.warning(f"Failed to analyze index {index_name}: {e}")
        
        # Check for missing critical indexes
        for index_def in self.CRITICAL_INDEXES:
            if index_def.name not in self.existing_indexes:
                analysis['missing_indexes'].append({
                    'name': index_def.name,
                    'table': index_def.table,
                    'priority': index_def.priority,
                    'description': index_def.description
                })
        
        # Generate recommendations
        if analysis['missing_indexes']:
            analysis['recommendations'].append(
                f"Create {len(analysis['missing_indexes'])} missing critical indexes"
            )
        
        cursor.close()
        return analysis
    
    def optimize_indexes(self) -> Dict[str, Any]:
        """Optimize existing indexes and create missing ones."""
        results = {
            'created': [],
            'dropped': [],
            'rebuilt': [],
            'errors': []
        }
        
        # Create missing critical indexes
        for index_def in self.CRITICAL_INDEXES:
            if index_def.name not in self.existing_indexes:
                if self.create_index(index_def):
                    results['created'].append(index_def.name)
                else:
                    results['errors'].append(f"Failed to create {index_def.name}")
        
        # Rebuild fragmented indexes (REINDEX in SQLite)
        try:
            logger.info("Reindexing database...")
            cursor = self.connection.cursor()
            cursor.execute("REINDEX")
            self.connection.commit()
            cursor.close()
            results['rebuilt'].append("ALL")
        except Exception as e:
            logger.error(f"Failed to reindex: {e}")
            results['errors'].append(f"Reindex failed: {e}")
        
        return results
    
    def get_index_recommendations(self, query: str) -> List[str]:
        """Get index recommendations for a specific query."""
        recommendations = []
        
        try:
            # Use EXPLAIN QUERY PLAN to analyze query
            cursor = self.connection.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            plan = cursor.fetchall()
            cursor.close()
            
            # Analyze the plan
            for row in plan:
                plan_text = str(row)
                
                # Check for full table scans
                if 'SCAN TABLE' in plan_text:
                    table_match = plan_text.split('SCAN TABLE')[1].split()[0]
                    recommendations.append(
                        f"Consider adding index on frequently filtered columns of table {table_match}"
                    )
                
                # Check for temporary B-trees
                if 'TEMP B-TREE' in plan_text:
                    if 'ORDER BY' in plan_text:
                        recommendations.append(
                            "Consider adding index on ORDER BY columns"
                        )
                    elif 'GROUP BY' in plan_text:
                        recommendations.append(
                            "Consider adding index on GROUP BY columns"
                        )
                
                # Check for suboptimal joins
                if 'SEARCH TABLE' in plan_text and 'USING INDEX' not in plan_text:
                    recommendations.append(
                        "Consider adding index on join columns"
                    )
            
        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            recommendations.append("Unable to analyze query")
        
        return recommendations
    
    def drop_index(self, index_name: str) -> bool:
        """Drop an index."""
        try:
            if index_name not in self.existing_indexes:
                logger.warning(f"Index {index_name} does not exist")
                return False
            
            cursor = self.connection.cursor()
            cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            self.connection.commit()
            cursor.close()
            
            del self.existing_indexes[index_name]
            logger.info(f"Dropped index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop index {index_name}: {e}")
            return False
    
    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get all indexes for a specific table."""
        indexes = []
        
        cursor = self.connection.cursor()
        cursor.execute(f"PRAGMA index_list({table_name})")
        
        for row in cursor.fetchall():
            index_name = row[1]
            unique = bool(row[2])
            
            # Get index columns
            cursor.execute(f"PRAGMA index_info({index_name})")
            columns = [col[2] for col in cursor.fetchall()]
            
            indexes.append({
                'name': index_name,
                'unique': unique,
                'columns': columns
            })
        
        cursor.close()
        return indexes
    
    def create_custom_index(
        self,
        table: str,
        columns: List[str],
        name: Optional[str] = None,
        unique: bool = False,
        partial: Optional[str] = None
    ) -> bool:
        """Create a custom index based on parameters."""
        if not name:
            name = f"idx_{table}_{'_'.join(columns)}"
        
        index_def = IndexDefinition(
            name=name,
            table=table,
            columns=columns,
            unique=unique,
            partial=partial
        )
        
        return self.create_index(index_def)


def auto_create_indexes(connection: sqlite3.Connection) -> Dict[str, Any]:
    """Automatically create optimal indexes for the database."""
    manager = IndexManager(connection)
    
    # Create critical indexes
    created = manager.create_critical_indexes()
    
    # Analyze and optimize
    analysis = manager.analyze_index_usage()
    optimization = manager.optimize_indexes()
    
    return {
        'created': created,
        'analysis': analysis,
        'optimization': optimization
    }