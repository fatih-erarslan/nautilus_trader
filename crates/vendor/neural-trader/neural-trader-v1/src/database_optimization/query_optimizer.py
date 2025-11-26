"""Query Optimization and Caching Module

Implements query optimization, result caching, and performance monitoring.
"""

import hashlib
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
import sqlite3
import redis
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Statistics for query performance tracking."""
    query_hash: str
    query_text: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_executed: Optional[datetime] = None
    
    def update(self, execution_time: float, cache_hit: bool = False):
        """Update statistics with new execution."""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.last_executed = datetime.now()
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1


class QueryOptimizer:
    """Query optimization with caching and performance monitoring."""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        cache_ttl: int = 300,
        enable_query_cache: bool = True,
        slow_query_threshold: float = 1.0
    ):
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self.enable_query_cache = enable_query_cache
        self.slow_query_threshold = slow_query_threshold
        self.query_stats: Dict[str, QueryStats] = {}
        self.query_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize Redis if not provided
        if self.enable_query_cache and not self.redis_client:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                logger.info("Redis cache connected")
            except:
                logger.warning("Redis not available, caching disabled")
                self.enable_query_cache = False
    
    def _generate_cache_key(self, query: str, params: Optional[Tuple] = None) -> str:
        """Generate cache key for query and parameters."""
        cache_data = {
            'query': query,
            'params': params if params else []
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return f"query_cache:{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query pattern (ignoring parameters)."""
        # Normalize query for pattern matching
        normalized = ' '.join(query.split()).upper()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def optimize_query(self, query: str) -> str:
        """Apply query optimizations."""
        optimized = query
        
        # Add EXPLAIN QUERY PLAN analysis
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Original query: {query}")
        
        # Common optimizations
        optimizations = [
            # Use covering indexes
            self._suggest_covering_index,
            # Rewrite subqueries as joins
            self._rewrite_subqueries,
            # Add index hints
            self._add_index_hints,
            # Optimize ORDER BY
            self._optimize_order_by,
            # Batch similar queries
            self._suggest_batching
        ]
        
        for optimization in optimizations:
            optimized = optimization(optimized)
        
        if optimized != query:
            logger.info(f"Query optimized: {query[:50]}...")
        
        return optimized
    
    def _suggest_covering_index(self, query: str) -> str:
        """Suggest covering indexes for SELECT queries."""
        # This would analyze the query and suggest indexes
        # For now, just return the query unchanged
        return query
    
    def _rewrite_subqueries(self, query: str) -> str:
        """Rewrite correlated subqueries as joins."""
        # Complex rewriting logic would go here
        return query
    
    def _add_index_hints(self, query: str) -> str:
        """Add index hints for better query planning."""
        # SQLite doesn't support index hints directly
        # But we can restructure queries to favor certain indexes
        return query
    
    def _optimize_order_by(self, query: str) -> str:
        """Optimize ORDER BY clauses."""
        # Check if ORDER BY columns have indexes
        return query
    
    def _suggest_batching(self, query: str) -> str:
        """Suggest query batching for similar queries."""
        query_hash = self._get_query_hash(query)
        self.query_patterns[query_hash].append(query)
        
        # If we see many similar queries, log a suggestion
        if len(self.query_patterns[query_hash]) > 10:
            logger.info(f"Consider batching similar queries: {query[:50]}...")
        
        return query
    
    def execute_with_cache(
        self,
        connection: sqlite3.Connection,
        query: str,
        params: Optional[Tuple] = None,
        cache_ttl: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Execute query with caching and optimization."""
        start_time = time.time()
        cache_hit = False
        result = None
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, params)
        query_hash = self._get_query_hash(query)
        
        # Initialize stats if needed
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = QueryStats(
                query_hash=query_hash,
                query_text=query[:200]  # Store first 200 chars
            )
        
        # Try to get from cache
        if self.enable_query_cache and self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = json.loads(cached_data)
                    cache_hit = True
                    logger.debug(f"Cache hit for query: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")
        
        # Execute query if not in cache
        if result is None:
            # Optimize query
            optimized_query = self.optimize_query(query)
            
            # Execute query
            cursor = connection.cursor()
            if params:
                cursor.execute(optimized_query, params)
            else:
                cursor.execute(optimized_query)
            
            # Fetch results
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            result = [dict(zip(columns, row)) for row in rows]
            cursor.close()
            
            # Cache result
            if self.enable_query_cache and self.redis_client and result:
                try:
                    ttl = cache_ttl or self.cache_ttl
                    self.redis_client.setex(
                        cache_key,
                        ttl,
                        json.dumps(result, default=str)
                    )
                    logger.debug(f"Cached query result for {ttl} seconds")
                except Exception as e:
                    logger.warning(f"Cache storage failed: {e}")
        
        # Update statistics
        execution_time = time.time() - start_time
        self.query_stats[query_hash].update(execution_time, cache_hit)
        
        # Log slow queries
        if execution_time > self.slow_query_threshold and not cache_hit:
            logger.warning(
                f"Slow query detected ({execution_time:.2f}s): {query[:100]}..."
            )
            self._analyze_slow_query(connection, query, params)
        
        return result
    
    def _analyze_slow_query(
        self,
        connection: sqlite3.Connection,
        query: str,
        params: Optional[Tuple] = None
    ):
        """Analyze slow query and suggest improvements."""
        try:
            # Get query plan
            explain_query = f"EXPLAIN QUERY PLAN {query}"
            cursor = connection.cursor()
            
            if params:
                cursor.execute(explain_query, params)
            else:
                cursor.execute(explain_query)
            
            plan = cursor.fetchall()
            cursor.close()
            
            # Analyze plan for issues
            issues = []
            for row in plan:
                plan_text = str(row)
                if 'SCAN TABLE' in plan_text:
                    issues.append("Full table scan detected")
                if 'TEMP B-TREE' in plan_text:
                    issues.append("Temporary B-tree created for sorting")
                if 'SEARCH TABLE' in plan_text and 'USING INDEX' not in plan_text:
                    issues.append("Table search without index")
            
            if issues:
                logger.warning(f"Query plan issues: {', '.join(issues)}")
                logger.debug(f"Query plan: {plan}")
                
        except Exception as e:
            logger.error(f"Failed to analyze slow query: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for all queries."""
        report = {
            'total_queries': sum(s.execution_count for s in self.query_stats.values()),
            'unique_queries': len(self.query_stats),
            'cache_hit_rate': 0.0,
            'avg_execution_time': 0.0,
            'slow_queries': [],
            'most_frequent': [],
            'optimization_suggestions': []
        }
        
        if not self.query_stats:
            return report
        
        # Calculate metrics
        total_hits = sum(s.cache_hits for s in self.query_stats.values())
        total_misses = sum(s.cache_misses for s in self.query_stats.values())
        
        if total_hits + total_misses > 0:
            report['cache_hit_rate'] = total_hits / (total_hits + total_misses)
        
        # Find slow and frequent queries
        for stats in self.query_stats.values():
            if stats.avg_time > self.slow_query_threshold:
                report['slow_queries'].append({
                    'query': stats.query_text,
                    'avg_time': stats.avg_time,
                    'execution_count': stats.execution_count
                })
        
        # Sort by execution count
        frequent_queries = sorted(
            self.query_stats.values(),
            key=lambda s: s.execution_count,
            reverse=True
        )[:10]
        
        report['most_frequent'] = [
            {
                'query': s.query_text,
                'count': s.execution_count,
                'avg_time': s.avg_time
            }
            for s in frequent_queries
        ]
        
        # Generate optimization suggestions
        if report['cache_hit_rate'] < 0.5:
            report['optimization_suggestions'].append(
                "Low cache hit rate - consider increasing cache TTL"
            )
        
        if len(report['slow_queries']) > 5:
            report['optimization_suggestions'].append(
                "Multiple slow queries detected - review indexes"
            )
        
        return report
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear query cache."""
        if not self.enable_query_cache or not self.redis_client:
            return
        
        try:
            if pattern:
                keys = self.redis_client.keys(f"query_cache:{pattern}*")
            else:
                keys = self.redis_client.keys("query_cache:*")
            
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached queries")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


def cached_query(ttl: int = 300):
    """Decorator for caching query results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            # (Implementation would use Redis or similar)
            
            # Execute function if not cached
            result = func(*args, **kwargs)
            
            # Cache result
            # (Implementation would store in Redis)
            
            return result
        return wrapper
    return decorator


# Example usage
@cached_query(ttl=600)
def get_portfolio_performance(user_id: int, period: str) -> Dict[str, Any]:
    """Get portfolio performance with caching."""
    # Query implementation
    pass