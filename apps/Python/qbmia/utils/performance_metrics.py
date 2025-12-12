"""
Performance tracking and metrics for QBMIA.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
import json

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Comprehensive performance tracking for QBMIA components.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize performance tracker.

        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size

        # Decision tracking
        self.decision_history = deque(maxlen=window_size)
        self.decision_outcomes = deque(maxlen=window_size)

        # Execution tracking
        self.execution_times = defaultdict(deque)
        self.execution_stats = defaultdict(dict)

        # Component performance
        self.component_metrics = {
            'quantum_nash': defaultdict(list),
            'machiavellian': defaultdict(list),
            'robin_hood': defaultdict(list),
            'temporal_nash': defaultdict(list),
            'antifragile': defaultdict(list)
        }

        # Resource usage
        self.resource_usage = deque(maxlen=window_size)

        # Financial metrics
        self.financial_metrics = {
            'returns': deque(maxlen=window_size),
            'positions': deque(maxlen=window_size),
            'risk_metrics': deque(maxlen=window_size)
        }

        # System health
        self.system_health = {
            'uptime_start': datetime.utcnow(),
            'errors': deque(maxlen=1000),
            'warnings': deque(maxlen=1000)
        }

        # Benchmarks
        self.benchmarks = {
            'baseline_accuracy': 0.5,
            'baseline_return': 0.0,
            'baseline_sharpe': 0.0
        }

    def record_decision(self, decision: Dict[str, Any], context: Dict[str, Any]):
        """
        Record a decision made by QBMIA.

        Args:
            decision: Decision details
            context: Market context at decision time
        """
        decision_record = {
            'timestamp': datetime.utcnow(),
            'decision': decision,
            'context': context,
            'confidence': decision.get('confidence', 0.5),
            'components_used': list(decision.get('component_results', {}).keys())
        }

        self.decision_history.append(decision_record)

        # Track unique insights
        if self._is_unique_insight(decision, context):
            self.unique_insights_count = getattr(self, 'unique_insights_count', 0) + 1

    def record_outcome(self, decision_id: str, outcome: Dict[str, Any]):
        """
        Record outcome of a decision.

        Args:
            decision_id: Decision identifier
            outcome: Outcome details
        """
        outcome_record = {
            'timestamp': datetime.utcnow(),
            'decision_id': decision_id,
            'outcome': outcome,
            'success': outcome.get('success', False),
            'profit_loss': outcome.get('profit_loss', 0.0)
        }

        self.decision_outcomes.append(outcome_record)

        # Update financial metrics
        if 'profit_loss' in outcome:
            self.financial_metrics['returns'].append(outcome['profit_loss'])

    def record_execution(self, operation: str, execution_time: float,
                        success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """
        Record execution performance.

        Args:
            operation: Operation name
            execution_time: Execution time in seconds
            success: Whether execution succeeded
            metadata: Additional metadata
        """
        self.execution_times[operation].append(execution_time)

        # Update statistics
        times = list(self.execution_times[operation])
        self.execution_stats[operation] = {
            'count': len(times),
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'p50': np.percentile(times, 50),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'success_rate': self._calculate_success_rate(operation)
        }

    def record_component_performance(self, component: str, metrics: Dict[str, Any]):
        """
        Record performance metrics for a specific component.

        Args:
            component: Component name
            metrics: Performance metrics
        """
        if component in self.component_metrics:
            for metric_name, value in metrics.items():
                self.component_metrics[component][metric_name].append({
                    'timestamp': datetime.utcnow(),
                    'value': value
                })

    def record_resource_usage(self, usage: Dict[str, Any]):
        """
        Record resource usage metrics.

        Args:
            usage: Resource usage data
        """
        usage_record = {
            'timestamp': datetime.utcnow(),
            'cpu_percent': usage.get('cpu_percent', 0),
            'memory_mb': usage.get('memory_mb', 0),
            'gpu_memory_mb': usage.get('gpu_memory_mb', 0),
            'gpu_utilization': usage.get('gpu_utilization', 0)
        }

        self.resource_usage.append(usage_record)

    def record_error(self, error: str, severity: str = 'error'):
        """
        Record system error or warning.

        Args:
            error: Error message
            severity: Error severity ('error' or 'warning')
        """
        error_record = {
            'timestamp': datetime.utcnow(),
            'message': error,
            'severity': severity
        }

        if severity == 'error':
            self.system_health['errors'].append(error_record)
        else:
            self.system_health['warnings'].append(error_record)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.

        Returns:
            Performance summary
        """
        return {
            'decision_performance': self._calculate_decision_performance(),
            'execution_performance': self.execution_stats,
            'component_performance': self._calculate_component_performance(),
            'financial_performance': self._calculate_financial_performance(),
            'resource_efficiency': self._calculate_resource_efficiency(),
            'system_health': self._calculate_system_health()
        }

    def _calculate_decision_performance(self) -> Dict[str, Any]:
        """Calculate decision-making performance metrics."""
        if not self.decision_history:
            return {'no_decisions': True}

        # Calculate accuracy based on outcomes
        if self.decision_outcomes:
            successful_outcomes = sum(1 for o in self.decision_outcomes if o['success'])
            accuracy = successful_outcomes / len(self.decision_outcomes)
        else:
            accuracy = 0.5  # No outcomes yet

        # Confidence calibration
        confidences = [d['confidence'] for d in self.decision_history]
        avg_confidence = np.mean(confidences)

        # Decision diversity
        components_used = defaultdict(int)
        for decision in self.decision_history:
            for component in decision['components_used']:
                components_used[component] += 1

        # Time-based metrics
        recent_decisions = [d for d in self.decision_history
                           if (datetime.utcnow() - d['timestamp']).seconds < 3600]

        return {
            'total_decisions': len(self.decision_history),
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'confidence_calibration': abs(accuracy - avg_confidence),
            'recent_decision_rate': len(recent_decisions),
            'component_usage': dict(components_used),
            'unique_insights': getattr(self, 'unique_insights_count', 0)
        }

    def _calculate_component_performance(self) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics for each component."""
        component_performance = {}

        for component, metrics in self.component_metrics.items():
            if not metrics:
                continue

            perf = {}

            # Calculate average metrics
            for metric_name, values in metrics.items():
                if values:
                    recent_values = [v['value'] for v in values[-100:]]
                    perf[f'{metric_name}_mean'] = np.mean(recent_values)
                    perf[f'{metric_name}_std'] = np.std(recent_values)

                    # Trend analysis
                    if len(recent_values) > 10:
                        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                        perf[f'{metric_name}_trend'] = 'improving' if trend > 0 else 'declining'

            component_performance[component] = perf

        return component_performance

    def _calculate_financial_performance(self) -> Dict[str, Any]:
        """Calculate financial performance metrics."""
        returns = list(self.financial_metrics['returns'])

        if not returns:
            return {'no_financial_data': True}

        # Basic statistics
        total_return = np.sum(returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        # Risk metrics
        sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0

        # Drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
        max_drawdown = np.min(drawdown)

        # Win rate
        winning_trades = sum(1 for r in returns if r > 0)
        win_rate = winning_trades / len(returns) if returns else 0

        # Risk-adjusted metrics
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            downside_std = np.std(downside_returns)
            sortino_ratio = (avg_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            sortino_ratio = sharpe_ratio  # No downside

        return {
            'total_return': total_return,
            'average_return': avg_return,
            'volatility': std_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': self._calculate_profit_factor(returns)
        }

    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        profits = sum(r for r in returns if r > 0)
        losses = abs(sum(r for r in returns if r < 0))

        return profits / losses if losses > 0 else float('inf') if profits > 0 else 0

    def _calculate_resource_efficiency(self) -> Dict[str, Any]:
        """Calculate resource usage efficiency metrics."""
        if not self.resource_usage:
            return {'no_resource_data': True}

        recent_usage = list(self.resource_usage)

        # Average usage
        avg_cpu = np.mean([u['cpu_percent'] for u in recent_usage])
        avg_memory = np.mean([u['memory_mb'] for u in recent_usage])
        avg_gpu_memory = np.mean([u['gpu_memory_mb'] for u in recent_usage])
        avg_gpu_util = np.mean([u['gpu_utilization'] for u in recent_usage])

        # Peak usage
        peak_cpu = np.max([u['cpu_percent'] for u in recent_usage])
        peak_memory = np.max([u['memory_mb'] for u in recent_usage])

        # Efficiency score (lower is better)
        cpu_efficiency = avg_cpu / max(1, len(self.decision_history))
        memory_efficiency = avg_memory / max(1, len(self.decision_history))

        return {
            'average_cpu_percent': avg_cpu,
            'average_memory_mb': avg_memory,
            'average_gpu_memory_mb': avg_gpu_memory,
            'average_gpu_utilization': avg_gpu_util,
            'peak_cpu_percent': peak_cpu,
            'peak_memory_mb': peak_memory,
            'cpu_efficiency': cpu_efficiency,
            'memory_efficiency': memory_efficiency,
            'resource_score': 1.0 / (1.0 + cpu_efficiency + memory_efficiency / 1000)
        }

    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health metrics."""
        uptime = datetime.utcnow() - self.system_health['uptime_start']

        # Error rates
        recent_errors = [e for e in self.system_health['errors']
                        if (datetime.utcnow() - e['timestamp']).seconds < 3600]
        recent_warnings = [w for w in self.system_health['warnings']
                          if (datetime.utcnow() - w['timestamp']).seconds < 3600]

        error_rate = len(recent_errors) / max(1, uptime.total_seconds() / 3600)
        warning_rate = len(recent_warnings) / max(1, uptime.total_seconds() / 3600)

        # Health score
        health_score = 1.0
        health_score -= min(0.5, error_rate * 0.1)  # Errors reduce health
        health_score -= min(0.3, warning_rate * 0.05)  # Warnings reduce health

        # Check execution success rates
        for op, stats in self.execution_stats.items():
            if stats.get('success_rate', 1.0) < 0.9:
                health_score -= 0.1

        return {
            'uptime_hours': uptime.total_seconds() / 3600,
            'total_errors': len(self.system_health['errors']),
            'total_warnings': len(self.system_health['warnings']),
            'error_rate_per_hour': error_rate,
            'warning_rate_per_hour': warning_rate,
            'health_score': max(0, health_score),
            'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.5 else 'unhealthy'
        }

    def _calculate_success_rate(self, operation: str) -> float:
        """Calculate success rate for an operation."""
        # Simplified - in practice would track successes/failures
        return 0.95

    def _is_unique_insight(self, decision: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Determine if decision represents a unique insight."""
        # Check if decision differs significantly from recent decisions
        if len(self.decision_history) < 10:
            return True

        recent_decisions = list(self.decision_history)[-10:-1]

        # Compare decision characteristics
        current_action = decision.get('integrated_decision', {}).get('action')
        current_confidence = decision.get('confidence', 0.5)

        # Check if action/confidence combination is unique
        for recent in recent_decisions:
            recent_action = recent['decision'].get('integrated_decision', {}).get('action')
            recent_confidence = recent['confidence']

            if (current_action == recent_action and
                abs(current_confidence - recent_confidence) < 0.1):
                return False

        return True

    def get_recent_performance(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance metrics for recent time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Recent performance metrics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Filter recent data
        recent_decisions = [d for d in self.decision_history
                           if d['timestamp'] > cutoff_time]
        recent_outcomes = [o for o in self.decision_outcomes
                          if o['timestamp'] > cutoff_time]

        # Calculate recent metrics
        if recent_outcomes:
            recent_accuracy = sum(1 for o in recent_outcomes if o['success']) / len(recent_outcomes)
            recent_returns = [o['profit_loss'] for o in recent_outcomes if 'profit_loss' in o]
            recent_total_return = sum(recent_returns) if recent_returns else 0
        else:
            recent_accuracy = 0.5
            recent_total_return = 0

        return {
            'time_period_hours': hours,
            'decisions_made': len(recent_decisions),
            'accuracy': recent_accuracy,
            'total_return': recent_total_return,
            'decision_rate_per_hour': len(recent_decisions) / hours
        }

    def export_metrics(self, filepath: str):
        """Export performance metrics to file."""
        metrics = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'summary': self.get_summary(),
            'recent_24h': self.get_recent_performance(24),
            'recent_1h': self.get_recent_performance(1),
            'execution_stats': self.execution_stats,
            'benchmarks': self.benchmarks
        }

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Performance metrics exported to {filepath}")

    def compare_to_benchmark(self) -> Dict[str, Any]:
        """Compare performance to benchmark."""
        current_performance = self._calculate_financial_performance()
        decision_performance = self._calculate_decision_performance()

        comparison = {}

        # Accuracy comparison
        if 'accuracy' in decision_performance:
            comparison['accuracy_vs_benchmark'] = (
                decision_performance['accuracy'] - self.benchmarks['baseline_accuracy']
            )

        # Return comparison
        if 'average_return' in current_performance:
            comparison['return_vs_benchmark'] = (
                current_performance['average_return'] - self.benchmarks['baseline_return']
            )

        # Sharpe comparison
        if 'sharpe_ratio' in current_performance:
            comparison['sharpe_vs_benchmark'] = (
                current_performance['sharpe_ratio'] - self.benchmarks['baseline_sharpe']
            )

        # Overall outperformance
        outperformance_factors = []
        if comparison.get('accuracy_vs_benchmark', 0) > 0:
            outperformance_factors.append('accuracy')
        if comparison.get('return_vs_benchmark', 0) > 0:
            outperformance_factors.append('returns')
        if comparison.get('sharpe_vs_benchmark', 0) > 0:
            outperformance_factors.append('risk_adjusted_returns')

        comparison['outperforming'] = outperformance_factors
        comparison['overall_assessment'] = (
            'outperforming' if len(outperformance_factors) >= 2 else
            'mixed' if len(outperformance_factors) == 1 else
            'underperforming'
        )

        return comparison
