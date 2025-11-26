"""
Advanced Auto-Scaling and Cost Optimization for GPU Trading Platform
Implements intelligent scaling based on multiple metrics and cost considerations
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import statistics

import psutil
import GPUtil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"
    EMERGENCY_SCALE = "emergency_scale"


class InstanceType(Enum):
    """GPU instance types with their specifications"""
    A10 = {"name": "a10", "memory": "16gb", "gpu_memory": "24gb", "cost_per_hour": 0.50}
    A100_40GB = {"name": "a100-40gb", "memory": "32gb", "gpu_memory": "40gb", "cost_per_hour": 2.40}
    A100_80GB = {"name": "a100-80gb", "memory": "64gb", "gpu_memory": "80gb", "cost_per_hour": 3.20}
    PERFORMANCE_4X = {"name": "performance-4x", "memory": "16gb", "cost_per_hour": 0.25}
    PERFORMANCE_8X = {"name": "performance-8x", "memory": "32gb", "cost_per_hour": 0.50}


@dataclass
class ScalingMetrics:
    """System metrics for scaling decisions"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    gpu_memory_percent: float
    gpu_temperature: float
    active_requests: int
    queue_length: int
    response_time_p95: float
    error_rate: float
    trading_volume: int
    profit_rate: float


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning"""
    action: ScalingAction
    target_instances: int
    target_instance_type: str
    reasoning: str
    confidence: float
    cost_impact: float
    metrics_used: Dict[str, float]
    timestamp: str


@dataclass
class CostOptimization:
    """Cost optimization recommendations"""
    current_cost_per_hour: float
    optimized_cost_per_hour: float
    potential_savings: float
    recommendations: List[str]
    implementation_effort: str
    risk_level: str


class MetricsCollector:
    """Collects metrics for scaling decisions"""
    
    def __init__(self, app_name: str):
        self.app_name = app_name
        self.flyctl_path = os.getenv('FLYCTL_PATH', '/home/codespace/.fly/bin/flyctl')
        
        # Setup HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    async def collect_metrics(self) -> ScalingMetrics:
        """Collect comprehensive system metrics"""
        timestamp = datetime.utcnow().isoformat()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU metrics
        gpu_utilization = 0
        gpu_memory_percent = 0
        gpu_temperature = 0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                gpu_utilization = gpu.load * 100
                gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                gpu_temperature = gpu.temperature
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")
        
        # Application metrics
        active_requests = await self._get_active_requests()
        queue_length = await self._get_queue_length()
        response_time_p95 = await self._get_response_time_p95()
        error_rate = await self._get_error_rate()
        
        # Trading metrics
        trading_volume = await self._get_trading_volume()
        profit_rate = await self._get_profit_rate()
        
        return ScalingMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_utilization=gpu_utilization,
            gpu_memory_percent=gpu_memory_percent,
            gpu_temperature=gpu_temperature,
            active_requests=active_requests,
            queue_length=queue_length,
            response_time_p95=response_time_p95,
            error_rate=error_rate,
            trading_volume=trading_volume,
            profit_rate=profit_rate
        )
    
    async def _get_active_requests(self) -> int:
        """Get number of active HTTP requests"""
        try:
            app_url = f"https://{self.app_name}.fly.dev"
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.session.get(f"{app_url}/metrics", timeout=10)
            )
            
            if response.status_code == 200:
                # Parse Prometheus metrics for active requests
                metrics_text = response.text
                for line in metrics_text.split('\n'):
                    if line.startswith('http_requests_active'):
                        return int(float(line.split(' ')[-1]))
            
            return 0
        except Exception as e:
            logger.warning(f"Failed to get active requests: {e}")
            return 0
    
    async def _get_queue_length(self) -> int:
        """Get current queue length"""
        try:
            # This would integrate with your actual queue system
            # For now, return a simulated value based on system load
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 80:
                return int(cpu_percent - 80) * 2
            return 0
        except Exception:
            return 0
    
    async def _get_response_time_p95(self) -> float:
        """Get 95th percentile response time"""
        try:
            app_url = f"https://{self.app_name}.fly.dev"
            start_time = time.time()
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get(f"{app_url}/health", timeout=10)
            )
            
            end_time = time.time()
            return end_time - start_time
        except Exception as e:
            logger.warning(f"Failed to get response time: {e}")
            return 0.0
    
    async def _get_error_rate(self) -> float:
        """Get current error rate"""
        try:
            app_url = f"https://{self.app_name}.fly.dev"
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get(f"{app_url}/metrics", timeout=10)
            )
            
            if response.status_code == 200:
                # Parse error metrics
                metrics_text = response.text
                total_requests = 0
                error_requests = 0
                
                for line in metrics_text.split('\n'):
                    if 'http_requests_total' in line:
                        value = float(line.split(' ')[-1])
                        if 'status="5' in line or 'status="4' in line:
                            error_requests += value
                        total_requests += value
                
                if total_requests > 0:
                    return (error_requests / total_requests) * 100
            
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to get error rate: {e}")
            return 0.0
    
    async def _get_trading_volume(self) -> int:
        """Get current trading volume"""
        try:
            # This would integrate with your trading system
            # For now, return a simulated value
            return int(time.time() % 100)
        except Exception:
            return 0
    
    async def _get_profit_rate(self) -> float:
        """Get current profit rate"""
        try:
            # This would integrate with your trading system
            # For now, return a simulated value
            return 2.5 + (time.time() % 10) / 10
        except Exception:
            return 0.0


class AutoScaler:
    """Intelligent auto-scaling engine"""
    
    def __init__(self, app_name: str, config: Dict[str, Any]):
        self.app_name = app_name
        self.config = config
        self.metrics_collector = MetricsCollector(app_name)
        self.flyctl_path = config.get('flyctl_path', '/home/codespace/.fly/bin/flyctl')
        
        # Scaling configuration
        self.min_instances = config.get('min_instances', 1)
        self.max_instances = config.get('max_instances', 5)
        self.scale_up_threshold = config.get('scale_up_threshold', 80)
        self.scale_down_threshold = config.get('scale_down_threshold', 30)
        self.emergency_threshold = config.get('emergency_threshold', 95)
        self.cooldown_period = config.get('cooldown_period', 300)  # 5 minutes
        
        # Cost optimization
        self.max_cost_per_hour = config.get('max_cost_per_hour', 10.0)
        self.target_cost_efficiency = config.get('target_cost_efficiency', 0.8)
        
        # State tracking
        self.last_scaling_action = None
        self.last_scaling_time = None
        self.metrics_history = []
        self.scaling_history = []
    
    async def evaluate_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Evaluate whether scaling action is needed"""
        logger.info("Evaluating scaling decision")
        
        # Store metrics in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:  # Keep last 100 metrics
            self.metrics_history.pop(0)
        
        # Get current instance count
        current_instances = await self._get_current_instances()
        
        # Check cooldown period
        if self._in_cooldown_period():
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                target_instances=current_instances,
                target_instance_type=await self._get_current_instance_type(),
                reasoning="In cooldown period",
                confidence=1.0,
                cost_impact=0.0,
                metrics_used={},
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Emergency scaling check
        if self._is_emergency_condition(metrics):
            return await self._make_emergency_scaling_decision(metrics, current_instances)
        
        # Regular scaling evaluation
        return await self._make_regular_scaling_decision(metrics, current_instances)
    
    def _is_emergency_condition(self, metrics: ScalingMetrics) -> bool:
        """Check if emergency scaling is needed"""
        conditions = [
            metrics.gpu_temperature > 90,  # GPU overheating
            metrics.error_rate > 50,       # High error rate
            metrics.response_time_p95 > 30, # Very slow responses
            metrics.gpu_utilization > self.emergency_threshold and metrics.gpu_memory_percent > 95
        ]
        
        return any(conditions)
    
    async def _make_emergency_scaling_decision(self, metrics: ScalingMetrics, current_instances: int) -> ScalingDecision:
        """Make emergency scaling decision"""
        target_instances = min(current_instances + 2, self.max_instances)
        
        if metrics.gpu_temperature > 90:
            reasoning = f"Emergency: GPU temperature critical ({metrics.gpu_temperature}°C)"
        elif metrics.error_rate > 50:
            reasoning = f"Emergency: High error rate ({metrics.error_rate}%)"
        elif metrics.response_time_p95 > 30:
            reasoning = f"Emergency: Slow response times ({metrics.response_time_p95}s)"
        else:
            reasoning = f"Emergency: GPU overload (util: {metrics.gpu_utilization}%, mem: {metrics.gpu_memory_percent}%)"
        
        cost_impact = await self._calculate_cost_impact(current_instances, target_instances)
        
        return ScalingDecision(
            action=ScalingAction.EMERGENCY_SCALE,
            target_instances=target_instances,
            target_instance_type=await self._get_current_instance_type(),
            reasoning=reasoning,
            confidence=1.0,
            cost_impact=cost_impact,
            metrics_used=asdict(metrics),
            timestamp=datetime.utcnow().isoformat()
        )
    
    async def _make_regular_scaling_decision(self, metrics: ScalingMetrics, current_instances: int) -> ScalingDecision:
        """Make regular scaling decision based on multiple factors"""
        
        # Calculate scaling score
        scale_up_score = self._calculate_scale_up_score(metrics)
        scale_down_score = self._calculate_scale_down_score(metrics)
        
        # Determine action
        if scale_up_score > 0.7 and current_instances < self.max_instances:
            action = ScalingAction.SCALE_UP
            target_instances = current_instances + 1
            confidence = scale_up_score
            reasoning = self._generate_scale_up_reasoning(metrics, scale_up_score)
        elif scale_down_score > 0.7 and current_instances > self.min_instances:
            action = ScalingAction.SCALE_DOWN
            target_instances = current_instances - 1
            confidence = scale_down_score
            reasoning = self._generate_scale_down_reasoning(metrics, scale_down_score)
        else:
            action = ScalingAction.NO_ACTION
            target_instances = current_instances
            confidence = 1.0 - max(scale_up_score, scale_down_score)
            reasoning = "Metrics within acceptable range"
        
        # Instance type optimization
        target_instance_type = await self._optimize_instance_type(metrics, target_instances)
        cost_impact = await self._calculate_cost_impact(current_instances, target_instances, target_instance_type)
        
        return ScalingDecision(
            action=action,
            target_instances=target_instances,
            target_instance_type=target_instance_type,
            reasoning=reasoning,
            confidence=confidence,
            cost_impact=cost_impact,
            metrics_used=asdict(metrics),
            timestamp=datetime.utcnow().isoformat()
        )
    
    def _calculate_scale_up_score(self, metrics: ScalingMetrics) -> float:
        """Calculate score for scaling up (0-1)"""
        factors = []
        
        # GPU utilization factor
        if metrics.gpu_utilization > self.scale_up_threshold:
            factors.append((metrics.gpu_utilization - self.scale_up_threshold) / (100 - self.scale_up_threshold))
        
        # GPU memory factor
        if metrics.gpu_memory_percent > 80:
            factors.append((metrics.gpu_memory_percent - 80) / 20)
        
        # CPU factor
        if metrics.cpu_percent > self.scale_up_threshold:
            factors.append((metrics.cpu_percent - self.scale_up_threshold) / (100 - self.scale_up_threshold))
        
        # Memory factor
        if metrics.memory_percent > 85:
            factors.append((metrics.memory_percent - 85) / 15)
        
        # Response time factor
        if metrics.response_time_p95 > 5:
            factors.append(min(metrics.response_time_p95 / 10, 1.0))
        
        # Queue length factor
        if metrics.queue_length > 10:
            factors.append(min(metrics.queue_length / 50, 1.0))
        
        # Trading volume factor (business logic)
        if metrics.trading_volume > 50:
            factors.append(min((metrics.trading_volume - 50) / 50, 1.0))
        
        return statistics.mean(factors) if factors else 0.0
    
    def _calculate_scale_down_score(self, metrics: ScalingMetrics) -> float:
        """Calculate score for scaling down (0-1)"""
        factors = []
        
        # GPU utilization factor
        if metrics.gpu_utilization < self.scale_down_threshold:
            factors.append((self.scale_down_threshold - metrics.gpu_utilization) / self.scale_down_threshold)
        
        # CPU factor
        if metrics.cpu_percent < self.scale_down_threshold:
            factors.append((self.scale_down_threshold - metrics.cpu_percent) / self.scale_down_threshold)
        
        # Memory factor
        if metrics.memory_percent < 50:
            factors.append((50 - metrics.memory_percent) / 50)
        
        # Response time factor
        if metrics.response_time_p95 < 2:
            factors.append((2 - metrics.response_time_p95) / 2)
        
        # Queue length factor
        if metrics.queue_length == 0:
            factors.append(1.0)
        
        # Trading volume factor
        if metrics.trading_volume < 20:
            factors.append((20 - metrics.trading_volume) / 20)
        
        # Time-based factor (scale down during low-traffic hours)
        current_hour = datetime.utcnow().hour
        if 2 <= current_hour <= 6:  # Early morning hours
            factors.append(0.5)
        
        return statistics.mean(factors) if factors else 0.0
    
    def _generate_scale_up_reasoning(self, metrics: ScalingMetrics, score: float) -> str:
        """Generate human-readable reasoning for scale up"""
        reasons = []
        
        if metrics.gpu_utilization > self.scale_up_threshold:
            reasons.append(f"GPU utilization high ({metrics.gpu_utilization:.1f}%)")
        
        if metrics.gpu_memory_percent > 80:
            reasons.append(f"GPU memory usage high ({metrics.gpu_memory_percent:.1f}%)")
        
        if metrics.cpu_percent > self.scale_up_threshold:
            reasons.append(f"CPU usage high ({metrics.cpu_percent:.1f}%)")
        
        if metrics.response_time_p95 > 5:
            reasons.append(f"Slow response times ({metrics.response_time_p95:.1f}s)")
        
        if metrics.queue_length > 10:
            reasons.append(f"Queue backlog ({metrics.queue_length} items)")
        
        if metrics.trading_volume > 50:
            reasons.append(f"High trading volume ({metrics.trading_volume})")
        
        return f"Scale up recommended (score: {score:.2f}): {', '.join(reasons)}"
    
    def _generate_scale_down_reasoning(self, metrics: ScalingMetrics, score: float) -> str:
        """Generate human-readable reasoning for scale down"""
        reasons = []
        
        if metrics.gpu_utilization < self.scale_down_threshold:
            reasons.append(f"GPU utilization low ({metrics.gpu_utilization:.1f}%)")
        
        if metrics.cpu_percent < self.scale_down_threshold:
            reasons.append(f"CPU usage low ({metrics.cpu_percent:.1f}%)")
        
        if metrics.response_time_p95 < 2:
            reasons.append(f"Fast response times ({metrics.response_time_p95:.1f}s)")
        
        if metrics.queue_length == 0:
            reasons.append("No queue backlog")
        
        if metrics.trading_volume < 20:
            reasons.append(f"Low trading volume ({metrics.trading_volume})")
        
        return f"Scale down recommended (score: {score:.2f}): {', '.join(reasons)}"
    
    async def _optimize_instance_type(self, metrics: ScalingMetrics, target_instances: int) -> str:
        """Optimize instance type based on workload"""
        current_type = await self._get_current_instance_type()
        
        # If GPU utilization is consistently high, consider A100
        if metrics.gpu_utilization > 85 and metrics.gpu_memory_percent > 80:
            if "a10" in current_type:
                return "a100-40gb"
            elif "a100-40gb" in current_type and metrics.gpu_memory_percent > 90:
                return "a100-80gb"
        
        # If GPU utilization is low, consider downgrading
        elif metrics.gpu_utilization < 30 and metrics.gpu_memory_percent < 40:
            if "a100-80gb" in current_type:
                return "a100-40gb"
            elif "a100-40gb" in current_type:
                return "a10"
        
        return current_type
    
    async def _calculate_cost_impact(self, current_instances: int, target_instances: int, 
                                   target_instance_type: Optional[str] = None) -> float:
        """Calculate cost impact of scaling decision"""
        if target_instance_type is None:
            target_instance_type = await self._get_current_instance_type()
        
        current_type = await self._get_current_instance_type()
        
        # Get hourly costs
        current_cost = self._get_instance_cost(current_type) * current_instances
        target_cost = self._get_instance_cost(target_instance_type) * target_instances
        
        return target_cost - current_cost
    
    def _get_instance_cost(self, instance_type: str) -> float:
        """Get hourly cost for instance type"""
        cost_map = {
            "a10": 0.50,
            "a100-40gb": 2.40,
            "a100-80gb": 3.20,
            "performance-4x": 0.25,
            "performance-8x": 0.50
        }
        
        for key, cost in cost_map.items():
            if key in instance_type:
                return cost
        
        return 1.0  # Default cost
    
    async def _get_current_instances(self) -> int:
        """Get current number of instances"""
        try:
            cmd = [self.flyctl_path, 'status', '--app', self.app_name, '--json']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            status_data = json.loads(result.stdout)
            machines = status_data.get('machines', [])
            
            # Count running machines
            running_machines = [m for m in machines if m.get('state') == 'started']
            return len(running_machines)
            
        except Exception as e:
            logger.error(f"Failed to get current instances: {e}")
            return 1
    
    async def _get_current_instance_type(self) -> str:
        """Get current instance type"""
        try:
            cmd = [self.flyctl_path, 'machine', 'list', '--app', self.app_name, '--json']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            machines = json.loads(result.stdout)
            if machines:
                # Get the type from the first machine
                config = machines[0].get('config', {})
                return config.get('size', 'performance-8x')
            
            return 'performance-8x'
            
        except Exception as e:
            logger.error(f"Failed to get current instance type: {e}")
            return 'performance-8x'
    
    def _in_cooldown_period(self) -> bool:
        """Check if we're in cooldown period"""
        if self.last_scaling_time is None:
            return False
        
        time_since_last_scaling = time.time() - self.last_scaling_time
        return time_since_last_scaling < self.cooldown_period
    
    async def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute the scaling decision"""
        if decision.action == ScalingAction.NO_ACTION:
            logger.info("No scaling action needed")
            return True
        
        logger.info(f"Executing scaling decision: {decision.action.value} to {decision.target_instances} instances")
        
        try:
            if decision.action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_DOWN, ScalingAction.EMERGENCY_SCALE]:
                success = await self._scale_instances(decision.target_instances)
                
                if success:
                    self.last_scaling_action = decision.action
                    self.last_scaling_time = time.time()
                    self.scaling_history.append(decision)
                    
                    # Keep scaling history limited
                    if len(self.scaling_history) > 50:
                        self.scaling_history.pop(0)
                    
                    logger.info(f"Scaling completed successfully: {decision.reasoning}")
                    return True
                else:
                    logger.error("Scaling failed")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
            return False
    
    async def _scale_instances(self, target_instances: int) -> bool:
        """Scale to target number of instances"""
        try:
            cmd = [
                self.flyctl_path, 'scale', 'count', str(target_instances),
                '--app', self.app_name
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Scaling command output: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Scaling command failed: {e.stderr}")
            return False


class CostOptimizer:
    """Cost optimization analyzer"""
    
    def __init__(self, app_name: str, auto_scaler: AutoScaler):
        self.app_name = app_name
        self.auto_scaler = auto_scaler
    
    async def analyze_cost_optimization(self, metrics_history: List[ScalingMetrics]) -> CostOptimization:
        """Analyze cost optimization opportunities"""
        if not metrics_history:
            return CostOptimization(
                current_cost_per_hour=0.0,
                optimized_cost_per_hour=0.0,
                potential_savings=0.0,
                recommendations=[],
                implementation_effort="low",
                risk_level="low"
            )
        
        # Calculate current costs
        current_instances = await self.auto_scaler._get_current_instances()
        current_type = await self.auto_scaler._get_current_instance_type()
        current_cost = self.auto_scaler._get_instance_cost(current_type) * current_instances
        
        # Analyze usage patterns
        avg_gpu_util = statistics.mean([m.gpu_utilization for m in metrics_history])
        avg_cpu_util = statistics.mean([m.cpu_percent for m in metrics_history])
        avg_memory_util = statistics.mean([m.memory_percent for m in metrics_history])
        
        recommendations = []
        optimized_cost = current_cost
        
        # GPU utilization optimization
        if avg_gpu_util < 40:
            if "a100-80gb" in current_type:
                recommendations.append("Consider downgrading from A100-80GB to A100-40GB")
                optimized_cost = self.auto_scaler._get_instance_cost("a100-40gb") * current_instances
            elif "a100-40gb" in current_type:
                recommendations.append("Consider downgrading from A100-40GB to A10")
                optimized_cost = self.auto_scaler._get_instance_cost("a10") * current_instances
        
        # Instance count optimization
        peak_hours = self._identify_peak_hours(metrics_history)
        if len(peak_hours) < 12:  # Less than 12 hours of peak usage
            recommendations.append("Consider using auto-stop during off-peak hours")
            optimized_cost *= 0.7  # 30% savings
        
        # Spot instance recommendations
        if current_instances > 1:
            recommendations.append("Consider using spot instances for non-critical workloads")
            optimized_cost *= 0.6  # 40% savings on spot instances
        
        # Resource efficiency
        if avg_memory_util < 50 and avg_cpu_util < 50:
            recommendations.append("Resources are underutilized - consider smaller instance types")
        
        # Trading-specific optimizations
        trading_volumes = [m.trading_volume for m in metrics_history]
        if max(trading_volumes) - min(trading_volumes) > 50:
            recommendations.append("High trading volume variance - implement predictive scaling")
        
        potential_savings = current_cost - optimized_cost
        
        # Determine implementation effort and risk
        implementation_effort = "medium"
        risk_level = "low"
        
        if len(recommendations) > 3:
            implementation_effort = "high"
        if any("downgrad" in r for r in recommendations):
            risk_level = "medium"
        
        return CostOptimization(
            current_cost_per_hour=current_cost,
            optimized_cost_per_hour=optimized_cost,
            potential_savings=potential_savings,
            recommendations=recommendations,
            implementation_effort=implementation_effort,
            risk_level=risk_level
        )
    
    def _identify_peak_hours(self, metrics_history: List[ScalingMetrics]) -> List[int]:
        """Identify peak usage hours"""
        hourly_usage = {}
        
        for metrics in metrics_history:
            hour = datetime.fromisoformat(metrics.timestamp.replace('Z', '+00:00')).hour
            if hour not in hourly_usage:
                hourly_usage[hour] = []
            hourly_usage[hour].append(metrics.gpu_utilization)
        
        # Calculate average usage per hour
        hourly_avg = {hour: statistics.mean(usage) for hour, usage in hourly_usage.items()}
        
        # Identify hours with above-average usage
        overall_avg = statistics.mean(hourly_avg.values())
        peak_hours = [hour for hour, avg in hourly_avg.items() if avg > overall_avg * 1.2]
        
        return peak_hours


async def main_scaling_loop(app_name: str, config: Dict[str, Any]):
    """Main auto-scaling loop"""
    auto_scaler = AutoScaler(app_name, config)
    cost_optimizer = CostOptimizer(app_name, auto_scaler)
    
    logger.info(f"Starting auto-scaling loop for {app_name}")
    
    while True:
        try:
            # Collect metrics
            metrics = await auto_scaler.metrics_collector.collect_metrics()
            logger.info(f"Collected metrics: GPU {metrics.gpu_utilization:.1f}%, CPU {metrics.cpu_percent:.1f}%")
            
            # Evaluate scaling decision
            decision = await auto_scaler.evaluate_scaling_decision(metrics)
            logger.info(f"Scaling decision: {decision.action.value} - {decision.reasoning}")
            
            # Execute scaling if needed
            if decision.action != ScalingAction.NO_ACTION:
                success = await auto_scaler.execute_scaling_decision(decision)
                if success:
                    logger.info("Scaling executed successfully")
                else:
                    logger.error("Scaling execution failed")
            
            # Periodic cost optimization analysis
            if len(auto_scaler.metrics_history) % 20 == 0:  # Every 20 cycles
                cost_analysis = await cost_optimizer.analyze_cost_optimization(auto_scaler.metrics_history)
                logger.info(f"Cost optimization: ${cost_analysis.potential_savings:.2f}/hour potential savings")
                
                if cost_analysis.recommendations:
                    logger.info("Cost optimization recommendations:")
                    for rec in cost_analysis.recommendations:
                        logger.info(f"  - {rec}")
            
            # Wait before next cycle
            await asyncio.sleep(config.get('scaling_interval', 60))
            
        except Exception as e:
            logger.error(f"Error in scaling loop: {e}")
            await asyncio.sleep(30)  # Shorter wait on error


if __name__ == "__main__":
    # CLI tool for auto-scaling
    import argparse
    
    def get_scaling_config() -> Dict[str, Any]:
        return {
            'min_instances': int(os.getenv('MIN_INSTANCES', '1')),
            'max_instances': int(os.getenv('MAX_INSTANCES', '5')),
            'scale_up_threshold': float(os.getenv('SCALE_UP_THRESHOLD', '80')),
            'scale_down_threshold': float(os.getenv('SCALE_DOWN_THRESHOLD', '30')),
            'emergency_threshold': float(os.getenv('EMERGENCY_THRESHOLD', '95')),
            'cooldown_period': int(os.getenv('COOLDOWN_PERIOD', '300')),
            'max_cost_per_hour': float(os.getenv('MAX_COST_PER_HOUR', '10.0')),
            'scaling_interval': int(os.getenv('SCALING_INTERVAL', '60')),
            'flyctl_path': os.getenv('FLYCTL_PATH', '/home/codespace/.fly/bin/flyctl')
        }
    
    async def cli_main():
        parser = argparse.ArgumentParser(description="Auto-Scaler for GPU Trading Platform")
        parser.add_argument('command', choices=['run', 'analyze', 'optimize'])
        parser.add_argument('--app', default='ruvtrade', help='App name')
        parser.add_argument('--duration', type=int, default=0, help='Run duration in minutes (0 for infinite)')
        
        args = parser.parse_args()
        
        config = get_scaling_config()
        
        if args.command == 'run':
            if args.duration > 0:
                # Run for specified duration
                end_time = time.time() + (args.duration * 60)
                while time.time() < end_time:
                    await main_scaling_loop(args.app, config)
            else:
                # Run indefinitely
                await main_scaling_loop(args.app, config)
        
        elif args.command == 'analyze':
            auto_scaler = AutoScaler(args.app, config)
            metrics = await auto_scaler.metrics_collector.collect_metrics()
            decision = await auto_scaler.evaluate_scaling_decision(metrics)
            
            print(f"Current metrics: {json.dumps(asdict(metrics), indent=2)}")
            print(f"Scaling decision: {json.dumps(asdict(decision), indent=2)}")
        
        elif args.command == 'optimize':
            auto_scaler = AutoScaler(args.app, config)
            cost_optimizer = CostOptimizer(args.app, auto_scaler)
            
            # Collect some metrics first
            metrics_history = []
            for _ in range(10):
                metrics = await auto_scaler.metrics_collector.collect_metrics()
                metrics_history.append(metrics)
                await asyncio.sleep(6)  # 1 minute of data
            
            cost_analysis = await cost_optimizer.analyze_cost_optimization(metrics_history)
            print(f"Cost optimization analysis: {json.dumps(asdict(cost_analysis), indent=2)}")
    
    asyncio.run(cli_main())