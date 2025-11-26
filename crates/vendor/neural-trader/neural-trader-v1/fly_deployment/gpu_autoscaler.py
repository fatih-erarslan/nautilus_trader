#!/usr/bin/env python3
"""
Fly.io GPU Auto-Scaler for AI News Trading Platform
Intelligent scaling based on GPU utilization, trading volume, and cost optimization
"""

import asyncio
import logging
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"
    EMERGENCY_SCALE = "emergency_scale"

@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: float
    gpu_utilization: float
    gpu_memory_usage: float
    cpu_utilization: float
    memory_usage: float
    request_rate: float
    response_time_p95: float
    error_rate: float
    prediction_queue_size: int
    trading_volume: float
    cost_per_hour: float
    instance_count: int

@dataclass
class ScalingRule:
    """Scaling rule configuration."""
    name: str
    condition: str
    action: ScalingAction
    priority: int
    cooldown_minutes: int
    min_instances: int
    max_instances: int

class FlyIOGPUAutoScaler:
    """Intelligent GPU auto-scaler for Fly.io deployment."""
    
    def __init__(self, app_name: str = "ai-news-trader-neural"):
        self.app_name = app_name
        self.api_token = os.getenv("FLY_API_TOKEN")
        self.base_url = "https://api.fly.io/v1"
        
        # Scaling configuration
        self.config = {
            "min_instances": 1,
            "max_instances": 5,
            "target_gpu_utilization": 75,
            "scale_up_threshold": 85,
            "scale_down_threshold": 40,
            "cooldown_minutes": 5,
            "emergency_threshold": 95,
            "cost_optimization_enabled": True,
            "trading_hours_scaling": True
        }
        
        # Scaling rules
        self.scaling_rules = [
            ScalingRule(
                name="emergency_gpu_overload",
                condition="gpu_utilization > 95 or gpu_memory_usage > 95",
                action=ScalingAction.EMERGENCY_SCALE,
                priority=1,
                cooldown_minutes=2,
                min_instances=2,
                max_instances=5
            ),
            ScalingRule(
                name="high_gpu_utilization",
                condition="gpu_utilization > 85 and request_rate > 50",
                action=ScalingAction.SCALE_UP,
                priority=2,
                cooldown_minutes=5,
                min_instances=1,
                max_instances=4
            ),
            ScalingRule(
                name="low_gpu_utilization", 
                condition="gpu_utilization < 40 and request_rate < 20",
                action=ScalingAction.SCALE_DOWN,
                priority=3,
                cooldown_minutes=10,
                min_instances=1,
                max_instances=5
            ),
            ScalingRule(
                name="trading_hours_scale_up",
                condition="trading_hours and request_rate > 30",
                action=ScalingAction.SCALE_UP,
                priority=4,
                cooldown_minutes=5,
                min_instances=2,
                max_instances=4
            ),
            ScalingRule(
                name="off_hours_scale_down",
                condition="not trading_hours and gpu_utilization < 30",
                action=ScalingAction.SCALE_DOWN,
                priority=5,
                cooldown_minutes=15,
                min_instances=1,
                max_instances=2
            )
        ]
        
        # State tracking
        self.last_scaling_action = None
        self.last_scaling_time = 0
        self.metrics_history = []
        self.scaling_history = []
        
        # HTTP client
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_token}"},
            timeout=30.0
        )
    
    async def get_app_metrics(self) -> ScalingMetrics:
        """Collect comprehensive metrics for scaling decisions."""
        try:
            # Get app status
            app_response = await self.client.get(f"{self.base_url}/apps/{self.app_name}")
            app_data = app_response.json()
            
            # Get machine metrics
            machines_response = await self.client.get(f"{self.base_url}/apps/{self.app_name}/machines")
            machines_data = machines_response.json()
            
            # Get application metrics from health endpoint
            app_metrics = await self._get_app_health_metrics()
            
            # Calculate current metrics
            current_time = time.time()
            instance_count = len([m for m in machines_data if m.get("state") == "started"])
            
            metrics = ScalingMetrics(
                timestamp=current_time,
                gpu_utilization=app_metrics.get("gpu_utilization", 0),
                gpu_memory_usage=app_metrics.get("gpu_memory_usage", 0),
                cpu_utilization=app_metrics.get("cpu_utilization", 0),
                memory_usage=app_metrics.get("memory_usage", 0),
                request_rate=app_metrics.get("request_rate", 0),
                response_time_p95=app_metrics.get("response_time_p95", 0),
                error_rate=app_metrics.get("error_rate", 0),
                prediction_queue_size=app_metrics.get("prediction_queue_size", 0),
                trading_volume=await self._get_trading_volume(),
                cost_per_hour=self._calculate_hourly_cost(instance_count),
                instance_count=instance_count
            )
            
            # Store metrics history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:  # Keep last 100 metrics
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            # Return default metrics
            return ScalingMetrics(
                timestamp=time.time(),
                gpu_utilization=0,
                gpu_memory_usage=0,
                cpu_utilization=0,
                memory_usage=0,
                request_rate=0,
                response_time_p95=0,
                error_rate=0,
                prediction_queue_size=0,
                trading_volume=0,
                cost_per_hour=0,
                instance_count=1
            )
    
    async def _get_app_health_metrics(self) -> Dict:
        """Get application health metrics from deployed app."""
        try:
            app_url = f"https://{self.app_name}.fly.dev"
            response = await self.client.get(f"{app_url}/metrics/gpu", timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get app metrics: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.warning(f"App metrics unavailable: {e}")
            return {}
    
    async def _get_trading_volume(self) -> float:
        """Get current trading volume indicator."""
        try:
            # This would integrate with your trading platform
            # For now, return a simulated value based on time
            current_hour = datetime.now().hour
            
            # Higher volume during trading hours (9-16 EST)
            if 9 <= current_hour <= 16:
                return 100.0 + (current_hour - 9) * 10
            else:
                return 20.0
                
        except Exception as e:
            logger.warning(f"Failed to get trading volume: {e}")
            return 0.0
    
    def _calculate_hourly_cost(self, instance_count: int) -> float:
        """Calculate current hourly cost."""
        # Fly.io A100-40GB GPU cost (approximate)
        gpu_cost_per_hour = 3.20
        cpu_cost_per_hour = 0.50
        memory_cost_per_hour = 0.25
        
        return instance_count * (gpu_cost_per_hour + cpu_cost_per_hour + memory_cost_per_hour)
    
    def _is_trading_hours(self) -> bool:
        """Check if we're in trading hours (9 AM - 4 PM EST)."""
        now = datetime.now()
        current_hour = now.hour
        current_weekday = now.weekday()
        
        # Monday = 0, Sunday = 6
        is_weekday = current_weekday < 5
        is_trading_time = 9 <= current_hour <= 16
        
        return is_weekday and is_trading_time
    
    def evaluate_scaling_rules(self, metrics: ScalingMetrics) -> Tuple[ScalingAction, str]:
        """Evaluate scaling rules against current metrics."""
        
        # Check cooldown period
        time_since_last_scaling = time.time() - self.last_scaling_time
        
        # Create evaluation context
        context = {
            **asdict(metrics),
            "trading_hours": self._is_trading_hours(),
            "time_since_last_scaling": time_since_last_scaling
        }
        
        # Sort rules by priority
        sorted_rules = sorted(self.scaling_rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            # Check cooldown
            if time_since_last_scaling < rule.cooldown_minutes * 60:
                continue
            
            # Evaluate condition
            try:
                if eval(rule.condition, {"__builtins__": {}}, context):
                    # Check instance limits
                    if rule.action == ScalingAction.SCALE_UP and metrics.instance_count >= rule.max_instances:
                        continue
                    if rule.action == ScalingAction.SCALE_DOWN and metrics.instance_count <= rule.min_instances:
                        continue
                    
                    return rule.action, rule.name
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
                continue
        
        return ScalingAction.NO_ACTION, "no_conditions_met"
    
    async def execute_scaling_action(self, action: ScalingAction, reason: str, current_instances: int) -> bool:
        """Execute the scaling action."""
        try:
            if action == ScalingAction.NO_ACTION:
                return True
            
            # Determine target instance count
            if action == ScalingAction.SCALE_UP:
                target_instances = min(current_instances + 1, self.config["max_instances"])
            elif action == ScalingAction.SCALE_DOWN:
                target_instances = max(current_instances - 1, self.config["min_instances"])
            elif action == ScalingAction.EMERGENCY_SCALE:
                target_instances = min(current_instances + 2, self.config["max_instances"])
            else:
                target_instances = current_instances
            
            if target_instances == current_instances:
                logger.info(f"Scaling action {action.value} would not change instance count")
                return True
            
            logger.info(f"Executing scaling action: {action.value} ({reason})")
            logger.info(f"Scaling from {current_instances} to {target_instances} instances")
            
            # Execute scaling via Fly API
            scaling_response = await self.client.post(
                f"{self.base_url}/apps/{self.app_name}/machines/scale",
                json={"count": target_instances}
            )
            
            if scaling_response.status_code in [200, 202]:
                self.last_scaling_action = action
                self.last_scaling_time = time.time()
                
                # Record scaling event
                scaling_event = {
                    "timestamp": time.time(),
                    "action": action.value,
                    "reason": reason,
                    "from_instances": current_instances,
                    "to_instances": target_instances,
                    "success": True
                }
                self.scaling_history.append(scaling_event)
                
                logger.info(f"Scaling successful: {action.value}")
                return True
            else:
                logger.error(f"Scaling failed: {scaling_response.status_code} - {scaling_response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute scaling action {action.value}: {e}")
            return False
    
    async def cost_optimization_check(self, metrics: ScalingMetrics) -> Tuple[ScalingAction, str]:
        """Check for cost optimization opportunities."""
        if not self.config["cost_optimization_enabled"]:
            return ScalingAction.NO_ACTION, "cost_optimization_disabled"
        
        # Calculate efficiency metrics
        gpu_efficiency = metrics.gpu_utilization / 100.0
        cost_per_prediction = metrics.cost_per_hour / max(1, metrics.request_rate * 3600)
        
        # High cost, low efficiency - scale down
        if gpu_efficiency < 0.3 and cost_per_prediction > 0.10 and metrics.instance_count > 1:
            return ScalingAction.SCALE_DOWN, "cost_optimization_low_efficiency"
        
        # High demand, good efficiency - can scale up
        if gpu_efficiency > 0.8 and cost_per_prediction < 0.05 and metrics.request_rate > 50:
            return ScalingAction.SCALE_UP, "cost_optimization_high_efficiency"
        
        return ScalingAction.NO_ACTION, "cost_optimization_no_action"
    
    async def generate_scaling_report(self) -> Dict:
        """Generate a comprehensive scaling report."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 metrics
        
        # Calculate averages
        avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_p95 for m in recent_metrics) / len(recent_metrics)
        avg_cost = sum(m.cost_per_hour for m in recent_metrics) / len(recent_metrics)
        
        # Generate recommendations
        recommendations = []
        
        if avg_gpu_util > 85:
            recommendations.append("Consider scaling up - high GPU utilization")
        elif avg_gpu_util < 30:
            recommendations.append("Consider scaling down - low GPU utilization")
        
        if avg_response_time > 100:  # ms
            recommendations.append("High response time - consider scaling up")
        
        if avg_cost > 20:  # per hour
            recommendations.append("High costs - review scaling strategy")
        
        return {
            "timestamp": time.time(),
            "current_instances": recent_metrics[-1].instance_count,
            "metrics_summary": {
                "avg_gpu_utilization": avg_gpu_util,
                "avg_response_time_ms": avg_response_time,
                "avg_cost_per_hour": avg_cost,
                "trading_hours": self._is_trading_hours()
            },
            "recent_scaling_actions": self.scaling_history[-5:],
            "recommendations": recommendations,
            "config": self.config
        }
    
    async def run_scaling_cycle(self) -> Dict:
        """Run a single scaling evaluation cycle."""
        try:
            logger.info("Starting scaling evaluation cycle")
            
            # Collect metrics
            metrics = await self.get_app_metrics()
            
            # Evaluate scaling rules
            action, reason = self.evaluate_scaling_rules(metrics)
            
            # Check cost optimization
            if action == ScalingAction.NO_ACTION:
                cost_action, cost_reason = await self.cost_optimization_check(metrics)
                if cost_action != ScalingAction.NO_ACTION:
                    action, reason = cost_action, cost_reason
            
            # Execute scaling action
            success = await self.execute_scaling_action(action, reason, metrics.instance_count)
            
            # Log results
            result = {
                "timestamp": time.time(),
                "metrics": asdict(metrics),
                "action": action.value,
                "reason": reason,
                "success": success,
                "trading_hours": self._is_trading_hours()
            }
            
            logger.info(f"Scaling cycle completed: {action.value} ({reason})")
            return result
            
        except Exception as e:
            logger.error(f"Scaling cycle failed: {e}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "success": False
            }
    
    async def run_continuous_scaling(self, interval_seconds: int = 60):
        """Run continuous auto-scaling."""
        logger.info(f"Starting continuous auto-scaling (interval: {interval_seconds}s)")
        
        while True:
            try:
                result = await self.run_scaling_cycle()
                
                # Log to file for monitoring
                with open("/app/logs/scaling.log", "a") as f:
                    f.write(f"{json.dumps(result)}\n")
                
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Auto-scaling stopped by user")
                break
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.client.aclose()

async def main():
    """Main entry point for auto-scaler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fly.io GPU Auto-Scaler")
    parser.add_argument("--app", default="ai-news-trader-neural", help="Fly.io app name")
    parser.add_argument("--interval", type=int, default=60, help="Scaling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once instead of continuously")
    parser.add_argument("--report", action="store_true", help="Generate scaling report")
    
    args = parser.parse_args()
    
    # Initialize auto-scaler
    scaler = FlyIOGPUAutoScaler(app_name=args.app)
    
    try:
        if args.report:
            # Generate and print report
            report = await scaler.generate_scaling_report()
            print(json.dumps(report, indent=2))
        elif args.once:
            # Run single scaling cycle
            result = await scaler.run_scaling_cycle()
            print(json.dumps(result, indent=2))
        else:
            # Run continuous scaling
            await scaler.run_continuous_scaling(args.interval)
            
    finally:
        await scaler.cleanup()

if __name__ == "__main__":
    asyncio.run(main())