#!/usr/bin/env python3
"""
Trading System Failover Manager
Automated failover procedures for critical trading components
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import psutil
from pathlib import Path

class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"

class FailoverAction(Enum):
    RESTART_SERVICE = "restart_service"
    SWITCH_DATACENTER = "switch_datacenter"
    ACTIVATE_STANDBY = "activate_standby"
    SCALE_RESOURCES = "scale_resources"
    ISOLATE_COMPONENT = "isolate_component"
    FULL_FAILOVER = "full_failover"

@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_id: str
    status: ComponentStatus
    last_check: datetime
    response_time_ms: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    custom_metrics: Dict[str, Any]
    alerts: List[str]

@dataclass
class FailoverConfig:
    """Configuration for failover procedures"""
    component_id: str
    primary_endpoint: str
    backup_endpoints: List[str]
    health_check_interval: int = 30
    failure_threshold: int = 3
    recovery_timeout: int = 300
    auto_failover: bool = True
    notification_channels: List[str] = None
    custom_checks: List[Callable] = None

class TradingSystemFailoverManager:
    """Comprehensive failover management for trading systems"""
    
    def __init__(self, config_file: str = "disaster_recovery/failover_config.json"):
        self.components: Dict[str, FailoverConfig] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.failover_history: List[Dict] = []
        self.active_failovers: Dict[str, Dict] = {}
        self.logger = self._setup_logging()
        self.monitoring_active = False
        
        # Load configuration
        self._load_config(config_file)
        
        # Initialize component health tracking
        self._initialize_health_tracking()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup failover manager logging"""
        logger = logging.getLogger("trading_failover")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        Path("disaster_recovery/logs").mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler("disaster_recovery/logs/failover.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_config(self, config_file: str) -> None:
        """Load failover configuration"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                for comp_id, comp_config in config_data.get('components', {}).items():
                    self.components[comp_id] = FailoverConfig(**comp_config)
            else:
                # Create default configuration
                self._create_default_config(config_file)
                
        except Exception as e:
            self.logger.error(f"Failed to load failover config: {e}")
            self._create_default_config(config_file)
    
    def _create_default_config(self, config_file: str) -> None:
        """Create default failover configuration"""
        default_config = {
            "components": {
                "trading_engine": {
                    "component_id": "trading_engine",
                    "primary_endpoint": "http://localhost:8080/health",
                    "backup_endpoints": [
                        "http://backup1:8080/health",
                        "http://backup2:8080/health"
                    ],
                    "health_check_interval": 30,
                    "failure_threshold": 3,
                    "recovery_timeout": 300,
                    "auto_failover": True,
                    "notification_channels": ["slack", "email", "pagerduty"]
                },
                "risk_manager": {
                    "component_id": "risk_manager",
                    "primary_endpoint": "http://localhost:8081/health",
                    "backup_endpoints": [
                        "http://backup1:8081/health",
                        "http://backup2:8081/health"
                    ],
                    "health_check_interval": 15,
                    "failure_threshold": 2,
                    "recovery_timeout": 180,
                    "auto_failover": True,
                    "notification_channels": ["slack", "email", "pagerduty"]
                },
                "data_feed": {
                    "component_id": "data_feed",
                    "primary_endpoint": "http://localhost:8082/health",
                    "backup_endpoints": [
                        "http://backup1:8082/health",
                        "http://backup2:8082/health"
                    ],
                    "health_check_interval": 10,
                    "failure_threshold": 2,
                    "recovery_timeout": 120,
                    "auto_failover": True,
                    "notification_channels": ["slack", "email"]
                },
                "neural_predictor": {
                    "component_id": "neural_predictor",
                    "primary_endpoint": "http://localhost:8083/health",
                    "backup_endpoints": [
                        "http://backup1:8083/health",
                        "http://backup2:8083/health"
                    ],
                    "health_check_interval": 60,
                    "failure_threshold": 3,
                    "recovery_timeout": 600,
                    "auto_failover": True,
                    "notification_channels": ["slack", "email"]
                }
            }
        }
        
        # Save default configuration
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        # Load the default configuration
        for comp_id, comp_config in default_config['components'].items():
            self.components[comp_id] = FailoverConfig(**comp_config)
    
    def _initialize_health_tracking(self) -> None:
        """Initialize health tracking for all components"""
        for comp_id in self.components:
            self.component_health[comp_id] = ComponentHealth(
                component_id=comp_id,
                status=ComponentStatus.HEALTHY,
                last_check=datetime.now(),
                response_time_ms=0.0,
                error_rate=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                custom_metrics={},
                alerts=[]
            )
    
    async def check_component_health(self, component_id: str) -> ComponentHealth:
        """Check health of a specific component"""
        if component_id not in self.components:
            raise ValueError(f"Unknown component: {component_id}")
        
        config = self.components[component_id]
        health = self.component_health[component_id]
        
        try:
            start_time = time.time()
            
            # Perform HTTP health check
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(config.primary_endpoint) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        health.status = ComponentStatus.HEALTHY
                        health.response_time_ms = response_time
                        health.error_rate = 0.0
                        health.alerts.clear()
                    else:
                        health.status = ComponentStatus.DEGRADED
                        health.error_rate += 0.1
                        health.alerts.append(f"HTTP {response.status} from {config.primary_endpoint}")
        
        except Exception as e:
            health.status = ComponentStatus.FAILED
            health.error_rate = 1.0
            health.alerts.append(f"Health check failed: {str(e)}")
            self.logger.error(f"Health check failed for {component_id}: {e}")
        
        # Get system metrics if component is local
        try:
            if "localhost" in config.primary_endpoint:
                health.cpu_usage = psutil.cpu_percent()
                health.memory_usage = psutil.virtual_memory().percent
        except Exception:
            pass
        
        health.last_check = datetime.now()
        return health
    
    async def monitor_all_components(self) -> None:
        """Monitor health of all components continuously"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Check all components in parallel
                health_checks = [
                    self.check_component_health(comp_id)
                    for comp_id in self.components
                ]
                
                await asyncio.gather(*health_checks, return_exceptions=True)
                
                # Evaluate failover conditions
                await self._evaluate_failover_conditions()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _evaluate_failover_conditions(self) -> None:
        """Evaluate if any components need failover"""
        for comp_id, health in self.component_health.items():
            config = self.components[comp_id]
            
            # Check if component has failed
            if (health.status == ComponentStatus.FAILED and 
                health.error_rate >= config.failure_threshold / 10.0 and
                config.auto_failover and
                comp_id not in self.active_failovers):
                
                await self.initiate_failover(comp_id, FailoverAction.ACTIVATE_STANDBY)
            
            # Check for degraded performance
            elif (health.status == ComponentStatus.DEGRADED and
                  health.response_time_ms > 5000):  # 5 second threshold
                
                await self.send_alert(comp_id, f"Component {comp_id} performance degraded")
    
    async def initiate_failover(self, 
                               component_id: str, 
                               action: FailoverAction,
                               manual: bool = False) -> bool:
        """Initiate failover for a component"""
        try:
            if component_id not in self.components:
                raise ValueError(f"Unknown component: {component_id}")
            
            config = self.components[component_id]
            
            # Record failover initiation
            failover_record = {
                'component_id': component_id,
                'action': action.value,
                'timestamp': datetime.now().isoformat(),
                'manual': manual,
                'reason': f"Component health: {self.component_health[component_id].status.value}"
            }
            
            self.active_failovers[component_id] = failover_record
            self.failover_history.append(failover_record)
            
            self.logger.warning(f"Initiating failover for {component_id}: {action.value}")
            
            # Execute failover action
            success = await self._execute_failover_action(component_id, action)
            
            if success:
                await self.send_notification(
                    component_id,
                    f"✅ Failover successful for {component_id} ({action.value})",
                    "success"
                )
                
                # Start recovery monitoring
                asyncio.create_task(self._monitor_recovery(component_id))
                
            else:
                await self.send_notification(
                    component_id,
                    f"❌ Failover failed for {component_id} ({action.value})",
                    "error"
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failover initiation failed for {component_id}: {e}")
            return False
    
    async def _execute_failover_action(self, 
                                      component_id: str, 
                                      action: FailoverAction) -> bool:
        """Execute specific failover action"""
        try:
            config = self.components[component_id]
            
            if action == FailoverAction.ACTIVATE_STANDBY:
                return await self._activate_standby(component_id)
            
            elif action == FailoverAction.RESTART_SERVICE:
                return await self._restart_service(component_id)
            
            elif action == FailoverAction.SWITCH_DATACENTER:
                return await self._switch_datacenter(component_id)
            
            elif action == FailoverAction.SCALE_RESOURCES:
                return await self._scale_resources(component_id)
            
            elif action == FailoverAction.ISOLATE_COMPONENT:
                return await self._isolate_component(component_id)
            
            elif action == FailoverAction.FULL_FAILOVER:
                return await self._full_failover(component_id)
            
            else:
                self.logger.error(f"Unknown failover action: {action}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to execute failover action {action} for {component_id}: {e}")
            return False
    
    async def _activate_standby(self, component_id: str) -> bool:
        """Activate standby instance for component"""
        config = self.components[component_id]
        
        # Try backup endpoints in order
        for backup_endpoint in config.backup_endpoints:
            try:
                async with aiohttp.ClientSession() as session:
                    # Check if backup is healthy
                    async with session.get(backup_endpoint) as response:
                        if response.status == 200:
                            # Update primary endpoint to backup
                            self.logger.info(f"Switching {component_id} to backup: {backup_endpoint}")
                            
                            # In real implementation, this would update load balancer
                            # or DNS records to point to backup
                            
                            return True
            except Exception as e:
                self.logger.warning(f"Backup endpoint {backup_endpoint} not available: {e}")
                continue
        
        return False
    
    async def _restart_service(self, component_id: str) -> bool:
        """Restart the service component"""
        try:
            # In real implementation, this would use systemctl, docker, or k8s APIs
            self.logger.info(f"Restarting service for {component_id}")
            
            # Placeholder for actual restart command
            # subprocess.run(['systemctl', 'restart', f'{component_id}.service'])
            
            # Wait for service to come back up
            await asyncio.sleep(30)
            
            # Verify service is healthy
            health = await self.check_component_health(component_id)
            return health.status == ComponentStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Failed to restart service {component_id}: {e}")
            return False
    
    async def _switch_datacenter(self, component_id: str) -> bool:
        """Switch component to different datacenter"""
        try:
            self.logger.info(f"Switching {component_id} to backup datacenter")
            
            # In real implementation, this would:
            # 1. Update DNS records
            # 2. Migrate data if necessary
            # 3. Update load balancer configuration
            # 4. Redirect traffic
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch datacenter for {component_id}: {e}")
            return False
    
    async def _scale_resources(self, component_id: str) -> bool:
        """Scale resources for component"""
        try:
            self.logger.info(f"Scaling resources for {component_id}")
            
            # In real implementation, this would:
            # 1. Increase CPU/memory allocation
            # 2. Add more instances
            # 3. Scale database connections
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to scale resources for {component_id}: {e}")
            return False
    
    async def _isolate_component(self, component_id: str) -> bool:
        """Isolate failed component"""
        try:
            self.logger.info(f"Isolating component {component_id}")
            
            # Remove from load balancer
            # Stop receiving new requests
            # Gracefully finish existing requests
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to isolate component {component_id}: {e}")
            return False
    
    async def _full_failover(self, component_id: str) -> bool:
        """Execute full failover for component"""
        try:
            self.logger.critical(f"Executing full failover for {component_id}")
            
            # Execute multiple failover actions
            actions = [
                self._isolate_component(component_id),
                self._activate_standby(component_id),
                self._scale_resources(component_id)
            ]
            
            results = await asyncio.gather(*actions, return_exceptions=True)
            return all(results)
            
        except Exception as e:
            self.logger.error(f"Failed to execute full failover for {component_id}: {e}")
            return False
    
    async def _monitor_recovery(self, component_id: str) -> None:
        """Monitor component recovery after failover"""
        config = self.components[component_id]
        start_time = datetime.now()
        timeout = timedelta(seconds=config.recovery_timeout)
        
        while datetime.now() - start_time < timeout:
            try:
                health = await self.check_component_health(component_id)
                
                if health.status == ComponentStatus.HEALTHY:
                    self.logger.info(f"Component {component_id} recovery successful")
                    
                    # Remove from active failovers
                    if component_id in self.active_failovers:
                        del self.active_failovers[component_id]
                    
                    await self.send_notification(
                        component_id,
                        f"✅ Component {component_id} has recovered",
                        "success"
                    )
                    return
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Recovery monitoring error for {component_id}: {e}")
                await asyncio.sleep(30)
        
        # Recovery timeout
        self.logger.error(f"Recovery timeout for {component_id}")
        await self.send_notification(
            component_id,
            f"⚠️ Recovery timeout for {component_id}",
            "warning"
        )
    
    async def send_notification(self, 
                               component_id: str, 
                               message: str,
                               severity: str = "info") -> None:
        """Send notifications through configured channels"""
        config = self.components.get(component_id)
        if not config or not config.notification_channels:
            return
        
        notification_data = {
            'component_id': component_id,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        for channel in config.notification_channels:
            try:
                if channel == "slack":
                    await self._send_slack_notification(notification_data)
                elif channel == "email":
                    await self._send_email_notification(notification_data)
                elif channel == "pagerduty":
                    await self._send_pagerduty_notification(notification_data)
                
            except Exception as e:
                self.logger.error(f"Failed to send {channel} notification: {e}")
    
    async def _send_slack_notification(self, data: Dict) -> None:
        """Send Slack notification"""
        # Placeholder for Slack integration
        self.logger.info(f"Slack notification: {data['message']}")
    
    async def _send_email_notification(self, data: Dict) -> None:
        """Send email notification"""
        # Placeholder for email integration
        self.logger.info(f"Email notification: {data['message']}")
    
    async def _send_pagerduty_notification(self, data: Dict) -> None:
        """Send PagerDuty notification"""
        # Placeholder for PagerDuty integration
        self.logger.info(f"PagerDuty notification: {data['message']}")
    
    async def send_alert(self, component_id: str, message: str) -> None:
        """Send alert for component issue"""
        await self.send_notification(component_id, message, "warning")
    
    def get_failover_status(self) -> Dict:
        """Get current failover status"""
        return {
            'active_failovers': self.active_failovers,
            'component_health': {
                comp_id: asdict(health)
                for comp_id, health in self.component_health.items()
            },
            'recent_history': self.failover_history[-10:]  # Last 10 events
        }
    
    async def manual_failover(self, 
                             component_id: str, 
                             action: FailoverAction) -> bool:
        """Manually trigger failover for a component"""
        return await self.initiate_failover(component_id, action, manual=True)
    
    def stop_monitoring(self) -> None:
        """Stop component monitoring"""
        self.monitoring_active = False

if __name__ == "__main__":
    # Example usage
    async def main():
        failover_manager = TradingSystemFailoverManager()
        
        # Start monitoring
        monitor_task = asyncio.create_task(failover_manager.monitor_all_components())
        
        # Wait for a bit to see monitoring in action
        await asyncio.sleep(60)
        
        # Stop monitoring
        failover_manager.stop_monitoring()
        await monitor_task
    
    # asyncio.run(main())