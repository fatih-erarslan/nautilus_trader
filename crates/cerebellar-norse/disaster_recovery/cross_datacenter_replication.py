#!/usr/bin/env python3
"""
Cross-Datacenter Replication System
High availability through multi-datacenter data and model synchronization
"""

import asyncio
import logging
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import aiofiles
from pathlib import Path

class ReplicationStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"
    SYNCING = "syncing"
    DEGRADED = "degraded"

class DatacenterType(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    DISASTER_RECOVERY = "disaster_recovery"
    EDGE = "edge"

class ReplicationMode(Enum):
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    SEMI_SYNCHRONOUS = "semi_synchronous"

@dataclass
class DatacenterConfig:
    """Configuration for a datacenter"""
    datacenter_id: str
    name: str
    location: str
    type: DatacenterType
    endpoints: Dict[str, str]  # service_name -> endpoint_url
    priority: int
    bandwidth_mbps: int
    latency_ms: float
    available: bool = True

@dataclass
class ReplicationTarget:
    """Configuration for replication target"""
    target_id: str
    source_datacenter: str
    target_datacenter: str
    data_types: List[str]
    mode: ReplicationMode
    compression_enabled: bool
    encryption_enabled: bool
    batch_size: int
    sync_interval_seconds: int
    retry_attempts: int
    status: ReplicationStatus = ReplicationStatus.ACTIVE

@dataclass
class ReplicationMetrics:
    """Metrics for replication performance"""
    target_id: str
    bytes_replicated: int
    records_replicated: int
    replication_lag_seconds: float
    last_sync_time: datetime
    success_rate: float
    error_count: int
    bandwidth_utilization_mbps: float

class CrossDatacenterReplicationManager:
    """Manages cross-datacenter replication for high availability"""
    
    def __init__(self, config_file: str = "disaster_recovery/replication_config.json"):
        self.datacenters: Dict[str, DatacenterConfig] = {}
        self.replication_targets: Dict[str, ReplicationTarget] = {}
        self.replication_metrics: Dict[str, ReplicationMetrics] = {}
        self.logger = self._setup_logging()
        self.monitoring_active = False
        
        # Load configuration
        self._load_config(config_file)
        
        # Initialize replication state
        self._initialize_replication_state()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup replication logging"""
        logger = logging.getLogger("cross_datacenter_replication")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        Path("disaster_recovery/logs").mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler("disaster_recovery/logs/replication.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_config(self, config_file: str) -> None:
        """Load replication configuration"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                # Load datacenters
                for dc_id, dc_config in config_data.get('datacenters', {}).items():
                    dc_config['type'] = DatacenterType(dc_config['type'])
                    self.datacenters[dc_id] = DatacenterConfig(**dc_config)
                
                # Load replication targets
                for target_id, target_config in config_data.get('replication_targets', {}).items():
                    target_config['mode'] = ReplicationMode(target_config['mode'])
                    target_config['status'] = ReplicationStatus(target_config.get('status', 'active'))
                    self.replication_targets[target_id] = ReplicationTarget(**target_config)
            else:
                self._create_default_config(config_file)
                
        except Exception as e:
            self.logger.error(f"Failed to load replication config: {e}")
            self._create_default_config(config_file)
    
    def _create_default_config(self, config_file: str) -> None:
        """Create default replication configuration"""
        default_config = {
            "datacenters": {
                "dc1_primary": {
                    "datacenter_id": "dc1_primary",
                    "name": "Primary Datacenter US-East",
                    "location": "Virginia, USA",
                    "type": "primary",
                    "endpoints": {
                        "trading_engine": "https://dc1.trading.company.com",
                        "risk_manager": "https://dc1.risk.company.com",
                        "neural_engine": "https://dc1.ml.company.com",
                        "data_storage": "https://dc1.data.company.com"
                    },
                    "priority": 1,
                    "bandwidth_mbps": 10000,
                    "latency_ms": 1.0,
                    "available": True
                },
                "dc2_secondary": {
                    "datacenter_id": "dc2_secondary",
                    "name": "Secondary Datacenter US-West",
                    "location": "California, USA",
                    "type": "secondary",
                    "endpoints": {
                        "trading_engine": "https://dc2.trading.company.com",
                        "risk_manager": "https://dc2.risk.company.com",
                        "neural_engine": "https://dc2.ml.company.com",
                        "data_storage": "https://dc2.data.company.com"
                    },
                    "priority": 2,
                    "bandwidth_mbps": 10000,
                    "latency_ms": 45.0,
                    "available": True
                },
                "dc3_dr": {
                    "datacenter_id": "dc3_dr",
                    "name": "Disaster Recovery EU",
                    "location": "Ireland, EU",
                    "type": "disaster_recovery",
                    "endpoints": {
                        "trading_engine": "https://dc3.trading.company.com",
                        "risk_manager": "https://dc3.risk.company.com",
                        "neural_engine": "https://dc3.ml.company.com",
                        "data_storage": "https://dc3.data.company.com"
                    },
                    "priority": 3,
                    "bandwidth_mbps": 5000,
                    "latency_ms": 120.0,
                    "available": True
                }
            },
            "replication_targets": {
                "neural_models_primary_to_secondary": {
                    "target_id": "neural_models_primary_to_secondary",
                    "source_datacenter": "dc1_primary",
                    "target_datacenter": "dc2_secondary",
                    "data_types": ["neural_models", "model_weights", "training_data"],
                    "mode": "asynchronous",
                    "compression_enabled": True,
                    "encryption_enabled": True,
                    "batch_size": 1000,
                    "sync_interval_seconds": 300,
                    "retry_attempts": 3,
                    "status": "active"
                },
                "trading_data_primary_to_secondary": {
                    "target_id": "trading_data_primary_to_secondary",
                    "source_datacenter": "dc1_primary",
                    "target_datacenter": "dc2_secondary",
                    "data_types": ["orders", "positions", "risk_metrics"],
                    "mode": "synchronous",
                    "compression_enabled": False,
                    "encryption_enabled": True,
                    "batch_size": 100,
                    "sync_interval_seconds": 1,
                    "retry_attempts": 5,
                    "status": "active"
                },
                "full_backup_to_dr": {
                    "target_id": "full_backup_to_dr",
                    "source_datacenter": "dc1_primary",
                    "target_datacenter": "dc3_dr",
                    "data_types": ["all_data", "neural_models", "configurations"],
                    "mode": "asynchronous",
                    "compression_enabled": True,
                    "encryption_enabled": True,
                    "batch_size": 5000,
                    "sync_interval_seconds": 3600,
                    "retry_attempts": 3,
                    "status": "active"
                }
            }
        }
        
        # Save default configuration
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        # Load the default configuration
        self._load_config(config_file)
    
    def _initialize_replication_state(self) -> None:
        """Initialize replication state and metrics"""
        for target_id in self.replication_targets:
            self.replication_metrics[target_id] = ReplicationMetrics(
                target_id=target_id,
                bytes_replicated=0,
                records_replicated=0,
                replication_lag_seconds=0.0,
                last_sync_time=datetime.now(),
                success_rate=1.0,
                error_count=0,
                bandwidth_utilization_mbps=0.0
            )
    
    async def start_replication_monitoring(self) -> None:
        """Start monitoring and managing all replication targets"""
        self.monitoring_active = True
        
        # Create replication tasks for each target
        replication_tasks = []
        for target_id, target_config in self.replication_targets.items():
            if target_config.status == ReplicationStatus.ACTIVE:
                task = asyncio.create_task(
                    self._manage_replication_target(target_id)
                )
                replication_tasks.append(task)
        
        # Create monitoring task
        monitor_task = asyncio.create_task(self._monitor_replication_health())
        
        try:
            # Wait for all tasks
            await asyncio.gather(*replication_tasks, monitor_task)
        except Exception as e:
            self.logger.error(f"Replication monitoring error: {e}")
        finally:
            self.monitoring_active = False
    
    async def _manage_replication_target(self, target_id: str) -> None:
        """Manage replication for a specific target"""
        target_config = self.replication_targets[target_id]
        
        while self.monitoring_active and target_config.status == ReplicationStatus.ACTIVE:
            try:
                start_time = time.time()
                
                # Check datacenter availability
                if not await self._check_datacenter_availability(target_config.source_datacenter):
                    self.logger.warning(f"Source datacenter {target_config.source_datacenter} unavailable")
                    await asyncio.sleep(target_config.sync_interval_seconds)
                    continue
                
                if not await self._check_datacenter_availability(target_config.target_datacenter):
                    self.logger.warning(f"Target datacenter {target_config.target_datacenter} unavailable")
                    await asyncio.sleep(target_config.sync_interval_seconds)
                    continue
                
                # Perform replication based on mode
                if target_config.mode == ReplicationMode.SYNCHRONOUS:
                    await self._synchronous_replication(target_id)
                elif target_config.mode == ReplicationMode.ASYNCHRONOUS:
                    await self._asynchronous_replication(target_id)
                elif target_config.mode == ReplicationMode.SEMI_SYNCHRONOUS:
                    await self._semi_synchronous_replication(target_id)
                
                # Update metrics
                self._update_replication_metrics(target_id, start_time, success=True)
                
                # Wait for next sync interval
                await asyncio.sleep(target_config.sync_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Replication error for {target_id}: {e}")
                self._update_replication_metrics(target_id, start_time, success=False)
                
                # Exponential backoff on failure
                await asyncio.sleep(min(target_config.sync_interval_seconds * 2, 300))
    
    async def _check_datacenter_availability(self, datacenter_id: str) -> bool:
        """Check if a datacenter is available"""
        if datacenter_id not in self.datacenters:
            return False
        
        datacenter = self.datacenters[datacenter_id]
        if not datacenter.available:
            return False
        
        try:
            # Check health endpoint
            health_endpoint = datacenter.endpoints.get('health', f"{datacenter.endpoints.get('data_storage', '')}/health")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(health_endpoint) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _synchronous_replication(self, target_id: str) -> None:
        """Perform synchronous replication"""
        target_config = self.replication_targets[target_id]
        
        try:
            # Get data changes since last sync
            changes = await self._get_data_changes(target_id)
            
            if not changes:
                return  # No changes to replicate
            
            # Replicate data synchronously
            for change in changes:
                success = await self._replicate_data_item(target_config, change)
                if not success:
                    raise Exception(f"Failed to replicate change: {change['id']}")
            
            # Update last sync time
            self.replication_metrics[target_id].last_sync_time = datetime.now()
            
            self.logger.debug(f"Synchronous replication completed for {target_id}: {len(changes)} changes")
            
        except Exception as e:
            self.logger.error(f"Synchronous replication failed for {target_id}: {e}")
            raise
    
    async def _asynchronous_replication(self, target_id: str) -> None:
        """Perform asynchronous replication"""
        target_config = self.replication_targets[target_id]
        
        try:
            # Get data changes since last sync
            changes = await self._get_data_changes(target_id)
            
            if not changes:
                return  # No changes to replicate
            
            # Process changes in batches
            batch_size = target_config.batch_size
            for i in range(0, len(changes), batch_size):
                batch = changes[i:i + batch_size]
                
                # Replicate batch asynchronously
                replication_tasks = [
                    self._replicate_data_item(target_config, change)
                    for change in batch
                ]
                
                results = await asyncio.gather(*replication_tasks, return_exceptions=True)
                
                # Count successes and failures
                successes = sum(1 for result in results if result is True)
                failures = len(results) - successes
                
                if failures > 0:
                    self.logger.warning(f"Batch replication for {target_id}: {successes} successes, {failures} failures")
            
            # Update last sync time
            self.replication_metrics[target_id].last_sync_time = datetime.now()
            
            self.logger.debug(f"Asynchronous replication completed for {target_id}: {len(changes)} changes")
            
        except Exception as e:
            self.logger.error(f"Asynchronous replication failed for {target_id}: {e}")
            raise
    
    async def _semi_synchronous_replication(self, target_id: str) -> None:
        """Perform semi-synchronous replication"""
        target_config = self.replication_targets[target_id]
        
        try:
            # Get data changes since last sync
            changes = await self._get_data_changes(target_id)
            
            if not changes:
                return  # No changes to replicate
            
            # Separate critical and non-critical changes
            critical_changes = [c for c in changes if c.get('critical', False)]
            normal_changes = [c for c in changes if not c.get('critical', False)]
            
            # Replicate critical changes synchronously
            for change in critical_changes:
                success = await self._replicate_data_item(target_config, change)
                if not success:
                    self.logger.error(f"Failed to replicate critical change: {change['id']}")
            
            # Replicate normal changes asynchronously
            if normal_changes:
                replication_tasks = [
                    self._replicate_data_item(target_config, change)
                    for change in normal_changes
                ]
                await asyncio.gather(*replication_tasks, return_exceptions=True)
            
            # Update last sync time
            self.replication_metrics[target_id].last_sync_time = datetime.now()
            
            self.logger.debug(f"Semi-synchronous replication completed for {target_id}: {len(changes)} changes")
            
        except Exception as e:
            self.logger.error(f"Semi-synchronous replication failed for {target_id}: {e}")
            raise
    
    async def _get_data_changes(self, target_id: str) -> List[Dict]:
        """Get data changes since last sync for a replication target"""
        target_config = self.replication_targets[target_id]
        metrics = self.replication_metrics[target_id]
        
        try:
            source_datacenter = self.datacenters[target_config.source_datacenter]
            data_endpoint = source_datacenter.endpoints.get('data_storage')
            
            if not data_endpoint:
                return []
            
            # Query for changes since last sync
            params = {
                'since': metrics.last_sync_time.isoformat(),
                'data_types': ','.join(target_config.data_types),
                'limit': target_config.batch_size
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{data_endpoint}/changes", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('changes', [])
                    else:
                        self.logger.warning(f"Failed to get changes for {target_id}: HTTP {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Failed to get data changes for {target_id}: {e}")
            return []
    
    async def _replicate_data_item(self, target_config: ReplicationTarget, change: Dict) -> bool:
        """Replicate a single data item to the target datacenter"""
        try:
            target_datacenter = self.datacenters[target_config.target_datacenter]
            data_endpoint = target_datacenter.endpoints.get('data_storage')
            
            if not data_endpoint:
                return False
            
            # Prepare data for replication
            data = change.copy()
            
            # Apply compression if enabled
            if target_config.compression_enabled:
                data = await self._compress_data(data)
            
            # Apply encryption if enabled
            if target_config.encryption_enabled:
                data = await self._encrypt_data(data)
            
            # Send data to target
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{data_endpoint}/replicate", json=data) as response:
                    if response.status in [200, 201]:
                        return True
                    else:
                        self.logger.warning(f"Replication failed: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to replicate data item {change.get('id', 'unknown')}: {e}")
            return False
    
    async def _compress_data(self, data: Dict) -> Dict:
        """Compress data for transmission"""
        # Placeholder for compression logic
        # In real implementation, this would use gzip, lz4, or zstd
        data['_compressed'] = True
        return data
    
    async def _encrypt_data(self, data: Dict) -> Dict:
        """Encrypt data for transmission"""
        # Placeholder for encryption logic
        # In real implementation, this would use AES or similar
        data['_encrypted'] = True
        return data
    
    def _update_replication_metrics(self, target_id: str, start_time: float, success: bool) -> None:
        """Update replication metrics"""
        metrics = self.replication_metrics[target_id]
        
        # Update timing
        duration = time.time() - start_time
        
        if success:
            # Update success metrics
            metrics.success_rate = (metrics.success_rate * 0.9) + (1.0 * 0.1)  # Exponential moving average
        else:
            # Update failure metrics
            metrics.error_count += 1
            metrics.success_rate = (metrics.success_rate * 0.9) + (0.0 * 0.1)
        
        # Update lag calculation
        if success:
            metrics.replication_lag_seconds = duration
    
    async def _monitor_replication_health(self) -> None:
        """Monitor overall replication health"""
        while self.monitoring_active:
            try:
                # Check each replication target
                for target_id, target_config in self.replication_targets.items():
                    metrics = self.replication_metrics[target_id]
                    
                    # Check for stale replication
                    time_since_sync = (datetime.now() - metrics.last_sync_time).total_seconds()
                    max_lag = target_config.sync_interval_seconds * 3  # 3x the normal interval
                    
                    if time_since_sync > max_lag:
                        self.logger.warning(f"Replication lag detected for {target_id}: {time_since_sync}s")
                        target_config.status = ReplicationStatus.DEGRADED
                    
                    # Check success rate
                    if metrics.success_rate < 0.8:  # Less than 80% success rate
                        self.logger.warning(f"Low success rate for {target_id}: {metrics.success_rate:.2%}")
                        target_config.status = ReplicationStatus.DEGRADED
                    
                    # Check error count
                    if metrics.error_count > 10:  # More than 10 consecutive errors
                        self.logger.error(f"High error count for {target_id}: {metrics.error_count}")
                        target_config.status = ReplicationStatus.FAILED
                
                # Wait before next health check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def trigger_manual_sync(self, target_id: str) -> bool:
        """Manually trigger synchronization for a target"""
        try:
            if target_id not in self.replication_targets:
                raise ValueError(f"Unknown replication target: {target_id}")
            
            target_config = self.replication_targets[target_id]
            
            self.logger.info(f"Manual sync triggered for {target_id}")
            
            # Execute replication based on mode
            if target_config.mode == ReplicationMode.SYNCHRONOUS:
                await self._synchronous_replication(target_id)
            elif target_config.mode == ReplicationMode.ASYNCHRONOUS:
                await self._asynchronous_replication(target_id)
            elif target_config.mode == ReplicationMode.SEMI_SYNCHRONOUS:
                await self._semi_synchronous_replication(target_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Manual sync failed for {target_id}: {e}")
            return False
    
    async def failover_to_datacenter(self, target_datacenter_id: str) -> bool:
        """Failover operations to a different datacenter"""
        try:
            if target_datacenter_id not in self.datacenters:
                raise ValueError(f"Unknown datacenter: {target_datacenter_id}")
            
            target_datacenter = self.datacenters[target_datacenter_id]
            
            self.logger.critical(f"Initiating failover to datacenter: {target_datacenter.name}")
            
            # Update datacenter priorities
            # Set target as primary
            target_datacenter.type = DatacenterType.PRIMARY
            target_datacenter.priority = 1
            
            # Demote other datacenters
            for dc_id, datacenter in self.datacenters.items():
                if dc_id != target_datacenter_id and datacenter.type == DatacenterType.PRIMARY:
                    datacenter.type = DatacenterType.SECONDARY
                    datacenter.priority += 1
            
            # Reconfigure replication targets
            await self._reconfigure_replication_for_failover(target_datacenter_id)
            
            self.logger.info(f"Failover to {target_datacenter.name} completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failover to {target_datacenter_id} failed: {e}")
            return False
    
    async def _reconfigure_replication_for_failover(self, new_primary_id: str) -> None:
        """Reconfigure replication targets after failover"""
        try:
            # Update replication targets to use new primary as source
            for target_id, target_config in self.replication_targets.items():
                if target_config.source_datacenter != new_primary_id:
                    # Update source to new primary
                    old_source = target_config.source_datacenter
                    target_config.source_datacenter = new_primary_id
                    
                    self.logger.info(f"Updated replication source for {target_id}: {old_source} -> {new_primary_id}")
            
            # Restart replication with new configuration
            await self._restart_replication_targets()
            
        except Exception as e:
            self.logger.error(f"Failed to reconfigure replication for failover: {e}")
    
    async def _restart_replication_targets(self) -> None:
        """Restart all replication targets"""
        # In real implementation, this would stop and restart replication tasks
        self.logger.info("Replication targets reconfigured for failover")
    
    def get_replication_status(self) -> Dict:
        """Get current replication status"""
        status = {
            'datacenters': {
                dc_id: asdict(datacenter)
                for dc_id, datacenter in self.datacenters.items()
            },
            'replication_targets': {
                target_id: asdict(target)
                for target_id, target in self.replication_targets.items()
            },
            'metrics': {
                target_id: asdict(metrics)
                for target_id, metrics in self.replication_metrics.items()
            },
            'overall_health': self._calculate_overall_health()
        }
        
        return status
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall replication health"""
        if not self.replication_metrics:
            return "unknown"
        
        # Calculate average success rate
        success_rates = [m.success_rate for m in self.replication_metrics.values()]
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        # Check for failed targets
        failed_targets = [
            t for t in self.replication_targets.values()
            if t.status == ReplicationStatus.FAILED
        ]
        
        if failed_targets:
            return "critical"
        elif avg_success_rate < 0.9:
            return "degraded"
        elif avg_success_rate < 0.95:
            return "warning"
        else:
            return "healthy"
    
    def stop_replication(self) -> None:
        """Stop all replication monitoring"""
        self.monitoring_active = False

if __name__ == "__main__":
    # Example usage
    async def main():
        replication_manager = CrossDatacenterReplicationManager()
        
        # Start replication monitoring
        monitor_task = asyncio.create_task(replication_manager.start_replication_monitoring())
        
        # Simulate some operations
        await asyncio.sleep(30)
        
        # Trigger manual sync
        await replication_manager.trigger_manual_sync("neural_models_primary_to_secondary")
        
        # Check status
        status = replication_manager.get_replication_status()
        print(f"Replication status: {status['overall_health']}")
        
        # Simulate failover
        await replication_manager.failover_to_datacenter("dc2_secondary")
        
        # Stop replication
        replication_manager.stop_replication()
        await monitor_task
    
    # asyncio.run(main())