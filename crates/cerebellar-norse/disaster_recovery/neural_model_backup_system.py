#!/usr/bin/env python3
"""
Neural Model Backup System
Automated backup and recovery for neural network models, weights, and configurations
"""

import os
import json
import shutil
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle
import torch
import numpy as np

@dataclass
class BackupMetadata:
    """Metadata for neural model backups"""
    model_id: str
    version: str
    timestamp: datetime
    file_size: int
    checksum: str
    model_type: str
    performance_metrics: Dict
    dependencies: List[str]
    backup_location: str
    compression_ratio: float = 1.0

class NeuralModelBackupSystem:
    """Comprehensive backup system for neural network models"""
    
    def __init__(self, 
                 primary_backup_path: str = "/backup/primary/neural_models",
                 secondary_backup_path: str = "/backup/secondary/neural_models",
                 remote_backup_path: str = "s3://neural-backups/models",
                 retention_days: int = 90):
        self.primary_backup_path = Path(primary_backup_path)
        self.secondary_backup_path = Path(secondary_backup_path)
        self.remote_backup_path = remote_backup_path
        self.retention_days = retention_days
        self.logger = self._setup_logging()
        
        # Create backup directories
        self.primary_backup_path.mkdir(parents=True, exist_ok=True)
        self.secondary_backup_path.mkdir(parents=True, exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup backup system logging"""
        logger = logging.getLogger("neural_backup")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("disaster_recovery/logs/neural_backup.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    async def backup_model(self, 
                          model_path: str, 
                          model_id: str,
                          model_type: str = "pytorch",
                          performance_metrics: Optional[Dict] = None) -> BackupMetadata:
        """Backup a neural network model with metadata"""
        try:
            source_path = Path(model_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Generate backup metadata
            timestamp = datetime.now()
            version = f"v{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Create backup filename
            backup_filename = f"{model_id}_{version}.{source_path.suffix}"
            primary_backup_file = self.primary_backup_path / backup_filename
            secondary_backup_file = self.secondary_backup_path / backup_filename
            
            # Copy model to primary backup location
            shutil.copy2(source_path, primary_backup_file)
            
            # Calculate file metrics
            file_size = primary_backup_file.stat().st_size
            checksum = self.calculate_checksum(primary_backup_file)
            
            # Copy to secondary backup location
            shutil.copy2(primary_backup_file, secondary_backup_file)
            
            # Create backup metadata
            metadata = BackupMetadata(
                model_id=model_id,
                version=version,
                timestamp=timestamp,
                file_size=file_size,
                checksum=checksum,
                model_type=model_type,
                performance_metrics=performance_metrics or {},
                dependencies=self._extract_dependencies(source_path),
                backup_location=str(primary_backup_file)
            )
            
            # Save metadata
            await self._save_metadata(metadata)
            
            # Schedule remote backup
            await self._schedule_remote_backup(primary_backup_file, metadata)
            
            self.logger.info(f"Successfully backed up model {model_id} version {version}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to backup model {model_id}: {e}")
            raise
    
    def _extract_dependencies(self, model_path: Path) -> List[str]:
        """Extract model dependencies and requirements"""
        dependencies = []
        
        # Check for requirements.txt in model directory
        req_file = model_path.parent / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                dependencies.extend([line.strip() for line in f.readlines()])
        
        # Check for conda environment file
        conda_file = model_path.parent / "environment.yml"
        if conda_file.exists():
            dependencies.append(f"conda_env:{conda_file}")
        
        return dependencies
    
    async def _save_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata to database and file"""
        metadata_file = self.primary_backup_path / f"{metadata.model_id}_{metadata.version}_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        # Also save to secondary location
        secondary_metadata_file = self.secondary_backup_path / f"{metadata.model_id}_{metadata.version}_metadata.json"
        shutil.copy2(metadata_file, secondary_metadata_file)
    
    async def _schedule_remote_backup(self, local_file: Path, metadata: BackupMetadata) -> None:
        """Schedule backup to remote storage (S3, Azure, etc.)"""
        try:
            # This would integrate with cloud storage APIs
            remote_path = f"{self.remote_backup_path}/{metadata.model_id}/{metadata.version}"
            
            # Placeholder for actual cloud upload
            self.logger.info(f"Scheduled remote backup to {remote_path}")
            
            # In real implementation, this would use boto3, azure-storage-blob, etc.
            # await upload_to_cloud(local_file, remote_path)
            
        except Exception as e:
            self.logger.error(f"Failed to schedule remote backup: {e}")
    
    async def restore_model(self, 
                           model_id: str, 
                           version: Optional[str] = None,
                           target_path: Optional[str] = None) -> Tuple[str, BackupMetadata]:
        """Restore a neural network model from backup"""
        try:
            # Find the backup to restore
            if version is None:
                # Get latest version
                metadata = await self._get_latest_backup_metadata(model_id)
            else:
                metadata = await self._get_backup_metadata(model_id, version)
            
            if not metadata:
                raise ValueError(f"No backup found for model {model_id}")
            
            # Verify backup integrity
            backup_file = Path(metadata.backup_location)
            if not backup_file.exists():
                # Try secondary backup
                backup_file = self.secondary_backup_path / backup_file.name
                if not backup_file.exists():
                    raise FileNotFoundError(f"Backup file not found: {metadata.backup_location}")
            
            # Verify checksum
            current_checksum = self.calculate_checksum(backup_file)
            if current_checksum != metadata.checksum:
                raise ValueError(f"Backup file corrupted: checksum mismatch")
            
            # Determine restore location
            if target_path is None:
                target_path = f"restored_models/{model_id}_{metadata.version}"
            
            target_file = Path(target_path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Restore the model
            shutil.copy2(backup_file, target_file)
            
            self.logger.info(f"Successfully restored model {model_id} version {metadata.version}")
            return str(target_file), metadata
            
        except Exception as e:
            self.logger.error(f"Failed to restore model {model_id}: {e}")
            raise
    
    async def _get_latest_backup_metadata(self, model_id: str) -> Optional[BackupMetadata]:
        """Get metadata for the latest backup of a model"""
        metadata_files = list(self.primary_backup_path.glob(f"{model_id}_*_metadata.json"))
        
        if not metadata_files:
            return None
        
        # Sort by timestamp to get latest
        latest_file = max(metadata_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return BackupMetadata(**data)
    
    async def _get_backup_metadata(self, model_id: str, version: str) -> Optional[BackupMetadata]:
        """Get metadata for a specific backup version"""
        metadata_file = self.primary_backup_path / f"{model_id}_{version}_metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return BackupMetadata(**data)
    
    async def cleanup_old_backups(self) -> int:
        """Clean up backups older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0
        
        try:
            # Clean primary backups
            for backup_file in self.primary_backup_path.glob("*"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    deleted_count += 1
            
            # Clean secondary backups
            for backup_file in self.secondary_backup_path.glob("*"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old backup files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")
            raise
    
    async def verify_backup_integrity(self) -> Dict[str, bool]:
        """Verify integrity of all backups"""
        integrity_results = {}
        
        try:
            metadata_files = list(self.primary_backup_path.glob("*_metadata.json"))
            
            for metadata_file in metadata_files:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    metadata = BackupMetadata(**data)
                
                backup_file = Path(metadata.backup_location)
                if backup_file.exists():
                    current_checksum = self.calculate_checksum(backup_file)
                    integrity_results[metadata.model_id] = (current_checksum == metadata.checksum)
                else:
                    integrity_results[metadata.model_id] = False
            
            return integrity_results
            
        except Exception as e:
            self.logger.error(f"Failed to verify backup integrity: {e}")
            raise

# Automated backup scheduler
class BackupScheduler:
    """Automated backup scheduling for neural models"""
    
    def __init__(self, backup_system: NeuralModelBackupSystem):
        self.backup_system = backup_system
        self.scheduled_models = {}
        self.running = False
    
    def schedule_model_backup(self, 
                             model_path: str, 
                             model_id: str,
                             interval_hours: int = 24,
                             model_type: str = "pytorch"):
        """Schedule regular backups for a model"""
        self.scheduled_models[model_id] = {
            'path': model_path,
            'interval': interval_hours,
            'type': model_type,
            'last_backup': None
        }
    
    async def start_scheduler(self):
        """Start the backup scheduler"""
        self.running = True
        
        while self.running:
            try:
                current_time = datetime.now()
                
                for model_id, config in self.scheduled_models.items():
                    last_backup = config.get('last_backup')
                    interval = timedelta(hours=config['interval'])
                    
                    if (last_backup is None or 
                        current_time - last_backup >= interval):
                        
                        # Perform backup
                        await self.backup_system.backup_model(
                            config['path'],
                            model_id,
                            config['type']
                        )
                        
                        config['last_backup'] = current_time
                
                # Wait before next check (check every hour)
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.backup_system.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(3600)  # Continue after error
    
    def stop_scheduler(self):
        """Stop the backup scheduler"""
        self.running = False

if __name__ == "__main__":
    # Example usage
    async def main():
        backup_system = NeuralModelBackupSystem()
        
        # Backup a model
        metadata = await backup_system.backup_model(
            "models/risk_prediction_model.pth",
            "risk_predictor",
            "pytorch",
            {"accuracy": 0.95, "f1_score": 0.92}
        )
        
        print(f"Backup completed: {metadata.model_id} v{metadata.version}")
        
        # Verify backup integrity
        integrity = await backup_system.verify_backup_integrity()
        print(f"Backup integrity: {integrity}")
        
        # Restore model
        restored_path, restored_metadata = await backup_system.restore_model("risk_predictor")
        print(f"Model restored to: {restored_path}")
    
    # asyncio.run(main())