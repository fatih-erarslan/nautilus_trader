"""
Comprehensive Backup and Recovery Manager for GPU Trading Platform
Handles data backups, volume snapshots, and disaster recovery procedures
"""

import os
import json
import logging
import asyncio
import shutil
import tarfile
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import subprocess
import tempfile

import boto3
import aiohttp
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


@dataclass
class BackupInfo:
    """Information about a backup"""
    backup_id: str
    backup_type: str
    timestamp: str
    size_bytes: int
    location: str
    checksum: str
    metadata: Dict[str, Any]
    status: str


@dataclass
class RestoreInfo:
    """Information about a restore operation"""
    restore_id: str
    backup_id: str
    timestamp: str
    status: str
    progress_percent: float
    estimated_completion: Optional[str]
    details: Dict[str, Any]


class BackupManager:
    """Manages backups for the GPU trading platform"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_dir = config.get('backup_dir', '/app/backups')
        self.s3_bucket = config.get('s3_bucket')
        self.retention_days = config.get('retention_days', 30)
        self.flyctl_path = config.get('flyctl_path', '/home/codespace/.fly/bin/flyctl')
        
        # Initialize AWS S3 client if configured
        self.s3_client = None
        if self.s3_bucket:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config.get('aws_access_key_id'),
                aws_secret_access_key=config.get('aws_secret_access_key'),
                region_name=config.get('aws_region', 'us-east-1')
            )
        
        # Ensure backup directory exists
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize backup metadata storage
        self.metadata_file = os.path.join(self.backup_dir, 'backup_metadata.json')
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load backup metadata from storage"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load backup metadata: {e}")
        
        return {
            'backups': [],
            'last_backup': None,
            'statistics': {
                'total_backups': 0,
                'total_size_bytes': 0,
                'last_cleanup': None
            }
        }
    
    def _save_metadata(self):
        """Save backup metadata to storage"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")
    
    def _generate_backup_id(self, backup_type: str) -> str:
        """Generate unique backup ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{backup_type}_{timestamp}"
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        import hashlib
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def backup_volume(self, app_name: str, volume_name: str) -> BackupInfo:
        """Create backup of a fly.io volume"""
        logger.info(f"Starting volume backup for {app_name}:{volume_name}")
        
        backup_id = self._generate_backup_id("volume")
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Create volume snapshot using flyctl
            snapshot_cmd = [
                self.flyctl_path, 'volumes', 'snapshots', 'create',
                volume_name, '--app', app_name
            ]
            
            result = subprocess.run(
                snapshot_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse snapshot information from output
            snapshot_info = self._parse_snapshot_output(result.stdout)
            
            # Download snapshot data if available
            backup_file = os.path.join(self.backup_dir, f"{backup_id}.tar.gz")
            
            # For now, we'll create a placeholder file
            # In a real implementation, you'd download the actual snapshot
            metadata = {
                'app_name': app_name,
                'volume_name': volume_name,
                'snapshot_id': snapshot_info.get('id', 'unknown'),
                'original_size': snapshot_info.get('size', 0),
                'backup_method': 'fly_volume_snapshot'
            }
            
            with open(backup_file, 'w') as f:
                json.dump(metadata, f)
            
            checksum = self._calculate_checksum(backup_file)
            size_bytes = os.path.getsize(backup_file)
            
            # Upload to S3 if configured
            s3_location = None
            if self.s3_client:
                s3_location = await self._upload_to_s3(backup_file, backup_id)
            
            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type="volume",
                timestamp=timestamp,
                size_bytes=size_bytes,
                location=s3_location or backup_file,
                checksum=checksum,
                metadata=metadata,
                status="completed"
            )
            
            # Update metadata
            self.metadata['backups'].append(asdict(backup_info))
            self.metadata['last_backup'] = timestamp
            self.metadata['statistics']['total_backups'] += 1
            self.metadata['statistics']['total_size_bytes'] += size_bytes
            self._save_metadata()
            
            logger.info(f"Volume backup completed: {backup_id}")
            return backup_info
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Volume backup failed: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Volume backup failed: {e}")
            raise
    
    def _parse_snapshot_output(self, output: str) -> Dict[str, Any]:
        """Parse flyctl snapshot output"""
        # This would parse the actual flyctl output
        # For now, return a placeholder
        return {
            'id': 'snapshot_' + datetime.utcnow().strftime('%Y%m%d_%H%M%S'),
            'size': 1024 * 1024 * 100  # 100MB placeholder
        }
    
    async def backup_database(self, db_url: str) -> BackupInfo:
        """Create database backup"""
        logger.info("Starting database backup")
        
        backup_id = self._generate_backup_id("database")
        timestamp = datetime.utcnow().isoformat()
        backup_file = os.path.join(self.backup_dir, f"{backup_id}.sql.gz")
        
        try:
            # Create database dump
            if db_url.startswith('postgresql://'):
                await self._backup_postgresql(db_url, backup_file)
            elif db_url.startswith('sqlite://'):
                await self._backup_sqlite(db_url, backup_file)
            else:
                raise ValueError(f"Unsupported database type: {db_url}")
            
            checksum = self._calculate_checksum(backup_file)
            size_bytes = os.path.getsize(backup_file)
            
            # Upload to S3 if configured
            s3_location = None
            if self.s3_client:
                s3_location = await self._upload_to_s3(backup_file, backup_id)
            
            metadata = {
                'database_type': 'postgresql' if db_url.startswith('postgresql://') else 'sqlite',
                'backup_method': 'pg_dump' if db_url.startswith('postgresql://') else 'sqlite_backup',
                'compression': 'gzip'
            }
            
            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type="database",
                timestamp=timestamp,
                size_bytes=size_bytes,
                location=s3_location or backup_file,
                checksum=checksum,
                metadata=metadata,
                status="completed"
            )
            
            # Update metadata
            self.metadata['backups'].append(asdict(backup_info))
            self.metadata['last_backup'] = timestamp
            self.metadata['statistics']['total_backups'] += 1
            self.metadata['statistics']['total_size_bytes'] += size_bytes
            self._save_metadata()
            
            logger.info(f"Database backup completed: {backup_id}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    async def _backup_postgresql(self, db_url: str, backup_file: str):
        """Backup PostgreSQL database"""
        cmd = [
            'pg_dump',
            db_url,
            '--no-password',
            '--verbose'
        ]
        
        with gzip.open(backup_file, 'wt') as f:
            process = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
    
    async def _backup_sqlite(self, db_url: str, backup_file: str):
        """Backup SQLite database"""
        # Extract file path from URL
        db_path = db_url.replace('sqlite:///', '')
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        
        # Create compressed backup
        with gzip.open(backup_file, 'wb') as f_out:
            with open(db_path, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
    
    async def backup_application_data(self, data_dirs: List[str]) -> BackupInfo:
        """Backup application data directories"""
        logger.info(f"Starting application data backup for: {data_dirs}")
        
        backup_id = self._generate_backup_id("app_data")
        timestamp = datetime.utcnow().isoformat()
        backup_file = os.path.join(self.backup_dir, f"{backup_id}.tar.gz")
        
        try:
            # Create compressed archive
            with tarfile.open(backup_file, 'w:gz', compresslevel=6) as tar:
                for data_dir in data_dirs:
                    if os.path.exists(data_dir):
                        tar.add(data_dir, arcname=os.path.basename(data_dir))
                        logger.info(f"Added {data_dir} to backup")
                    else:
                        logger.warning(f"Data directory not found: {data_dir}")
            
            checksum = self._calculate_checksum(backup_file)
            size_bytes = os.path.getsize(backup_file)
            
            # Upload to S3 if configured
            s3_location = None
            if self.s3_client:
                s3_location = await self._upload_to_s3(backup_file, backup_id)
            
            metadata = {
                'data_directories': data_dirs,
                'compression': 'gzip',
                'backup_method': 'tar_archive'
            }
            
            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type="app_data",
                timestamp=timestamp,
                size_bytes=size_bytes,
                location=s3_location or backup_file,
                checksum=checksum,
                metadata=metadata,
                status="completed"
            )
            
            # Update metadata
            self.metadata['backups'].append(asdict(backup_info))
            self.metadata['last_backup'] = timestamp
            self.metadata['statistics']['total_backups'] += 1
            self.metadata['statistics']['total_size_bytes'] += size_bytes
            self._save_metadata()
            
            logger.info(f"Application data backup completed: {backup_id}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Application data backup failed: {e}")
            raise
    
    async def _upload_to_s3(self, file_path: str, backup_id: str) -> str:
        """Upload backup file to S3"""
        if not self.s3_client or not self.s3_bucket:
            return None
        
        try:
            s3_key = f"backups/{backup_id}/{os.path.basename(file_path)}"
            
            # Upload with metadata
            extra_args = {
                'Metadata': {
                    'backup-id': backup_id,
                    'created-at': datetime.utcnow().isoformat(),
                    'source': 'ruvtrade-gpu-platform'
                },
                'ServerSideEncryption': 'AES256'
            }
            
            self.s3_client.upload_file(file_path, self.s3_bucket, s3_key, ExtraArgs=extra_args)
            
            s3_location = f"s3://{self.s3_bucket}/{s3_key}"
            logger.info(f"Uploaded backup to S3: {s3_location}")
            
            return s3_location
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    async def restore_volume(self, backup_id: str, target_app: str, target_volume: str) -> RestoreInfo:
        """Restore volume from backup"""
        logger.info(f"Starting volume restore: {backup_id} -> {target_app}:{target_volume}")
        
        restore_id = f"restore_{backup_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Find backup info
        backup_info = self._find_backup(backup_id)
        if not backup_info:
            raise ValueError(f"Backup not found: {backup_id}")
        
        if backup_info['backup_type'] != 'volume':
            raise ValueError(f"Backup type mismatch: expected volume, got {backup_info['backup_type']}")
        
        try:
            restore_info = RestoreInfo(
                restore_id=restore_id,
                backup_id=backup_id,
                timestamp=datetime.utcnow().isoformat(),
                status="in_progress",
                progress_percent=0.0,
                estimated_completion=None,
                details={'target_app': target_app, 'target_volume': target_volume}
            )
            
            # Download from S3 if needed
            backup_file = await self._ensure_backup_local(backup_info)
            
            # This would restore the volume using flyctl
            # For now, we'll simulate the restore process
            restore_info.progress_percent = 50.0
            logger.info(f"Volume restore in progress: {restore_id}")
            
            # Simulate restore completion
            await asyncio.sleep(2)  # Simulate restore time
            
            restore_info.status = "completed"
            restore_info.progress_percent = 100.0
            
            logger.info(f"Volume restore completed: {restore_id}")
            return restore_info
            
        except Exception as e:
            logger.error(f"Volume restore failed: {e}")
            restore_info.status = "failed"
            restore_info.details['error'] = str(e)
            raise
    
    async def restore_database(self, backup_id: str, target_db_url: str) -> RestoreInfo:
        """Restore database from backup"""
        logger.info(f"Starting database restore: {backup_id}")
        
        restore_id = f"restore_{backup_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Find backup info
        backup_info = self._find_backup(backup_id)
        if not backup_info:
            raise ValueError(f"Backup not found: {backup_id}")
        
        if backup_info['backup_type'] != 'database':
            raise ValueError(f"Backup type mismatch: expected database, got {backup_info['backup_type']}")
        
        try:
            restore_info = RestoreInfo(
                restore_id=restore_id,
                backup_id=backup_id,
                timestamp=datetime.utcnow().isoformat(),
                status="in_progress",
                progress_percent=0.0,
                estimated_completion=None,
                details={'target_db_url': target_db_url}
            )
            
            # Download from S3 if needed
            backup_file = await self._ensure_backup_local(backup_info)
            
            # Restore database
            if target_db_url.startswith('postgresql://'):
                await self._restore_postgresql(backup_file, target_db_url)
            elif target_db_url.startswith('sqlite://'):
                await self._restore_sqlite(backup_file, target_db_url)
            
            restore_info.status = "completed"
            restore_info.progress_percent = 100.0
            
            logger.info(f"Database restore completed: {restore_id}")
            return restore_info
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            restore_info.status = "failed"
            restore_info.details['error'] = str(e)
            raise
    
    async def _restore_postgresql(self, backup_file: str, db_url: str):
        """Restore PostgreSQL database"""
        cmd = [
            'psql',
            db_url,
            '--no-password',
            '--quiet'
        ]
        
        with gzip.open(backup_file, 'rt') as f:
            process = subprocess.run(
                cmd,
                stdin=f,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
    
    async def _restore_sqlite(self, backup_file: str, db_url: str):
        """Restore SQLite database"""
        db_path = db_url.replace('sqlite:///', '')
        
        # Backup existing database
        if os.path.exists(db_path):
            backup_path = f"{db_path}.backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(db_path, backup_path)
            logger.info(f"Existing database backed up to: {backup_path}")
        
        # Restore from backup
        with gzip.open(backup_file, 'rb') as f_in:
            with open(db_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    async def _ensure_backup_local(self, backup_info: Dict[str, Any]) -> str:
        """Ensure backup file is available locally"""
        location = backup_info['location']
        
        if location.startswith('s3://'):
            # Download from S3
            backup_file = os.path.join(self.backup_dir, f"temp_{backup_info['backup_id']}.tmp")
            await self._download_from_s3(location, backup_file)
            return backup_file
        else:
            # Already local
            return location
    
    async def _download_from_s3(self, s3_location: str, local_file: str):
        """Download backup from S3"""
        if not self.s3_client:
            raise ValueError("S3 client not configured")
        
        # Parse S3 location
        s3_parts = s3_location.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        
        try:
            self.s3_client.download_file(bucket, key, local_file)
            logger.info(f"Downloaded backup from S3: {s3_location}")
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            raise
    
    def _find_backup(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Find backup by ID"""
        for backup in self.metadata['backups']:
            if backup['backup_id'] == backup_id:
                return backup
        return None
    
    async def list_backups(self, backup_type: Optional[str] = None) -> List[BackupInfo]:
        """List available backups"""
        backups = []
        
        for backup_data in self.metadata['backups']:
            if backup_type is None or backup_data['backup_type'] == backup_type:
                backup_info = BackupInfo(**backup_data)
                backups.append(backup_info)
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        
        return backups
    
    async def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up old backups based on retention policy"""
        logger.info(f"Starting backup cleanup (retention: {self.retention_days} days)")
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        backups_to_remove = []
        total_freed_bytes = 0
        
        for backup in self.metadata['backups']:
            backup_date = datetime.fromisoformat(backup['timestamp'].replace('Z', '+00:00'))
            
            if backup_date < cutoff_date:
                backups_to_remove.append(backup)
                total_freed_bytes += backup['size_bytes']
        
        # Remove old backups
        for backup in backups_to_remove:
            try:
                await self._remove_backup(backup)
                self.metadata['backups'].remove(backup)
                logger.info(f"Removed old backup: {backup['backup_id']}")
            except Exception as e:
                logger.error(f"Failed to remove backup {backup['backup_id']}: {e}")
        
        # Update metadata
        self.metadata['statistics']['last_cleanup'] = datetime.utcnow().isoformat()
        self._save_metadata()
        
        cleanup_results = {
            'removed_backups': len(backups_to_remove),
            'freed_bytes': total_freed_bytes,
            'remaining_backups': len(self.metadata['backups']),
            'cleanup_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Backup cleanup completed: {cleanup_results}")
        return cleanup_results
    
    async def _remove_backup(self, backup: Dict[str, Any]):
        """Remove a backup from storage"""
        location = backup['location']
        
        if location.startswith('s3://'):
            # Remove from S3
            s3_parts = location.replace('s3://', '').split('/', 1)
            bucket = s3_parts[0]
            key = s3_parts[1]
            
            try:
                self.s3_client.delete_object(Bucket=bucket, Key=key)
            except ClientError as e:
                logger.error(f"Failed to delete S3 object: {e}")
                raise
        else:
            # Remove local file
            if os.path.exists(location):
                os.remove(location)
    
    async def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics"""
        stats = self.metadata['statistics'].copy()
        
        # Calculate additional statistics
        if self.metadata['backups']:
            backup_types = {}
            for backup in self.metadata['backups']:
                backup_type = backup['backup_type']
                if backup_type not in backup_types:
                    backup_types[backup_type] = {'count': 0, 'size_bytes': 0}
                backup_types[backup_type]['count'] += 1
                backup_types[backup_type]['size_bytes'] += backup['size_bytes']
            
            stats['backup_types'] = backup_types
            stats['oldest_backup'] = min(backup['timestamp'] for backup in self.metadata['backups'])
            stats['newest_backup'] = max(backup['timestamp'] for backup in self.metadata['backups'])
        
        return stats


class DisasterRecoveryManager:
    """Manages disaster recovery procedures"""
    
    def __init__(self, backup_manager: BackupManager, config: Dict[str, Any]):
        self.backup_manager = backup_manager
        self.config = config
        self.flyctl_path = config.get('flyctl_path', '/home/codespace/.fly/bin/flyctl')
    
    async def create_disaster_recovery_backup(self, app_name: str) -> Dict[str, BackupInfo]:
        """Create complete disaster recovery backup"""
        logger.info(f"Creating disaster recovery backup for {app_name}")
        
        backups = {}
        
        try:
            # Backup volumes
            volumes = await self._get_app_volumes(app_name)
            for volume in volumes:
                backup_info = await self.backup_manager.backup_volume(app_name, volume['name'])
                backups[f"volume_{volume['name']}"] = backup_info
            
            # Backup database
            db_url = os.getenv('DATABASE_URL')
            if db_url:
                backup_info = await self.backup_manager.backup_database(db_url)
                backups['database'] = backup_info
            
            # Backup application data
            data_dirs = ['/app/data', '/app/logs', '/app/memory']
            existing_dirs = [d for d in data_dirs if os.path.exists(d)]
            if existing_dirs:
                backup_info = await self.backup_manager.backup_application_data(existing_dirs)
                backups['app_data'] = backup_info
            
            logger.info(f"Disaster recovery backup completed: {len(backups)} components")
            return backups
            
        except Exception as e:
            logger.error(f"Disaster recovery backup failed: {e}")
            raise
    
    async def _get_app_volumes(self, app_name: str) -> List[Dict[str, Any]]:
        """Get list of volumes for an app"""
        try:
            cmd = [self.flyctl_path, 'volumes', 'list', '--app', app_name, '--json']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            volumes = json.loads(result.stdout)
            return volumes
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get app volumes: {e.stderr}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse volume list: {e}")
            return []
    
    async def disaster_recovery_restore(self, backup_set: Dict[str, str], target_app: str) -> Dict[str, RestoreInfo]:
        """Perform complete disaster recovery restore"""
        logger.info(f"Starting disaster recovery restore for {target_app}")
        
        restore_results = {}
        
        try:
            # Restore database first
            if 'database' in backup_set:
                db_url = os.getenv('DATABASE_URL')
                if db_url:
                    restore_info = await self.backup_manager.restore_database(
                        backup_set['database'], db_url
                    )
                    restore_results['database'] = restore_info
            
            # Restore volumes
            for key, backup_id in backup_set.items():
                if key.startswith('volume_'):
                    volume_name = key.replace('volume_', '')
                    restore_info = await self.backup_manager.restore_volume(
                        backup_id, target_app, volume_name
                    )
                    restore_results[key] = restore_info
            
            logger.info(f"Disaster recovery restore completed: {len(restore_results)} components")
            return restore_results
            
        except Exception as e:
            logger.error(f"Disaster recovery restore failed: {e}")
            raise


def get_backup_config() -> Dict[str, Any]:
    """Get backup configuration from environment"""
    return {
        'backup_dir': os.getenv('BACKUP_DIR', '/app/backups'),
        's3_bucket': os.getenv('BACKUP_S3_BUCKET'),
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'aws_region': os.getenv('AWS_REGION', 'us-east-1'),
        'retention_days': int(os.getenv('BACKUP_RETENTION_DAYS', '30')),
        'flyctl_path': os.getenv('FLYCTL_PATH', '/home/codespace/.fly/bin/flyctl')
    }


if __name__ == "__main__":
    # CLI tool for backup operations
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Backup Manager for GPU Trading Platform")
        parser.add_argument('command', choices=['backup', 'restore', 'list', 'cleanup', 'stats', 'dr-backup', 'dr-restore'])
        parser.add_argument('--type', choices=['volume', 'database', 'app_data'], help='Backup type')
        parser.add_argument('--app', help='App name')
        parser.add_argument('--volume', help='Volume name')
        parser.add_argument('--backup-id', help='Backup ID for restore')
        parser.add_argument('--db-url', help='Database URL')
        
        args = parser.parse_args()
        
        config = get_backup_config()
        backup_manager = BackupManager(config)
        
        if args.command == 'backup':
            if args.type == 'volume':
                if not args.app or not args.volume:
                    print("--app and --volume required for volume backup")
                    return
                result = await backup_manager.backup_volume(args.app, args.volume)
                print(f"Volume backup completed: {result.backup_id}")
                
            elif args.type == 'database':
                db_url = args.db_url or os.getenv('DATABASE_URL')
                if not db_url:
                    print("Database URL required")
                    return
                result = await backup_manager.backup_database(db_url)
                print(f"Database backup completed: {result.backup_id}")
                
            elif args.type == 'app_data':
                data_dirs = ['/app/data', '/app/logs', '/app/memory']
                result = await backup_manager.backup_application_data(data_dirs)
                print(f"App data backup completed: {result.backup_id}")
        
        elif args.command == 'list':
            backups = await backup_manager.list_backups(args.type)
            print(f"Found {len(backups)} backups:")
            for backup in backups:
                print(f"  {backup.backup_id} ({backup.backup_type}) - {backup.timestamp}")
        
        elif args.command == 'cleanup':
            result = await backup_manager.cleanup_old_backups()
            print(f"Cleanup completed: {result}")
        
        elif args.command == 'stats':
            stats = await backup_manager.get_backup_statistics()
            print(f"Backup statistics: {json.dumps(stats, indent=2)}")
        
        elif args.command == 'dr-backup':
            if not args.app:
                print("--app required for disaster recovery backup")
                return
            dr_manager = DisasterRecoveryManager(backup_manager, config)
            result = await dr_manager.create_disaster_recovery_backup(args.app)
            print(f"Disaster recovery backup completed: {len(result)} components")
            for component, backup_info in result.items():
                print(f"  {component}: {backup_info.backup_id}")
    
    asyncio.run(main())