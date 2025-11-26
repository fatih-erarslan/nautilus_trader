"""
Secrets Management System

Provides secure handling of API keys, database credentials, and other sensitive
configuration data with encryption, access control, and audit logging.
"""

import os
import json
import base64
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import getpass

logger = logging.getLogger(__name__)


@dataclass
class SecretMetadata:
    """Metadata for a stored secret."""
    name: str
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    expires_at: Optional[datetime] = None
    environment: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = None


class SecretsManager:
    """
    Secure secrets management with encryption, access control, and audit logging.
    Supports multiple backends including file-based, keyring, and environment variables.
    """
    
    def __init__(self, 
                 secrets_dir: Optional[str] = None,
                 master_key: Optional[str] = None,
                 use_keyring: bool = True,
                 encryption_enabled: bool = True):
        """
        Initialize secrets manager.
        
        Args:
            secrets_dir: Directory for storing encrypted secrets
            master_key: Master key for encryption (will prompt if not provided)
            use_keyring: Use system keyring for key storage
            encryption_enabled: Enable encryption for stored secrets
        """
        self.secrets_dir = Path(secrets_dir) if secrets_dir else Path.home() / ".neural_forecast" / "secrets"
        self.use_keyring = use_keyring
        self.encryption_enabled = encryption_enabled
        self.cipher_suite = None
        self.secrets_cache = {}
        self.metadata_cache = {}
        self.audit_log = []
        
        # Ensure secrets directory exists with proper permissions
        self._setup_secrets_directory()
        
        # Initialize encryption
        if self.encryption_enabled:
            self._initialize_encryption(master_key)
        
        # Load existing secrets metadata
        self._load_metadata()
        
        logger.info("Secrets manager initialized")
    
    def _setup_secrets_directory(self):
        """Setup secrets directory with secure permissions."""
        self.secrets_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Set restrictive permissions on secrets directory
        try:
            os.chmod(self.secrets_dir, 0o700)
        except OSError as e:
            logger.warning(f"Could not set secure permissions on secrets directory: {e}")
        
        # Create subdirectories
        (self.secrets_dir / "encrypted").mkdir(exist_ok=True, mode=0o700)
        (self.secrets_dir / "metadata").mkdir(exist_ok=True, mode=0o700)
        (self.secrets_dir / "audit").mkdir(exist_ok=True, mode=0o700)
    
    def _initialize_encryption(self, master_key: Optional[str]):
        """Initialize encryption cipher suite."""
        if master_key is None:
            # Try to get master key from keyring
            if self.use_keyring:
                try:
                    master_key = keyring.get_password("neural_forecast", "master_key")
                except Exception as e:
                    logger.debug(f"Could not retrieve master key from keyring: {e}")
            
            # If still no key, prompt user or generate new one
            if master_key is None:
                if os.getenv('NEURAL_FORECAST_MASTER_KEY'):
                    master_key = os.getenv('NEURAL_FORECAST_MASTER_KEY')
                else:
                    master_key = self._prompt_for_master_key()
        
        # Derive encryption key from master key
        salt = self._get_or_create_salt()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.cipher_suite = Fernet(key)
        
        # Store master key in keyring if enabled
        if self.use_keyring and master_key:
            try:
                keyring.set_password("neural_forecast", "master_key", master_key)
            except Exception as e:
                logger.warning(f"Could not store master key in keyring: {e}")
        
        logger.debug("Encryption initialized successfully")
    
    def _get_or_create_salt(self) -> bytes:
        """Get or create encryption salt."""
        salt_file = self.secrets_dir / ".salt"
        
        if salt_file.exists():
            with open(salt_file, 'rb') as f:
                return f.read()
        else:
            salt = os.urandom(16)
            with open(salt_file, 'wb') as f:
                f.write(salt)
            os.chmod(salt_file, 0o600)
            return salt
    
    def _prompt_for_master_key(self) -> str:
        """Prompt user for master key."""
        print("Enter master key for secrets encryption:")
        master_key = getpass.getpass("Master key: ")
        
        if not master_key:
            # Generate random master key
            master_key = secrets.token_urlsafe(32)
            print(f"Generated new master key: {master_key}")
            print("Please store this key securely!")
        
        return master_key
    
    def _load_metadata(self):
        """Load secrets metadata from disk."""
        metadata_file = self.secrets_dir / "metadata" / "secrets.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata_data = json.load(f)
                
                for name, data in metadata_data.items():
                    self.metadata_cache[name] = SecretMetadata(
                        name=data['name'],
                        created_at=datetime.fromisoformat(data['created_at']),
                        last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None,
                        access_count=data.get('access_count', 0),
                        expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
                        environment=data.get('environment'),
                        description=data.get('description'),
                        tags=data.get('tags', [])
                    )
                
                logger.debug(f"Loaded metadata for {len(self.metadata_cache)} secrets")
            except Exception as e:
                logger.error(f"Error loading secrets metadata: {e}")
    
    def _save_metadata(self):
        """Save secrets metadata to disk."""
        metadata_file = self.secrets_dir / "metadata" / "secrets.json"
        
        try:
            metadata_data = {}
            for name, metadata in self.metadata_cache.items():
                metadata_dict = asdict(metadata)
                # Convert datetime objects to ISO format
                for key, value in metadata_dict.items():
                    if isinstance(value, datetime):
                        metadata_dict[key] = value.isoformat()
                metadata_data[name] = metadata_dict
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_data, f, indent=2)
            
            os.chmod(metadata_file, 0o600)
            
        except Exception as e:
            logger.error(f"Error saving secrets metadata: {e}")
    
    def _log_access(self, action: str, secret_name: str, success: bool = True, details: Optional[str] = None):
        """Log secret access for audit purposes."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'secret_name': secret_name,
            'success': success,
            'details': details,
            'user': os.getenv('USER', 'unknown')
        }
        
        self.audit_log.append(audit_entry)
        
        # Save to audit file
        audit_file = self.secrets_dir / "audit" / f"audit_{datetime.now().strftime('%Y%m')}.json"
        
        try:
            audit_entries = []
            if audit_file.exists():
                with open(audit_file, 'r') as f:
                    audit_entries = json.load(f)
            
            audit_entries.append(audit_entry)
            
            with open(audit_file, 'w') as f:
                json.dump(audit_entries, f, indent=2)
            
            os.chmod(audit_file, 0o600)
            
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")
    
    def store_secret(self, 
                    name: str, 
                    value: Union[str, Dict[str, Any]], 
                    environment: Optional[str] = None,
                    description: Optional[str] = None,
                    expires_in_days: Optional[int] = None,
                    tags: Optional[List[str]] = None,
                    overwrite: bool = False) -> bool:
        """
        Store a secret securely.
        
        Args:
            name: Secret name/identifier
            value: Secret value (string or dictionary)
            environment: Environment this secret belongs to
            description: Description of the secret
            expires_in_days: Number of days until secret expires
            tags: Tags for categorization
            overwrite: Whether to overwrite existing secret
            
        Returns:
            True if stored successfully
        """
        if name in self.metadata_cache and not overwrite:
            logger.warning(f"Secret '{name}' already exists. Use overwrite=True to replace.")
            return False
        
        try:
            # Prepare secret data
            secret_data = {
                'value': value,
                'stored_at': datetime.now().isoformat(),
                'environment': environment
            }
            
            # Encrypt if enabled
            if self.encryption_enabled and self.cipher_suite:
                encrypted_data = self.cipher_suite.encrypt(json.dumps(secret_data).encode())
                storage_data = base64.b64encode(encrypted_data).decode()
            else:
                storage_data = json.dumps(secret_data)
            
            # Store to file
            secret_file = self.secrets_dir / "encrypted" / f"{name}.enc"
            with open(secret_file, 'w') as f:
                f.write(storage_data)
            
            os.chmod(secret_file, 0o600)
            
            # Update metadata
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)
            
            self.metadata_cache[name] = SecretMetadata(
                name=name,
                created_at=datetime.now(),
                expires_at=expires_at,
                environment=environment,
                description=description,
                tags=tags or []
            )
            
            self._save_metadata()
            self._log_access('store', name, True, f"Stored secret for {environment or 'no environment'}")
            
            logger.info(f"Stored secret '{name}' successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error storing secret '{name}': {e}")
            self._log_access('store', name, False, str(e))
            return False
    
    def get_secret(self, name: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a secret.
        
        Args:
            name: Secret name
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        # Check cache first
        if name in self.secrets_cache:
            metadata = self.metadata_cache.get(name)
            if metadata:
                metadata.last_accessed = datetime.now()
                metadata.access_count += 1
                self._save_metadata()
            
            self._log_access('get', name, True, "Retrieved from cache")
            return self.secrets_cache[name]
        
        # Check if secret exists
        if name not in self.metadata_cache:
            self._log_access('get', name, False, "Secret not found")
            return default
        
        # Check if secret has expired
        metadata = self.metadata_cache[name]
        if metadata.expires_at and datetime.now() > metadata.expires_at:
            logger.warning(f"Secret '{name}' has expired")
            self._log_access('get', name, False, "Secret expired")
            return default
        
        try:
            # Load from file
            secret_file = self.secrets_dir / "encrypted" / f"{name}.enc"
            if not secret_file.exists():
                self._log_access('get', name, False, "Secret file not found")
                return default
            
            with open(secret_file, 'r') as f:
                storage_data = f.read()
            
            # Decrypt if enabled
            if self.encryption_enabled and self.cipher_suite:
                encrypted_data = base64.b64decode(storage_data.encode())
                decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                secret_data = json.loads(decrypted_data.decode())
            else:
                secret_data = json.loads(storage_data)
            
            value = secret_data['value']
            
            # Update metadata
            metadata.last_accessed = datetime.now()
            metadata.access_count += 1
            self._save_metadata()
            
            # Cache the secret
            self.secrets_cache[name] = value
            
            self._log_access('get', name, True)
            return value
            
        except Exception as e:
            logger.error(f"Error retrieving secret '{name}': {e}")
            self._log_access('get', name, False, str(e))
            return default
    
    def delete_secret(self, name: str) -> bool:
        """
        Delete a secret.
        
        Args:
            name: Secret name
            
        Returns:
            True if deleted successfully
        """
        if name not in self.metadata_cache:
            logger.warning(f"Secret '{name}' not found")
            return False
        
        try:
            # Remove from file system
            secret_file = self.secrets_dir / "encrypted" / f"{name}.enc"
            if secret_file.exists():
                secret_file.unlink()
            
            # Remove from caches and metadata
            self.secrets_cache.pop(name, None)
            del self.metadata_cache[name]
            self._save_metadata()
            
            self._log_access('delete', name, True)
            logger.info(f"Deleted secret '{name}' successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting secret '{name}': {e}")
            self._log_access('delete', name, False, str(e))
            return False
    
    def list_secrets(self, environment: Optional[str] = None, 
                    include_expired: bool = False) -> List[Dict[str, Any]]:
        """
        List available secrets.
        
        Args:
            environment: Filter by environment
            include_expired: Include expired secrets
            
        Returns:
            List of secret information
        """
        secrets_list = []
        current_time = datetime.now()
        
        for name, metadata in self.metadata_cache.items():
            # Filter by environment
            if environment and metadata.environment != environment:
                continue
            
            # Filter expired secrets
            if not include_expired and metadata.expires_at and current_time > metadata.expires_at:
                continue
            
            secret_info = {
                'name': metadata.name,
                'environment': metadata.environment,
                'description': metadata.description,
                'created_at': metadata.created_at.isoformat(),
                'last_accessed': metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                'access_count': metadata.access_count,
                'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                'expired': metadata.expires_at and current_time > metadata.expires_at if metadata.expires_at else False,
                'tags': metadata.tags
            }
            
            secrets_list.append(secret_info)
        
        return sorted(secrets_list, key=lambda x: x['name'])
    
    def rotate_secret(self, name: str, new_value: Union[str, Dict[str, Any]]) -> bool:
        """
        Rotate (update) a secret value.
        
        Args:
            name: Secret name
            new_value: New secret value
            
        Returns:
            True if rotated successfully
        """
        if name not in self.metadata_cache:
            logger.error(f"Cannot rotate secret '{name}': not found")
            return False
        
        # Get existing metadata
        metadata = self.metadata_cache[name]
        
        # Store new value with same metadata
        success = self.store_secret(
            name=name,
            value=new_value,
            environment=metadata.environment,
            description=metadata.description,
            tags=metadata.tags,
            overwrite=True
        )
        
        if success:
            # Clear from cache to force reload
            self.secrets_cache.pop(name, None)
            self._log_access('rotate', name, True)
            logger.info(f"Rotated secret '{name}' successfully")
        
        return success
    
    def export_secrets(self, environment: Optional[str] = None, 
                      output_file: Optional[str] = None,
                      include_values: bool = False) -> str:
        """
        Export secrets metadata and optionally values.
        
        Args:
            environment: Environment to export
            output_file: Output file path
            include_values: Include secret values (DANGEROUS)
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            env_suffix = f"_{environment}" if environment else ""
            output_file = f"secrets_export{env_suffix}_{timestamp}.json"
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'environment_filter': environment,
            'include_values': include_values,
            'secrets': []
        }
        
        for secret_info in self.list_secrets(environment=environment, include_expired=True):
            secret_export = secret_info.copy()
            
            if include_values:
                secret_export['value'] = self.get_secret(secret_info['name'])
            
            export_data['secrets'].append(secret_export)
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        os.chmod(output_path, 0o600)
        
        logger.info(f"Exported {len(export_data['secrets'])} secrets to {output_path}")
        self._log_access('export', f"{len(export_data['secrets'])} secrets", True, 
                        f"Environment: {environment}, Include values: {include_values}")
        
        return str(output_path)
    
    def cleanup_expired_secrets(self) -> int:
        """
        Remove expired secrets.
        
        Returns:
            Number of secrets cleaned up
        """
        current_time = datetime.now()
        expired_secrets = []
        
        for name, metadata in self.metadata_cache.items():
            if metadata.expires_at and current_time > metadata.expires_at:
                expired_secrets.append(name)
        
        cleaned_count = 0
        for name in expired_secrets:
            if self.delete_secret(name):
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired secrets")
            self._log_access('cleanup', f"{cleaned_count} secrets", True)
        
        return cleaned_count
    
    def get_audit_log(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            List of audit log entries
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get recent audit entries from memory
        recent_entries = [
            entry for entry in self.audit_log
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_date
        ]
        
        # Load from audit files
        audit_files = list((self.secrets_dir / "audit").glob("audit_*.json"))
        
        for audit_file in audit_files:
            try:
                with open(audit_file, 'r') as f:
                    file_entries = json.load(f)
                
                for entry in file_entries:
                    if datetime.fromisoformat(entry['timestamp']) >= cutoff_date:
                        recent_entries.append(entry)
            
            except Exception as e:
                logger.error(f"Error reading audit file {audit_file}: {e}")
        
        # Sort by timestamp
        recent_entries.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return recent_entries
    
    def validate_secrets_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of stored secrets.
        
        Returns:
            Validation results
        """
        results = {
            'total_secrets': len(self.metadata_cache),
            'valid_secrets': 0,
            'corrupted_secrets': [],
            'missing_files': [],
            'expired_secrets': [],
            'orphaned_files': []
        }
        
        current_time = datetime.now()
        
        # Check each secret in metadata
        for name, metadata in self.metadata_cache.items():
            secret_file = self.secrets_dir / "encrypted" / f"{name}.enc"
            
            if not secret_file.exists():
                results['missing_files'].append(name)
                continue
            
            # Check if expired
            if metadata.expires_at and current_time > metadata.expires_at:
                results['expired_secrets'].append(name)
            
            # Try to decrypt/read
            try:
                value = self.get_secret(name)
                if value is not None:
                    results['valid_secrets'] += 1
                else:
                    results['corrupted_secrets'].append(name)
            except Exception as e:
                results['corrupted_secrets'].append(f"{name}: {str(e)}")
        
        # Check for orphaned files
        encrypted_files = list((self.secrets_dir / "encrypted").glob("*.enc"))
        for file_path in encrypted_files:
            secret_name = file_path.stem
            if secret_name not in self.metadata_cache:
                results['orphaned_files'].append(secret_name)
        
        results['integrity_score'] = results['valid_secrets'] / results['total_secrets'] if results['total_secrets'] > 0 else 1.0
        
        return results