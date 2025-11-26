"""
Data Encryption Module for Syndicate Sensitive Information
Handles encryption/decryption of member credentials and financial data
"""

import os
import base64
import json
from typing import Any, Dict, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import secrets


class DataEncryptor:
    """Handles encryption and decryption of sensitive data"""
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize encryptor with master key"""
        if master_key:
            self.master_key = master_key.encode()
        else:
            # Try to get from environment, otherwise generate
            env_key = os.environ.get('SYNDICATE_MASTER_KEY')
            if env_key:
                self.master_key = env_key.encode()
            else:
                self.master_key = self._generate_master_key()
        
        # Initialize cipher suite
        self.cipher_suite = self._create_cipher_suite()
        
        # Field-level encryption keys
        self.field_keys = {}
    
    def _generate_master_key(self) -> bytes:
        """Generate a new master key"""
        return Fernet.generate_key()
    
    def _create_cipher_suite(self) -> Fernet:
        """Create Fernet cipher suite from master key"""
        if len(self.master_key) != 44:  # Fernet keys are 44 bytes when base64 encoded
            # Derive a proper key from the master key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'syndicate_salt_v1',  # In production, use a random salt
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        else:
            key = self.master_key
        
        return Fernet(key)
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data and return base64 encoded string"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self.cipher_suite.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded data"""
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.cipher_suite.decrypt(decoded)
            return decrypted.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data"""
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data"""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)
    
    def encrypt_field(self, field_name: str, value: str) -> str:
        """Encrypt a specific field with field-specific key"""
        # Get or create field-specific key
        if field_name not in self.field_keys:
            self.field_keys[field_name] = self._derive_field_key(field_name)
        
        field_cipher = Fernet(self.field_keys[field_name])
        encrypted = field_cipher.encrypt(value.encode('utf-8'))
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt_field(self, field_name: str, encrypted_value: str) -> str:
        """Decrypt a field-encrypted value"""
        # Get field-specific key
        if field_name not in self.field_keys:
            self.field_keys[field_name] = self._derive_field_key(field_name)
        
        field_cipher = Fernet(self.field_keys[field_name])
        decoded = base64.urlsafe_b64decode(encrypted_value.encode('utf-8'))
        decrypted = field_cipher.decrypt(decoded)
        return decrypted.decode('utf-8')
    
    def _derive_field_key(self, field_name: str) -> bytes:
        """Derive a field-specific key from master key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=f'field_{field_name}'.encode('utf-8'),
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(self.master_key))


class SecureDataStore:
    """Secure storage for sensitive syndicate data"""
    
    def __init__(self, encryptor: Optional[DataEncryptor] = None):
        self.encryptor = encryptor or DataEncryptor()
        self.sensitive_fields = [
            'password', 'api_key', 'secret_key', 'private_key',
            'bank_account', 'credit_card', 'ssn', 'tax_id',
            'wallet_address', 'seed_phrase'
        ]
    
    def store_member_credentials(self, member_id: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Store member credentials with encryption"""
        encrypted_creds = {}
        
        for field, value in credentials.items():
            if field in self.sensitive_fields and value:
                # Encrypt sensitive fields
                encrypted_creds[field] = {
                    'encrypted': True,
                    'value': self.encryptor.encrypt_field(field, str(value))
                }
            else:
                # Store non-sensitive fields as-is
                encrypted_creds[field] = {
                    'encrypted': False,
                    'value': value
                }
        
        # Add metadata
        encrypted_creds['_metadata'] = {
            'member_id': member_id,
            'stored_at': datetime.now(timezone.utc).isoformat(),
            'version': '1.0'
        }
        
        return encrypted_creds
    
    def retrieve_member_credentials(self, encrypted_creds: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve and decrypt member credentials"""
        decrypted_creds = {}
        
        for field, data in encrypted_creds.items():
            if field == '_metadata':
                decrypted_creds[field] = data
                continue
            
            if isinstance(data, dict) and data.get('encrypted'):
                # Decrypt sensitive fields
                try:
                    decrypted_creds[field] = self.encryptor.decrypt_field(field, data['value'])
                except Exception as e:
                    # Log decryption failure without exposing the value
                    logger.error(f"Failed to decrypt field {field}: {str(e)}")
                    decrypted_creds[field] = None
            else:
                # Return non-sensitive fields as-is
                decrypted_creds[field] = data.get('value') if isinstance(data, dict) else data
        
        return decrypted_creds
    
    def encrypt_financial_data(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt financial transaction data"""
        sensitive_transaction_fields = [
            'amount', 'account_number', 'routing_number',
            'card_number', 'cvv', 'recipient_details'
        ]
        
        encrypted_data = {}
        
        for field, value in transaction_data.items():
            if field in sensitive_transaction_fields and value:
                encrypted_data[field] = self.encryptor.encrypt(str(value))
            else:
                encrypted_data[field] = value
        
        # Add encryption marker
        encrypted_data['_encrypted'] = True
        encrypted_data['_encryption_version'] = '1.0'
        
        return encrypted_data
    
    def decrypt_financial_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt financial transaction data"""
        if not encrypted_data.get('_encrypted'):
            return encrypted_data
        
        sensitive_transaction_fields = [
            'amount', 'account_number', 'routing_number',
            'card_number', 'cvv', 'recipient_details'
        ]
        
        decrypted_data = {}
        
        for field, value in encrypted_data.items():
            if field.startswith('_'):
                continue
                
            if field in sensitive_transaction_fields and value:
                try:
                    decrypted_data[field] = self.encryptor.decrypt(value)
                except Exception as e:
                    logger.error(f"Failed to decrypt transaction field {field}: {str(e)}")
                    decrypted_data[field] = None
            else:
                decrypted_data[field] = value
        
        return decrypted_data
    
    def create_encrypted_backup(self, data: Dict[str, Any]) -> str:
        """Create an encrypted backup of syndicate data"""
        backup_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '1.0',
            'data': data
        }
        
        # Encrypt the entire backup
        return self.encryptor.encrypt_dict(backup_data)
    
    def restore_from_backup(self, encrypted_backup: str) -> Dict[str, Any]:
        """Restore data from encrypted backup"""
        try:
            backup_data = self.encryptor.decrypt_dict(encrypted_backup)
            return backup_data['data']
        except Exception as e:
            raise ValueError(f"Failed to restore backup: {str(e)}")


class TokenEncryption:
    """Specialized encryption for authentication tokens"""
    
    def __init__(self):
        self.token_key = self._get_or_create_token_key()
        self.cipher = Fernet(self.token_key)
    
    def _get_or_create_token_key(self) -> bytes:
        """Get or create token encryption key"""
        key_file = ".token_key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            return key
    
    def encrypt_token_data(self, token_data: Dict[str, Any]) -> str:
        """Encrypt token data for storage"""
        json_data = json.dumps(token_data)
        encrypted = self.cipher.encrypt(json_data.encode('utf-8'))
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt_token_data(self, encrypted_token: str) -> Dict[str, Any]:
        """Decrypt stored token data"""
        try:
            decoded = base64.urlsafe_b64decode(encrypted_token.encode('utf-8'))
            decrypted = self.cipher.decrypt(decoded)
            return json.loads(decrypted.decode('utf-8'))
        except Exception as e:
            raise ValueError(f"Invalid encrypted token: {str(e)}")


# Import required modules
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)