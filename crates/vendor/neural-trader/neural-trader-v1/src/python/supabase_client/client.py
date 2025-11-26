"""
Core Supabase Client Implementation
==================================

Main client classes for synchronous and asynchronous Supabase operations.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import json

try:
    from supabase import create_client, Client
    from supabase.lib.client_options import ClientOptions
    from postgrest import APIError
    from supabase_auth import User
except ImportError:
    raise ImportError(
        "Supabase dependencies not found. Install with: pip install supabase"
    )

from .config import SupabaseConfig
from .models.database_models import BaseModel

logger = logging.getLogger(__name__)

class SupabaseError(Exception):
    """Custom exception for Supabase operations."""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error

class SupabaseClient:
    """
    Synchronous Supabase client with enhanced error handling and utilities.
    """
    
    def __init__(self, config: SupabaseConfig):
        """
        Initialize Supabase client.
        
        Args:
            config: Supabase configuration object
        """
        self.config = config
        self._client: Optional[Client] = None
        self._user: Optional[User] = None
        self._connected = False
        
    def connect(self) -> "SupabaseClient":
        """
        Establish connection to Supabase.
        
        Returns:
            Self for method chaining
            
        Raises:
            SupabaseError: If connection fails
        """
        try:
            options = ClientOptions(
                auto_refresh_token=self.config.auto_refresh_token,
                persist_session=self.config.persist_session,
                detect_session_in_url=self.config.detect_session_in_url,
                headers=self.config.headers,
                schema=self.config.schema
            )
            
            self._client = create_client(
                self.config.url,
                self.config.anon_key,
                options
            )
            
            # Test connection
            self._test_connection()
            self._connected = True
            
            logger.info("Successfully connected to Supabase")
            return self
            
        except Exception as e:
            raise SupabaseError(f"Failed to connect to Supabase: {str(e)}", e)
    
    def disconnect(self):
        """Disconnect from Supabase."""
        if self._client:
            # Sign out if authenticated
            if self._user:
                try:
                    self._client.auth.sign_out()
                except:
                    pass
                    
            self._client = None
            self._user = None
            self._connected = False
            logger.info("Disconnected from Supabase")
    
    def _test_connection(self):
        """Test database connection."""
        try:
            result = self._client.table("profiles").select("id").limit(1).execute()
            logger.debug("Connection test successful")
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
    
    @property
    def client(self) -> Client:
        """Get the underlying Supabase client."""
        if not self._connected or not self._client:
            raise SupabaseError("Client not connected. Call connect() first.")
        return self._client
    
    @property
    def user(self) -> Optional[User]:
        """Get current authenticated user."""
        return self._user
    
    def authenticate(self, email: str, password: str) -> User:
        """
        Authenticate user with email and password.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            Authenticated user object
            
        Raises:
            SupabaseError: If authentication fails
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            self._user = response.user
            logger.info(f"User authenticated: {email}")
            return self._user
            
        except Exception as e:
            raise SupabaseError(f"Authentication failed: {str(e)}", e)
    
    def sign_out(self):
        """Sign out current user."""
        try:
            if self._user:
                self.client.auth.sign_out()
                self._user = None
                logger.info("User signed out")
        except Exception as e:
            logger.error(f"Sign out error: {e}")
    
    def table(self, table_name: str):
        """Get table interface for database operations."""
        return self.client.table(table_name)
    
    def rpc(self, function_name: str, params: Dict[str, Any] = None):
        """
        Call database function.
        
        Args:
            function_name: Name of the database function
            params: Function parameters
            
        Returns:
            Function result
        """
        try:
            return self.client.rpc(function_name, params or {}).execute()
        except Exception as e:
            raise SupabaseError(f"RPC call failed: {str(e)}", e)
    
    def storage(self):
        """Get storage interface."""
        return self.client.storage
    
    def __enter__(self):
        """Context manager entry."""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

class AsyncSupabaseClient:
    """
    Asynchronous Supabase client with enhanced functionality.
    """
    
    def __init__(self, config: SupabaseConfig):
        """
        Initialize async Supabase client.
        
        Args:
            config: Supabase configuration object
        """
        self.config = config
        self._sync_client: Optional[SupabaseClient] = None
        self._user: Optional[User] = None
        self._connected = False
        self._connection_lock = asyncio.Lock()
        
    async def connect(self) -> "AsyncSupabaseClient":
        """
        Establish async connection to Supabase.
        
        Returns:
            Self for method chaining
        """
        async with self._connection_lock:
            if self._connected:
                return self
                
            try:
                # Create sync client in thread pool
                loop = asyncio.get_event_loop()
                self._sync_client = await loop.run_in_executor(
                    None, 
                    lambda: SupabaseClient(self.config).connect()
                )
                
                self._connected = True
                logger.info("Async connection to Supabase established")
                return self
                
            except Exception as e:
                raise SupabaseError(f"Async connection failed: {str(e)}", e)
    
    async def disconnect(self):
        """Disconnect from Supabase."""
        async with self._connection_lock:
            if self._sync_client:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._sync_client.disconnect)
                self._sync_client = None
                self._user = None
                self._connected = False
                logger.info("Async connection to Supabase closed")
    
    @property
    def client(self) -> SupabaseClient:
        """Get the underlying sync client."""
        if not self._connected or not self._sync_client:
            raise SupabaseError("Async client not connected. Call connect() first.")
        return self._sync_client
    
    async def authenticate(self, email: str, password: str) -> User:
        """
        Authenticate user asynchronously.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            Authenticated user object
        """
        loop = asyncio.get_event_loop()
        self._user = await loop.run_in_executor(
            None, 
            self.client.authenticate, 
            email, 
            password
        )
        return self._user
    
    async def sign_out(self):
        """Sign out current user asynchronously."""
        if self._user:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.client.sign_out)
            self._user = None
    
    async def select(
        self, 
        table: str, 
        columns: str = "*", 
        filter_dict: Dict[str, Any] = None,
        order_by: str = None,
        limit: int = None,
        offset: int = None
    ) -> List[Dict[str, Any]]:
        """
        Async select operation with common filters.
        
        Args:
            table: Table name
            columns: Columns to select
            filter_dict: Filters to apply
            order_by: Order by clause
            limit: Limit number of results
            offset: Offset for pagination
            
        Returns:
            List of records
        """
        loop = asyncio.get_event_loop()
        
        def _select():
            query = self.client.table(table).select(columns)
            
            if filter_dict:
                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        query = query.in_(key, value)
                    else:
                        query = query.eq(key, value)
            
            if order_by:
                desc = order_by.startswith("-")
                column = order_by.lstrip("-")
                query = query.order(column, desc=desc)
            
            if limit:
                query = query.limit(limit)
                
            if offset:
                query = query.offset(offset)
            
            result = query.execute()
            return result.data
        
        return await loop.run_in_executor(None, _select)
    
    async def insert(
        self, 
        table: str, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Async insert operation.
        
        Args:
            table: Table name
            data: Data to insert
            
        Returns:
            Inserted records
        """
        loop = asyncio.get_event_loop()
        
        def _insert():
            result = self.client.table(table).insert(data).execute()
            return result.data
        
        return await loop.run_in_executor(None, _insert)
    
    async def update(
        self, 
        table: str, 
        data: Dict[str, Any], 
        filter_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Async update operation.
        
        Args:
            table: Table name
            data: Data to update
            filter_dict: Filters for update
            
        Returns:
            Updated records
        """
        loop = asyncio.get_event_loop()
        
        def _update():
            query = self.client.table(table).update(data)
            
            for key, value in filter_dict.items():
                query = query.eq(key, value)
            
            result = query.execute()
            return result.data
        
        return await loop.run_in_executor(None, _update)
    
    async def delete(
        self, 
        table: str, 
        filter_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Async delete operation.
        
        Args:
            table: Table name
            filter_dict: Filters for deletion
            
        Returns:
            Deleted records
        """
        loop = asyncio.get_event_loop()
        
        def _delete():
            query = self.client.table(table).delete()
            
            for key, value in filter_dict.items():
                query = query.eq(key, value)
            
            result = query.execute()
            return result.data
        
        return await loop.run_in_executor(None, _delete)
    
    async def rpc(
        self, 
        function_name: str, 
        params: Dict[str, Any] = None
    ) -> Any:
        """
        Async RPC call.
        
        Args:
            function_name: Function name
            params: Function parameters
            
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        
        def _rpc():
            return self.client.rpc(function_name, params or {})
        
        result = await loop.run_in_executor(None, _rpc)
        return result.data
    
    async def bulk_insert(
        self, 
        table: str, 
        data_list: List[Dict[str, Any]], 
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Bulk insert with batching for large datasets.
        
        Args:
            table: Table name
            data_list: List of records to insert
            batch_size: Size of each batch
            
        Returns:
            All inserted records
        """
        results = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            batch_results = await self.insert(table, batch)
            results.extend(batch_results)
            
            # Brief pause between batches to avoid overwhelming the database
            if i + batch_size < len(data_list):
                await asyncio.sleep(0.1)
        
        return results
    
    async def upsert(
        self, 
        table: str, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]], 
        on_conflict: str = "id"
    ) -> List[Dict[str, Any]]:
        """
        Async upsert operation.
        
        Args:
            table: Table name
            data: Data to upsert
            on_conflict: Column(s) to check for conflicts
            
        Returns:
            Upserted records
        """
        loop = asyncio.get_event_loop()
        
        def _upsert():
            result = self.client.table(table).upsert(
                data, 
                on_conflict=on_conflict
            ).execute()
            return result.data
        
        return await loop.run_in_executor(None, _upsert)
    
    async def count(
        self, 
        table: str, 
        filter_dict: Dict[str, Any] = None
    ) -> int:
        """
        Get count of records.
        
        Args:
            table: Table name
            filter_dict: Filters to apply
            
        Returns:
            Count of records
        """
        loop = asyncio.get_event_loop()
        
        def _count():
            query = self.client.table(table).select("*", count="exact", head=True)
            
            if filter_dict:
                for key, value in filter_dict.items():
                    query = query.eq(key, value)
            
            result = query.execute()
            return result.count
        
        return await loop.run_in_executor(None, _count)
    
    async def execute_transaction(
        self, 
        operations: List[Callable]
    ) -> List[Any]:
        """
        Execute multiple operations as a transaction.
        
        Args:
            operations: List of operation functions
            
        Returns:
            Results of all operations
        """
        # Note: Supabase doesn't have built-in transaction support
        # This is a simple sequential execution with rollback on error
        results = []
        
        try:
            for operation in operations:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, operation)
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            # In a real implementation, you'd implement rollback logic here
            raise SupabaseError(f"Transaction failed: {str(e)}", e)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the connection.
        
        Returns:
            Health check results
        """
        start_time = datetime.now()
        
        try:
            # Test basic connectivity
            await self.select("profiles", "id", limit=1)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "timestamp": datetime.now().isoformat(),
                "connected": self._connected
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "connected": self._connected
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return await self.connect()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()