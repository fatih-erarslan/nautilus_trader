"""
Core Client Tests
===============

Test suite for the core Supabase client functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from supabase_client.client import SupabaseClient, AsyncSupabaseClient, SupabaseError
from supabase_client.config import SupabaseConfig

class TestSupabaseClient:
    """Test suite for synchronous SupabaseClient."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SupabaseConfig(
            url="https://test.supabase.co",
            anon_key="test-anon-key",
            service_key="test-service-key"
        )
    
    @pytest.fixture
    def client(self, config):
        """Create client instance."""
        return SupabaseClient(config)
    
    def test_client_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client._client is None
        assert client._user is None
        assert client._connected is False
    
    @patch('supabase_client.client.create_client')
    def test_connect_success(self, mock_create_client, client):
        """Test successful connection."""
        # Arrange
        mock_supabase = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        # Mock table method for connection test
        mock_table = MagicMock()
        mock_result = MagicMock()
        mock_result.execute.return_value = MagicMock()
        mock_table.select.return_value.limit.return_value = mock_result
        mock_supabase.table.return_value = mock_table
        
        # Act
        result = client.connect()
        
        # Assert
        assert result == client
        assert client._connected is True
        assert client._client == mock_supabase
        mock_create_client.assert_called_once()
    
    @patch('supabase_client.client.create_client')
    def test_connect_failure(self, mock_create_client, client):
        """Test connection failure."""
        # Arrange
        mock_create_client.side_effect = Exception("Connection failed")
        
        # Act & Assert
        with pytest.raises(SupabaseError) as exc_info:
            client.connect()
        
        assert "Failed to connect to Supabase" in str(exc_info.value)
        assert client._connected is False
    
    def test_disconnect(self, client):
        """Test disconnection."""
        # Arrange
        client._connected = True
        client._client = MagicMock()
        client._user = MagicMock()
        
        # Act
        client.disconnect()
        
        # Assert
        assert client._connected is False
        assert client._client is None
        assert client._user is None
    
    def test_client_property_when_connected(self, client):
        """Test client property when connected."""
        # Arrange
        mock_client = MagicMock()
        client._connected = True
        client._client = mock_client
        
        # Act
        result = client.client
        
        # Assert
        assert result == mock_client
    
    def test_client_property_when_not_connected(self, client):
        """Test client property when not connected."""
        # Act & Assert
        with pytest.raises(SupabaseError) as exc_info:
            client.client
        
        assert "Client not connected" in str(exc_info.value)
    
    def test_context_manager(self, client):
        """Test context manager functionality."""
        with patch.object(client, 'connect') as mock_connect, \
             patch.object(client, 'disconnect') as mock_disconnect:
            
            mock_connect.return_value = client
            
            # Act
            with client as ctx_client:
                assert ctx_client == client
            
            # Assert
            mock_connect.assert_called_once()
            mock_disconnect.assert_called_once()

class TestAsyncSupabaseClient:
    """Test suite for asynchronous AsyncSupabaseClient."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SupabaseConfig(
            url="https://test.supabase.co",
            anon_key="test-anon-key"
        )
    
    @pytest.fixture
    def async_client(self, config):
        """Create async client instance."""
        return AsyncSupabaseClient(config)
    
    def test_async_client_initialization(self, async_client, config):
        """Test async client initialization."""
        assert async_client.config == config
        assert async_client._sync_client is None
        assert async_client._user is None
        assert async_client._connected is False
    
    @pytest.mark.asyncio
    async def test_connect_success(self, async_client):
        """Test successful async connection."""
        # Arrange
        mock_sync_client = MagicMock()
        
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = asyncio.Future()
            mock_loop.run_in_executor.return_value.set_result(mock_sync_client)
            
            # Act
            result = await async_client.connect()
            
            # Assert
            assert result == async_client
            assert async_client._connected is True
            assert async_client._sync_client == mock_sync_client
    
    @pytest.mark.asyncio
    async def test_connect_already_connected(self, async_client):
        """Test connecting when already connected."""
        # Arrange
        async_client._connected = True
        
        # Act
        result = await async_client.connect()
        
        # Assert
        assert result == async_client
    
    @pytest.mark.asyncio
    async def test_disconnect(self, async_client):
        """Test async disconnection."""
        # Arrange
        mock_sync_client = MagicMock()
        async_client._connected = True
        async_client._sync_client = mock_sync_client
        
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = asyncio.Future()
            mock_loop.run_in_executor.return_value.set_result(None)
            
            # Act
            await async_client.disconnect()
            
            # Assert
            assert async_client._connected is False
            assert async_client._sync_client is None
    
    @pytest.mark.asyncio
    async def test_select_operation(self, async_client):
        """Test async select operation."""
        # Arrange
        mock_sync_client = MagicMock()
        async_client._connected = True
        async_client._sync_client = mock_sync_client
        
        mock_table = MagicMock()
        mock_query = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{"id": 1, "name": "test"}]
        
        mock_sync_client.table.return_value = mock_table
        mock_table.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.execute.return_value = mock_result
        
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = asyncio.Future()
            mock_loop.run_in_executor.return_value.set_result([{"id": 1, "name": "test"}])
            
            # Act
            result = await async_client.select(
                "test_table",
                columns="id,name",
                filter_dict={"active": True},
                order_by="name",
                limit=10
            )
            
            # Assert
            assert result == [{"id": 1, "name": "test"}]
    
    @pytest.mark.asyncio
    async def test_insert_operation(self, async_client):
        """Test async insert operation."""
        # Arrange
        mock_sync_client = MagicMock()
        async_client._connected = True
        async_client._sync_client = mock_sync_client
        
        test_data = {"name": "test", "value": 123}
        expected_result = [{"id": 1, **test_data}]
        
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = asyncio.Future()
            mock_loop.run_in_executor.return_value.set_result(expected_result)
            
            # Act
            result = await async_client.insert("test_table", test_data)
            
            # Assert
            assert result == expected_result
    
    @pytest.mark.asyncio
    async def test_update_operation(self, async_client):
        """Test async update operation."""
        # Arrange
        mock_sync_client = MagicMock()
        async_client._connected = True
        async_client._sync_client = mock_sync_client
        
        update_data = {"name": "updated"}
        filter_dict = {"id": 1}
        expected_result = [{"id": 1, "name": "updated"}]
        
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = asyncio.Future()
            mock_loop.run_in_executor.return_value.set_result(expected_result)
            
            # Act
            result = await async_client.update("test_table", update_data, filter_dict)
            
            # Assert
            assert result == expected_result
    
    @pytest.mark.asyncio
    async def test_delete_operation(self, async_client):
        """Test async delete operation."""
        # Arrange
        mock_sync_client = MagicMock()
        async_client._connected = True
        async_client._sync_client = mock_sync_client
        
        filter_dict = {"id": 1}
        expected_result = [{"id": 1}]
        
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = asyncio.Future()
            mock_loop.run_in_executor.return_value.set_result(expected_result)
            
            # Act
            result = await async_client.delete("test_table", filter_dict)
            
            # Assert
            assert result == expected_result
    
    @pytest.mark.asyncio
    async def test_bulk_insert(self, async_client):
        """Test bulk insert operation."""
        # Arrange
        async_client._connected = True
        async_client._sync_client = MagicMock()
        
        data_list = [{"name": f"item{i}"} for i in range(5)]
        
        # Mock the insert method to return the data
        async def mock_insert(table, data):
            return data
        
        async_client.insert = mock_insert
        
        # Act
        result = await async_client.bulk_insert("test_table", data_list, batch_size=2)
        
        # Assert
        assert len(result) == 5
        assert all(item["name"].startswith("item") for item in result)
    
    @pytest.mark.asyncio
    async def test_count_operation(self, async_client):
        """Test count operation."""
        # Arrange
        mock_sync_client = MagicMock()
        async_client._connected = True
        async_client._sync_client = mock_sync_client
        
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = asyncio.Future()
            mock_loop.run_in_executor.return_value.set_result(42)
            
            # Act
            result = await async_client.count("test_table", {"active": True})
            
            # Assert
            assert result == 42
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, async_client):
        """Test health check when system is healthy."""
        # Arrange
        async_client._connected = True
        
        async def mock_select(*args, **kwargs):
            return [{"id": 1}]
        
        async_client.select = mock_select
        
        # Act
        result = await async_client.health_check()
        
        # Assert
        assert result["status"] == "healthy"
        assert "response_time_ms" in result
        assert "timestamp" in result
        assert result["connected"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, async_client):
        """Test health check when system is unhealthy."""
        # Arrange
        async_client._connected = True
        
        async def mock_select(*args, **kwargs):
            raise Exception("Database error")
        
        async_client.select = mock_select
        
        # Act
        result = await async_client.health_check()
        
        # Assert
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, async_client):
        """Test async context manager functionality."""
        with patch.object(async_client, 'connect') as mock_connect, \
             patch.object(async_client, 'disconnect') as mock_disconnect:
            
            mock_connect.return_value = asyncio.Future()
            mock_connect.return_value.set_result(async_client)
            mock_disconnect.return_value = asyncio.Future()
            mock_disconnect.return_value.set_result(None)
            
            # Act
            async with async_client as ctx_client:
                assert ctx_client == async_client
            
            # Assert
            mock_connect.assert_called_once()
            mock_disconnect.assert_called_once()

class TestSupabaseError:
    """Test suite for SupabaseError exception."""
    
    def test_error_creation(self):
        """Test error creation with message only."""
        error = SupabaseError("Test error message")
        assert str(error) == "Test error message"
        assert error.original_error is None
    
    def test_error_creation_with_original(self):
        """Test error creation with original exception."""
        original = ValueError("Original error")
        error = SupabaseError("Wrapper error", original)
        
        assert str(error) == "Wrapper error"
        assert error.original_error == original

if __name__ == "__main__":
    pytest.main([__file__])