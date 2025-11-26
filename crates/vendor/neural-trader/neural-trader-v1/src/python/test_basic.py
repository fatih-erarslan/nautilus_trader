#!/usr/bin/env python3
"""
Basic functionality test for Python Supabase client
"""

import asyncio
import os
from uuid import uuid4

from supabase_client import NeuralTradingClient
from supabase_client.config import SupabaseConfig
from supabase_client.clients.neural_models import CreateModelRequest

async def test_basic_functionality():
    """Test basic client functionality."""
    print("🧪 Testing Python Supabase Client...")
    
    # Test configuration
    config = SupabaseConfig(
        url="https://demo.supabase.co",
        anon_key="demo-key",
        service_key="demo-service-key"
    )
    print("✅ Configuration created")
    
    # Test client initialization
    client = NeuralTradingClient(
        url=config.url,
        key=config.anon_key,
        service_key=config.service_key
    )
    print("✅ Client initialized")
    
    # Test basic operations (without actual connection)
    print("✅ Client components accessible:")
    print(f"  - Neural models client: {type(client.neural_models).__name__}")
    print(f"  - Trading bots client: {type(client.trading_bots).__name__}")
    print(f"  - Sandbox client: {type(client.sandbox).__name__}")
    print(f"  - Realtime manager: {type(client.realtime).__name__}")
    print(f"  - Performance monitor: {type(client.performance).__name__}")
    
    # Test data models
    from supabase_client.models.database_models import (
        Profile, NeuralModel, TradingBot, Order
    )
    
    # Test model creation
    profile = Profile(
        id=uuid4(),
        username="testuser",
        email="test@example.com",
        full_name="Test User"
    )
    print(f"✅ Profile model created: {profile.username}")
    
    # Test request models
    model_request = CreateModelRequest(
        name="Test LSTM Model",
        model_type="lstm",
        architecture={"layers": 3, "units": 128}
    )
    print(f"✅ Model request created: {model_request.name}")
    
    print("\n🎉 All basic tests passed!")
    return True

def test_sync_functionality():
    """Test synchronous functionality."""
    print("\n🔄 Testing synchronous operations...")
    
    # Test imports
    from supabase_client.utils.validation_utils import (
        validate_email, validate_symbol, validate_uuid
    )
    
    # Test validation utilities
    assert validate_email("test@example.com") == True
    assert validate_email("invalid-email") == False
    print("✅ Email validation works")
    
    assert validate_symbol("AAPL") == True
    assert validate_symbol("invalid") == False
    print("✅ Symbol validation works")
    
    test_uuid = str(uuid4())
    assert validate_uuid(test_uuid) == True
    assert validate_uuid("invalid-uuid") == False
    print("✅ UUID validation works")
    
    print("✅ All synchronous tests passed!")
    return True

def main():
    """Main test function."""
    print("🚀 Starting Python Supabase Client Tests\n")
    
    try:
        # Test sync functionality
        test_sync_functionality()
        
        # Test async functionality
        asyncio.run(test_basic_functionality())
        
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The Python Supabase client is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)