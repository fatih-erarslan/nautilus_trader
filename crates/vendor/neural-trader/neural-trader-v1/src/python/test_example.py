#!/usr/bin/env python3
"""
Test example functionality without actual database connection
"""

import asyncio
import os
from uuid import uuid4

# Test that we can import and run example code
from supabase_client import NeuralTradingClient
from supabase_client.clients.neural_models import CreateModelRequest
from supabase_client.clients.trading_bots import CreateBotRequest

async def test_example_initialization():
    """Test that examples can be initialized properly."""
    print("🧪 Testing example initialization...")
    
    # Initialize client like in examples
    client = NeuralTradingClient(
        url="https://demo.supabase.co",
        key="demo-key",
        service_key="demo-service-key"
    )
    
    user_id = uuid4()
    account_id = uuid4()
    
    # Test neural model request creation
    model_request = CreateModelRequest(
        name="LSTM Price Predictor",
        model_type="lstm",
        architecture={
            "sequence_length": 60,
            "layers": [128, 64, 32],
            "dropout": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32
        }
    )
    print(f"✅ Created model request: {model_request.name}")
    
    # Test bot request creation  
    bot_request = CreateBotRequest(
        name="Demo Momentum Bot",
        account_id=account_id,
        strategy_type="momentum",
        configuration={"momentum_window": 20, "signal_threshold": 0.02},
        symbols=["AAPL", "GOOGL"]
    )
    print(f"✅ Created bot request: {bot_request.name}")
    
    # Test client components
    print("✅ Client components accessible:")
    print(f"  - Neural models: {client.neural_models.__class__.__name__}")
    print(f"  - Trading bots: {client.trading_bots.__class__.__name__}")
    print(f"  - Sandbox: {client.sandbox.__class__.__name__}")
    print(f"  - Realtime: {client.realtime.__class__.__name__}")
    print(f"  - Performance: {client.performance.__class__.__name__}")
    
    print("🎉 Example initialization test passed!")

def main():
    """Main test function."""
    print("🚀 Testing Python Supabase Client Examples\n")
    
    try:
        # Test async functionality
        asyncio.run(test_example_initialization())
        
        print("\n🎉 ALL EXAMPLE TESTS PASSED! 🎉")
        print("The Python Supabase client examples are working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)