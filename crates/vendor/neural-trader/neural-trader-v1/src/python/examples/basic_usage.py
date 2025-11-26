"""
Basic Usage Example
==================

This example demonstrates the basic usage of the Python Supabase client
for neural trading platform operations.
"""

import asyncio
import os
from uuid import uuid4
from supabase_client import NeuralTradingClient
from supabase_client.clients.neural_models import CreateModelRequest
from supabase_client.clients.trading_bots import CreateBotRequest

async def main():
    """Main example function."""
    
    # Initialize the client
    client = NeuralTradingClient(
        url=os.getenv("SUPABASE_URL", "https://your-project.supabase.co"),
        key=os.getenv("SUPABASE_ANON_KEY", "your-anon-key"),
        service_key=os.getenv("SUPABASE_SERVICE_KEY")  # Optional
    )
    
    # Example user ID (in real usage, this would come from authentication)
    user_id = uuid4()
    
    try:
        # Connect to Supabase
        await client.connect()
        print("✅ Connected to Supabase")
        
        # 1. Create a user profile
        profile_data = {
            "id": str(user_id),
            "email": "demo@example.com",
            "full_name": "Demo User",
            "is_active": True
        }
        
        profiles = await client.supabase.insert("profiles", profile_data)
        print(f"✅ Created user profile: {profiles[0]['email']}")
        
        # 2. Create a trading account
        account_data = {
            "user_id": str(user_id),
            "account_name": "Demo Trading Account",
            "account_type": "demo",
            "broker": "demo_broker",
            "balance": 10000.0,
            "currency": "USD",
            "is_active": True
        }
        
        accounts = await client.supabase.insert("trading_accounts", account_data)
        account_id = accounts[0]["id"]
        print(f"✅ Created trading account: {accounts[0]['account_name']}")
        
        # 3. Create a neural model
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
        
        model, error = await client.neural_models.create_model(user_id, model_request)
        if error:
            print(f"❌ Error creating model: {error}")
            return
        
        print(f"✅ Created neural model: {model['name']}")
        
        # 4. List user models
        models, error = await client.neural_models.list_user_models(user_id)
        if error:
            print(f"❌ Error listing models: {error}")
        else:
            print(f"📊 User has {len(models)} neural models")
        
        # 5. Create a trading bot
        bot_request = CreateBotRequest(
            name="Demo Momentum Bot",
            strategy="momentum",
            account_id=account_id,
            neural_model_id=model["id"],
            symbols=["AAPL", "GOOGL"],
            risk_params={
                "max_position_size": 0.05,  # 5% max position
                "stop_loss": 0.02,          # 2% stop loss
                "take_profit": 0.06,        # 6% take profit
                "max_daily_trades": 10
            },
            strategy_params={
                "momentum_window": 20,
                "signal_threshold": 0.02,
                "volume_filter": True
            }
        )
        
        bot, error = await client.trading_bots.create_bot(user_id, bot_request)
        if error:
            print(f"❌ Error creating bot: {error}")
            return
        
        print(f"✅ Created trading bot: {bot['name']}")
        
        # 6. List user bots
        bots, error = await client.trading_bots.list_user_bots(user_id)
        if error:
            print(f"❌ Error listing bots: {error}")
        else:
            print(f"🤖 User has {len(bots)} trading bots")
        
        # 7. Get bot status
        status, error = await client.trading_bots.get_bot_status(bot["id"])
        if error:
            print(f"❌ Error getting bot status: {error}")
        else:
            print(f"📈 Bot status: {status['bot']['status']}")
        
        # 8. Perform health check
        health = await client.supabase.health_check()
        print(f"🏥 System health: {health['status']} (response time: {health.get('response_time_ms', 0):.1f}ms)")
        
        print("\n🎉 Basic usage example completed successfully!")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        
    finally:
        # Disconnect from Supabase
        await client.disconnect()
        print("👋 Disconnected from Supabase")

if __name__ == "__main__":
    # Set up environment variables or use default values
    if not os.getenv("SUPABASE_URL"):
        print("⚠️  SUPABASE_URL not set, using placeholder")
    if not os.getenv("SUPABASE_ANON_KEY"):
        print("⚠️  SUPABASE_ANON_KEY not set, using placeholder")
    
    print("🚀 Starting basic usage example...")
    asyncio.run(main())