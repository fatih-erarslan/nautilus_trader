"""
Neural Model Training Example
============================

This example demonstrates how to create, train, and use neural models
for trading predictions using the Python Supabase client.
"""

import asyncio
import os
import json
from uuid import uuid4
from datetime import datetime, timedelta
import random

from supabase_client import NeuralTradingClient
from supabase_client.clients.neural_models import (
    CreateModelRequest,
    StartTrainingRequest,
    PredictionRequest,
    ModelStatus,
    TrainingStatus
)

async def generate_sample_training_data():
    """Generate sample training data for demonstration."""
    # In a real scenario, this would be historical market data
    data = {
        "features": [],
        "targets": [],
        "symbols": ["AAPL", "GOOGL"],
        "timeframe": "1h",
        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "end_date": datetime.now().isoformat()
    }
    
    # Generate 1000 sample data points
    for i in range(1000):
        # Sample features: [price, volume, volatility, rsi, macd]
        features = [
            random.uniform(100, 200),  # price
            random.uniform(1000000, 5000000),  # volume
            random.uniform(0.1, 0.5),  # volatility
            random.uniform(20, 80),  # rsi
            random.uniform(-2, 2)  # macd
        ]
        
        # Sample target: next period return
        target = random.uniform(-0.05, 0.05)
        
        data["features"].append(features)
        data["targets"].append(target)
    
    return data

async def monitor_training_progress(neural_client, training_id):
    """Monitor training progress until completion."""
    print("📊 Monitoring training progress...")
    
    while True:
        status, error = await neural_client.get_training_status(training_id)
        if error:
            print(f"❌ Error getting training status: {error}")
            break
        
        print(f"Status: {status['status']}")
        print(f"Progress: {status.get('progress', 0):.1%}")
        print(f"Epoch: {status.get('current_epoch', 0)}/{status.get('total_epochs', 0)}")
        
        if status.get('metrics'):
            print(f"Loss: {status['metrics'].get('loss', 'N/A')}")
            print(f"Accuracy: {status['metrics'].get('accuracy', 'N/A')}")
        
        if status["status"] in [TrainingStatus.COMPLETED.value, TrainingStatus.FAILED.value]:
            break
        
        print("---")
        await asyncio.sleep(5)  # Check every 5 seconds
    
    return status

async def make_sample_predictions(neural_client, model_id):
    """Make sample predictions with the trained model."""
    print("🔮 Making sample predictions...")
    
    # Generate sample input data
    sample_inputs = [
        {
            "features": [150.0, 2000000, 0.25, 45.0, 0.5],
            "symbol": "AAPL",
            "timestamp": datetime.now().isoformat()
        },
        {
            "features": [2800.0, 1500000, 0.30, 60.0, -0.2],
            "symbol": "GOOGL", 
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    predictions = []
    for input_data in sample_inputs:
        prediction_request = PredictionRequest(
            model_id=model_id,
            input_data=input_data,
            symbols=[input_data["symbol"]]
        )
        
        prediction, error = await neural_client.make_prediction(prediction_request)
        if error:
            print(f"❌ Error making prediction: {error}")
            continue
        
        predictions.append(prediction)
        print(f"Prediction for {input_data['symbol']}: {prediction.get('prediction', 'N/A')}")
        print(f"Confidence: {prediction.get('confidence', 'N/A')}")
    
    return predictions

async def main():
    """Main function demonstrating neural model training workflow."""
    
    # Initialize client
    client = NeuralTradingClient(
        url=os.getenv("SUPABASE_URL", "https://your-project.supabase.co"),
        key=os.getenv("SUPABASE_ANON_KEY", "your-anon-key"),
        service_key=os.getenv("SUPABASE_SERVICE_KEY")
    )
    
    user_id = uuid4()
    
    try:
        await client.connect()
        print("✅ Connected to Supabase")
        
        # Create user profile
        profile_data = {
            "id": str(user_id),
            "email": "ml-trader@example.com",
            "full_name": "ML Trader",
            "is_active": True
        }
        await client.supabase.insert("profiles", profile_data)
        print("✅ Created user profile")
        
        # 1. Create a sophisticated neural model
        print("\n🧠 Creating neural model...")
        
        model_request = CreateModelRequest(
            name="LSTM Momentum Predictor",
            model_type="lstm",
            architecture={
                "architecture": "lstm",
                "sequence_length": 60,
                "layers": [
                    {"type": "lstm", "units": 128, "dropout": 0.3, "recurrent_dropout": 0.3},
                    {"type": "lstm", "units": 64, "dropout": 0.3, "recurrent_dropout": 0.3},
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dropout", "rate": 0.5},
                    {"type": "dense", "units": 1, "activation": "tanh"}
                ],
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999
                },
                "loss": "mse",
                "metrics": ["mae", "mape"],
                "early_stopping": {
                    "patience": 10,
                    "monitor": "val_loss",
                    "restore_best_weights": True
                },
                "feature_engineering": {
                    "technical_indicators": ["rsi", "macd", "bollinger_bands", "moving_averages"],
                    "lookback_periods": [5, 10, 20, 50],
                    "normalize": True,
                    "remove_outliers": True
                }
            }
        )
        
        model, error = await client.neural_models.create_model(user_id, model_request)
        if error:
            print(f"❌ Error creating model: {error}")
            return
        
        print(f"✅ Created model: {model['name']} (ID: {model['id']})")
        
        # 2. Generate and prepare training data
        print("\n📊 Preparing training data...")
        training_data = await generate_sample_training_data()
        print(f"✅ Generated {len(training_data['features'])} training samples")
        
        # 3. Start model training
        print("\n🚀 Starting model training...")
        
        training_request = StartTrainingRequest(
            model_id=model["id"],
            training_data=training_data,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            training_config={
                "shuffle": True,
                "verbose": 1,
                "callbacks": ["early_stopping", "model_checkpoint", "reduce_lr"],
                "class_weight": None,
                "sample_weight": None
            }
        )
        
        training_run, error = await client.neural_models.start_training(training_request)
        if error:
            print(f"❌ Error starting training: {error}")
            return
        
        print(f"✅ Training started (ID: {training_run['id']})")
        
        # 4. Monitor training progress
        final_status = await monitor_training_progress(
            client.neural_models, 
            training_run["id"]
        )
        
        if final_status["status"] == TrainingStatus.COMPLETED.value:
            print("🎉 Training completed successfully!")
            
            # Display final metrics
            if final_status.get("final_metrics"):
                metrics = final_status["final_metrics"]
                print(f"Final Training Loss: {metrics.get('loss', 'N/A')}")
                print(f"Final Validation Loss: {metrics.get('val_loss', 'N/A')}")
                print(f"Final Training MAE: {metrics.get('mae', 'N/A')}")
                print(f"Final Validation MAE: {metrics.get('val_mae', 'N/A')}")
        else:
            print(f"❌ Training failed: {final_status.get('error_message', 'Unknown error')}")
            return
        
        # 5. Get updated model information
        model_info, error = await client.neural_models.get_model(model["id"])
        if error:
            print(f"❌ Error getting model info: {error}")
        else:
            print(f"Model Status: {model_info['status']}")
            print(f"Model Version: {model_info.get('version', 'N/A')}")
        
        # 6. Make predictions with the trained model
        if model_info and model_info["status"] == ModelStatus.TRAINED.value:
            predictions = await make_sample_predictions(
                client.neural_models,
                model["id"]
            )
            print(f"✅ Made {len(predictions)} predictions")
        
        # 7. Get model performance metrics
        print("\n📈 Getting model performance...")
        performance, error = await client.neural_models.get_model_performance(
            model["id"],
            days=7
        )
        if error:
            print(f"❌ Error getting performance: {error}")
        else:
            print(f"Average Accuracy: {performance.get('average_accuracy', 'N/A')}")
            print(f"Prediction Count: {performance.get('prediction_count', 0)}")
            print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A')}")
        
        # 8. Update model configuration (example)
        print("\n🔧 Updating model configuration...")
        updates = {
            "configuration": {
                **model["configuration"],
                "inference_threshold": 0.6,
                "max_position_size": 0.1
            }
        }
        
        updated_model, error = await client.neural_models.update_model(
            model["id"], 
            updates
        )
        if error:
            print(f"❌ Error updating model: {error}")
        else:
            print("✅ Model configuration updated")
        
        print("\n🎉 Neural model training example completed successfully!")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await client.disconnect()
        print("👋 Disconnected from Supabase")

if __name__ == "__main__":
    print("🧠 Starting neural model training example...")
    asyncio.run(main())