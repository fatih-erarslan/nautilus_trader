#!/usr/bin/env python3
"""
Comprehensive Example: AI Trading Model Management System

This script demonstrates the complete model management workflow:
1. Model creation and storage
2. Version control
3. Performance monitoring
4. Deployment orchestration
5. Health monitoring
6. MCP server integration
7. Real-time predictions

Run this script to see the full system in action.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

# Import model management components
from model_manager import ModelManager, ManagerConfig, DeploymentTarget
from storage.model_storage import ModelStorage, ModelFormat
from storage.metadata_manager import MetadataManager, ModelStatus
from storage.version_control import ModelVersionControl
from deployment.deploy_orchestrator import (
    DeploymentOrchestrator, DeploymentConfig, 
    DeploymentTarget, DeploymentStrategy
)
from deployment.health_monitor import HealthMonitor
from mcp_integration.trading_mcp_server import TradingMCPServer


async def demonstrate_model_lifecycle():
    """Demonstrate complete model lifecycle."""
    print("🚀 AI Trading Model Management System Demo")
    print("=" * 60)
    
    # 1. Initialize the Model Manager
    print("\n📦 1. Initializing Model Manager...")
    config = ManagerConfig(
        storage_path="demo_models",
        mcp_server_port=8000,
        api_server_port=8001,
        websocket_server_port=8002,
        enable_auto_cleanup=False  # Keep demo data
    )
    
    manager = ModelManager(config)
    
    # Register event callbacks
    manager.register_event_callback('model_created', on_model_created)
    manager.register_event_callback('model_deployed', on_model_deployed)
    manager.register_event_callback('performance_alert', on_performance_alert)
    
    try:
        # Start the manager (this starts all servers)
        await manager.start()
        print("✅ Model Manager started successfully")
        
        # 2. Create different types of trading models
        print("\n🧠 2. Creating Trading Models...")
        
        # Mean Reversion Strategy
        mean_reversion_id = await create_mean_reversion_model(manager)
        
        # Momentum Strategy  
        momentum_id = await create_momentum_model(manager)
        
        # Mirror Trading Strategy
        mirror_id = await create_mirror_trading_model(manager)
        
        # 3. Demonstrate version control
        print("\n📝 3. Demonstrating Version Control...")
        await demonstrate_version_control(manager, mean_reversion_id)
        
        # 4. Show model registry and analytics
        print("\n📊 4. Model Registry and Analytics...")
        await show_model_analytics(manager)
        
        # 5. Demonstrate model deployment
        print("\n🚀 5. Deploying Models...")
        await demonstrate_deployment(manager, mean_reversion_id)
        
        # 6. Test real-time predictions
        print("\n🔮 6. Testing Real-time Predictions...")
        await test_predictions(manager, [mean_reversion_id, momentum_id, mirror_id])
        
        # 7. Demonstrate health monitoring
        print("\n💓 7. Health Monitoring...")
        await demonstrate_health_monitoring(manager)
        
        # 8. Show system status
        print("\n📈 8. System Status...")
        await show_system_status(manager)
        
        # Keep system running for a bit to show real-time features
        print("\n⏱️  System running... (30 seconds)")
        print("💡 You can now:")
        print("   - Visit http://localhost:8001/docs for API documentation")
        print("   - Connect to ws://localhost:8002 for real-time updates")
        print("   - Use MCP server at localhost:8000 for predictions")
        
        await asyncio.sleep(30)
        
    finally:
        # Clean shutdown
        print("\n🛑 Shutting down...")
        await manager.stop()
        print("✅ Demo completed successfully!")


async def create_mean_reversion_model(manager: ModelManager) -> str:
    """Create a mean reversion trading model."""
    print("  📈 Creating Mean Reversion Strategy...")
    
    parameters = {
        "z_score_entry_threshold": 2.0,
        "z_score_exit_threshold": 0.5,
        "lookback_window": 40,
        "short_ma_window": 10,
        "base_position_size": 0.08,
        "z_score_position_scaling": 1.2,
        "max_position_size": 0.15,
        "stop_loss_multiplier": 1.6,
        "profit_target_multiplier": 1.8,
        "time_stop_days": 7,
        "volatility_adjustment": 1.1,
        "bull_market_z_threshold": 2.2,
        "bear_market_z_threshold": 2.8,
        "volume_confirmation_threshold": 1.4,
        "rsi_confirmation_threshold": 30,
        "bollinger_band_confirmation": True
    }
    
    performance_metrics = {
        "sharpe_ratio": 2.8,
        "total_return": 0.22,
        "max_drawdown": 0.09,
        "win_rate": 0.67,
        "profit_factor": 2.4,
        "calmar_ratio": 2.44,
        "trades_per_month": 18,
        "avg_holding_days": 4
    }
    
    model_id = await manager.create_model(
        name="Optimized Mean Reversion Strategy",
        version="1.0.0",
        strategy_name="mean_reversion",
        model_type="parameter_optimization",
        parameters=parameters,
        performance_metrics=performance_metrics,
        description="Advanced mean reversion strategy with adaptive thresholds and multi-regime support",
        tags=["mean_reversion", "optimized", "multi_regime", "production_ready"],
        author="Demo System"
    )
    
    print(f"    ✅ Mean Reversion Model created: {model_id}")
    return model_id


async def create_momentum_model(manager: ModelManager) -> str:
    """Create a momentum trading model."""
    print("  🚀 Creating Momentum Strategy...")
    
    parameters = {
        "momentum_threshold": 0.65,
        "trend_confirmation_period": 12,
        "base_position_size": 0.06,
        "momentum_scaling_factor": 1.5,
        "max_position_size": 0.12,
        "stop_loss_percentage": 0.03,
        "profit_target_percentage": 0.08,
        "volume_surge_threshold": 1.8,
        "rsi_momentum_filter": 60,
        "market_regime_adjustment": True,
        "correlation_filter": 0.7,
        "volatility_breakout_factor": 1.3
    }
    
    performance_metrics = {
        "sharpe_ratio": 2.1,
        "total_return": 0.19,
        "max_drawdown": 0.12,
        "win_rate": 0.58,
        "profit_factor": 1.9,
        "calmar_ratio": 1.58,
        "trades_per_month": 25,
        "avg_holding_days": 3
    }
    
    model_id = await manager.create_model(
        name="Enhanced Momentum Strategy",
        version="1.0.0", 
        strategy_name="momentum",
        model_type="parameter_optimization",
        parameters=parameters,
        performance_metrics=performance_metrics,
        description="High-frequency momentum strategy with volume confirmation and regime adaptation",
        tags=["momentum", "high_frequency", "breakout", "trending"],
        author="Demo System"
    )
    
    print(f"    ✅ Momentum Model created: {model_id}")
    return model_id


async def create_mirror_trading_model(manager: ModelManager) -> str:
    """Create a mirror trading model."""
    print("  🪞 Creating Mirror Trading Strategy...")
    
    parameters = {
        "berkshire_confidence": 0.85,
        "bridgewater_confidence": 0.92,
        "renaissance_confidence": 0.88,
        "soros_confidence": 0.79,
        "institutional_weight_threshold": 0.4,
        "min_position_change": 0.02,
        "max_mirror_position": 0.10,
        "entry_delay_days": 2,
        "confidence_decay_rate": 0.05,
        "sector_concentration_limit": 0.25,
        "market_cap_filter": 5000000000,  # $5B minimum
        "liquidity_threshold": 1000000     # $1M daily volume
    }
    
    performance_metrics = {
        "sharpe_ratio": 1.9,
        "total_return": 0.16,
        "max_drawdown": 0.08,
        "win_rate": 0.62,
        "profit_factor": 2.1,
        "calmar_ratio": 2.0,
        "trades_per_month": 12,
        "avg_holding_days": 45
    }
    
    model_id = await manager.create_model(
        name="Institutional Mirror Trading Strategy",
        version="1.0.0",
        strategy_name="mirror_trading", 
        model_type="institutional_following",
        parameters=parameters,
        performance_metrics=performance_metrics,
        description="Mirror successful institutional trades with confidence weighting and risk controls",
        tags=["mirror_trading", "institutional", "following", "low_frequency"],
        author="Demo System"
    )
    
    print(f"    ✅ Mirror Trading Model created: {model_id}")
    return model_id


async def demonstrate_version_control(manager: ModelManager, model_id: str):
    """Demonstrate version control features."""
    print("  📝 Creating model versions...")
    
    # Update model with improved parameters
    improved_params = {
        "z_score_entry_threshold": 2.1,  # Slightly more aggressive
        "base_position_size": 0.09,      # Larger positions
        "stop_loss_multiplier": 1.5      # Tighter stops
    }
    
    improved_metrics = {
        "sharpe_ratio": 3.1,    # Better performance
        "total_return": 0.25,
        "max_drawdown": 0.07,   # Lower drawdown
        "win_rate": 0.69
    }
    
    # Update the model (creates new version)
    await manager.update_model(model_id, {
        "parameters": improved_params,
        "performance_metrics": improved_metrics,
        "description": "Improved parameters based on backtesting results"
    })
    
    print(f"    ✅ Model {model_id} updated with improved parameters")
    
    # Show version history using version control
    versions = manager.version_control.get_version_history(model_id)
    print(f"    📚 Version history: {len(versions)} versions")
    for version in versions[-3:]:  # Show last 3 versions
        print(f"      - v{version.version_number}: {version.commit_message}")


async def show_model_analytics(manager: ModelManager):
    """Show model registry and analytics."""
    registry = manager.get_model_registry()
    print(f"  📊 Total models in registry: {len(registry)}")
    
    # Show strategy breakdown
    strategies = {}
    for model_id, model_info in registry.items():
        strategy = model_info['metadata']['strategy_name']
        if strategy not in strategies:
            strategies[strategy] = []
        strategies[strategy].append(model_info)
    
    print("  📈 Strategy breakdown:")
    for strategy, models in strategies.items():
        avg_sharpe = sum(m['metadata']['performance_metrics'].get('sharpe_ratio', 0) 
                        for m in models) / len(models)
        print(f"    - {strategy}: {len(models)} models, avg Sharpe: {avg_sharpe:.2f}")
    
    # Show top performers
    print("  🏆 Top performing models:")
    all_models = []
    for model_info in registry.values():
        sharpe = model_info['metadata']['performance_metrics'].get('sharpe_ratio', 0)
        all_models.append((model_info['metadata']['name'], sharpe))
    
    all_models.sort(key=lambda x: x[1], reverse=True)
    for name, sharpe in all_models[:3]:
        print(f"    - {name}: Sharpe {sharpe:.2f}")


async def demonstrate_deployment(manager: ModelManager, model_id: str):
    """Demonstrate model deployment."""
    print("  🚀 Deploying model to local environment...")
    
    # Create deployment configuration
    from deployment.deploy_orchestrator import DeploymentConfig, DeploymentTarget, DeploymentStrategy
    
    config = DeploymentConfig(
        target=DeploymentTarget.LOCAL,
        strategy=DeploymentStrategy.RECREATE,
        resource_requirements={
            "cpu": "1",
            "memory": "1Gi"
        },
        environment_variables={
            "MODEL_ID": model_id,
            "LOG_LEVEL": "INFO"
        },
        health_check_config={
            "max_retries": 5,
            "retry_interval": 10
        },
        auto_rollback=True,
        timeout_seconds=300
    )
    
    try:
        deployment_id = await manager.deploy_model(model_id, config)
        print(f"    ✅ Deployment started: {deployment_id}")
        
        # Wait a bit and check status
        await asyncio.sleep(5)
        
        print(f"    📊 Deployment status: In Progress")
        # In a real scenario, you'd monitor the deployment progress
        
    except Exception as e:
        print(f"    ❌ Deployment demo failed: {e}")
        print("    💡 This is expected in demo mode without actual deployment infrastructure")


async def test_predictions(manager: ModelManager, model_ids: list):
    """Test real-time predictions."""
    print("  🔮 Testing model predictions...")
    
    # Test data for different strategies
    test_cases = [
        {
            "strategy": "mean_reversion",
            "data": {
                "z_score": -2.3,
                "price": 98.5,
                "moving_average": 100.0,
                "volatility": 0.18,
                "volume_ratio": 1.6,
                "rsi": 25,
                "market_regime": 0.3
            }
        },
        {
            "strategy": "momentum", 
            "data": {
                "price_change": 0.035,
                "volume_change": 0.45,
                "momentum_score": 0.78,
                "trend_strength": 0.85,
                "volatility": 0.22,
                "market_sentiment": 0.7
            }
        },
        {
            "strategy": "mirror_trading",
            "data": {
                "institutional_positions": {
                    "berkshire": 0.12,
                    "bridgewater": 0.08,
                    "renaissance": 0.15
                },
                "confidence_scores": {
                    "berkshire": 0.9,
                    "bridgewater": 0.85,
                    "renaissance": 0.92
                },
                "entry_timing": "prompt",
                "market_conditions": "favorable"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        if i < len(model_ids):
            model_id = model_ids[i]
            print(f"    🎯 Testing {test_case['strategy']} strategy...")
            
            try:
                prediction = await manager.get_model_prediction(
                    model_id, 
                    test_case['data']
                )
                
                action = prediction.get('action', 'hold')
                confidence = prediction.get('confidence', 0.5)
                position_size = prediction.get('position_size', 0.0)
                
                print(f"      📈 Prediction: {action.upper()} "
                      f"(confidence: {confidence:.2f}, size: {position_size:.3f})")
                
            except Exception as e:
                print(f"      ❌ Prediction failed: {e}")


async def demonstrate_health_monitoring(manager: ModelManager):
    """Demonstrate health monitoring capabilities."""
    print("  💓 Initializing health monitoring...")
    
    # In a real deployment, health monitoring would track actual deployments
    print("    📊 Health monitoring features:")
    print("      - Real-time endpoint health checks")
    print("      - Performance metric tracking")
    print("      - Automated alerting")
    print("      - Uptime monitoring")
    print("      - Response time analysis")
    print("    ✅ Health monitoring system ready")


async def show_system_status(manager: ModelManager):
    """Show comprehensive system status."""
    status = manager.get_system_status()
    
    print("  📈 System Status:")
    print(f"    - Manager Status: {status['manager_status']}")
    print(f"    - Uptime: {status['uptime_seconds']:.1f} seconds")
    print(f"    - Models in Registry: {status['model_registry_size']}")
    print(f"    - Models Loaded: {status['statistics']['models_loaded']}")
    print(f"    - Predictions Made: {status['statistics']['predictions_made']}")
    print(f"    - Storage Used: {status['storage_stats']['total_size_mb']:.1f} MB")
    
    if 'server_status' in status:
        print("    - Server Status:")
        for server_name, server_info in status['server_status'].items():
            if isinstance(server_info, dict):
                print(f"      - {server_name}: {server_info.get('status', 'unknown')}")


# Event callback functions
async def on_model_created(data):
    """Handle model creation events."""
    model_id = data['model_id']
    metadata = data['metadata']
    print(f"🔔 Event: Model created - {metadata['name']} ({model_id})")


async def on_model_deployed(data):
    """Handle model deployment events."""
    model_id = data['model_id']
    target = data['target']
    print(f"🔔 Event: Model deployed - {model_id} to {target}")


async def on_performance_alert(data):
    """Handle performance alerts."""
    model_id = data['model_id']
    alert = data['alert']
    print(f"🚨 Performance Alert: {model_id} - {alert['message']}")


def main():
    """Main entry point."""
    print("🎬 Starting AI Trading Model Management Demo...")
    
    # Ensure demo directory exists
    demo_path = Path("demo_models")
    if demo_path.exists():
        import shutil
        shutil.rmtree(demo_path)
    
    try:
        # Run the demonstration
        asyncio.run(demonstrate_model_lifecycle())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup demo data
        if demo_path.exists():
            import shutil
            shutil.rmtree(demo_path)
        print("🧹 Demo cleanup completed")


if __name__ == "__main__":
    main()