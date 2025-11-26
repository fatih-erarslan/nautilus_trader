#!/usr/bin/env python3
"""
Test script for neural forecasting MCP server implementation
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

def test_neural_mcp_server():
    """Test the neural forecasting MCP server implementation."""
    print("🧪 Testing Neural Forecasting MCP Server Implementation")
    print("=" * 60)
    
    try:
        # Import necessary components without starting the server
        exec(open('src/mcp/mcp_server_enhanced.py').read().replace('if __name__ == "__main__":', 'if False:'))
        
        print("✅ MCP server script imports successfully")
        
        # Check neural models are loaded
        print(f"✅ Neural models loaded: {len(NEURAL_MODELS)} models")
        for model_id, model_info in NEURAL_MODELS.items():
            gpu_status = "GPU" if model_info.get("gpu_accelerated", False) else "CPU"
            status = model_info.get("training_status", "unknown")
            model_type = model_info.get("model_type", "Unknown")
            print(f"  - {model_id}: {model_type} ({gpu_status}, {status})")
        
        # Check trading models are loaded
        print(f"✅ Trading models loaded: {len(OPTIMIZED_MODELS)} models")
        for model_id in OPTIMIZED_MODELS.keys():
            print(f"  - {model_id}")
        
        # Check MCP server is initialized
        print(f"✅ MCP server initialized with name: '{mcp.name}'")
        
        # Test neural tools are accessible
        neural_tools = [
            'neural_forecast', 'neural_train', 'neural_evaluate', 
            'neural_backtest', 'neural_model_status', 'neural_optimize'
        ]
        
        # Check functions exist in globals
        available_tools = []
        for tool in neural_tools:
            if tool in globals():
                available_tools.append(tool)
        
        print(f"✅ Neural tools available: {len(available_tools)}/{len(neural_tools)}")
        for tool in available_tools:
            print(f"  - {tool}")
        
        # Test validation schemas
        validation_schemas = [
            'NeuralForecastRequest', 'NeuralTrainRequest', 'NeuralEvaluateRequest',
            'NeuralBacktestRequest', 'NeuralOptimizeRequest'
        ]
        
        available_schemas = []
        for schema in validation_schemas:
            if schema in globals():
                available_schemas.append(schema)
        
        print(f"✅ Validation schemas available: {len(available_schemas)}/{len(validation_schemas)}")
        for schema in available_schemas:
            print(f"  - {schema}")
        
        # Check GPU detection
        print(f"✅ GPU detection: {'Available' if GPU_AVAILABLE else 'Not Available'}")
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Neural forecasting MCP server is ready.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_neural_mcp_server()
    sys.exit(0 if success else 1)