#!/usr/bin/env python3
"""
Test E2B Integration Implementation
"""

import asyncio
import json
import requests
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test E2B health endpoint"""
    print("Testing E2B health check...")
    response = requests.get(f"{BASE_URL}/e2b/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed: {data}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False


def test_create_sandbox():
    """Test sandbox creation"""
    print("\nTesting sandbox creation...")
    
    config = {
        "name": "test_sandbox",
        "timeout": 300,
        "memory_mb": 512,
        "cpu_count": 1,
        "allow_internet": True,
        "envs": {
            "TEST_VAR": "test_value"
        },
        "metadata": {
            "purpose": "testing"
        }
    }
    
    response = requests.post(f"{BASE_URL}/e2b/sandbox/create", json=config)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Sandbox created: {data}")
        return data.get("sandbox_id")
    else:
        print(f"❌ Failed to create sandbox: {response.text}")
        return None


def test_execute_command(sandbox_id: str):
    """Test command execution in sandbox"""
    print(f"\nTesting command execution in sandbox {sandbox_id}...")
    
    params = {
        "command": "echo 'Hello from E2B sandbox!' && python --version",
        "timeout": 10
    }
    
    response = requests.post(
        f"{BASE_URL}/e2b/sandbox/{sandbox_id}/execute",
        params=params
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Command executed successfully:")
        print(f"   Output: {data.get('stdout', '')}")
        return True
    else:
        print(f"❌ Failed to execute command: {response.text}")
        return False


def test_run_agent():
    """Test running a trading agent"""
    print("\nTesting agent execution...")
    
    config = {
        "agent_type": "momentum_trader",
        "symbols": ["AAPL", "GOOGL"],
        "strategy_params": {
            "window": 10,
            "threshold": 0.01
        },
        "execution_mode": "simulation",
        "use_gpu": False
    }
    
    response = requests.post(f"{BASE_URL}/e2b/agent/run", json=config)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Agent executed successfully:")
        print(f"   Status: {data.get('status')}")
        print(f"   Trades: {len(data.get('trades', []))}")
        return True
    else:
        print(f"❌ Failed to run agent: {response.text}")
        return False


def test_execute_script():
    """Test script execution"""
    print("\nTesting script execution...")
    
    script = """
import numpy as np
import pandas as pd
from datetime import datetime

# Generate sample data
data = {
    'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    'price': np.random.uniform(100, 200, 10),
    'volume': np.random.uniform(1000000, 5000000, 10)
}

df = pd.DataFrame(data)
print("Sample trading data generated:")
print(df.head())
print(f"\\nMean price: ${df['price'].mean():.2f}")
print(f"Total volume: {df['volume'].sum():,.0f}")
"""
    
    payload = {
        "script_content": script,
        "language": "python"
    }
    
    response = requests.post(f"{BASE_URL}/e2b/script/execute", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Script executed successfully:")
        print(f"   Output:\n{data.get('stdout', '')}")
        return True
    else:
        print(f"❌ Failed to execute script: {response.text}")
        return False


def test_list_sandboxes():
    """Test listing sandboxes"""
    print("\nTesting sandbox listing...")
    
    response = requests.get(f"{BASE_URL}/e2b/sandbox/list")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Active sandboxes: {len(data)}")
        for sandbox in data:
            print(f"   - {sandbox['name']} ({sandbox['sandbox_id']}): {sandbox['status']}")
        return True
    else:
        print(f"❌ Failed to list sandboxes: {response.text}")
        return False


def test_cleanup_sandbox(sandbox_id: str):
    """Test sandbox termination"""
    print(f"\nTesting sandbox termination for {sandbox_id}...")
    
    response = requests.delete(f"{BASE_URL}/e2b/sandbox/{sandbox_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Sandbox terminated: {data}")
        return True
    else:
        print(f"❌ Failed to terminate sandbox: {response.text}")
        return False


def run_all_tests():
    """Run all E2B integration tests"""
    print("=" * 60)
    print("E2B Integration Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test health check
    results.append(("Health Check", test_health_check()))
    
    # Test sandbox creation
    sandbox_id = test_create_sandbox()
    results.append(("Create Sandbox", sandbox_id is not None))
    
    if sandbox_id:
        # Test command execution
        results.append(("Execute Command", test_execute_command(sandbox_id)))
        
        # Test sandbox termination
        results.append(("Terminate Sandbox", test_cleanup_sandbox(sandbox_id)))
    
    # Test agent execution
    results.append(("Run Agent", test_run_agent()))
    
    # Test script execution
    results.append(("Execute Script", test_execute_script()))
    
    # Test listing sandboxes
    results.append(("List Sandboxes", test_list_sandboxes()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code != 200:
            print("❌ FastAPI server is not running on port 8000")
            print("   Please start the server first: python src/main.py")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to FastAPI server on port 8000")
        print("   Please start the server first: python src/main.py")
        exit(1)
    
    # Run tests
    success = run_all_tests()
    exit(0 if success else 1)