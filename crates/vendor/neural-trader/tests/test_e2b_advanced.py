#!/usr/bin/env python3
"""
Advanced E2B endpoint testing - file operations, pipelines, etc.
"""

import time
import json
import requests
import tempfile
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_advanced_features():
    """Test advanced E2B features"""
    print("=" * 60)
    print("ADVANCED E2B FEATURE TESTING")
    print("=" * 60)
    
    # 1. Create a sandbox for testing
    print("\n1. Creating test sandbox...")
    sandbox_response = requests.post(f"{BASE_URL}/e2b/sandbox/create", json={
        "name": "advanced_test",
        "timeout": 120,
        "memory_mb": 512,
        "cpu_count": 1,
        "allow_internet": True,
        "envs": {"TEST_MODE": "advanced"},
        "metadata": {"type": "testing"}
    })
    
    if sandbox_response.status_code != 200:
        print(f"❌ Failed to create sandbox: {sandbox_response.text}")
        return
    
    sandbox_id = sandbox_response.json()["sandbox_id"]
    print(f"✅ Sandbox created: {sandbox_id}")
    
    # 2. Test file upload
    print("\n2. Testing file upload...")
    test_content = "This is a test file for E2B upload"
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(test_content)
        temp_file = f.name
    
    upload_response = requests.post(
        f"{BASE_URL}/e2b/sandbox/{sandbox_id}/upload",
        params={
            "local_path": temp_file,
            "sandbox_path": "/tmp/uploaded_test.txt"
        }
    )
    print(f"   Upload status: {upload_response.status_code}")
    
    # 3. Test file download
    print("\n3. Testing file download...")
    download_response = requests.get(
        f"{BASE_URL}/e2b/sandbox/{sandbox_id}/download",
        params={"sandbox_path": "/tmp/uploaded_test.txt"}
    )
    
    if download_response.status_code == 200:
        content = download_response.json().get("content", "")
        print(f"✅ Downloaded content: {content}")
    else:
        print(f"❌ Download failed: {download_response.text}")
    
    # 4. Test pipeline execution
    print("\n4. Testing pipeline execution...")
    pipeline_configs = [
        {
            "command": "echo",
            "args": ["Starting pipeline"],
            "working_dir": "/tmp",
            "env_vars": {},
            "capture_output": True,
            "timeout": 10
        },
        {
            "command": "python",
            "args": ["-c", "print('Pipeline step 2: Python')"],
            "working_dir": "/tmp",
            "env_vars": {},
            "capture_output": True,
            "timeout": 10
        },
        {
            "command": "echo",
            "args": ["Pipeline complete"],
            "working_dir": "/tmp",
            "env_vars": {},
            "capture_output": True,
            "timeout": 10
        }
    ]
    
    pipeline_response = requests.post(
        f"{BASE_URL}/e2b/process/pipeline",
        json=pipeline_configs,
        params={"sandbox_id": sandbox_id}
    )
    
    if pipeline_response.status_code == 200:
        results = pipeline_response.json()
        print(f"✅ Pipeline executed: {len(results)} steps")
        for i, result in enumerate(results, 1):
            output = result.get("stdout", "").strip()
            print(f"   Step {i}: {output}")
    else:
        print(f"❌ Pipeline failed: {pipeline_response.text}")
    
    # 5. Test background process
    print("\n5. Testing background process...")
    bg_process_response = requests.post(
        f"{BASE_URL}/e2b/process/background",
        json={
            "command": "python",
            "args": ["-c", "import time; time.sleep(2); print('Background complete')"],
            "working_dir": "/tmp",
            "env_vars": {},
            "capture_output": True,
            "timeout": 60
        },
        params={"sandbox_id": sandbox_id}
    )
    
    if bg_process_response.status_code == 200:
        process_id = bg_process_response.json()["process_id"]
        print(f"✅ Background process started: {process_id}")
        
        # Check process status
        time.sleep(1)
        status_response = requests.get(f"{BASE_URL}/e2b/process/{process_id}/status")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"   Process status: {status.get('status', 'unknown')}")
    else:
        print(f"❌ Background process failed: {bg_process_response.text}")
    
    # 6. Test complex script execution
    print("\n6. Testing complex script execution...")
    complex_script = '''
import json
import random
from datetime import datetime

# Generate mock trading data
data = {
    "timestamp": datetime.now().isoformat(),
    "trades": [
        {"symbol": "AAPL", "action": "buy", "price": random.uniform(150, 200)},
        {"symbol": "GOOGL", "action": "sell", "price": random.uniform(100, 150)}
    ],
    "summary": {
        "total_trades": 2,
        "profit": random.uniform(-100, 500)
    }
}

print(json.dumps(data, indent=2))
'''
    
    script_response = requests.post(
        f"{BASE_URL}/e2b/script/execute",
        json={
            "script_content": complex_script,
            "language": "python",
            "sandbox_id": sandbox_id
        }
    )
    
    if script_response.status_code == 200:
        output = script_response.json().get("stdout", "")
        print(f"✅ Complex script executed")
        if output:
            try:
                data = json.loads(output)
                print(f"   Generated {len(data.get('trades', []))} trades")
                print(f"   Profit: ${data.get('summary', {}).get('profit', 0):.2f}")
            except:
                print(f"   Output: {output[:100]}...")
    else:
        print(f"❌ Script failed: {script_response.text}")
    
    # 7. Test batch process execution
    print("\n7. Testing batch process execution...")
    batch_configs = [
        {
            "command": "echo",
            "args": ["Process 1"],
            "working_dir": "/tmp",
            "env_vars": {},
            "capture_output": True,
            "timeout": 10
        },
        {
            "command": "echo",
            "args": ["Process 2"],
            "working_dir": "/tmp",
            "env_vars": {},
            "capture_output": True,
            "timeout": 10
        }
    ]
    
    batch_response = requests.post(
        f"{BASE_URL}/e2b/process/batch",
        json=batch_configs,
        params={"parallel": True}
    )
    
    if batch_response.status_code == 200:
        results = batch_response.json()
        print(f"✅ Batch executed: {len(results)} processes")
    else:
        print(f"❌ Batch failed: {batch_response.text}")
    
    # 8. Test sandbox info retrieval
    print("\n8. Testing sandbox info retrieval...")
    info_response = requests.get(f"{BASE_URL}/e2b/sandbox/{sandbox_id}")
    
    if info_response.status_code == 200:
        info = info_response.json()
        print(f"✅ Sandbox info retrieved")
        print(f"   Name: {info.get('name')}")
        print(f"   Status: {info.get('status')}")
        print(f"   Created: {info.get('created_at')}")
    else:
        print(f"❌ Info retrieval failed: {info_response.text}")
    
    # 9. Test multiple agent types
    print("\n9. Testing different agent types...")
    agent_types = ["momentum_trader", "mean_reversion_trader", "neural_forecaster", "risk_manager"]
    
    for agent_type in agent_types:
        agent_response = requests.post(
            f"{BASE_URL}/e2b/agent/run",
            json={
                "agent_type": agent_type,
                "symbols": ["AAPL"] if agent_type != "risk_manager" else [],
                "strategy_params": {},
                "execution_mode": "simulation",
                "use_gpu": False
            }
        )
        
        status = "✅" if agent_response.status_code == 200 else "❌"
        result = agent_response.json() if agent_response.status_code == 200 else {}
        agent_status = result.get("status", "failed")
        print(f"   {status} {agent_type}: {agent_status}")
    
    # 10. Clean up
    print("\n10. Cleaning up test sandbox...")
    cleanup_response = requests.delete(f"{BASE_URL}/e2b/sandbox/{sandbox_id}")
    
    if cleanup_response.status_code == 200:
        print(f"✅ Sandbox {sandbox_id} terminated")
    else:
        print(f"❌ Cleanup failed: {cleanup_response.text}")
    
    print("\n" + "=" * 60)
    print("Advanced testing complete!")

if __name__ == "__main__":
    test_advanced_features()