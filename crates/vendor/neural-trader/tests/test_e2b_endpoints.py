#!/usr/bin/env python3
"""
Comprehensive E2B endpoint testing
"""

import time
import json
import requests
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"

def test_endpoint(method: str, endpoint: str, data: Any = None, params: Any = None) -> Dict[str, Any]:
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data, params=params)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        return {
            "status_code": response.status_code,
            "success": response.status_code in [200, 201],
            "data": response.json() if response.text else None,
            "error": None if response.status_code in [200, 201] else response.text
        }
    except Exception as e:
        return {
            "status_code": 0,
            "success": False,
            "data": None,
            "error": str(e)
        }

def run_all_tests():
    """Run comprehensive endpoint tests"""
    results = {}
    
    print("=" * 60)
    print("E2B ENDPOINT TESTING")
    print("=" * 60)
    
    # 1. Health Check
    print("\n1. Testing Health Check...")
    results["health"] = test_endpoint("GET", "/e2b/health")
    print(f"   Status: {results['health']['status_code']}")
    if results['health']['success']:
        print(f"   Data: {results['health']['data']}")
    
    # 2. List Agent Types
    print("\n2. Testing Agent Types...")
    results["agent_types"] = test_endpoint("GET", "/e2b/agent/types")
    print(f"   Status: {results['agent_types']['status_code']}")
    if results['agent_types']['success']:
        print(f"   Available agents: {results['agent_types']['data']}")
    
    # 3. List Sandboxes (should be empty initially)
    print("\n3. Testing List Sandboxes...")
    results["list_sandboxes"] = test_endpoint("GET", "/e2b/sandbox/list")
    print(f"   Status: {results['list_sandboxes']['status_code']}")
    if results['list_sandboxes']['success']:
        print(f"   Active sandboxes: {len(results['list_sandboxes']['data'])}")
    
    # 4. Create Sandbox (simplified)
    print("\n4. Testing Create Sandbox (simplified)...")
    sandbox_config = {
        "name": "test_minimal",
        "timeout": 60,  # Shorter timeout
        "memory_mb": 256,
        "cpu_count": 1,
        "allow_internet": False,  # No internet to speed up
        "envs": {},
        "metadata": {"test": "minimal"}
    }
    results["create_sandbox"] = test_endpoint("POST", "/e2b/sandbox/create", sandbox_config)
    print(f"   Status: {results['create_sandbox']['status_code']}")
    
    sandbox_id = None
    if results['create_sandbox']['success']:
        sandbox_id = results['create_sandbox']['data'].get('sandbox_id')
        print(f"   Sandbox ID: {sandbox_id}")
    else:
        print(f"   Error: {results['create_sandbox']['error'][:100]}...")
    
    # 5. Execute Command (if sandbox created)
    if sandbox_id:
        print("\n5. Testing Command Execution...")
        command_params = {
            "command": "echo 'Testing E2B' && pwd",
            "timeout": 10
        }
        results["execute_command"] = test_endpoint(
            "POST", 
            f"/e2b/sandbox/{sandbox_id}/execute",
            params=command_params
        )
        print(f"   Status: {results['execute_command']['status_code']}")
        if results['execute_command']['success']:
            print(f"   Output: {results['execute_command']['data'].get('stdout', '')}")
    
    # 6. Script Execution
    print("\n6. Testing Script Execution...")
    script_data = {
        "script_content": "print('Hello from E2B!')\nprint(2 + 2)",
        "language": "python"
    }
    results["execute_script"] = test_endpoint("POST", "/e2b/script/execute", script_data)
    print(f"   Status: {results['execute_script']['status_code']}")
    if results['execute_script']['success']:
        output = results['execute_script']['data'].get('stdout', '')
        print(f"   Output: {output}")
    
    # 7. Run Agent
    print("\n7. Testing Agent Execution...")
    agent_config = {
        "agent_type": "news_analyzer",
        "symbols": ["AAPL"],
        "strategy_params": {},
        "execution_mode": "simulation",
        "use_gpu": False
    }
    results["run_agent"] = test_endpoint("POST", "/e2b/agent/run", agent_config)
    print(f"   Status: {results['run_agent']['status_code']}")
    if results['run_agent']['success']:
        print(f"   Agent status: {results['run_agent']['data'].get('status')}")
    
    # 8. Process Execution
    print("\n8. Testing Process Execution...")
    process_config = {
        "command": "ls",
        "args": ["-la", "/tmp"],
        "working_dir": "/tmp",
        "env_vars": {},
        "capture_output": True,
        "timeout": 10
    }
    results["execute_process"] = test_endpoint("POST", "/e2b/process/execute", process_config)
    print(f"   Status: {results['execute_process']['status_code']}")
    
    # 9. List Active Processes
    print("\n9. Testing List Processes...")
    results["list_processes"] = test_endpoint("GET", "/e2b/process/list")
    print(f"   Status: {results['list_processes']['status_code']}")
    if results['list_processes']['success']:
        print(f"   Active processes: {len(results['list_processes']['data'])}")
    
    # 10. Cleanup Sandbox (if created)
    if sandbox_id:
        print("\n10. Testing Sandbox Termination...")
        results["terminate_sandbox"] = test_endpoint("DELETE", f"/e2b/sandbox/{sandbox_id}")
        print(f"   Status: {results['terminate_sandbox']['status_code']}")
    
    # 11. Batch Agent Execution
    print("\n11. Testing Batch Agent Execution...")
    batch_configs = [
        {
            "agent_type": "momentum_trader",
            "symbols": ["AAPL"],
            "strategy_params": {},
            "execution_mode": "simulation",
            "use_gpu": False
        },
        {
            "agent_type": "risk_manager",
            "symbols": [],
            "strategy_params": {},
            "execution_mode": "simulation",
            "use_gpu": False
        }
    ]
    results["run_batch_agents"] = test_endpoint("POST", "/e2b/agent/run-batch", batch_configs)
    print(f"   Status: {results['run_batch_agents']['status_code']}")
    if results['run_batch_agents']['success']:
        print(f"   Agents run: {len(results['run_batch_agents']['data'])}")
    
    # 12. Cleanup All
    print("\n12. Testing Cleanup All...")
    results["cleanup_all"] = test_endpoint("DELETE", "/e2b/cleanup")
    print(f"   Status: {results['cleanup_all']['status_code']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r['success'])
    
    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        code = result['status_code']
        print(f"{status} {name:.<40} [{code}]")
    
    print(f"\nTotal: {passed}/{total} endpoints working")
    
    if passed == total:
        print("\n🎉 All endpoints are working!")
    else:
        print(f"\n⚠️  {total - passed} endpoint(s) have issues")
        
        # Show errors
        print("\nErrors:")
        for name, result in results.items():
            if not result['success'] and result['error']:
                error_msg = result['error']
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                print(f"\n{name}:")
                print(f"  {error_msg}")
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(r['success'] for r in results.values())
    exit(0 if all_passed else 1)