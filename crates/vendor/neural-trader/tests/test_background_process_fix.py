#!/usr/bin/env python3
"""
Test background process fix for complex commands
"""

import time
import json
import requests

BASE_URL = "http://localhost:8000"

def test_background_processes():
    """Test various background process scenarios"""
    print("=" * 60)
    print("BACKGROUND PROCESS TESTING")
    print("=" * 60)
    
    # Create a sandbox
    print("\n1. Creating test sandbox...")
    sandbox_response = requests.post(f"{BASE_URL}/e2b/sandbox/create", json={
        "name": "bg_test",
        "timeout": 120,
        "memory_mb": 512,
        "cpu_count": 1,
        "allow_internet": False,
        "envs": {},
        "metadata": {"test": "background"}
    })
    
    if sandbox_response.status_code != 200:
        print(f"❌ Failed to create sandbox: {sandbox_response.text}")
        return False
    
    sandbox_id = sandbox_response.json()["sandbox_id"]
    print(f"✅ Sandbox created: {sandbox_id}")
    
    all_tests_passed = True
    
    # Test 1: Simple background command
    print("\n2. Testing simple background command...")
    simple_response = requests.post(
        f"{BASE_URL}/e2b/process/background",
        json={
            "command": "sleep",
            "args": ["2"],
            "working_dir": "/tmp",
            "env_vars": {},
            "capture_output": True,
            "timeout": 60
        },
        params={"sandbox_id": sandbox_id}
    )
    
    if simple_response.status_code == 200:
        process_id = simple_response.json()["process_id"]
        print(f"✅ Simple background process started: {process_id}")
    else:
        print(f"❌ Simple background failed: {simple_response.text}")
        all_tests_passed = False
    
    # Test 2: Python command with quotes
    print("\n3. Testing Python command with quotes...")
    python_response = requests.post(
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
    
    if python_response.status_code == 200:
        process_id = python_response.json()["process_id"]
        print(f"✅ Python background process started: {process_id}")
        
        # Wait and check status
        time.sleep(3)
        status_response = requests.get(f"{BASE_URL}/e2b/process/{process_id}/status")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"   Process status: {status.get('status', 'unknown')}")
    else:
        print(f"❌ Python background failed: {python_response.text}")
        all_tests_passed = False
    
    # Test 3: Complex command with special characters
    print("\n4. Testing complex command with special characters...")
    complex_response = requests.post(
        f"{BASE_URL}/e2b/process/background",
        json={
            "command": "bash",
            "args": ["-c", "echo 'Test with $VAR and \"quotes\" and spaces' > /tmp/test.txt; sleep 1"],
            "working_dir": "/tmp",
            "env_vars": {"VAR": "test_value"},
            "capture_output": True,
            "timeout": 60
        },
        params={"sandbox_id": sandbox_id}
    )
    
    if complex_response.status_code == 200:
        process_id = complex_response.json()["process_id"]
        print(f"✅ Complex background process started: {process_id}")
        
        # Wait and verify file was created
        time.sleep(2)
        verify_response = requests.post(
            f"{BASE_URL}/e2b/sandbox/{sandbox_id}/execute",
            params={"command": "cat /tmp/test.txt", "timeout": 10}
        )
        if verify_response.status_code == 200:
            content = verify_response.json().get("stdout", "").strip()
            if content:
                print(f"   File content: {content}")
    else:
        print(f"❌ Complex background failed: {complex_response.text}")
        all_tests_passed = False
    
    # Test 4: JSON output command
    print("\n5. Testing JSON output background command...")
    json_response = requests.post(
        f"{BASE_URL}/e2b/process/background",
        json={
            "command": "python",
            "args": ["-c", """
import json
import time
time.sleep(1)
data = {"status": "complete", "value": 42}
with open('/tmp/output.json', 'w') as f:
    json.dump(data, f)
"""],
            "working_dir": "/tmp",
            "env_vars": {},
            "capture_output": True,
            "timeout": 60
        },
        params={"sandbox_id": sandbox_id}
    )
    
    if json_response.status_code == 200:
        process_id = json_response.json()["process_id"]
        print(f"✅ JSON output process started: {process_id}")
        
        # Wait and read the output
        time.sleep(2)
        verify_response = requests.post(
            f"{BASE_URL}/e2b/sandbox/{sandbox_id}/execute",
            params={"command": "cat /tmp/output.json", "timeout": 10}
        )
        if verify_response.status_code == 200:
            output = verify_response.json().get("stdout", "")
            if output:
                try:
                    data = json.loads(output)
                    print(f"   JSON data: {data}")
                except:
                    print(f"   Raw output: {output}")
    else:
        print(f"❌ JSON output failed: {json_response.text}")
        all_tests_passed = False
    
    # Test 5: List all processes
    print("\n6. Testing process listing...")
    list_response = requests.get(f"{BASE_URL}/e2b/process/list")
    
    if list_response.status_code == 200:
        processes = list_response.json()
        print(f"✅ Active processes: {len(processes)}")
        for proc in processes[:3]:  # Show first 3
            print(f"   - {proc.get('process_id')}: {proc.get('command', 'unknown')[:50]}...")
    else:
        print(f"❌ Process listing failed: {list_response.text}")
        all_tests_passed = False
    
    # Clean up
    print("\n7. Cleaning up...")
    cleanup_response = requests.delete(f"{BASE_URL}/e2b/sandbox/{sandbox_id}")
    
    if cleanup_response.status_code == 200:
        print(f"✅ Sandbox {sandbox_id} terminated")
    else:
        print(f"❌ Cleanup failed: {cleanup_response.text}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ All background process tests PASSED!")
    else:
        print("❌ Some background process tests FAILED")
    print("=" * 60)
    
    return all_tests_passed

if __name__ == "__main__":
    success = test_background_processes()
    exit(0 if success else 1)