#!/usr/bin/env python3
"""
DIAGNOSTIC TEST: Verify Pairlist-CDFA Integration
This script tests the complete data flow from Data App -> CDFA -> Pairlist
to identify the exact failure point.
"""
import asyncio
import httpx
import json
import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent / "tengri" / "pairlist_app"))

def test_data_app_connection():
    """Test 1: Verify Data App is responding"""
    print("🔍 Test 1: Data App Connection")
    try:
        import requests
        response = requests.get("http://localhost:8010/api/v1/symbols", timeout=10)
        if response.status_code == 200:
            data = response.json()
            symbols = data.get('symbols', [])
            print(f"✅ Data App responding: {len(symbols)} symbols available")
            return True, symbols[:5]  # Return first 5 for testing
        else:
            print(f"❌ Data App error: HTTP {response.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ Data App connection failed: {e}")
        return False, []

def test_cdfa_api_connection():
    """Test 2: Verify CDFA API is responding with real analysis"""
    print("\n🔍 Test 2: CDFA API Connection")
    try:
        import requests
        response = requests.post(
            "http://localhost:8020/api/v1/signals/batch",
            json={"symbols": ["BTC/USDT", "ETH/USDT"]},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            signals = data.get('signals', [])
            if signals:
                first_signal = signals[0]
                print(f"✅ CDFA API responding: {len(signals)} signals")
                print(f"   Sample: {first_signal.get('symbol')} = signal:{first_signal.get('signal'):.3f}, strength:{first_signal.get('strength'):.3f}")
                # Check if it's real analysis (not mock data)
                if first_signal.get('signal') != 0.5 or first_signal.get('strength') != 0.5:
                    print("✅ Real CDFA analysis detected (non-default values)")
                    return True, signals
                else:
                    print("⚠️ Possible mock data detected (default 0.5 values)")
                    return False, signals
            else:
                print("❌ No signals in CDFA response")
                return False, []
        else:
            print(f"❌ CDFA API error: HTTP {response.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ CDFA API connection failed: {e}")
        return False, []

def test_pairlist_update_mechanism():
    """Test 3: Test pairlist update endpoint"""
    print("\n🔍 Test 3: Pairlist Update Mechanism")
    try:
        import requests
        response = requests.post("http://localhost:8030/pairlists/default/update", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Pairlist update endpoint responding: {data}")
            return True, data
        else:
            print(f"❌ Pairlist update error: HTTP {response.status_code}")
            return False, {}
    except Exception as e:
        print(f"❌ Pairlist update failed: {e}")
        return False, {}

def test_pairlist_data_quality():
    """Test 4: Check pairlist data quality"""
    print("\n🔍 Test 4: Pairlist Data Quality")
    try:
        import requests
        response = requests.get("http://localhost:8030/pairlists/default/pairs?page=1&page_size=3", timeout=10)
        if response.status_code == 200:
            data = response.json()
            metadata = data.get('metadata', {})
            if metadata:
                # Check first pair for CDFA analysis data
                first_pair_key = list(metadata.keys())[0]
                first_pair = metadata[first_pair_key]
                print(f"✅ Pairlist responding with metadata for {len(metadata)} pairs")
                print(f"   Sample pair: {first_pair_key}")
                
                # Check for CDFA analysis fields
                cdfa_fields = ['cdfa_signal', 'cdfa_strength', 'cdfa_confidence', 'opportunity_score']
                has_cdfa = all(field in first_pair for field in cdfa_fields)
                
                if has_cdfa:
                    print(f"✅ CDFA analysis present: signal={first_pair.get('cdfa_signal')}, strength={first_pair.get('cdfa_strength')}")
                    # Check if values are real (not defaults)
                    if (first_pair.get('cdfa_strength', 0) != 0.5 or 
                        first_pair.get('opportunity_score', 0) not in [4.0, 5.0]):
                        print("✅ Real CDFA analysis data detected")
                        return True, "real_data"
                    else:
                        print("⚠️ Default/mock CDFA values detected")
                        return True, "mock_data"
                else:
                    print("❌ No CDFA analysis fields in pairlist data")
                    return False, "no_cdfa"
            else:
                print("❌ No metadata in pairlist response")
                return False, "no_metadata"
        else:
            print(f"❌ Pairlist data error: HTTP {response.status_code}")
            return False, "api_error"
    except Exception as e:
        print(f"❌ Pairlist data check failed: {e}")
        return False, "exception"

def test_pairlist_json_file():
    """Test 5: Check the pairlist.json file directly"""
    print("\n🔍 Test 5: Pairlist JSON File")
    try:
        pairlist_path = Path("user_data/pairlist.json")
        if pairlist_path.exists():
            with open(pairlist_path, 'r') as f:
                data = json.load(f)
            
            print(f"✅ Pairlist file exists with {len(data.get('pairs', []))} pairs")
            
            if 'metadata' in data and data['metadata']:
                print(f"✅ Metadata present for {len(data['metadata'])} pairs")
                # Check for CDFA fields in metadata
                first_pair_key = list(data['metadata'].keys())[0]
                first_pair = data['metadata'][first_pair_key]
                has_cdfa = 'cdfa_signal' in first_pair
                print(f"   CDFA analysis in file: {'✅ Yes' if has_cdfa else '❌ No'}")
                return True, data
            else:
                print("❌ No metadata in pairlist file")
                return False, data
        else:
            print("❌ Pairlist file does not exist")
            return False, {}
    except Exception as e:
        print(f"❌ Pairlist file check failed: {e}")
        return False, {}

def main():
    """Run complete diagnostic test suite"""
    print("🚀 TENGRI SYSTEM DIAGNOSTIC TEST")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Data App
    results['data_app'] = test_data_app_connection()
    
    # Test 2: CDFA API
    results['cdfa_api'] = test_cdfa_api_connection()
    
    # Test 3: Pairlist Update
    results['pairlist_update'] = test_pairlist_update_mechanism()
    
    # Test 4: Pairlist Data
    results['pairlist_data'] = test_pairlist_data_quality()
    
    # Test 5: Pairlist File
    results['pairlist_file'] = test_pairlist_json_file()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    all_working = True
    for test_name, (success, _) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20}: {status}")
        if not success:
            all_working = False
    
    if all_working:
        print("\n🎉 ALL TESTS PASSED - System working correctly!")
    else:
        print("\n❌ SYSTEM ISSUES DETECTED")
        print("\nNext steps:")
        if not results['data_app'][0]:
            print("1. Start data app: cd tengri/data_app && python server.py")
        if not results['cdfa_api'][0]:
            print("2. Fix CDFA API or start server: cd tengri/cdfa_app && python server.py")
        if not results['pairlist_update'][0]:
            print("3. Fix pairlist server: cd tengri/pairlist_app && python server.py")
        if not results['pairlist_data'][0] or not results['pairlist_file'][0]:
            print("4. Fix pairlist-CDFA integration in core.py")
    
    return results

if __name__ == "__main__":
    main()