#!/usr/bin/env python3
"""
Verification script for Trading API Base Infrastructure

This script verifies that all components are correctly implemented
and can be imported without errors.
"""

import sys
import traceback
from pathlib import Path

def verify_imports():
    """Verify all components can be imported"""
    print("🔍 Verifying imports...")
    
    try:
        # Test core components
        from api_interface import TradingAPIInterface, OrderRequest, OrderResponse, MarketData
        print("✅ API Interface imported successfully")
        
        from connection_pool import ConnectionPool, PooledConnection, ConnectionState
        print("✅ Connection Pool imported successfully")
        
        from latency_monitor import LatencyMonitor, LatencyMeasurement, LatencyLevel
        print("✅ Latency Monitor imported successfully")
        
        from config_loader import ConfigLoader, APIConfig, TradingConfig
        print("✅ Config Loader imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def verify_structure():
    """Verify directory structure"""
    print("\n🏗️  Verifying structure...")
    
    base_dir = Path(__file__).parent
    required_files = [
        "__init__.py",
        "api_interface.py",
        "connection_pool.py",
        "latency_monitor.py",
        "config_loader.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not (base_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def verify_config():
    """Verify configuration file exists"""
    print("\n⚙️  Verifying configuration...")
    
    config_file = Path("/workspaces/ai-news-trader/config/trading_apis.yaml")
    if not config_file.exists():
        print("❌ Configuration file missing")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['global_settings', 'connection_pool', 'monitoring', 'apis']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
        
        print("✅ Configuration file valid")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def verify_dependencies():
    """Verify dependencies are available"""
    print("\n📦 Verifying dependencies...")
    
    required_deps = [
        'asyncio',
        'typing',
        'dataclasses',
        'datetime',
        'threading',
        'concurrent.futures',
        'collections',
        'enum',
        'json',
        'hashlib',
        'pathlib',
        'os',
        'time',
        'yaml',
        'numpy'
    ]
    
    missing_deps = []
    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        return False
    else:
        print("✅ All dependencies available")
        return True

def main():
    """Main verification function"""
    print("🚀 Trading API Base Infrastructure Verification\n")
    
    checks = [
        verify_structure,
        verify_imports,
        verify_config,
        verify_dependencies
    ]
    
    passed = 0
    for check in checks:
        if check():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(checks)} checks passed")
    
    if passed == len(checks):
        print("✅ All verifications passed! Infrastructure is ready.")
        return 0
    else:
        print("❌ Some verifications failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())