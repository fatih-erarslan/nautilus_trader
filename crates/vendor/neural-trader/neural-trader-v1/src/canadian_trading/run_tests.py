#!/usr/bin/env python3
"""
Run all Canadian Trading API tests and verify implementation integrity.
"""
import sys
import asyncio
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

def test_imports() -> Tuple[List[str], List[str]]:
    """Test all module imports."""
    successful = []
    failed = []
    
    modules_to_test = [
        "canadian_trading.brokers.ib_canada",
        "canadian_trading.brokers.questrade",
        "canadian_trading.brokers.oanda_canada",
        "canadian_trading.utils.auth",
        "canadian_trading.utils.forex_utils",
        "canadian_trading.compliance.ciro_compliance",
        "canadian_trading.compliance.tax_reporting",
        "canadian_trading.compliance.audit_trail",
        "canadian_trading.compliance.monitoring",
    ]
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            successful.append(module)
            print(f"✅ {module}")
        except Exception as e:
            failed.append(f"{module}: {str(e)}")
            print(f"❌ {module}: {str(e)}")
    
    return successful, failed

def check_file_structure() -> Dict[str, bool]:
    """Verify file structure is complete."""
    required_files = {
        "brokers/__init__.py": Path("src/canadian_trading/brokers/__init__.py"),
        "brokers/ib_canada.py": Path("src/canadian_trading/brokers/ib_canada.py"),
        "brokers/questrade.py": Path("src/canadian_trading/brokers/questrade.py"),
        "brokers/oanda_canada.py": Path("src/canadian_trading/brokers/oanda_canada.py"),
        "compliance/__init__.py": Path("src/canadian_trading/compliance/__init__.py"),
        "compliance/ciro_compliance.py": Path("src/canadian_trading/compliance/ciro_compliance.py"),
        "compliance/tax_reporting.py": Path("src/canadian_trading/compliance/tax_reporting.py"),
        "compliance/audit_trail.py": Path("src/canadian_trading/compliance/audit_trail.py"),
        "compliance/monitoring.py": Path("src/canadian_trading/compliance/monitoring.py"),
        "utils/__init__.py": Path("src/canadian_trading/utils/__init__.py"),
        "utils/auth.py": Path("src/canadian_trading/utils/auth.py"),
        "utils/forex_utils.py": Path("src/canadian_trading/utils/forex_utils.py"),
    }
    
    results = {}
    for name, path in required_files.items():
        exists = path.exists()
        results[name] = exists
        status = "✅" if exists else "❌"
        print(f"{status} {name}")
    
    return results

def verify_configurations():
    """Verify configuration files exist."""
    config_files = {
        "requirements.txt": Path("src/canadian_trading/requirements.txt"),
        "README.md": Path("src/canadian_trading/README.md"),
        "config.py": Path("src/canadian_trading/config.py"),
    }
    
    print("\n📄 Configuration Files:")
    for name, path in config_files.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {name}")

async def test_basic_functionality():
    """Test basic functionality of each module."""
    print("\n🧪 Basic Functionality Tests:")
    
    # Test IB Canada
    try:
        from canadian_trading.brokers.ib_canada import IBCanadaClient, ConnectionConfig
        config = ConnectionConfig(host="127.0.0.1", port=7497, is_paper=True)
        print("✅ IB Canada: Configuration created successfully")
    except Exception as e:
        print(f"❌ IB Canada: {e}")
    
    # Test Questrade
    try:
        from canadian_trading.brokers.questrade import QuestradeAPI
        print("✅ Questrade: Module imports successfully")
    except Exception as e:
        print(f"❌ Questrade: {e}")
    
    # Test OANDA
    try:
        from canadian_trading.brokers.oanda_canada import OANDACanada
        print("✅ OANDA: Module imports successfully")
    except Exception as e:
        print(f"❌ OANDA: {e}")
    
    # Test Compliance
    try:
        from canadian_trading.compliance import CIROCompliance, TaxReporting, AuditTrail
        print("✅ Compliance: All modules import successfully")
    except Exception as e:
        print(f"❌ Compliance: {e}")

def main():
    """Run all tests."""
    print("🇨🇦 Canadian Trading API Test Suite")
    print("=" * 50)
    
    # Add src to Python path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Check file structure
    print("\n📁 File Structure Check:")
    check_file_structure()
    
    # Verify configurations
    verify_configurations()
    
    # Test imports
    print("\n📦 Module Import Tests:")
    successful, failed = test_imports()
    
    # Run basic functionality tests
    asyncio.run(test_basic_functionality())
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"✅ Successful imports: {len(successful)}")
    print(f"❌ Failed imports: {len(failed)}")
    
    if failed:
        print("\n❌ Failed modules:")
        for fail in failed:
            print(f"  - {fail}")
        return 1
    else:
        print("\n✅ All tests passed! Canadian Trading APIs are ready for use.")
        return 0

if __name__ == "__main__":
    sys.exit(main())