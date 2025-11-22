#!/usr/bin/env python3
"""
Verify CWTS Ultra FreqTrade integration setup
"""

import sys
import os
import importlib.util

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_module(module_name, file_path=None):
    """Check if a module can be imported"""
    try:
        if file_path and os.path.exists(file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True, "File exists and loads"
        else:
            __import__(module_name)
            return True, "Importable"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

print("=" * 60)
print("CWTS Ultra FreqTrade Integration Verification")
print("=" * 60)

# Check directory structure
print("\n📁 Directory Structure:")
base_dir = os.path.dirname(os.path.abspath(__file__))
strategies_dir = os.path.join(base_dir, "strategies")

files_to_check = [
    ("cwts_client_simple.py", os.path.join(base_dir, "cwts_client_simple.py")),
    ("CWTSUltraStrategy.py", os.path.join(strategies_dir, "CWTSUltraStrategy.py")),
    ("CWTSMomentumStrategy.py", os.path.join(strategies_dir, "CWTSMomentumStrategy.py")),
]

all_files_exist = True
for name, path in files_to_check:
    exists = os.path.exists(path)
    all_files_exist = all_files_exist and exists
    status = "✅" if exists else "❌"
    print(f"  {status} {name}: {'Found' if exists else 'Missing'}")
    if exists:
        print(f"     Path: {path}")

# Check imports
print("\n📦 Module Imports:")

# First add strategies directory to path
sys.path.insert(0, strategies_dir)

# Test client import
print("\n  Testing cwts_client_simple...")
success, msg = check_module("cwts_client_simple", os.path.join(base_dir, "cwts_client_simple.py"))
print(f"  {'✅' if success else '❌'} cwts_client_simple: {msg}")

# Test strategy imports
print("\n  Testing strategies...")
success, msg = check_module("CWTSUltraStrategy", os.path.join(strategies_dir, "CWTSUltraStrategy.py"))
print(f"  {'✅' if success else '❌'} CWTSUltraStrategy: {msg}")

success, msg = check_module("CWTSMomentumStrategy", os.path.join(strategies_dir, "CWTSMomentumStrategy.py"))
print(f"  {'✅' if success else '❌'} CWTSMomentumStrategy: {msg}")

# Check dependencies
print("\n📚 Dependencies:")
dependencies = [
    "numpy",
    "pandas",
    "websockets",
    "msgpack",
    "aiofiles",
]

all_deps_installed = True
for dep in dependencies:
    try:
        __import__(dep)
        print(f"  ✅ {dep}: Installed")
    except ImportError:
        print(f"  ❌ {dep}: Not installed")
        all_deps_installed = False

# Check FreqTrade
print("\n🤖 FreqTrade Check:")
try:
    import freqtrade
    print(f"  ✅ FreqTrade installed: {freqtrade.__version__}")
    
    # Try to import with FreqTrade context
    from freqtrade.strategy import IStrategy
    print(f"  ✅ FreqTrade IStrategy importable")
except ImportError:
    print("  ⚠️  FreqTrade not found in current environment")
    print("     This is OK if you're running this outside FreqTrade's environment")

# Summary
print("\n" + "=" * 60)
print("📊 Summary:")
print("=" * 60)

if all_files_exist and all_deps_installed:
    print("✅ All files and dependencies are in place!")
    print("\n🎯 Next Steps:")
    print("1. Run the setup script: ./setup_freqtrade.sh")
    print("2. The script will:")
    print("   - Find your FreqTrade installation")
    print("   - Copy/link the strategy files")
    print("   - Create configuration templates")
    print("   - Set up launcher scripts")
    print("\n3. Then you can run:")
    print("   freqtrade trade --strategy CWTSMomentumStrategy")
else:
    print("⚠️  Some issues detected. Please check above.")
    if not all_files_exist:
        print("\n❌ Missing files - please ensure all files are present")
    if not all_deps_installed:
        print("\n❌ Missing dependencies - install with:")
        print("   pip install numpy pandas websockets msgpack-python aiofiles")

print("\n" + "=" * 60)