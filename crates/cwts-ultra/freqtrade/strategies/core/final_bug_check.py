#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final comprehensive bug check for the Integrated Quantum Trading System
"""

import sys
import traceback

def test_imports():
    """Test all key imports."""
    print("Testing imports...")
    
    tests = [
        ("Unified Messaging", "from unified_messaging import UnifiedMessenger, Message, MessageType, AgentType"),
        ("PADS Messaging", "from pads_messaging_integration import PADSMessagingIntegration"),
        ("QUASAR Adapter", "from quasar_messaging_adapter import QUASARMessagingAdapter"),
        ("Quantum AMOS Adapter", "from quantum_amos_messaging_adapter import QuantumAMOSMessagingAdapter"),
        ("QBMIA Connector", "from qbmia.integration.pads_connector import PADSConnector"),
        ("Main System", "from integrated_quantum_trading_system import IntegratedQuantumTradingSystem, SystemConfiguration")
    ]
    
    results = {}
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            results[name] = True
            print(f"✓ {name}")
        except Exception as e:
            results[name] = False
            print(f"✗ {name}: {e}")
    
    return results

def test_enum_consistency():
    """Test enum consistency across modules."""
    print("\nTesting enum consistency...")
    
    try:
        # Test MarketPhase consistency
        from quantum_amos import MarketPhase as QuantumAMOSPhase
        from cdfa_extensions.analyzers.panarchy_analyzer import MarketPhase as PanarchyPhase
        
        qamos_values = set(phase.value for phase in QuantumAMOSPhase)
        panarchy_values = set(phase.value for phase in PanarchyPhase)
        
        if qamos_values == panarchy_values:
            print("✓ MarketPhase enums are consistent")
            return True
        else:
            print(f"✗ MarketPhase inconsistency: {qamos_values} vs {panarchy_values}")
            return False
            
    except Exception as e:
        print(f"✗ Enum consistency test failed: {e}")
        return False

def test_message_types():
    """Test message type definitions."""
    print("\nTesting message types...")
    
    try:
        from unified_messaging import MessageType, AgentType
        
        # Check all required message types exist
        required_types = [
            'DECISION_REQUEST', 'DECISION_RESPONSE', 'PHASE_TRANSITION',
            'RISK_ALERT', 'PERFORMANCE_FEEDBACK', 'MARKET_UPDATE',
            'AGENT_STATUS', 'SYSTEM_COMMAND'
        ]
        
        missing = []
        for msg_type in required_types:
            if not hasattr(MessageType, msg_type):
                missing.append(msg_type)
        
        if missing:
            print(f"✗ Missing message types: {missing}")
            return False
        else:
            print("✓ All message types available")
            
        # Check all required agent types exist
        required_agents = ['PADS', 'QBMIA', 'QUASAR', 'QUANTUM_AMOS']
        
        missing = []
        for agent_type in required_agents:
            if not hasattr(AgentType, agent_type):
                missing.append(agent_type)
        
        if missing:
            print(f"✗ Missing agent types: {missing}")
            return False
        else:
            print("✓ All agent types available")
            return True
            
    except Exception as e:
        print(f"✗ Message type test failed: {e}")
        return False

def test_config_validity():
    """Test configuration file validity."""
    print("\nTesting configuration...")
    
    try:
        import json
        import os
        
        config_file = "quantum_system_config.json"
        if not os.path.exists(config_file):
            print(f"✗ Configuration file missing: {config_file}")
            return False
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check required keys
        required_keys = ['redis_url', 'use_real_messaging', 'zmq_ports']
        missing = [key for key in required_keys if key not in config]
        
        if missing:
            print(f"✗ Missing config keys: {missing}")
            return False
        
        # Check zmq_ports structure
        if not isinstance(config['zmq_ports'], dict):
            print("✗ zmq_ports should be a dictionary")
            return False
        
        required_agents = ['pads', 'qbmia', 'quasar', 'quantum_amos']
        missing_ports = [agent for agent in required_agents if agent not in config['zmq_ports']]
        
        if missing_ports:
            print(f"✗ Missing ZMQ ports for: {missing_ports}")
            return False
        
        print("✓ Configuration file valid")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_startup_script():
    """Test startup script exists and is executable."""
    print("\nTesting startup script...")
    
    try:
        import os
        import stat
        
        script_file = "start_quantum_system.sh"
        if not os.path.exists(script_file):
            print(f"✗ Startup script missing: {script_file}")
            return False
        
        # Check if executable
        file_stat = os.stat(script_file)
        if not file_stat.st_mode & stat.S_IEXEC:
            print(f"✗ Startup script not executable: {script_file}")
            return False
        
        print("✓ Startup script ready")
        return True
        
    except Exception as e:
        print(f"✗ Startup script test failed: {e}")
        return False

def test_class_interfaces():
    """Test key class interfaces."""
    print("\nTesting class interfaces...")
    
    try:
        from unified_messaging import Message, MessageType, AgentType
        
        # Test Message creation
        msg = Message(
            message_type=MessageType.DECISION_REQUEST,
            sender=AgentType.PADS,
            recipient=AgentType.QBMIA,
            data={'test': 'data'}
        )
        
        # Test serialization
        msg_dict = msg.to_dict()
        msg_back = Message.from_dict(msg_dict)
        
        if msg.id != msg_back.id:
            print("✗ Message serialization failed")
            return False
        
        print("✓ Message interface working")
        return True
        
    except Exception as e:
        print(f"✗ Class interface test failed: {e}")
        return False

def main():
    """Run comprehensive bug check."""
    print("COMPREHENSIVE BUG CHECK")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Enum Consistency", test_enum_consistency),
        ("Message Types", test_message_types),
        ("Configuration", test_config_validity),
        ("Startup Script", test_startup_script),
        ("Class Interfaces", test_class_interfaces)
    ]
    
    all_results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_name == "Import Tests":
                result = test_func()
                # For import tests, check if all passed
                all_results[test_name] = all(result.values()) if isinstance(result, dict) else result
            else:
                result = test_func()
                all_results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            all_results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("BUG CHECK SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in all_results.values() if result)
    total = len(all_results)
    
    for test_name, result in all_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 No critical bugs found! System is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} critical issues found. Please review and fix.")
        return 1

if __name__ == "__main__":
    sys.exit(main())