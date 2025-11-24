#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Integrated Quantum Trading System

This script tests the messaging integration and agent coordination
without requiring all agent implementations to be fully functional.
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any

# Test unified messaging
try:
    from unified_messaging import (
        UnifiedMessenger, Message, MessageType, AgentType,
        create_pads_messenger, create_qbmia_messenger, 
        create_quasar_messenger, create_quantum_amos_messenger
    )
    MESSAGING_AVAILABLE = True
    print("✓ Unified messaging available")
except ImportError as e:
    MESSAGING_AVAILABLE = False
    print(f"✗ Unified messaging not available: {e}")

# Test messaging adapters
try:
    from pads_messaging_integration import PADSMessagingIntegration
    from quasar_messaging_adapter import QUASARMessagingAdapter
    from quantum_amos_messaging_adapter import QuantumAMOSMessagingAdapter
    ADAPTERS_AVAILABLE = True
    print("✓ Messaging adapters available")
except ImportError as e:
    ADAPTERS_AVAILABLE = False
    print(f"✗ Messaging adapters not available: {e}")

# Test QBMIA PADS connector
try:
    from qbmia.integration.pads_connector import PADSConnector
    QBMIA_CONNECTOR_AVAILABLE = True
    print("✓ QBMIA PADS connector available")
except ImportError as e:
    QBMIA_CONNECTOR_AVAILABLE = False
    print(f"✗ QBMIA PADS connector not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationTest")

class MockAgent:
    """Mock agent for testing messaging without full implementation."""
    
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.decisions_received = []
        self.messages_sent = []
        self.connected = False
    
    def get_status(self):
        return {
            'healthy': True,
            'connected': self.connected,
            'decisions_count': len(self.decisions_received),
            'messages_sent': len(self.messages_sent)
        }
    
    async def make_decision(self, market_data: Dict[str, Any], position_state: Dict[str, Any] = None):
        """Mock decision making."""
        decision = {
            'action': 'HOLD',
            'confidence': 0.7,
            'reasoning': f'{self.name} mock decision',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.decisions_received.append({
            'market_data': market_data,
            'decision': decision
        })
        return decision

async def test_basic_messaging():
    """Test basic messaging functionality."""
    print("\n" + "="*50)
    print("TESTING BASIC MESSAGING")
    print("="*50)
    
    if not MESSAGING_AVAILABLE:
        print("Skipping messaging tests - unified messaging not available")
        return False
    
    try:
        # Create test messengers
        pads_messenger = create_pads_messenger()
        qbmia_messenger = create_qbmia_messenger()
        
        # Track received messages
        received_messages = []
        
        async def message_handler(message: Message):
            received_messages.append(message)
            logger.info(f"Received message: {message.message_type.value} from {message.sender.value}")
        
        # Register handlers
        pads_messenger.register_handler(MessageType.DECISION_RESPONSE, message_handler)
        qbmia_messenger.register_handler(MessageType.DECISION_REQUEST, message_handler)
        
        # Connect messengers
        pads_connected = await pads_messenger.connect()
        qbmia_connected = await qbmia_messenger.connect()
        
        if not (pads_connected or qbmia_connected):
            print("✗ Could not connect any messengers")
            return False
        
        print(f"✓ Connected messengers: PADS={pads_connected}, QBMIA={qbmia_connected}")
        
        # Start listening
        if qbmia_connected:
            asyncio.create_task(qbmia_messenger.start_listening())
        if pads_connected:
            asyncio.create_task(pads_messenger.start_listening())
        
        # Send test message
        if pads_connected:
            test_message = Message(
                message_type=MessageType.DECISION_REQUEST,
                sender=AgentType.PADS,
                recipient=AgentType.QBMIA,
                data={
                    'market_data': {'symbol': 'BTC/USDT', 'price': 50000},
                    'request_id': 'test_001'
                }
            )
            
            success = await pads_messenger.send_message(test_message)
            print(f"✓ Test message sent: {success}")
        
        # Wait for message processing
        await asyncio.sleep(2)
        
        # Check results
        print(f"✓ Messages received: {len(received_messages)}")
        
        # Cleanup
        await pads_messenger.disconnect()
        await qbmia_messenger.disconnect()
        
        return len(received_messages) > 0
        
    except Exception as e:
        logger.error(f"Basic messaging test failed: {e}")
        return False

async def test_adapter_integration():
    """Test messaging adapter integration."""
    print("\n" + "="*50)
    print("TESTING ADAPTER INTEGRATION")
    print("="*50)
    
    if not ADAPTERS_AVAILABLE:
        print("Skipping adapter tests - adapters not available")
        return False
    
    try:
        # Create mock agents
        mock_pads = MockAgent("MockPADS", "pads")
        mock_quasar = MockAgent("MockQUASAR", "quasar")
        mock_amos = MockAgent("MockQuantumAMOS", "quantum_amos")
        
        # Create adapters
        config = {
            'redis_url': 'redis://localhost:6379',
            'use_real_messaging': True,
            'zmq_ports': {}
        }
        
        pads_integration = PADSMessagingIntegration(mock_pads, config)
        quasar_adapter = QUASARMessagingAdapter(mock_quasar, config)
        amos_adapter = QuantumAMOSMessagingAdapter(mock_amos, config)
        
        # Test connections
        pads_connected = await pads_integration.connect()
        quasar_connected = await quasar_adapter.connect()
        amos_connected = await amos_adapter.connect()
        
        print(f"✓ Adapter connections: PADS={pads_connected}, QUASAR={quasar_connected}, AMOS={amos_connected}")
        
        # Test decision request if PADS connected
        if pads_connected:
            market_data = {
                'symbol': 'BTC/USDT',
                'price': 50000,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Request decisions from connected agents
            agent_types = []
            if quasar_connected:
                agent_types.append(AgentType.QUASAR)
            if amos_connected:
                agent_types.append(AgentType.QUANTUM_AMOS)
            
            if agent_types:
                responses = await pads_integration.request_agent_decisions(
                    market_data=market_data,
                    agents=agent_types,
                    timeout=3.0
                )
                print(f"✓ Decision responses received: {len(responses)}")
        
        # Cleanup
        await pads_integration.disconnect()
        await quasar_adapter.disconnect()
        await amos_adapter.disconnect()
        
        return True
        
    except Exception as e:
        logger.error(f"Adapter integration test failed: {e}")
        return False

async def test_qbmia_pads_connector():
    """Test QBMIA PADS connector."""
    print("\n" + "="*50)
    print("TESTING QBMIA PADS CONNECTOR")
    print("="*50)
    
    if not QBMIA_CONNECTOR_AVAILABLE:
        print("Skipping QBMIA connector test - connector not available")
        return False
    
    try:
        # Create mock QBMIA agent
        mock_qbmia = MockAgent("MockQBMIA", "qbmia")
        
        # Create PADS connector
        pads_config = {
            'endpoint': 'http://localhost:9090',
            'system_id': 'QBMIA_TEST_001',
            'redis_url': 'redis://localhost:6379',
            'use_real_messaging': True
        }
        
        connector = PADSConnector(mock_qbmia, pads_config)
        
        # Test connection
        connected = await connector.connect_to_pads()
        print(f"✓ QBMIA PADS connector connected: {connected}")
        
        if connected:
            # Test status
            status = connector.get_pads_status()
            print(f"✓ PADS status retrieved: {status['connected']}")
            
            # Test decision submission
            test_decision = {
                'action': 'BUY',
                'confidence': 0.8,
                'reasoning': 'Test decision from QBMIA connector'
            }
            
            response = await connector.submit_decision_to_pads(test_decision)
            print(f"✓ Decision submitted: {response['status']}")
        
        # Cleanup
        await connector.disconnect_from_pads()
        
        return connected
        
    except Exception as e:
        logger.error(f"QBMIA PADS connector test failed: {e}")
        return False

async def test_configuration_loading():
    """Test configuration loading."""
    print("\n" + "="*50)
    print("TESTING CONFIGURATION")
    print("="*50)
    
    try:
        # Test config file existence
        import os
        config_file = "quantum_system_config.json"
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ Configuration loaded: {len(config)} settings")
            
            # Validate required settings
            required_settings = ['redis_url', 'use_real_messaging', 'zmq_ports']
            missing = [setting for setting in required_settings if setting not in config]
            
            if missing:
                print(f"✗ Missing required settings: {missing}")
                return False
            else:
                print("✓ All required settings present")
                return True
        else:
            print(f"✗ Configuration file not found: {config_file}")
            return False
            
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False

async def main():
    """Run all integration tests."""
    print("INTEGRATED QUANTUM TRADING SYSTEM - INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Basic Messaging", test_basic_messaging),
        ("Adapter Integration", test_adapter_integration),
        ("QBMIA PADS Connector", test_qbmia_pads_connector)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning test: {test_name}")
            result = await test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            print(f"Test {test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"Test {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The integration is ready.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())