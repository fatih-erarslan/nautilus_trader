#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple messaging test for the Integrated Quantum Trading System
"""

import asyncio
import logging
from datetime import datetime

try:
    from unified_messaging import (
        UnifiedMessenger, Message, MessageType, AgentType,
        create_pads_messenger, create_qbmia_messenger
    )
    MESSAGING_AVAILABLE = True
except ImportError as e:
    MESSAGING_AVAILABLE = False
    print(f"✗ Unified messaging not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MessagingTest")

async def test_basic_messaging():
    """Test basic messaging between PADS and QBMIA."""
    print("Testing basic messaging...")
    
    if not MESSAGING_AVAILABLE:
        print("✗ Cannot test - messaging not available")
        return False
    
    try:
        # Create messengers
        pads_messenger = create_pads_messenger()
        qbmia_messenger = create_qbmia_messenger()
        
        # Track messages
        received_messages = []
        
        async def message_handler(message: Message):
            received_messages.append(message)
            print(f"✓ Received: {message.message_type.value} from {message.sender.value}")
        
        # Register handlers
        qbmia_messenger.register_handler(MessageType.DECISION_REQUEST, message_handler)
        
        # Connect
        pads_connected = await pads_messenger.connect()
        qbmia_connected = await qbmia_messenger.connect()
        
        print(f"✓ PADS connected: {pads_connected}")
        print(f"✓ QBMIA connected: {qbmia_connected}")
        
        if qbmia_connected:
            # Start listening
            listen_task = asyncio.create_task(qbmia_messenger.start_listening())
            
            # Give it a moment to start
            await asyncio.sleep(0.5)
            
            # Send test message
            if pads_connected:
                test_message = Message(
                    message_type=MessageType.DECISION_REQUEST,
                    sender=AgentType.PADS,
                    recipient=AgentType.QBMIA,
                    data={
                        'symbol': 'BTC/USDT',
                        'price': 50000,
                        'test': True
                    }
                )
                
                success = await pads_messenger.send_message(test_message)
                print(f"✓ Message sent: {success}")
            
            # Wait for message processing
            await asyncio.sleep(2)
            
            # Cancel listening
            listen_task.cancel()
        
        # Cleanup
        await pads_messenger.disconnect()
        await qbmia_messenger.disconnect()
        
        print(f"✓ Test complete - Messages received: {len(received_messages)}")
        return len(received_messages) > 0
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

async def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        import os
        import json
        
        config_file = "quantum_system_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ Configuration loaded: {len(config)} settings")
            return True
        else:
            print(f"✗ Configuration file not found: {config_file}")
            return False
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

async def main():
    print("SIMPLE MESSAGING TEST")
    print("=" * 40)
    
    tests = [
        ("Configuration", test_configuration),
        ("Basic Messaging", test_basic_messaging)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results[test_name] = result
            print(f"Result: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results[test_name] = False
            print(f"Result: ERROR - {e}")
    
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Messaging system is working.")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")

if __name__ == "__main__":
    asyncio.run(main())