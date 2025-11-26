#!/usr/bin/env python3
"""Quick test of syndicate MCP integration"""

import asyncio
import sys
sys.path.append('.')

from src.mcp.tools.syndicate_tools import (
    syndicate_create, syndicate_add_member, syndicate_list_members,
    syndicate_allocate_funds, syndicate_get_balance, syndicate_member_performance
)


async def test_syndicate_tools():
    """Test basic syndicate functionality"""
    print("🧪 Testing Syndicate MCP Integration\n")
    
    # Test 1: Create syndicate
    print("1️⃣ Creating syndicate...")
    try:
        result = await syndicate_create({
            'syndicate_id': 'test-syndicate-001',
            'name': 'Test Trading Syndicate',
            'description': 'Testing MCP integration',
            'initial_capital': 0  # Will be updated when members contribute
        })
        print(f"✅ Syndicate created: {result['syndicate_id']}")
        print(f"   Name: {result['name']}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    # Test 2: Add members
    print("\n2️⃣ Adding members...")
    members = [
        ('Alice Johnson', 'alice@test.com', 'lead_investor', 100000),
        ('Bob Smith', 'bob@test.com', 'senior_analyst', 50000),
        ('Charlie Brown', 'charlie@test.com', 'contributing_member', 25000)
    ]
    
    for name, email, role, contribution in members:
        try:
            result = await syndicate_add_member({
                'syndicate_id': 'test-syndicate-001',
                'name': name,
                'email': email,
                'role': role,
                'initial_contribution': contribution
            })
            print(f"✅ Added {name} as {role} with ${contribution:,}")
        except Exception as e:
            print(f"❌ Failed to add {name}: {e}")
    
    # Test 3: List members
    print("\n3️⃣ Listing members...")
    try:
        result = await syndicate_list_members({
            'syndicate_id': 'test-syndicate-001'
        })
        print(f"✅ Found {len(result['members'])} members:")
        for member in result['members']:
            print(f"   - {member['name']} ({member['role']}): ${member['capital_contribution']:,}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 4: Get balance
    print("\n4️⃣ Getting syndicate balance...")
    try:
        result = await syndicate_get_balance({
            'syndicate_id': 'test-syndicate-001'
        })
        print(f"✅ Total capital: ${result['total_capital']:,}")
        print(f"   Available: ${result['available_capital']:,}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 5: Allocate funds
    print("\n5️⃣ Testing fund allocation...")
    try:
        result = await syndicate_allocate_funds({
            'syndicate_id': 'test-syndicate-001',
            'opportunity': {
                'sport': 'NBA',
                'event': 'Lakers vs Celtics',
                'bet_type': 'spread',
                'selection': 'Lakers -3.5',
                'odds': 1.91,
                'probability': 0.58,
                'edge': 0.05,
                'confidence': 0.75,
                'model_agreement': 0.82,
                'liquidity': 50000
            },
            'strategy': 'kelly_criterion'
        })
        print(f"✅ Allocation recommendation:")
        print(f"   Amount: ${result['allocation']['amount']}")
        print(f"   % of bankroll: {result['allocation']['percentage_of_bankroll']:.2%}")
        if result['allocation']['warnings']:
            print(f"   Warnings: {', '.join(result['allocation']['warnings'])}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 6: Member performance
    print("\n6️⃣ Getting member performance...")
    try:
        # Get the first member's ID
        members_result = await syndicate_list_members({
            'syndicate_id': 'test-syndicate-001'
        })
        if members_result['members']:
            member_id = members_result['members'][0]['id']
            result = await syndicate_member_performance({
                'syndicate_id': 'test-syndicate-001',
                'member_id': member_id
            })
            print(f"✅ Performance for {result['member_info']['name']}:")
            print(f"   ROI: {result['financial_summary']['roi']:.1%}")
            print(f"   Win Rate: {result['betting_performance']['win_rate']:.1%}")
            print(f"   Capital: ${result['financial_summary']['capital_contribution']:,}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_syndicate_tools())