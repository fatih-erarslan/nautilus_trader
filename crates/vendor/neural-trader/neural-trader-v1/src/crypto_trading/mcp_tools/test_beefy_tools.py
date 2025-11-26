"""
Test script for Beefy Finance MCP tools
"""

import asyncio
import json
from beefy_tools import BeefyToolHandlers

async def test_beefy_tools():
    """Test all Beefy Finance MCP tools"""
    
    print("🐄 Testing Beefy Finance MCP Tools\n")
    
    # Initialize handlers
    handlers = BeefyToolHandlers()
    
    try:
        # Test 1: Get vaults
        print("1️⃣ Testing beefy_get_vaults...")
        vaults_params = {
            "chain": "bsc",
            "min_apy": 10,
            "sort_by": "apy",
            "limit": 5
        }
        vaults_result = await handlers.handle_beefy_get_vaults(vaults_params)
        print(f"✅ Found {vaults_result.get('total_count', 0)} vaults")
        if vaults_result.get('vaults'):
            print(f"   Top vault: {vaults_result['vaults'][0]['name']} - {vaults_result['vaults'][0]['apy']}% APY")
        print()
        
        # Test 2: Analyze vault
        if vaults_result.get('vaults'):
            vault_id = vaults_result['vaults'][0]['vault_id']
            print(f"2️⃣ Testing beefy_analyze_vault for {vault_id}...")
            analyze_params = {
                "vault_id": vault_id,
                "include_history": True
            }
            analysis_result = await handlers.handle_beefy_analyze_vault(analyze_params)
            if 'analysis' in analysis_result:
                print(f"✅ Analysis complete:")
                print(f"   Current APY: {analysis_result['analysis']['current_apy']}%")
                print(f"   Risk Score: {analysis_result['risk_metrics']['overall_risk_score']}/100")
            print()
        
        # Test 3: Simulate investment
        print("3️⃣ Testing beefy_invest (simulation)...")
        invest_params = {
            "vault_id": "beefy-bnb-btcb",  # Example vault
            "amount": 1000,
            "slippage": 0.01,
            "simulate": True
        }
        invest_result = await handlers.handle_beefy_invest(invest_params)
        if 'result' in invest_result:
            print(f"✅ Investment simulation complete")
            print(f"   Amount: ${invest_params['amount']}")
            print(f"   Estimated shares: {invest_result['result'].get('estimated_shares', 'N/A')}")
        print()
        
        # Test 4: Get harvestable yields
        print("4️⃣ Testing beefy_harvest_yields...")
        harvest_params = {
            "vault_ids": None,  # Check all vaults
            "auto_compound": True,
            "simulate": True
        }
        harvest_result = await handlers.handle_beefy_harvest_yields(harvest_params)
        if 'total_harvested' in harvest_result:
            print(f"✅ Harvestable yields: ${harvest_result['total_harvested']:.2f}")
        print()
        
        # Test 5: Portfolio rebalancing
        print("5️⃣ Testing beefy_rebalance_portfolio...")
        rebalance_params = {
            "strategy": "equal_weight",
            "simulate": True
        }
        rebalance_result = await handlers.handle_beefy_rebalance_portfolio(rebalance_params)
        if 'rebalance_actions' in rebalance_result:
            print(f"✅ Rebalancing plan created:")
            print(f"   Actions needed: {len(rebalance_result['rebalance_actions'])}")
        print()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
    finally:
        await handlers.close()

if __name__ == "__main__":
    asyncio.run(test_beefy_tools())