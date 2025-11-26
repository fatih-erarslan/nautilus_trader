"""
Example usage of Beefy Finance API integration.

This script demonstrates how to use the BeefyFinanceAPI client
to fetch vault data, APY information, and prepare transactions.
"""

import asyncio
import logging
from decimal import Decimal
from pprint import pprint

from beefy_client import BeefyFinanceAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Demonstrate Beefy Finance API usage."""
    
    # Initialize the API client
    async with BeefyFinanceAPI() as api:
        
        # 1. Fetch all vaults
        print("\n=== Fetching All Vaults ===")
        vaults = await api.get_vaults()
        print(f"Total vaults available: {len(vaults)}")
        
        # Show first 3 vaults
        for vault in vaults[:3]:
            print(f"\nVault: {vault.name}")
            print(f"  ID: {vault.id}")
            print(f"  Chain: {vault.chain}")
            print(f"  Platform: {vault.platformId}")
            print(f"  Status: {vault.status}")
            print(f"  Token: {vault.token}")
        
        # 2. Fetch vaults for specific chain
        print("\n=== Fetching BSC Vaults ===")
        bsc_vaults = await api.get_vaults(chain="bsc")
        print(f"BSC vaults: {len(bsc_vaults)}")
        
        # 3. Fetch APY data
        print("\n=== Fetching APY Data ===")
        apy_data = await api.get_apy()
        print(f"APY data for {len(apy_data)} vaults")
        
        # Show top 5 APY vaults
        sorted_apys = sorted(
            [(k, v) for k, v in apy_data.items()],
            key=lambda x: x[1].totalApy,
            reverse=True
        )
        
        print("\nTop 5 APY Vaults:")
        for vault_id, apy in sorted_apys[:5]:
            print(f"  {vault_id}: {apy.totalApy:.2f}% APY")
        
        # 4. Fetch TVL data
        print("\n=== Fetching TVL Data ===")
        tvl_data = await api.get_tvl()
        print(f"TVL data for {len(tvl_data)} chains/protocols")
        
        # Show TVL by chain
        total_tvl = sum(v.tvl for v in tvl_data.values())
        print(f"Total TVL across all chains: ${total_tvl:,.2f}")
        
        # 5. Fetch token prices
        print("\n=== Fetching Token Prices ===")
        prices = await api.get_prices()
        print(f"Prices for {len(prices)} tokens")
        
        # Show some major token prices
        major_tokens = ["BTC", "ETH", "BNB", "MATIC", "AVAX"]
        for token in major_tokens:
            if token in prices:
                print(f"  {token}: ${prices[token].price:,.2f}")
        
        # 6. Search for specific vaults
        print("\n=== Searching for USDC Vaults ===")
        usdc_vaults = await api.search_vaults(
            query="USDC",
            chain="polygon",
            min_apy=5.0
        )
        
        print(f"Found {len(usdc_vaults)} USDC vaults on Polygon with >5% APY")
        for vault in usdc_vaults[:3]:
            print(f"\n  {vault['name']}")
            print(f"    APY: {vault['apy']:.2f}%")
            print(f"    TVL: ${float(vault['tvl']):,.2f}")
            print(f"    Platform: {vault['platform']}")
        
        # 7. Prepare a deposit transaction (example)
        print("\n=== Preparing Deposit Transaction ===")
        
        # Example vault and addresses (would need real values)
        example_vault = bsc_vaults[0] if bsc_vaults else None
        
        if example_vault:
            # Example parameters (would need real wallet)
            user_address = "0x742d35Cc6634C0532925a3b844Bc9e7595f5b06E"  # Example address
            amount_to_deposit = Web3.to_wei(100, 'ether')  # 100 tokens
            
            deposit_tx = api.prepare_deposit_transaction(
                vault_id=example_vault.id,
                vault_address=example_vault.earnContractAddress,
                token_address=example_vault.tokenAddress or "0x0",
                amount=amount_to_deposit,
                chain=example_vault.chain,
                user_address=user_address
            )
            
            if deposit_tx:
                print(f"Deposit transaction prepared for {example_vault.name}")
                print(f"  Vault: {deposit_tx.vaultId}")
                print(f"  Amount: {deposit_tx.amount}")
                print(f"  Function: {deposit_tx.functionName}")
                if deposit_tx.estimatedGas:
                    print(f"  Estimated Gas: {deposit_tx.estimatedGas}")
            else:
                print("Failed to prepare deposit transaction")
        
        # 8. Estimate gas costs
        print("\n=== Estimating Gas Costs ===")
        
        if example_vault:
            gas_estimate = api.estimate_gas_costs(
                chain=example_vault.chain,
                vault_address=example_vault.earnContractAddress,
                action="deposit",
                user_address=user_address,
                amount=amount_to_deposit
            )
            
            if gas_estimate:
                print(f"Gas estimate for deposit on {gas_estimate.chain}:")
                print(f"  Estimated Gas: {gas_estimate.estimatedGas}")
                print(f"  Gas Price: {gas_estimate.gasPrice} wei")
                print(f"  Total Cost: {gas_estimate.totalCostETH} ETH")
                print(f"  Total Cost USD: ${gas_estimate.totalCostUSD}")
            else:
                print("Failed to estimate gas costs")
        
        # 9. Get active chains
        print("\n=== Active Web3 Connections ===")
        active_chains = api.web3_manager.get_active_chains()
        print(f"Connected to {len(active_chains)} chains: {', '.join(active_chains)}")


async def test_error_handling():
    """Test error handling and retry logic."""
    print("\n=== Testing Error Handling ===")
    
    async with BeefyFinanceAPI() as api:
        # Test with invalid vault ID
        try:
            apy_data = await api.get_apy(["invalid-vault-id"])
            print("APY data fetched (even for invalid IDs)")
        except Exception as e:
            print(f"Error handled: {str(e)}")
        
        # Test rate limiting behavior
        print("\nTesting rate limiting...")
        tasks = []
        for i in range(5):
            tasks.append(api.get_ticker(f"TEST-{i}"))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"Completed {len(results)} concurrent requests")


if __name__ == "__main__":
    # Import Web3 for the example
    from web3 import Web3
    
    # Run the main example
    asyncio.run(main())
    
    # Optionally run error handling tests
    # asyncio.run(test_error_handling())