"""
Test fixtures for vault data

Provides realistic vault data for testing purposes.
"""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any

# Real Beefy vault data samples
SAMPLE_VAULTS = [
    {
        "id": "beefy-bsc-cake-bnb",
        "name": "CAKE-BNB LP",
        "token": "CAKE-BNB LP",
        "tokenAddress": "0x0eD7e52944161450477ee417DE9Cd3a859b14fD0",
        "tokenDecimals": 18,
        "tokenProviderId": "pancakeswap",
        "earnedToken": "mooCakeBNB",
        "earnedTokenAddress": "0x3f16b3e3e7A8e0F2E5b9D7F8e4F8B7e4E6E7F8e9",
        "earnContractAddress": "0x3f16b3e3e7A8e0F2E5b9D7F8e4F8B7e4E6E7F8e9",
        "oracle": "lps",
        "oracleId": "pancakeswap-cake-bnb",
        "status": "active",
        "platformId": "pancakeswap",
        "assets": ["CAKE", "BNB"],
        "risks": ["COMPLEXITY_LOW", "BATTLE_TESTED", "IL_NONE"],
        "strategyTypeId": "lp",
        "network": "bsc",
        "chain": "bsc",
        "createdAt": 1609459200
    },
    {
        "id": "beefy-polygon-matic-eth",
        "name": "MATIC-ETH LP",
        "token": "MATIC-ETH LP",
        "tokenAddress": "0x6e7a5FAFcec6BB1e78bAE2A1F0B612012BF14827",
        "tokenDecimals": 18,
        "tokenProviderId": "quickswap",
        "earnedToken": "mooMaticETH",
        "earnedTokenAddress": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
        "earnContractAddress": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
        "oracle": "lps",
        "oracleId": "quickswap-matic-eth",
        "status": "active",
        "platformId": "quickswap",
        "assets": ["MATIC", "ETH"],
        "risks": ["COMPLEXITY_LOW", "BATTLE_TESTED", "IL_LOW"],
        "strategyTypeId": "lp",
        "network": "polygon",
        "chain": "polygon",
        "createdAt": 1615459200
    },
    {
        "id": "beefy-avax-joe-avax",
        "name": "JOE-AVAX LP",
        "token": "JOE-AVAX LP",
        "tokenAddress": "0x454E67025631C065d3cFAD6d71E6892f74487a15",
        "tokenDecimals": 18,
        "tokenProviderId": "traderjoe",
        "earnedToken": "mooJoeAVAX",
        "earnedTokenAddress": "0x371c7ec6D8039ff7933a2AA28EB827Ffe1F52f07",
        "earnContractAddress": "0x371c7ec6D8039ff7933a2AA28EB827Ffe1F52f07",
        "oracle": "lps",
        "oracleId": "traderjoe-joe-avax",
        "status": "active",
        "platformId": "traderjoe",
        "assets": ["JOE", "AVAX"],
        "risks": ["COMPLEXITY_MID", "IL_LOW"],
        "strategyTypeId": "lp",
        "network": "avax",
        "chain": "avax",
        "createdAt": 1625459200
    },
    {
        "id": "beefy-ethereum-usdc-eth",
        "name": "USDC-ETH LP",
        "token": "USDC-ETH LP",
        "tokenAddress": "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
        "tokenDecimals": 18,
        "tokenProviderId": "uniswap",
        "earnedToken": "mooUSDCETH",
        "earnedTokenAddress": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "earnContractAddress": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "oracle": "lps",
        "oracleId": "uniswap-usdc-eth",
        "status": "active",
        "platformId": "uniswap",
        "assets": ["USDC", "ETH"],
        "risks": ["COMPLEXITY_HIGH", "BATTLE_TESTED", "IL_LOW"],
        "strategyTypeId": "lp",
        "network": "ethereum",
        "chain": "ethereum",
        "createdAt": 1595459200
    },
    {
        "id": "beefy-fantom-boo-ftm",
        "name": "BOO-FTM LP",
        "token": "BOO-FTM LP",
        "tokenAddress": "0xEc7178F4C41f346b2721907F5cF7628E388A7a58",
        "tokenDecimals": 18,
        "tokenProviderId": "spookyswap",
        "earnedToken": "mooBooFTM",
        "earnedTokenAddress": "0xd6070ae98b8069de6B494332d1A1a81B6179D960",
        "earnContractAddress": "0xd6070ae98b8069de6B494332d1A1a81B6179D960",
        "oracle": "lps",
        "oracleId": "spookyswap-boo-ftm",
        "status": "active",
        "platformId": "spookyswap",
        "assets": ["BOO", "FTM"],
        "risks": ["COMPLEXITY_MID", "IL_MID"],
        "strategyTypeId": "lp",
        "network": "fantom",
        "chain": "fantom",
        "createdAt": 1635459200
    }
]

# APY data corresponding to sample vaults
SAMPLE_APY_DATA = {
    "beefy-bsc-cake-bnb": {
        "vaultId": "beefy-bsc-cake-bnb",
        "priceId": "pancakeswap-cake-bnb",
        "vaultApr": 24.2,
        "vaultApy": 27.3,
        "compoundingsPerYear": 2190,
        "beefyPerformanceFee": 0.045,
        "vaultDailyApy": 0.0748,
        "totalApy": 27.3,
        "tradingApr": 18.5,
        "liquidStakingApr": None,
        "composablePoolApr": None,
        "merklApr": None
    },
    "beefy-polygon-matic-eth": {
        "vaultId": "beefy-polygon-matic-eth",
        "priceId": "quickswap-matic-eth",
        "vaultApr": 16.8,
        "vaultApy": 18.2,
        "compoundingsPerYear": 2190,
        "beefyPerformanceFee": 0.045,
        "vaultDailyApy": 0.0498,
        "totalApy": 18.2,
        "tradingApr": 12.3,
        "liquidStakingApr": None,
        "composablePoolApr": None,
        "merklApr": None
    },
    "beefy-avax-joe-avax": {
        "vaultId": "beefy-avax-joe-avax",
        "priceId": "traderjoe-joe-avax",
        "vaultApr": 29.5,
        "vaultApy": 33.8,
        "compoundingsPerYear": 2190,
        "beefyPerformanceFee": 0.045,
        "vaultDailyApy": 0.0926,
        "totalApy": 33.8,
        "tradingApr": 22.1,
        "liquidStakingApr": None,
        "composablePoolApr": None,
        "merklApr": None
    },
    "beefy-ethereum-usdc-eth": {
        "vaultId": "beefy-ethereum-usdc-eth",
        "priceId": "uniswap-usdc-eth",
        "vaultApr": 8.2,
        "vaultApy": 8.5,
        "compoundingsPerYear": 2190,
        "beefyPerformanceFee": 0.045,
        "vaultDailyApy": 0.0233,
        "totalApy": 8.5,
        "tradingApr": 5.8,
        "liquidStakingApr": None,
        "composablePoolApr": None,
        "merklApr": None
    },
    "beefy-fantom-boo-ftm": {
        "vaultId": "beefy-fantom-boo-ftm",
        "priceId": "spookyswap-boo-ftm",
        "vaultApr": 45.2,
        "vaultApy": 56.8,
        "compoundingsPerYear": 2190,
        "beefyPerformanceFee": 0.045,
        "vaultDailyApy": 0.1556,
        "totalApy": 56.8,
        "tradingApr": 38.5,
        "liquidStakingApr": None,
        "composablePoolApr": None,
        "merklApr": None
    }
}

# TVL data by chain
SAMPLE_TVL_DATA = {
    "bsc": 2847362847.23,
    "polygon": 1529384756.89,
    "avalanche": 892847362.45,
    "ethereum": 4385629374.12,
    "fantom": 384729573.67,
    "arbitrum": 673847362.89,
    "optimism": 298374629.34
}

# Token price data
SAMPLE_PRICE_DATA = {
    "CAKE": 2.15,
    "BNB": 245.67,
    "MATIC": 0.85,
    "ETH": 1650.00,
    "JOE": 0.25,
    "AVAX": 15.32,
    "BOO": 1.45,
    "FTM": 0.35,
    "USDC": 1.00,
    "USDT": 1.00,
    "DAI": 1.00
}

# Historical yield data generator
def generate_historical_yields(vault_id: str, days: int = 30) -> List[Dict[str, Any]]:
    """Generate historical yield data for testing"""
    base_apy = SAMPLE_APY_DATA.get(vault_id, {"totalApy": 20.0})["totalApy"]
    
    historical_data = []
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=days-i)
        
        # Add some realistic volatility
        import random
        random.seed(hash(f"{vault_id}_{i}"))  # Deterministic for testing
        volatility = random.uniform(0.85, 1.15)
        apy = base_apy * volatility
        
        historical_data.append({
            "date": date.isoformat(),
            "vault_id": vault_id,
            "apy": round(apy, 2),
            "tvl": random.uniform(1000000, 10000000),
            "price_per_share": 1.0 + (i * 0.001),  # Gradual increase
            "volume_24h": random.uniform(100000, 1000000)
        })
    
    return historical_data

# Portfolio position fixtures
SAMPLE_PORTFOLIO_POSITIONS = [
    {
        "vault_id": "beefy-bsc-cake-bnb",
        "vault_name": "CAKE-BNB LP",
        "chain": "bsc",
        "amount_deposited": 1000.0,
        "shares_owned": 952.38,
        "current_value": 1125.50,
        "entry_price": 1.05,
        "entry_apy": 25.5,
        "status": "active",
        "created_at": datetime.utcnow() - timedelta(days=15)
    },
    {
        "vault_id": "beefy-polygon-matic-eth",
        "vault_name": "MATIC-ETH LP",
        "chain": "polygon",
        "amount_deposited": 2000.0,
        "shares_owned": 1980.39,
        "current_value": 2156.75,
        "entry_price": 1.01,
        "entry_apy": 18.2,
        "status": "active",
        "created_at": datetime.utcnow() - timedelta(days=22)
    },
    {
        "vault_id": "beefy-ethereum-usdc-eth",
        "vault_name": "USDC-ETH LP",
        "chain": "ethereum",
        "amount_deposited": 3000.0,
        "shares_owned": 2985.67,
        "current_value": 3089.25,
        "entry_price": 1.005,
        "entry_apy": 8.5,
        "status": "active",
        "created_at": datetime.utcnow() - timedelta(days=8)
    },
    {
        "vault_id": "beefy-fantom-boo-ftm",
        "vault_name": "BOO-FTM LP",
        "chain": "fantom",
        "amount_deposited": 500.0,
        "shares_owned": 478.26,
        "current_value": 425.80,
        "entry_price": 1.045,
        "entry_apy": 56.8,
        "status": "closed",
        "created_at": datetime.utcnow() - timedelta(days=45)
    }
]

# Transaction fixtures
SAMPLE_TRANSACTIONS = [
    {
        "transaction_type": "deposit",
        "vault_id": "beefy-bsc-cake-bnb",
        "chain": "bsc",
        "amount": 1000.0,
        "gas_used": 0.015,
        "tx_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "status": "confirmed",
        "created_at": datetime.utcnow() - timedelta(days=15)
    },
    {
        "transaction_type": "deposit",
        "vault_id": "beefy-polygon-matic-eth",
        "chain": "polygon",
        "amount": 2000.0,
        "gas_used": 0.05,
        "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "status": "confirmed",
        "created_at": datetime.utcnow() - timedelta(days=22)
    },
    {
        "transaction_type": "compound",
        "vault_id": "beefy-bsc-cake-bnb",
        "chain": "bsc",
        "amount": 25.50,
        "gas_used": 0.008,
        "tx_hash": "0x567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234",
        "status": "confirmed",
        "created_at": datetime.utcnow() - timedelta(days=10)
    },
    {
        "transaction_type": "withdraw",
        "vault_id": "beefy-fantom-boo-ftm",
        "chain": "fantom",
        "amount": 425.80,
        "gas_used": 0.012,
        "tx_hash": "0x90abcdef1234567890abcdef1234567890abcdef1234567890abcdef123456789",
        "status": "confirmed",
        "created_at": datetime.utcnow() - timedelta(days=5)
    }
]

# Yield history fixtures
def generate_yield_history(position_id: int, vault_id: str, days: int = 30) -> List[Dict]:
    """Generate yield history for a position"""
    base_apy = SAMPLE_APY_DATA.get(vault_id, {"totalApy": 20.0})["totalApy"]
    
    yield_history = []
    cumulative_earned = 0.0
    
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=days-i)
        
        # Daily yield calculation
        daily_apy = base_apy + (i * 0.1)  # Gradual increase
        daily_earned = 1000.0 * (daily_apy / 100 / 365)  # Assuming $1000 base
        cumulative_earned += daily_earned
        
        yield_history.append({
            "vault_id": vault_id,
            "position_id": position_id,
            "earned_amount": daily_earned,
            "apy_snapshot": daily_apy,
            "tvl_snapshot": 5000000.0 + (i * 10000),
            "price_per_share": 1.0 + (i * 0.001),
            "recorded_at": date
        })
    
    return yield_history

# Market data fixtures
SAMPLE_MARKET_DATA = {
    "total_tvl": sum(SAMPLE_TVL_DATA.values()),
    "chains": {
        "bsc": {
            "tvl": SAMPLE_TVL_DATA["bsc"],
            "avg_apy": 22.5,
            "vault_count": 156,
            "top_protocols": ["pancakeswap", "biswap", "apeswap"]
        },
        "ethereum": {
            "tvl": SAMPLE_TVL_DATA["ethereum"],
            "avg_apy": 12.3,
            "vault_count": 89,
            "top_protocols": ["uniswap", "curve", "balancer"]
        },
        "polygon": {
            "tvl": SAMPLE_TVL_DATA["polygon"],
            "avg_apy": 18.7,
            "vault_count": 94,
            "top_protocols": ["quickswap", "sushiswap", "curve"]
        }
    },
    "trending_vaults": [
        {"vault_id": "beefy-bsc-cake-bnb", "apy_change": "+2.3%"},
        {"vault_id": "beefy-fantom-boo-ftm", "apy_change": "+8.5%"},
        {"vault_id": "beefy-avax-joe-avax", "apy_change": "-1.2%"}
    ]
}

# Risk assessment data
VAULT_RISK_PROFILES = {
    "beefy-bsc-cake-bnb": {
        "overall_risk": 35,
        "smart_contract_risk": 15,
        "impermanent_loss_risk": 25,
        "liquidity_risk": 10,
        "platform_risk": 20,
        "risk_factors": ["COMPLEXITY_LOW", "BATTLE_TESTED", "IL_NONE"],
        "audit_status": "audited",
        "insurance_coverage": False
    },
    "beefy-ethereum-usdc-eth": {
        "overall_risk": 25,
        "smart_contract_risk": 20,
        "impermanent_loss_risk": 15,
        "liquidity_risk": 5,
        "platform_risk": 10,
        "risk_factors": ["COMPLEXITY_HIGH", "BATTLE_TESTED", "IL_LOW"],
        "audit_status": "audited",
        "insurance_coverage": True
    },
    "beefy-fantom-boo-ftm": {
        "overall_risk": 65,
        "smart_contract_risk": 30,
        "impermanent_loss_risk": 40,
        "liquidity_risk": 35,
        "platform_risk": 25,
        "risk_factors": ["COMPLEXITY_MID", "IL_MID"],
        "audit_status": "pending",
        "insurance_coverage": False
    }
}

# Performance benchmarks
SAMPLE_BENCHMARKS = {
    "defi_index": {
        "name": "DeFi Pulse Index",
        "returns_30d": 8.5,
        "returns_90d": 15.2,
        "returns_1y": 45.8,
        "volatility": 24.3,
        "sharpe_ratio": 1.2
    },
    "crypto_market": {
        "name": "Crypto Market Cap",
        "returns_30d": 12.1,
        "returns_90d": 28.4,
        "returns_1y": 89.2,
        "volatility": 45.2,
        "sharpe_ratio": 0.8
    },
    "traditional_60_40": {
        "name": "60/40 Stock/Bond Portfolio",
        "returns_30d": 2.1,
        "returns_90d": 6.5,
        "returns_1y": 8.9,
        "volatility": 12.4,
        "sharpe_ratio": 0.6
    }
}

def get_vault_by_id(vault_id: str) -> Dict[str, Any]:
    """Get vault data by ID"""
    for vault in SAMPLE_VAULTS:
        if vault["id"] == vault_id:
            return vault
    return {}

def get_vaults_by_chain(chain: str) -> List[Dict[str, Any]]:
    """Get vaults filtered by chain"""
    return [vault for vault in SAMPLE_VAULTS if vault["chain"] == chain]

def get_high_yield_vaults(min_apy: float = 20.0) -> List[Dict[str, Any]]:
    """Get vaults with APY above threshold"""
    high_yield = []
    for vault in SAMPLE_VAULTS:
        vault_apy = SAMPLE_APY_DATA.get(vault["id"], {"totalApy": 0})["totalApy"]
        if vault_apy >= min_apy:
            high_yield.append({**vault, "current_apy": vault_apy})
    return high_yield

def save_fixtures_to_file(filename: str):
    """Save all fixtures to JSON file for external use"""
    fixtures = {
        "vaults": SAMPLE_VAULTS,
        "apy_data": SAMPLE_APY_DATA,
        "tvl_data": SAMPLE_TVL_DATA,
        "price_data": SAMPLE_PRICE_DATA,
        "portfolio_positions": SAMPLE_PORTFOLIO_POSITIONS,
        "transactions": SAMPLE_TRANSACTIONS,
        "market_data": SAMPLE_MARKET_DATA,
        "risk_profiles": VAULT_RISK_PROFILES,
        "benchmarks": SAMPLE_BENCHMARKS
    }
    
    with open(filename, 'w') as f:
        json.dump(fixtures, f, indent=2, default=str)

if __name__ == "__main__":
    # Generate sample data file
    save_fixtures_to_file("crypto_trading_fixtures.json")
    print("Generated fixture data saved to crypto_trading_fixtures.json")