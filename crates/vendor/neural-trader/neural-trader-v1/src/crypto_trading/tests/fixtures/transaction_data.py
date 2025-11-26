"""
Transaction test fixtures

Provides sample transaction data for testing blockchain interactions.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any
from decimal import Decimal

# Gas price data by chain (in Gwei for ETH-based, native token units for others)
GAS_PRICES = {
    "ethereum": {
        "slow": 20,
        "standard": 30,
        "fast": 50,
        "instant": 80
    },
    "bsc": {
        "slow": 3,
        "standard": 5,
        "fast": 10,
        "instant": 20
    },
    "polygon": {
        "slow": 30,
        "standard": 40,
        "fast": 60,
        "instant": 100
    },
    "avalanche": {
        "slow": 25,
        "standard": 30,
        "fast": 40,
        "instant": 60
    },
    "fantom": {
        "slow": 50,
        "standard": 80,
        "fast": 120,
        "instant": 200
    },
    "arbitrum": {
        "slow": 0.1,
        "standard": 0.2,
        "fast": 0.5,
        "instant": 1.0
    },
    "optimism": {
        "slow": 0.001,
        "standard": 0.002,
        "fast": 0.005,
        "instant": 0.01
    }
}

# Transaction templates
DEPOSIT_TRANSACTION_TEMPLATE = {
    "transaction_type": "deposit",
    "function_name": "deposit",
    "function_signature": "deposit(uint256)",
    "gas_estimate": 150000,
    "gas_limit": 200000,
    "estimated_cost_usd": 0.0,
    "success_probability": 0.98,
    "required_approvals": 1
}

WITHDRAWAL_TRANSACTION_TEMPLATE = {
    "transaction_type": "withdraw",
    "function_name": "withdraw",
    "function_signature": "withdraw(uint256)",
    "gas_estimate": 180000,
    "gas_limit": 250000,
    "estimated_cost_usd": 0.0,
    "success_probability": 0.99,
    "required_approvals": 0
}

COMPOUND_TRANSACTION_TEMPLATE = {
    "transaction_type": "compound",
    "function_name": "earn",
    "function_signature": "earn()",
    "gas_estimate": 120000,
    "gas_limit": 160000,
    "estimated_cost_usd": 0.0,
    "success_probability": 0.97,
    "required_approvals": 0
}

APPROVAL_TRANSACTION_TEMPLATE = {
    "transaction_type": "approval",
    "function_name": "approve",
    "function_signature": "approve(address,uint256)",
    "gas_estimate": 46000,
    "gas_limit": 60000,
    "estimated_cost_usd": 0.0,
    "success_probability": 0.995,
    "required_approvals": 0
}

# Sample transaction history
SAMPLE_TRANSACTION_HISTORY = [
    {
        "id": 1,
        "transaction_type": "deposit",
        "vault_id": "beefy-bsc-cake-bnb",
        "chain": "bsc",
        "amount": 1000.0,
        "token_amount": "1000000000000000000000",  # 1000 tokens in wei
        "gas_used": 142856,
        "gas_price": 5000000000,  # 5 Gwei
        "gas_cost": 0.000714,  # in BNB
        "gas_cost_usd": 0.175,
        "tx_hash": "0x1a2b3c4d5e6f7890abcdef1234567890abcdef1234567890abcdef1234567890",
        "block_number": 25847362,
        "status": "confirmed",
        "confirmations": 145,
        "created_at": datetime.utcnow() - timedelta(days=15),
        "confirmed_at": datetime.utcnow() - timedelta(days=15, minutes=-3),
        "from_address": "0x742b15C0B0b17A5D0E0D8EfEFA5b5B2c2C1D4A5B",
        "to_address": "0x3f16b3e3e7A8e0F2E5b9D7F8e4F8B7e4E6E7F8e9",
        "nonce": 42,
        "receipt": {
            "status": 1,
            "gasUsed": 142856,
            "effectiveGasPrice": 5000000000,
            "logs": [
                {
                    "address": "0x3f16b3e3e7A8e0F2E5b9D7F8e4F8B7e4E6E7F8e9",
                    "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"],
                    "data": "0x00000000000000000000000000000000000000000000003635c9adc5dea00000"
                }
            ]
        }
    },
    {
        "id": 2,
        "transaction_type": "approval",
        "vault_id": "beefy-polygon-matic-eth",
        "chain": "polygon",
        "amount": 2000.0,
        "token_amount": "2000000000000000000000",
        "gas_used": 46102,
        "gas_price": 40000000000,  # 40 Gwei
        "gas_cost": 0.001844,  # in MATIC
        "gas_cost_usd": 0.0016,
        "tx_hash": "0x9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba",
        "block_number": 38456789,
        "status": "confirmed",
        "confirmations": 67,
        "created_at": datetime.utcnow() - timedelta(days=22),
        "confirmed_at": datetime.utcnow() - timedelta(days=22, minutes=-1),
        "from_address": "0x742b15C0B0b17A5D0E0D8EfEFA5b5B2c2C1D4A5B",
        "to_address": "0x6e7a5FAFcec6BB1e78bAE2A1F0B612012BF14827",
        "nonce": 28,
        "receipt": {
            "status": 1,
            "gasUsed": 46102,
            "effectiveGasPrice": 40000000000,
            "logs": [
                {
                    "address": "0x6e7a5FAFcec6BB1e78bAE2A1F0B612012BF14827",
                    "topics": ["0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"],
                    "data": "0x00000000000000000000000000000000000000000000006c6b935b8bbd400000"
                }
            ]
        }
    },
    {
        "id": 3,
        "transaction_type": "compound",
        "vault_id": "beefy-bsc-cake-bnb",
        "chain": "bsc",
        "amount": 25.50,
        "token_amount": "25500000000000000000",
        "gas_used": 115623,
        "gas_price": 5000000000,
        "gas_cost": 0.000578,
        "gas_cost_usd": 0.142,
        "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "block_number": 25923847,
        "status": "confirmed",
        "confirmations": 89,
        "created_at": datetime.utcnow() - timedelta(days=10),
        "confirmed_at": datetime.utcnow() - timedelta(days=10, minutes=-2),
        "from_address": "0x742b15C0B0b17A5D0E0D8EfEFA5b5B2c2C1D4A5B",
        "to_address": "0x3f16b3e3e7A8e0F2E5b9D7F8e4F8B7e4E6E7F8e9",
        "nonce": 57,
        "receipt": {
            "status": 1,
            "gasUsed": 115623,
            "effectiveGasPrice": 5000000000,
            "logs": []  # Compound logs would be here
        }
    },
    {
        "id": 4,
        "transaction_type": "withdraw",
        "vault_id": "beefy-fantom-boo-ftm",
        "chain": "fantom",
        "amount": 425.80,
        "token_amount": "425800000000000000000",
        "gas_used": 167894,
        "gas_price": 80000000000,  # 80 Gwei equivalent
        "gas_cost": 0.0134,  # in FTM
        "gas_cost_usd": 0.0047,
        "tx_hash": "0x567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234",
        "block_number": 52847362,
        "status": "confirmed",
        "confirmations": 234,
        "created_at": datetime.utcnow() - timedelta(days=5),
        "confirmed_at": datetime.utcnow() - timedelta(days=5, minutes=-4),
        "from_address": "0x742b15C0B0b17A5D0E0D8EfEFA5b5B2c2C1D4A5B",
        "to_address": "0xd6070ae98b8069de6B494332d1A1a81B6179D960",
        "nonce": 73,
        "receipt": {
            "status": 1,
            "gasUsed": 167894,
            "effectiveGasPrice": 80000000000,
            "logs": [
                {
                    "address": "0xd6070ae98b8069de6B494332d1A1a81B6179D960",
                    "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"],
                    "data": "0x0000000000000000000000000000000000000000000000171651ba2f7ae40000"
                }
            ]
        }
    },
    {
        "id": 5,
        "transaction_type": "deposit",
        "vault_id": "beefy-ethereum-usdc-eth",
        "chain": "ethereum",
        "amount": 3000.0,
        "token_amount": "3000000000000000000000",
        "gas_used": 156789,
        "gas_price": 30000000000,  # 30 Gwei
        "gas_cost": 0.0047,  # in ETH
        "gas_cost_usd": 7.76,
        "tx_hash": "0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
        "block_number": 16847362,
        "status": "confirmed",
        "confirmations": 45,
        "created_at": datetime.utcnow() - timedelta(days=8),
        "confirmed_at": datetime.utcnow() - timedelta(days=8, minutes=-8),
        "from_address": "0x742b15C0B0b17A5D0E0D8EfEFA5b5B2c2C1D4A5B",
        "to_address": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "nonce": 134,
        "receipt": {
            "status": 1,
            "gasUsed": 156789,
            "effectiveGasPrice": 30000000000,
            "logs": [
                {
                    "address": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
                    "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"],
                    "data": "0x00000000000000000000000000000000000000000000009b18ab5df7180b2000"
                }
            ]
        }
    }
]

# Failed transaction examples
FAILED_TRANSACTIONS = [
    {
        "id": 6,
        "transaction_type": "deposit",
        "vault_id": "beefy-bsc-cake-bnb",
        "chain": "bsc",
        "amount": 500.0,
        "token_amount": "500000000000000000000",
        "gas_used": 21000,  # Failed early
        "gas_price": 5000000000,
        "gas_cost": 0.000105,
        "gas_cost_usd": 0.026,
        "tx_hash": "0x111222333444555666777888999aaabbbcccdddeeefffaaa111222333444555",
        "block_number": 25847380,
        "status": "failed",
        "error_message": "Insufficient token allowance",
        "error_code": "INSUFFICIENT_ALLOWANCE",
        "confirmations": 67,
        "created_at": datetime.utcnow() - timedelta(days=12),
        "confirmed_at": datetime.utcnow() - timedelta(days=12, minutes=-2),
        "from_address": "0x742b15C0B0b17A5D0E0D8EfEFA5b5B2c2C1D4A5B",
        "to_address": "0x3f16b3e3e7A8e0F2E5b9D7F8e4F8B7e4E6E7F8e9",
        "nonce": 48,
        "receipt": {
            "status": 0,
            "gasUsed": 21000,
            "effectiveGasPrice": 5000000000,
            "logs": []
        }
    },
    {
        "id": 7,
        "transaction_type": "withdraw",
        "vault_id": "beefy-polygon-matic-eth",
        "chain": "polygon",
        "amount": 1000.0,
        "token_amount": "1000000000000000000000",
        "gas_used": 85430,
        "gas_price": 40000000000,
        "gas_cost": 0.0034,
        "gas_cost_usd": 0.0029,
        "tx_hash": "0x999888777666555444333222111000aaabbbcccdddeeefffaaa111222333444",
        "block_number": 38456820,
        "status": "failed",
        "error_message": "Vault is paused",
        "error_code": "VAULT_PAUSED",
        "confirmations": 89,
        "created_at": datetime.utcnow() - timedelta(days=18),
        "confirmed_at": datetime.utcnow() - timedelta(days=18, minutes=-3),
        "from_address": "0x742b15C0B0b17A5D0E0D8EfEFA5b5B2c2C1D4A5B",
        "to_address": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
        "nonce": 31,
        "receipt": {
            "status": 0,
            "gasUsed": 85430,
            "effectiveGasPrice": 40000000000,
            "logs": []
        }
    }
]

# Pending transactions
PENDING_TRANSACTIONS = [
    {
        "id": 8,
        "transaction_type": "deposit",
        "vault_id": "beefy-avax-joe-avax",
        "chain": "avalanche",
        "amount": 750.0,
        "token_amount": "750000000000000000000",
        "gas_used": None,
        "gas_price": 30000000000,
        "gas_cost": None,
        "gas_cost_usd": None,
        "tx_hash": "0xpending123456789abcdefpending123456789abcdefpending123456789abcdef",
        "block_number": None,
        "status": "pending",
        "confirmations": 0,
        "created_at": datetime.utcnow() - timedelta(minutes=5),
        "confirmed_at": None,
        "from_address": "0x742b15C0B0b17A5D0E0D8EfEFA5b5B2c2C1D4A5B",
        "to_address": "0x371c7ec6D8039ff7933a2AA28EB827Ffe1F52f07",
        "nonce": 89,
        "receipt": None
    }
]

# Transaction cost estimates by chain and operation
TRANSACTION_COST_ESTIMATES = {
    "ethereum": {
        "deposit": {"gas": 150000, "cost_gwei": 30, "cost_usd": 7.40},
        "withdraw": {"gas": 180000, "cost_gwei": 30, "cost_usd": 8.88},
        "compound": {"gas": 120000, "cost_gwei": 30, "cost_usd": 5.92},
        "approval": {"gas": 46000, "cost_gwei": 30, "cost_usd": 2.27}
    },
    "bsc": {
        "deposit": {"gas": 150000, "cost_gwei": 5, "cost_usd": 0.18},
        "withdraw": {"gas": 180000, "cost_gwei": 5, "cost_usd": 0.22},
        "compound": {"gas": 120000, "cost_gwei": 5, "cost_usd": 0.15},
        "approval": {"gas": 46000, "cost_gwei": 5, "cost_usd": 0.06}
    },
    "polygon": {
        "deposit": {"gas": 150000, "cost_gwei": 40, "cost_usd": 0.0051},
        "withdraw": {"gas": 180000, "cost_gwei": 40, "cost_usd": 0.0061},
        "compound": {"gas": 120000, "cost_gwei": 40, "cost_usd": 0.0041},
        "approval": {"gas": 46000, "cost_gwei": 40, "cost_usd": 0.0016}
    },
    "avalanche": {
        "deposit": {"gas": 150000, "cost_gwei": 30, "cost_usd": 0.069},
        "withdraw": {"gas": 180000, "cost_gwei": 30, "cost_usd": 0.083},
        "compound": {"gas": 120000, "cost_gwei": 30, "cost_usd": 0.055},
        "approval": {"gas": 46000, "cost_gwei": 30, "cost_usd": 0.021}
    },
    "fantom": {
        "deposit": {"gas": 150000, "cost_gwei": 80, "cost_usd": 0.0042},
        "withdraw": {"gas": 180000, "cost_gwei": 80, "cost_usd": 0.0050},
        "compound": {"gas": 120000, "cost_gwei": 80, "cost_usd": 0.0034},
        "approval": {"gas": 46000, "cost_gwei": 80, "cost_usd": 0.0013}
    }
}

# MEV (Maximum Extractable Value) data
MEV_DATA = {
    "ethereum": {
        "average_mev_per_block": 0.045,  # ETH
        "sandwich_attacks_per_day": 234,
        "front_running_risk": "high",
        "protection_available": True
    },
    "bsc": {
        "average_mev_per_block": 0.0021,  # BNB
        "sandwich_attacks_per_day": 89,
        "front_running_risk": "medium",
        "protection_available": False
    },
    "polygon": {
        "average_mev_per_block": 0.12,  # MATIC
        "sandwich_attacks_per_day": 156,
        "front_running_risk": "medium",
        "protection_available": True
    }
}

def generate_transaction_receipt(tx_hash: str, success: bool = True) -> Dict[str, Any]:
    """Generate a mock transaction receipt"""
    return {
        "transactionHash": tx_hash,
        "transactionIndex": 42,
        "blockHash": "0xblock123456789abcdefblock123456789abcdefblock123456789abcdefblock",
        "blockNumber": 12345678,
        "from": "0x742b15C0B0b17A5D0E0D8EfEFA5b5B2c2C1D4A5B",
        "to": "0x3f16b3e3e7A8e0F2E5b9D7F8e4F8B7e4E6E7F8e9",
        "gasUsed": 142856 if success else 21000,
        "effectiveGasPrice": 5000000000,
        "status": 1 if success else 0,
        "logs": [
            {
                "address": "0x3f16b3e3e7A8e0F2E5b9D7F8e4F8B7e4E6E7F8e9",
                "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"],
                "data": "0x00000000000000000000000000000000000000000000003635c9adc5dea00000"
            }
        ] if success else []
    }

def calculate_transaction_cost(chain: str, operation: str, gas_price_tier: str = "standard") -> Dict[str, Any]:
    """Calculate transaction cost for given parameters"""
    if chain not in GAS_PRICES or operation not in TRANSACTION_COST_ESTIMATES.get(chain, {}):
        return {"error": "Unsupported chain or operation"}
    
    gas_estimate = TRANSACTION_COST_ESTIMATES[chain][operation]["gas"]
    gas_price = GAS_PRICES[chain][gas_price_tier]
    
    # Convert to actual cost (simplified)
    if chain == "ethereum":
        gas_cost_eth = (gas_estimate * gas_price) / 1e9
        cost_usd = gas_cost_eth * 1650  # Assume ETH = $1650
    elif chain == "bsc":
        gas_cost_bnb = (gas_estimate * gas_price) / 1e9
        cost_usd = gas_cost_bnb * 245  # Assume BNB = $245
    elif chain == "polygon":
        gas_cost_matic = (gas_estimate * gas_price) / 1e9
        cost_usd = gas_cost_matic * 0.85  # Assume MATIC = $0.85
    else:
        cost_usd = TRANSACTION_COST_ESTIMATES[chain][operation]["cost_usd"]
    
    return {
        "chain": chain,
        "operation": operation,
        "gas_estimate": gas_estimate,
        "gas_price": gas_price,
        "cost_usd": round(cost_usd, 4),
        "gas_price_tier": gas_price_tier
    }

def get_transaction_by_hash(tx_hash: str) -> Dict[str, Any]:
    """Get transaction by hash from sample data"""
    all_transactions = SAMPLE_TRANSACTION_HISTORY + FAILED_TRANSACTIONS + PENDING_TRANSACTIONS
    
    for tx in all_transactions:
        if tx["tx_hash"] == tx_hash:
            return tx
    
    return {"error": "Transaction not found"}

def get_transactions_by_vault(vault_id: str) -> List[Dict[str, Any]]:
    """Get all transactions for a specific vault"""
    all_transactions = SAMPLE_TRANSACTION_HISTORY + FAILED_TRANSACTIONS + PENDING_TRANSACTIONS
    
    return [tx for tx in all_transactions if tx["vault_id"] == vault_id]

def get_transactions_by_status(status: str) -> List[Dict[str, Any]]:
    """Get transactions by status"""
    all_transactions = SAMPLE_TRANSACTION_HISTORY + FAILED_TRANSACTIONS + PENDING_TRANSACTIONS
    
    return [tx for tx in all_transactions if tx["status"] == status]

if __name__ == "__main__":
    # Example usage
    print("Sample transaction cost for BSC deposit:")
    print(calculate_transaction_cost("bsc", "deposit", "standard"))
    
    print("\nTransactions for CAKE-BNB vault:")
    cake_txs = get_transactions_by_vault("beefy-bsc-cake-bnb")
    for tx in cake_txs:
        print(f"- {tx['transaction_type']}: {tx['amount']} (${tx['gas_cost_usd']:.4f} gas)")