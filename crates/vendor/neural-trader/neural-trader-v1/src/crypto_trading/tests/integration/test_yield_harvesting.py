"""
Integration test for yield harvesting workflows

Tests automated yield harvesting, compounding, and optimization.
"""
import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.crypto_trading.beefy_client import BeefyClient
from src.crypto_trading.database import Database, Vault, Investment, Transaction
from src.crypto_trading.strategies import YieldFarmingStrategy


class TestYieldHarvesting:
    """Test yield harvesting and compounding workflows"""
    
    @pytest.fixture
    async def setup(self):
        """Setup test environment with yield-generating positions"""
        client = BeefyClient()
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.init_db()
        
        strategy = YieldFarmingStrategy(
            min_apy=Decimal("0.05"),
            max_risk_score=Decimal("3"),
            diversification_factor=5
        )
        
        # Create portfolio with accumulated yields
        portfolio_data = await self._create_yielding_portfolio(db)
        
        yield {
            "client": client,
            "db": db,
            "strategy": strategy,
            "portfolio": portfolio_data
        }
        
        await client.close()
        await db.close()
    
    async def _create_yielding_portfolio(self, db: Database) -> Dict[str, Any]:
        """Create portfolio with positions that have accumulated yields"""
        async with db.get_session() as session:
            # Create vaults with different yield characteristics
            vaults_data = [
                {
                    "vault_id": "high-yield-daily",
                    "name": "High Yield Daily Compound",
                    "chain": "polygon",
                    "apy": Decimal("0.25"),  # 25% APY
                    "compound_frequency": "daily",
                    "investment": Decimal("5000"),
                    "days_invested": 30
                },
                {
                    "vault_id": "stable-weekly",
                    "name": "Stable Weekly Harvest",
                    "chain": "bsc",
                    "apy": Decimal("0.12"),  # 12% APY
                    "compound_frequency": "weekly",
                    "investment": Decimal("10000"),
                    "days_invested": 60
                },
                {
                    "vault_id": "moderate-monthly",
                    "name": "Moderate Monthly Yield",
                    "chain": "ethereum",
                    "apy": Decimal("0.08"),  # 8% APY
                    "compound_frequency": "monthly",
                    "investment": Decimal("15000"),
                    "days_invested": 90
                },
                {
                    "vault_id": "autocompound-vault",
                    "name": "Auto-Compound Vault",
                    "chain": "arbitrum",
                    "apy": Decimal("0.18"),  # 18% APY
                    "compound_frequency": "auto",
                    "investment": Decimal("8000"),
                    "days_invested": 45
                }
            ]
            
            portfolio = []
            total_invested = Decimal("0")
            
            for data in vaults_data:
                # Create vault
                vault = Vault(
                    vault_id=data["vault_id"],
                    name=data["name"],
                    chain=data["chain"],
                    apy=data["apy"],
                    tvl=Decimal("1000000")
                )
                session.add(vault)
                await session.commit()
                
                # Calculate accumulated value
                daily_rate = data["apy"] / Decimal("365")
                if data["compound_frequency"] == "daily":
                    compounds_per_year = 365
                elif data["compound_frequency"] == "weekly":
                    compounds_per_year = 52
                elif data["compound_frequency"] == "monthly":
                    compounds_per_year = 12
                else:  # auto
                    compounds_per_year = 365
                
                # Calculate current value with compound interest
                periods = data["days_invested"] * compounds_per_year / 365
                rate_per_period = data["apy"] / compounds_per_year
                current_value = data["investment"] * (Decimal("1") + rate_per_period) ** periods
                
                # Create investment
                investment = Investment(
                    vault_id=vault.id,
                    amount=data["investment"],
                    shares=data["investment"],
                    entry_share_price=Decimal("1.0"),
                    status="active",
                    invested_at=datetime.utcnow() - timedelta(days=data["days_invested"])
                )
                session.add(investment)
                
                total_invested += data["investment"]
                
                portfolio.append({
                    "vault": vault,
                    "investment": investment,
                    "compound_frequency": data["compound_frequency"],
                    "current_value": current_value,
                    "accumulated_yield": current_value - data["investment"]
                })
            
            await session.commit()
            
            return {
                "positions": portfolio,
                "total_invested": total_invested
            }
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_manual_harvest_workflow(self, setup):
        """Test manual harvest and reinvestment workflow"""
        db = setup["db"]
        portfolio = setup["portfolio"]
        
        print("\n=== Manual Harvest Workflow Test ===")
        print(f"Total Invested: ${portfolio['total_invested']:,.2f}")
        
        async with db.get_session() as session:
            total_harvested = Decimal("0")
            harvest_transactions = []
            
            for position in portfolio["positions"]:
                if position["compound_frequency"] != "auto":
                    print(f"\nVault: {position['vault'].name}")
                    print(f"  Chain: {position['vault'].chain}")
                    print(f"  APY: {position['vault'].apy:.2%}")
                    print(f"  Invested: ${position['investment'].amount:,.2f}")
                    print(f"  Current Value: ${position['current_value']:,.2f}")
                    print(f"  Accumulated Yield: ${position['accumulated_yield']:,.2f}")
                    
                    # Calculate harvestable amount
                    harvestable = position['accumulated_yield'] * Decimal("0.95")  # 5% left for gas
                    
                    if harvestable > Decimal("50"):  # Minimum harvest threshold
                        print(f"  ✅ Harvesting: ${harvestable:,.2f}")
                        
                        # Record harvest transaction
                        harvest_tx = Transaction(
                            vault_id=position['vault'].id,
                            type="harvest",
                            amount=harvestable,
                            shares=Decimal("0"),  # No shares for harvest
                            share_price=position['current_value'] / position['investment'].shares,
                            gas_fee=Decimal("10"),
                            tx_hash=f"0xharvest{position['vault'].id}",
                            status="confirmed",
                            timestamp=datetime.utcnow()
                        )
                        session.add(harvest_tx)
                        harvest_transactions.append(harvest_tx)
                        total_harvested += harvestable
                    else:
                        print(f"  ⏸️  Below minimum threshold (${harvestable:,.2f} < $50)")
            
            await session.commit()
            
            print(f"\n=== Harvest Summary ===")
            print(f"Total Harvested: ${total_harvested:,.2f}")
            print(f"Number of Harvests: {len(harvest_transactions)}")
            print(f"Average Harvest: ${total_harvested / len(harvest_transactions):,.2f}" if harvest_transactions else "N/A")
            
            # Test reinvestment options
            if total_harvested > Decimal("100"):
                print("\n=== Reinvestment Options ===")
                
                # Option 1: Reinvest proportionally
                print("\nOption 1: Proportional Reinvestment")
                for position in portfolio["positions"]:
                    weight = position['investment'].amount / portfolio['total_invested']
                    reinvest_amount = total_harvested * weight
                    print(f"  {position['vault'].vault_id}: ${reinvest_amount:,.2f} ({weight:.1%})")
                
                # Option 2: Reinvest in highest APY
                print("\nOption 2: Highest APY Focus")
                highest_apy_position = max(portfolio["positions"], key=lambda p: p['vault'].apy)
                print(f"  All ${total_harvested:,.2f} → {highest_apy_position['vault'].vault_id} (APY: {highest_apy_position['vault'].apy:.2%})")
                
                # Option 3: New opportunity
                print("\nOption 3: New Opportunity Investment")
                print(f"  Explore new vaults with harvested ${total_harvested:,.2f}")
                
                # Execute Option 1 for test
                print("\n✅ Executing proportional reinvestment...")
                
                for position in portfolio["positions"]:
                    weight = position['investment'].amount / portfolio['total_invested']
                    reinvest_amount = total_harvested * weight
                    
                    if reinvest_amount > Decimal("10"):
                        reinvest_tx = Transaction(
                            vault_id=position['vault'].id,
                            type="reinvest",
                            amount=reinvest_amount,
                            shares=reinvest_amount,  # Simplified
                            share_price=Decimal("1.0"),
                            gas_fee=Decimal("5"),
                            status="confirmed",
                            timestamp=datetime.utcnow()
                        )
                        session.add(reinvest_tx)
                
                await session.commit()
                print("✅ Reinvestment completed")
        
        assert total_harvested > Decimal("0")
        assert len(harvest_transactions) >= 2
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_automated_compound_strategy(self, setup):
        """Test automated compounding strategies"""
        db = setup["db"]
        portfolio = setup["portfolio"]
        
        print("\n=== Automated Compound Strategy Test ===")
        
        # Define compound strategies
        compound_strategies = {
            "aggressive": {
                "frequency": "daily",
                "min_amount": Decimal("10"),
                "gas_threshold": Decimal("0.01")  # 1% gas threshold
            },
            "balanced": {
                "frequency": "weekly",
                "min_amount": Decimal("50"),
                "gas_threshold": Decimal("0.02")  # 2% gas threshold
            },
            "conservative": {
                "frequency": "monthly",
                "min_amount": Decimal("100"),
                "gas_threshold": Decimal("0.03")  # 3% gas threshold
            }
        }
        
        async with db.get_session() as session:
            for strategy_name, strategy in compound_strategies.items():
                print(f"\n=== Testing {strategy_name.upper()} Strategy ===")
                print(f"Frequency: {strategy['frequency']}")
                print(f"Min Amount: ${strategy['min_amount']}")
                print(f"Gas Threshold: {strategy['gas_threshold']:.1%}")
                
                compound_count = 0
                total_compounded = Decimal("0")
                gas_spent = Decimal("0")
                
                for position in portfolio["positions"]:
                    if position["compound_frequency"] != "auto":
                        # Check if compounding is due
                        days_since_investment = (datetime.utcnow() - position['investment'].invested_at).days
                        
                        if strategy['frequency'] == "daily":
                            compounds_due = days_since_investment
                        elif strategy['frequency'] == "weekly":
                            compounds_due = days_since_investment // 7
                        else:  # monthly
                            compounds_due = days_since_investment // 30
                        
                        if compounds_due > 0:
                            # Calculate compound amount
                            pending_yield = position['accumulated_yield'] / compounds_due
                            
                            # Check if meets minimum and gas efficiency
                            gas_fee = Decimal("5")
                            gas_ratio = gas_fee / pending_yield if pending_yield > 0 else Decimal("1")
                            
                            if pending_yield >= strategy['min_amount'] and gas_ratio <= strategy['gas_threshold']:
                                print(f"\n  {position['vault'].vault_id}:")
                                print(f"    Pending: ${pending_yield:,.2f}")
                                print(f"    Gas Fee: ${gas_fee:,.2f} ({gas_ratio:.1%})")
                                print(f"    ✅ Compounding")
                                
                                compound_count += 1
                                total_compounded += pending_yield
                                gas_spent += gas_fee
                            else:
                                print(f"\n  {position['vault'].vault_id}:")
                                print(f"    Pending: ${pending_yield:,.2f}")
                                if pending_yield < strategy['min_amount']:
                                    print(f"    ❌ Below minimum (${strategy['min_amount']})")
                                else:
                                    print(f"    ❌ Gas inefficient ({gas_ratio:.1%} > {strategy['gas_threshold']:.1%})")
                
                print(f"\n{strategy_name.upper()} Strategy Results:")
                print(f"  Compounds: {compound_count}")
                print(f"  Total Compounded: ${total_compounded:,.2f}")
                print(f"  Gas Spent: ${gas_spent:,.2f}")
                print(f"  Net Gain: ${total_compounded - gas_spent:,.2f}")
                
                if total_compounded > 0:
                    efficiency = (total_compounded - gas_spent) / total_compounded
                    print(f"  Efficiency: {efficiency:.1%}")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_optimal_harvest_timing(self, setup):
        """Test optimal harvest timing calculations"""
        portfolio = setup["portfolio"]
        
        print("\n=== Optimal Harvest Timing Analysis ===")
        
        # Gas prices at different times (simplified)
        gas_prices = {
            "low": Decimal("5"),      # Off-peak
            "medium": Decimal("15"),  # Normal
            "high": Decimal("30")     # Peak congestion
        }
        
        for position in portfolio["positions"]:
            if position["compound_frequency"] != "auto":
                print(f"\n{position['vault'].name}:")
                print(f"  Daily Yield: ${position['accumulated_yield'] / 30:,.2f}")
                
                # Calculate optimal harvest frequency for each gas price
                daily_yield = position['accumulated_yield'] / 30
                
                for gas_level, gas_price in gas_prices.items():
                    # Find days where gas is <= 2% of harvest
                    days_to_wait = 1
                    while (gas_price / (daily_yield * days_to_wait)) > Decimal("0.02"):
                        days_to_wait += 1
                    
                    harvest_amount = daily_yield * days_to_wait
                    net_profit = harvest_amount - gas_price
                    
                    print(f"\n  Gas Price: {gas_level.upper()} (${gas_price})")
                    print(f"    Optimal Wait: {days_to_wait} days")
                    print(f"    Harvest Amount: ${harvest_amount:,.2f}")
                    print(f"    Net Profit: ${net_profit:,.2f}")
                    print(f"    Gas Ratio: {(gas_price/harvest_amount)*100:.1f}%")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_chain_harvest_optimization(self, setup):
        """Test harvest optimization across multiple chains"""
        db = setup["db"]
        portfolio = setup["portfolio"]
        
        print("\n=== Multi-Chain Harvest Optimization ===")
        
        # Chain-specific gas costs and times
        chain_data = {
            "ethereum": {"gas_cost": Decimal("30"), "confirmation_time": 15},
            "polygon": {"gas_cost": Decimal("0.10"), "confirmation_time": 2},
            "bsc": {"gas_cost": Decimal("0.50"), "confirmation_time": 3},
            "arbitrum": {"gas_cost": Decimal("5"), "confirmation_time": 1}
        }
        
        async with db.get_session() as session:
            # Group positions by chain
            chain_positions = {}
            for position in portfolio["positions"]:
                chain = position['vault'].chain
                if chain not in chain_positions:
                    chain_positions[chain] = []
                chain_positions[chain].append(position)
            
            print("Positions by Chain:")
            for chain, positions in chain_positions.items():
                total_yield = sum(p['accumulated_yield'] for p in positions)
                print(f"\n{chain.upper()}:")
                print(f"  Positions: {len(positions)}")
                print(f"  Total Yield: ${total_yield:,.2f}")
                print(f"  Gas Cost: ${chain_data[chain]['gas_cost']}")
                print(f"  Confirmation: {chain_data[chain]['confirmation_time']} min")
            
            # Optimize harvest order
            print("\n=== Optimal Harvest Sequence ===")
            
            harvest_plan = []
            
            for chain, positions in chain_positions.items():
                chain_yield = sum(p['accumulated_yield'] for p in positions)
                gas_cost = chain_data[chain]['gas_cost']
                
                # Calculate efficiency score
                efficiency = (chain_yield - gas_cost) / gas_cost if gas_cost > 0 else float('inf')
                
                harvest_plan.append({
                    "chain": chain,
                    "positions": len(positions),
                    "total_yield": chain_yield,
                    "gas_cost": gas_cost,
                    "net_profit": chain_yield - gas_cost,
                    "efficiency": efficiency,
                    "time": chain_data[chain]['confirmation_time']
                })
            
            # Sort by efficiency
            harvest_plan.sort(key=lambda x: x['efficiency'], reverse=True)
            
            print("\nRecommended Harvest Order (by efficiency):")
            total_time = 0
            total_profit = Decimal("0")
            
            for i, plan in enumerate(harvest_plan, 1):
                if plan['net_profit'] > 0:
                    print(f"\n{i}. {plan['chain'].upper()}")
                    print(f"   Yield: ${plan['total_yield']:,.2f}")
                    print(f"   Gas: ${plan['gas_cost']:,.2f}")
                    print(f"   Net: ${plan['net_profit']:,.2f}")
                    print(f"   Efficiency: {plan['efficiency']:.1f}x")
                    print(f"   Time: {plan['time']} min")
                    
                    total_time += plan['time']
                    total_profit += plan['net_profit']
                    
                    # Execute harvest
                    for position in chain_positions[plan['chain']]:
                        if position['accumulated_yield'] > plan['gas_cost']:
                            tx = Transaction(
                                vault_id=position['vault'].id,
                                type="batch_harvest",
                                amount=position['accumulated_yield'],
                                shares=Decimal("0"),
                                share_price=Decimal("1.0"),
                                gas_fee=plan['gas_cost'] / len(chain_positions[plan['chain']]),
                                status="confirmed",
                                timestamp=datetime.utcnow()
                            )
                            session.add(tx)
            
            await session.commit()
            
            print(f"\n=== Batch Harvest Summary ===")
            print(f"Total Time: {total_time} minutes")
            print(f"Total Net Profit: ${total_profit:,.2f}")
            print(f"Average Profit per Minute: ${total_profit/total_time:,.2f}")
        
        assert len(harvest_plan) >= 3
        assert total_profit > Decimal("0")