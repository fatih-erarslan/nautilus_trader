"""
Integration test for portfolio rebalancing

Tests automated rebalancing strategies and execution.
"""
import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.crypto_trading.beefy_client import BeefyClient
from src.crypto_trading.database import Database, Vault, Investment, Transaction
from src.crypto_trading.strategies import PortfolioOptimizer, RiskManager


class TestPortfolioRebalancing:
    """Test portfolio rebalancing workflows"""
    
    @pytest.fixture
    async def setup(self):
        """Setup test environment with existing portfolio"""
        client = BeefyClient()
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.init_db()
        
        optimizer = PortfolioOptimizer(
            target_return=Decimal("0.12"),
            max_risk=Decimal("0.15"),
            min_position_size=Decimal("500")
        )
        
        risk_manager = RiskManager(
            max_portfolio_risk=Decimal("0.20"),
            max_single_position=Decimal("0.30"),
            max_correlation=Decimal("0.70"),
            max_drawdown=Decimal("0.15")
        )
        
        # Create initial portfolio
        portfolio_data = await self._create_test_portfolio(db)
        
        yield {
            "client": client,
            "db": db,
            "optimizer": optimizer,
            "risk_manager": risk_manager,
            "portfolio": portfolio_data
        }
        
        await client.close()
        await db.close()
    
    async def _create_test_portfolio(self, db: Database) -> Dict[str, Any]:
        """Create a test portfolio with multiple positions"""
        async with db.get_session() as session:
            # Create vaults
            vaults_data = [
                {
                    "vault_id": "beefy-eth-eth",
                    "name": "Beefy ETH",
                    "chain": "ethereum",
                    "apy": Decimal("0.08"),
                    "tvl": Decimal("5000000"),
                    "allocation": Decimal("4000")  # 40%
                },
                {
                    "vault_id": "aave-polygon-usdc",
                    "name": "Aave USDC",
                    "chain": "polygon",
                    "apy": Decimal("0.10"),
                    "tvl": Decimal("3000000"),
                    "allocation": Decimal("3000")  # 30%
                },
                {
                    "vault_id": "curve-bsc-3pool",
                    "name": "Curve 3pool",
                    "chain": "bsc",
                    "apy": Decimal("0.12"),
                    "tvl": Decimal("8000000"),
                    "allocation": Decimal("2000")  # 20%
                },
                {
                    "vault_id": "sushi-arbitrum-wbtc",
                    "name": "Sushi WBTC",
                    "chain": "arbitrum",
                    "apy": Decimal("0.15"),
                    "tvl": Decimal("1500000"),
                    "allocation": Decimal("1000")  # 10%
                }
            ]
            
            vaults = []
            investments = []
            
            for vault_data in vaults_data:
                vault = Vault(
                    vault_id=vault_data["vault_id"],
                    name=vault_data["name"],
                    chain=vault_data["chain"],
                    apy=vault_data["apy"],
                    tvl=vault_data["tvl"]
                )
                session.add(vault)
                await session.commit()
                vaults.append(vault)
                
                investment = Investment(
                    vault_id=vault.id,
                    amount=vault_data["allocation"],
                    shares=vault_data["allocation"],
                    entry_share_price=Decimal("1.0"),
                    status="active",
                    invested_at=datetime.utcnow() - timedelta(days=30)
                )
                session.add(investment)
                investments.append(investment)
            
            await session.commit()
            
            return {
                "vaults": vaults,
                "investments": investments,
                "total_invested": Decimal("10000")
            }
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_threshold_based_rebalancing(self, setup):
        """Test rebalancing when allocations drift beyond threshold"""
        db = setup["db"]
        optimizer = setup["optimizer"]
        portfolio = setup["portfolio"]
        
        print("\n=== Threshold-Based Rebalancing Test ===")
        
        # Define target allocations
        target_allocations = {
            "beefy-eth-eth": Decimal("0.30"),      # Was 40%, reduce
            "aave-polygon-usdc": Decimal("0.30"),   # Was 30%, maintain
            "curve-bsc-3pool": Decimal("0.25"),     # Was 20%, increase
            "sushi-arbitrum-wbtc": Decimal("0.15")  # Was 10%, increase
        }
        
        rebalance_threshold = Decimal("0.05")  # 5% threshold
        
        async with db.get_session() as session:
            # Calculate current allocations
            current_allocations = {}
            total_value = Decimal("0")
            
            for inv in portfolio["investments"]:
                vault = await session.get(Vault, inv.vault_id)
                # Simulate price appreciation
                days_held = 30
                daily_rate = vault.apy / Decimal("365")
                current_share_price = Decimal("1.0") * (Decimal("1") + daily_rate) ** days_held
                
                current_value = inv.shares * current_share_price
                total_value += current_value
                current_allocations[vault.vault_id] = current_value
            
            print(f"Total Portfolio Value: ${total_value:.2f}")
            print("\nCurrent Allocations:")
            
            # Calculate percentage allocations
            rebalance_needed = False
            rebalance_actions = []
            
            for vault_id, current_value in current_allocations.items():
                current_pct = current_value / total_value
                target_pct = target_allocations.get(vault_id, Decimal("0"))
                drift = abs(current_pct - target_pct)
                
                print(f"  {vault_id}:")
                print(f"    Current: {current_pct:.1%} (${current_value:.2f})")
                print(f"    Target:  {target_pct:.1%}")
                print(f"    Drift:   {drift:.1%}")
                
                if drift > rebalance_threshold:
                    rebalance_needed = True
                    target_value = total_value * target_pct
                    adjustment = target_value - current_value
                    
                    if adjustment > 0:
                        rebalance_actions.append({
                            "vault_id": vault_id,
                            "action": "buy",
                            "amount": adjustment
                        })
                    else:
                        rebalance_actions.append({
                            "vault_id": vault_id,
                            "action": "sell",
                            "amount": abs(adjustment)
                        })
            
            if rebalance_needed:
                print("\n⚠️  Rebalancing Required!")
                print("\nRebalancing Actions:")
                
                total_sells = Decimal("0")
                total_buys = Decimal("0")
                
                for action in rebalance_actions:
                    if action["action"] == "sell":
                        print(f"  SELL ${action['amount']:.2f} from {action['vault_id']}")
                        total_sells += action["amount"]
                    else:
                        print(f"  BUY  ${action['amount']:.2f} into {action['vault_id']}")
                        total_buys += action["amount"]
                
                print(f"\nTotal to Sell: ${total_sells:.2f}")
                print(f"Total to Buy:  ${total_buys:.2f}")
                
                # Execute rebalancing
                for action in rebalance_actions:
                    tx = Transaction(
                        vault_id=next(v.id for v in portfolio["vaults"] if v.vault_id == action["vault_id"]),
                        type="rebalance",
                        amount=action["amount"],
                        shares=action["amount"],  # Simplified
                        share_price=Decimal("1.0"),
                        gas_fee=Decimal("10"),
                        status="confirmed",
                        timestamp=datetime.utcnow()
                    )
                    session.add(tx)
                
                await session.commit()
                print("\n✅ Rebalancing executed successfully")
            else:
                print("\n✅ Portfolio is within target allocations - no rebalancing needed")
        
        assert rebalance_needed  # Should need rebalancing in this test
        assert len(rebalance_actions) >= 2  # Multiple positions should need adjustment
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_risk_based_rebalancing(self, setup):
        """Test rebalancing based on risk metrics"""
        db = setup["db"]
        risk_manager = setup["risk_manager"]
        portfolio = setup["portfolio"]
        
        print("\n=== Risk-Based Rebalancing Test ===")
        
        # Simulate risk score changes
        risk_updates = {
            "beefy-eth-eth": {"old": 2.0, "new": 2.5},      # Risk increased
            "aave-polygon-usdc": {"old": 1.5, "new": 1.5},  # No change
            "curve-bsc-3pool": {"old": 2.0, "new": 2.0},    # No change
            "sushi-arbitrum-wbtc": {"old": 3.0, "new": 4.5} # Risk increased significantly
        }
        
        async with db.get_session() as session:
            print("Risk Score Changes:")
            risk_actions = []
            
            for vault in portfolio["vaults"]:
                risk_data = risk_updates[vault.vault_id]
                risk_change = risk_data["new"] - risk_data["old"]
                
                print(f"\n{vault.vault_id}:")
                print(f"  Previous Risk: {risk_data['old']}")
                print(f"  Current Risk:  {risk_data['new']}")
                print(f"  Change:        {'+' if risk_change > 0 else ''}{risk_change}")
                
                # Check if risk exceeds threshold
                if risk_data["new"] > risk_manager.max_portfolio_risk * 20:  # Scaled
                    investment = next(i for i in portfolio["investments"] if i.vault_id == vault.id)
                    
                    # Calculate reduction needed
                    current_allocation = investment.amount / portfolio["total_invested"]
                    max_allowed = Decimal("0.10")  # Max 10% for high-risk
                    
                    if current_allocation > max_allowed:
                        reduction_pct = (current_allocation - max_allowed) / current_allocation
                        reduction_amount = investment.amount * reduction_pct
                        
                        risk_actions.append({
                            "vault_id": vault.vault_id,
                            "action": "reduce_exposure",
                            "amount": reduction_amount,
                            "reason": "risk_exceeded"
                        })
                        
                        print(f"  ⚠️  ACTION: Reduce position by ${reduction_amount:.2f}")
            
            if risk_actions:
                print("\n=== Executing Risk-Based Adjustments ===")
                
                for action in risk_actions:
                    print(f"Reducing {action['vault_id']} by ${action['amount']:.2f}")
                    
                    # Record transaction
                    vault = next(v for v in portfolio["vaults"] if v.vault_id == action["vault_id"])
                    tx = Transaction(
                        vault_id=vault.id,
                        type="risk_adjustment",
                        amount=action["amount"],
                        shares=action["amount"],
                        share_price=Decimal("1.0"),
                        gas_fee=Decimal("15"),
                        status="confirmed",
                        timestamp=datetime.utcnow()
                    )
                    session.add(tx)
                
                await session.commit()
                print("\n✅ Risk adjustments completed")
            else:
                print("\n✅ All positions within risk limits")
        
        assert len(risk_actions) >= 1  # At least one high-risk position
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_yield_optimization_rebalancing(self, setup):
        """Test rebalancing to optimize yield"""
        client = setup["client"]
        db = setup["db"]
        optimizer = setup["optimizer"]
        portfolio = setup["portfolio"]
        
        print("\n=== Yield Optimization Rebalancing Test ===")
        
        # Get current market opportunities
        new_vaults = await client.get_filtered_vaults(
            min_apy=0.15,  # High yield opportunities
            min_tvl=1000000
        )
        
        if new_vaults:
            print(f"Found {len(new_vaults)} high-yield opportunities")
            
            # Compare with current portfolio
            async with db.get_session() as session:
                current_avg_apy = Decimal("0")
                
                for inv in portfolio["investments"]:
                    vault = await session.get(Vault, inv.vault_id)
                    weight = inv.amount / portfolio["total_invested"]
                    current_avg_apy += vault.apy * weight
                
                print(f"\nCurrent Portfolio APY: {current_avg_apy:.2%}")
                
                # Find replacement candidates
                replacement_candidates = []
                
                for new_vault in new_vaults[:5]:  # Top 5
                    new_apy = Decimal(str(new_vault["apy"]))
                    
                    # Find lowest performing current position
                    min_apy_vault = min(portfolio["vaults"], key=lambda v: v.apy)
                    
                    if new_apy > min_apy_vault.apy * Decimal("1.5"):  # 50% higher
                        replacement_candidates.append({
                            "old_vault": min_apy_vault,
                            "new_vault": new_vault,
                            "apy_improvement": new_apy - min_apy_vault.apy
                        })
                
                if replacement_candidates:
                    print("\n=== Yield Optimization Opportunities ===")
                    
                    for candidate in replacement_candidates[:2]:  # Top 2
                        print(f"\nReplace {candidate['old_vault'].vault_id}:")
                        print(f"  Current APY: {candidate['old_vault'].apy:.2%}")
                        print(f"  New APY:     {candidate['new_vault']['apy']:.2%}")
                        print(f"  Improvement: +{candidate['apy_improvement']:.2%}")
                        
                        # Calculate expected portfolio improvement
                        old_investment = next(i for i in portfolio["investments"] 
                                            if i.vault_id == candidate['old_vault'].id)
                        position_weight = old_investment.amount / portfolio["total_invested"]
                        portfolio_improvement = candidate['apy_improvement'] * position_weight
                        
                        print(f"  Portfolio Impact: +{portfolio_improvement:.2%}")
                    
                    # Execute best replacement
                    best_replacement = replacement_candidates[0]
                    
                    print(f"\n✅ Executing replacement:")
                    print(f"   OUT: {best_replacement['old_vault'].vault_id}")
                    print(f"   IN:  {best_replacement['new_vault']['id']}")
                    
                    # Record transactions
                    old_inv = next(i for i in portfolio["investments"] 
                                 if i.vault_id == best_replacement['old_vault'].id)
                    
                    # Withdrawal
                    tx_out = Transaction(
                        vault_id=best_replacement['old_vault'].id,
                        type="withdraw",
                        amount=old_inv.amount,
                        shares=old_inv.shares,
                        share_price=Decimal("1.05"),  # Some appreciation
                        gas_fee=Decimal("20"),
                        status="confirmed",
                        timestamp=datetime.utcnow()
                    )
                    session.add(tx_out)
                    
                    # Create new vault record
                    new_vault_record = Vault(
                        vault_id=best_replacement['new_vault']['id'],
                        name=best_replacement['new_vault'].get('name', 'New Vault'),
                        chain=best_replacement['new_vault']['chain'],
                        apy=Decimal(str(best_replacement['new_vault']['apy'])),
                        tvl=Decimal(str(best_replacement['new_vault']['tvl']))
                    )
                    session.add(new_vault_record)
                    await session.commit()
                    
                    # Deposit to new vault
                    tx_in = Transaction(
                        vault_id=new_vault_record.id,
                        type="deposit",
                        amount=old_inv.amount * Decimal("1.05"),  # Include profits
                        shares=old_inv.amount * Decimal("1.05"),
                        share_price=Decimal("1.0"),
                        gas_fee=Decimal("20"),
                        status="confirmed",
                        timestamp=datetime.utcnow()
                    )
                    session.add(tx_in)
                    
                    await session.commit()
                    
                    print("\n✅ Yield optimization complete")
                else:
                    print("\n✅ Current portfolio is already well-optimized")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_automated_rebalancing_schedule(self, setup):
        """Test automated rebalancing on schedule"""
        db = setup["db"]
        optimizer = setup["optimizer"]
        
        print("\n=== Automated Rebalancing Schedule Test ===")
        
        # Define rebalancing schedule
        rebalance_schedule = {
            "frequency": "weekly",
            "day": "Monday",
            "time": "00:00 UTC",
            "rules": {
                "threshold": Decimal("0.05"),
                "min_trade_size": Decimal("100"),
                "max_trades_per_session": 5
            }
        }
        
        print("Rebalancing Schedule:")
        print(f"  Frequency: {rebalance_schedule['frequency']}")
        print(f"  Execution: {rebalance_schedule['day']} at {rebalance_schedule['time']}")
        print(f"  Threshold: {rebalance_schedule['rules']['threshold']:.1%}")
        
        # Simulate weekly rebalancing for 4 weeks
        for week in range(1, 5):
            print(f"\n=== Week {week} Rebalancing ===")
            
            async with db.get_session() as session:
                # Get all active investments
                from sqlalchemy import select
                stmt = select(Investment).where(Investment.status == "active")
                result = await session.execute(stmt)
                investments = result.scalars().all()
                
                if investments:
                    # Calculate if rebalancing is needed
                    total_value = sum(inv.amount for inv in investments)
                    
                    rebalance_count = 0
                    for inv in investments:
                        current_allocation = inv.amount / total_value
                        target_allocation = Decimal("0.25")  # Equal weight for simplicity
                        
                        if abs(current_allocation - target_allocation) > rebalance_schedule['rules']['threshold']:
                            rebalance_count += 1
                    
                    if rebalance_count > 0:
                        print(f"Rebalancing needed for {rebalance_count} positions")
                    else:
                        print("No rebalancing needed this week")
                    
                    # Record rebalancing check
                    from src.crypto_trading.database import Alert
                    alert = Alert(
                        type="rebalance_check",
                        condition={"week": week, "positions_needing_rebalance": rebalance_count},
                        message=f"Week {week} rebalancing check completed",
                        status="completed",
                        triggered_at=datetime.utcnow()
                    )
                    session.add(alert)
                    await session.commit()
        
        print("\n✅ Automated rebalancing schedule test completed")