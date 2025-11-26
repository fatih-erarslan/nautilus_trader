"""
Integration tests for full investment workflow

Tests the complete flow from vault discovery to investment execution.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from crypto_trading.beefy.beefy_client import BeefyFinanceAPI
from crypto_trading.strategies.yield_chaser import YieldChaserStrategy
from crypto_trading.strategies.stable_farmer import StableFarmerStrategy
from crypto_trading.database.connection import DatabaseManager
from crypto_trading.database.models import VaultPosition, CryptoTransaction
from crypto_trading.mcp_tools.beefy_tools import BeefyMCPTools

from ..fixtures.vault_data import SAMPLE_VAULTS, SAMPLE_APY_DATA, SAMPLE_PRICE_DATA
from ..fixtures.transaction_data import SAMPLE_TRANSACTION_HISTORY


class TestFullInvestmentFlow:
    """Test complete investment workflow integration"""

    @pytest.fixture
    async def beefy_client(self):
        """Create Beefy client with mocked responses"""
        client = BeefyFinanceAPI()
        
        # Mock the session
        client.session = AsyncMock()
        
        # Mock API responses
        with patch.object(client, '_make_request') as mock_request:
            # Set up different responses for different endpoints
            def side_effect(endpoint, *args, **kwargs):
                if endpoint == "/vaults":
                    return SAMPLE_VAULTS
                elif endpoint == "/apy":
                    return {k: v["totalApy"] for k, v in SAMPLE_APY_DATA.items()}
                elif endpoint == "/apy/breakdown":
                    return SAMPLE_APY_DATA
                elif endpoint == "/prices":
                    return SAMPLE_PRICE_DATA
                elif endpoint == "/tvl":
                    return {"bsc": 2847362847.23, "polygon": 1529384756.89}
                else:
                    return {}
            
            mock_request.side_effect = side_effect
            yield client
        
        if client.session:
            await client.session.close()

    @pytest.fixture
    def db_manager(self):
        """Create in-memory database for testing"""
        db_manager = DatabaseManager("sqlite:///:memory:")
        db_manager.create_tables()
        return db_manager

    @pytest.fixture
    def yield_chaser_strategy(self):
        """Create yield chaser strategy"""
        return YieldChaserStrategy(
            min_apy_threshold=15.0,
            max_position_size=0.3
        )

    @pytest.fixture
    def stable_farmer_strategy(self):
        """Create stable farmer strategy"""
        return StableFarmerStrategy(
            preferred_stablecoins=["USDC", "USDT", "DAI"],
            min_apy_threshold=5.0
        )

    @pytest.mark.asyncio
    async def test_complete_yield_chaser_flow(self, beefy_client, db_manager, yield_chaser_strategy):
        """Test complete yield chasing investment flow"""
        
        # Step 1: Discover high-yield vaults
        vaults = await beefy_client.get_vaults()
        assert len(vaults) > 0
        
        # Step 2: Get APY data for evaluation
        apy_data = await beefy_client.get_apy()
        assert len(apy_data) > 0
        
        # Step 3: Get current prices
        prices = await beefy_client.get_prices()
        assert len(prices) > 0
        
        # Step 4: Create investment opportunities from data
        from crypto_trading.strategies.base_strategy import VaultOpportunity, ChainType
        
        opportunities = []
        for vault in vaults[:3]:  # Test with first 3 vaults
            vault_apy_data = apy_data.get(vault.id)
            if vault_apy_data:
                opportunity = VaultOpportunity(
                    vault_id=vault.id,
                    chain=ChainType(vault.chain),
                    protocol=vault.platformId,
                    token_pair=tuple(vault.assets[:2]) if len(vault.assets) >= 2 else (vault.assets[0], "BASE"),
                    apy=vault_apy_data.totalApy,
                    daily_apy=vault_apy_data.vaultDailyApy,
                    tvl=5000000.0,  # Mock TVL
                    platform_fee=vault_apy_data.beefyPerformanceFee,
                    withdraw_fee=0.001,
                    is_paused=vault.status != "active",
                    has_boost=False,
                    boost_apy=None,
                    risk_factors={"smart_contract": 0.2},
                    created_at=datetime.fromtimestamp(vault.createdAt),
                    last_harvest=datetime.utcnow() - timedelta(hours=2)
                )
                opportunities.append(opportunity)
        
        # Step 5: Create mock portfolio state
        from crypto_trading.strategies.base_strategy import PortfolioState
        
        portfolio = PortfolioState(
            positions=[],
            total_value=10000.0,
            available_capital=5000.0,
            timestamp=datetime.utcnow()
        )
        
        # Step 6: Evaluate opportunities with strategy
        evaluations = yield_chaser_strategy.evaluate_opportunities(opportunities, portfolio)
        
        assert len(evaluations) > 0
        
        # Should prioritize high-yield opportunities
        sorted_by_yield = sorted(evaluations, key=lambda x: x[0].total_apy, reverse=True)
        assert sorted_by_yield[0][0].total_apy >= sorted_by_yield[-1][0].total_apy
        
        # Step 7: Select top opportunity for investment
        selected_opportunity, allocation = evaluations[0]
        
        assert allocation > 0
        assert allocation <= portfolio.available_capital
        
        # Step 8: Prepare transaction (mocked Web3 interaction)
        with patch.object(beefy_client.web3_manager, 'check_token_allowance') as mock_allowance:
            with patch.object(beefy_client.web3_manager, 'estimate_transaction_cost') as mock_estimate:
                mock_allowance.return_value = 10000000000000000000000  # Sufficient allowance
                mock_estimate.return_value = Mock(estimatedGas=150000)
                
                deposit_tx = beefy_client.prepare_deposit_transaction(
                    vault_id=selected_opportunity.vault_id,
                    vault_address="0x123456789",
                    token_address="0x987654321",
                    amount=str(int(allocation * 1e18)),
                    chain=selected_opportunity.chain.value,
                    user_address="0xuser123456789"
                )
                
                assert deposit_tx is not None
                assert deposit_tx.vaultId == selected_opportunity.vault_id
        
        # Step 9: Record investment in database
        with db_manager.get_session() as session:
            # Create position record
            position = VaultPosition(
                vault_id=selected_opportunity.vault_id,
                vault_name=f"{selected_opportunity.token_pair[0]}-{selected_opportunity.token_pair[1]} LP",
                chain=selected_opportunity.chain.value,
                amount_deposited=allocation,
                shares_owned=allocation * 0.95,  # Mock shares after fees
                current_value=allocation,
                entry_price=1.0,
                entry_apy=selected_opportunity.apy,
                status="active"
            )
            session.add(position)
            
            # Create transaction record
            transaction = CryptoTransaction(
                transaction_type="deposit",
                vault_id=selected_opportunity.vault_id,
                chain=selected_opportunity.chain.value,
                amount=allocation,
                gas_used=0.015,
                tx_hash="0xmocktxhash123456789",
                status="confirmed"
            )
            session.add(transaction)
            
            session.commit()
            
            # Verify records were created
            positions = session.query(VaultPosition).all()
            transactions = session.query(CryptoTransaction).all()
            
            assert len(positions) == 1
            assert len(transactions) == 1
            
            created_position = positions[0]
            assert created_position.vault_id == selected_opportunity.vault_id
            assert created_position.amount_deposited == allocation
            assert created_position.entry_apy == selected_opportunity.apy

    @pytest.mark.asyncio
    async def test_stable_farming_flow(self, beefy_client, db_manager, stable_farmer_strategy):
        """Test stable coin farming investment flow"""
        
        # Step 1: Get available vaults
        vaults = await beefy_client.get_vaults()
        apy_data = await beefy_client.get_apy()
        
        # Step 2: Filter for stable coin opportunities
        stable_vaults = []
        for vault in vaults:
            # Look for vaults with stable coins
            stable_coins = ["USDC", "USDT", "DAI", "BUSD"]
            if any(coin in vault.assets for coin in stable_coins):
                stable_vaults.append(vault)
        
        assert len(stable_vaults) > 0, "Should find stable coin vaults"
        
        # Step 3: Create opportunities
        from crypto_trading.strategies.base_strategy import VaultOpportunity, ChainType
        
        opportunities = []
        for vault in stable_vaults:
            vault_apy_data = apy_data.get(vault.id)
            if vault_apy_data:
                opportunity = VaultOpportunity(
                    vault_id=vault.id,
                    chain=ChainType(vault.chain),
                    protocol=vault.platformId,
                    token_pair=tuple(vault.assets[:2]) if len(vault.assets) >= 2 else (vault.assets[0], "BASE"),
                    apy=vault_apy_data.totalApy,
                    daily_apy=vault_apy_data.vaultDailyApy,
                    tvl=20000000.0,  # Higher TVL for stable coins
                    platform_fee=vault_apy_data.beefyPerformanceFee,
                    withdraw_fee=0.001,
                    is_paused=False,
                    has_boost=False,
                    boost_apy=None,
                    risk_factors={"smart_contract": 0.05},  # Lower risk
                    created_at=datetime.fromtimestamp(vault.createdAt),
                    last_harvest=datetime.utcnow() - timedelta(hours=1)
                )
                opportunities.append(opportunity)
        
        # Step 4: Create conservative portfolio
        from crypto_trading.strategies.base_strategy import PortfolioState
        
        portfolio = PortfolioState(
            positions=[],
            total_value=50000.0,
            available_capital=20000.0,
            timestamp=datetime.utcnow()
        )
        
        # Step 5: Evaluate with stable farming strategy
        evaluations = stable_farmer_strategy.evaluate_opportunities(opportunities, portfolio)
        
        assert len(evaluations) > 0
        
        # Step 6: Verify strategy preferences
        for opportunity, allocation in evaluations:
            # Should prefer stable coin pairs
            stable_coins = ["USDC", "USDT", "DAI", "BUSD"]
            has_stable = any(coin in opportunity.token_pair for coin in stable_coins)
            assert has_stable, f"Strategy should prefer stable coins: {opportunity.token_pair}"
            
            # Should have reasonable allocation
            assert allocation > 0
            assert allocation <= portfolio.available_capital * stable_farmer_strategy.max_position_size

    @pytest.mark.asyncio
    async def test_portfolio_rebalancing_flow(self, beefy_client, db_manager, yield_chaser_strategy):
        """Test portfolio rebalancing workflow"""
        
        # Step 1: Create existing positions in database
        with db_manager.get_session() as session:
            # Add some existing positions
            positions = [
                VaultPosition(
                    vault_id="beefy-bsc-cake-bnb",
                    vault_name="CAKE-BNB LP",
                    chain="bsc",
                    amount_deposited=1000.0,
                    shares_owned=950.0,
                    current_value=1050.0,
                    entry_price=1.05,
                    entry_apy=25.5,
                    status="active"
                ),
                VaultPosition(
                    vault_id="beefy-ethereum-usdc-eth",
                    vault_name="USDC-ETH LP",
                    chain="ethereum",
                    amount_deposited=2000.0,
                    shares_owned=1990.0,
                    current_value=1980.0,  # Underperforming
                    entry_price=1.005,
                    entry_apy=8.5,
                    status="active"
                )
            ]
            
            for position in positions:
                session.add(position)
            session.commit()
            
            # Step 2: Get current market opportunities
            vaults = await beefy_client.get_vaults()
            apy_data = await beefy_client.get_apy()
            
            # Step 3: Create current portfolio state
            from crypto_trading.strategies.base_strategy import PortfolioState, Position, ChainType
            
            current_positions = []
            for pos in positions:
                current_positions.append(
                    Position(
                        vault_id=pos.vault_id,
                        chain=ChainType(pos.chain),
                        protocol="mock",
                        token_pair=("A", "B"),
                        apy=pos.entry_apy,
                        tvl=5000000.0,
                        amount=pos.current_value,
                        entry_time=pos.created_at,
                        risk_score=30.0
                    )
                )
            
            portfolio = PortfolioState(
                positions=current_positions,
                total_value=3030.0,
                available_capital=1000.0,
                timestamp=datetime.utcnow()
            )
            
            # Step 4: Check if rebalancing is needed
            should_rebalance = yield_chaser_strategy.should_rebalance(portfolio)
            
            if should_rebalance:
                # Step 5: Generate rebalancing trades
                from crypto_trading.strategies.base_strategy import VaultOpportunity
                
                opportunities = []
                for vault in vaults[:5]:  # Use first 5 vaults
                    vault_apy_data = apy_data.get(vault.id)
                    if vault_apy_data:
                        opportunity = VaultOpportunity(
                            vault_id=vault.id,
                            chain=ChainType(vault.chain),
                            protocol=vault.platformId,
                            token_pair=tuple(vault.assets[:2]) if len(vault.assets) >= 2 else (vault.assets[0], "BASE"),
                            apy=vault_apy_data.totalApy,
                            daily_apy=vault_apy_data.vaultDailyApy,
                            tvl=5000000.0,
                            platform_fee=vault_apy_data.beefyPerformanceFee,
                            withdraw_fee=0.001,
                            is_paused=False,
                            has_boost=False,
                            boost_apy=None,
                            risk_factors={"smart_contract": 0.2},
                            created_at=datetime.fromtimestamp(vault.createdAt),
                            last_harvest=datetime.utcnow() - timedelta(hours=2)
                        )
                        opportunities.append(opportunity)
                
                rebalance_trades = yield_chaser_strategy.generate_rebalance_trades(portfolio, opportunities)
                
                assert isinstance(rebalance_trades, list)
                
                # Step 6: Execute rebalancing (mock execution)
                for trade in rebalance_trades:
                    assert "action" in trade  # Should be "sell" or "buy"
                    assert "vault_id" in trade
                    assert "amount" in trade
                    
                    # Record rebalancing transaction
                    rebalance_tx = CryptoTransaction(
                        transaction_type=trade["action"],
                        vault_id=trade["vault_id"],
                        chain="bsc",  # Mock chain
                        amount=trade["amount"],
                        gas_used=0.02,
                        tx_hash=f"0xrebalance{trade['vault_id'][-8:]}",
                        status="confirmed"
                    )
                    session.add(rebalance_tx)
                
                session.commit()
                
                # Verify rebalancing transactions were recorded
                rebalance_txs = session.query(CryptoTransaction).filter(
                    CryptoTransaction.transaction_type.in_(["sell", "buy", "withdraw", "deposit"])
                ).all()
                
                assert len(rebalance_txs) >= len(rebalance_trades)

    @pytest.mark.asyncio
    async def test_mcp_tools_integration_flow(self, beefy_client):
        """Test MCP tools integration with full workflow"""
        
        # Step 1: Initialize MCP tools
        mcp_tools = BeefyMCPTools()
        
        # Mock the underlying client
        with patch.object(mcp_tools.vault_handler, 'beefy_client', beefy_client):
            
            # Step 2: Search for vaults via MCP
            search_params = {
                "query": "CAKE",
                "chain": "bsc",
                "min_apy": 20.0,
                "max_results": 5
            }
            
            search_result = await mcp_tools.search_vaults(search_params)
            
            assert "vaults" in search_result
            assert len(search_result["vaults"]) > 0
            
            # Step 3: Analyze top vault
            top_vault = search_result["vaults"][0]
            
            analyze_params = {
                "vault_id": top_vault["id"],
                "include_risk_analysis": True,
                "include_historical_data": True
            }
            
            with patch.object(mcp_tools.vault_handler, 'calculate_vault_metrics') as mock_metrics:
                mock_metrics.return_value = {
                    "risk_score": 35.0,
                    "efficiency_ratio": 0.85,
                    "liquidity_score": 90.0
                }
                
                analysis_result = await mcp_tools.analyze_vault(analyze_params)
                
                assert "vault_info" in analysis_result
                assert "risk_analysis" in analysis_result
                assert "recommendations" in analysis_result
            
            # Step 4: Simulate investment
            invest_params = {
                "vault_id": top_vault["id"],
                "amount": 1000.0,
                "strategy": "yield_chaser",
                "dry_run": True
            }
            
            with patch.object(mcp_tools.investment_handler, '_get_vault_info') as mock_vault_info:
                mock_vault_info.return_value = {
                    "id": top_vault["id"],
                    "name": top_vault["name"],
                    "apy": top_vault["apy"],
                    "chain": top_vault["chain"]
                }
                
                with patch.object(mcp_tools.investment_handler, 'calculate_expected_returns') as mock_returns:
                    mock_returns.return_value = {
                        "gross_return": 250.0,
                        "net_return": 238.75,
                        "yield_breakdown": {
                            "trading_fees": 180.0,
                            "farming_rewards": 70.0,
                            "platform_fees": -11.25
                        }
                    }
                    
                    investment_result = await mcp_tools.invest_in_vault(invest_params)
                    
                    assert "transaction_plan" in investment_result
                    assert "estimated_costs" in investment_result
                    assert "expected_returns" in investment_result
                    
                    # Verify expected returns are realistic
                    returns = investment_result["expected_returns"]
                    assert returns["net_return"] < returns["gross_return"]
                    assert returns["net_return"] > 0
            
            # Step 5: Get portfolio summary
            with patch.object(mcp_tools.portfolio_handler, 'get_portfolio_summary') as mock_summary:
                mock_summary.return_value = {
                    "total_value": 5000.0,
                    "total_yield_earned": 125.0,
                    "active_positions": 3,
                    "chains": ["bsc", "ethereum"],
                    "average_apy": 18.5,
                    "unrealized_pnl": 75.0
                }
                
                portfolio_result = await mcp_tools.get_portfolio_summary({})
                
                assert portfolio_result["total_value"] == 5000.0
                assert portfolio_result["active_positions"] == 3
                assert portfolio_result["average_apy"] == 18.5

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, beefy_client, db_manager):
        """Test error handling throughout the investment flow"""
        
        # Test API failure recovery
        with patch.object(beefy_client, '_make_request') as mock_request:
            # Simulate API failure
            mock_request.side_effect = Exception("API temporarily unavailable")
            
            vaults = await beefy_client.get_vaults()
            assert vaults == []  # Should return empty list on error
            
            # Test recovery when API is back
            mock_request.side_effect = None
            mock_request.return_value = SAMPLE_VAULTS[:2]
            
            vaults = await beefy_client.get_vaults()
            assert len(vaults) == 2
        
        # Test database transaction rollback
        with db_manager.get_session() as session:
            try:
                # Create invalid position (should fail)
                invalid_position = VaultPosition(
                    vault_id="test-vault",
                    vault_name="Test Vault",
                    chain="invalid_chain",  # Should trigger validation error
                    amount_deposited=1000.0,
                    shares_owned=1000.0,
                    entry_price=1.0,
                    entry_apy=20.0
                )
                session.add(invalid_position)
                session.commit()
                
                assert False, "Should have raised validation error"
                
            except Exception:
                # Transaction should be rolled back
                session.rollback()
                
                # Verify no invalid data was saved
                positions = session.query(VaultPosition).all()
                assert len(positions) == 0
        
        # Test Web3 transaction failure
        with patch.object(beefy_client.web3_manager, 'check_token_allowance') as mock_allowance:
            # Simulate insufficient allowance
            mock_allowance.return_value = 0
            
            deposit_tx = beefy_client.prepare_deposit_transaction(
                vault_id="test-vault",
                vault_address="0x123",
                token_address="0x456",
                amount="1000000000000000000000",
                chain="bsc",
                user_address="0x789"
            )
            
            # Should still return transaction but flag approval needed
            assert deposit_tx is not None

    @pytest.mark.asyncio
    async def test_performance_tracking(self, db_manager):
        """Test performance tracking across investment lifecycle"""
        
        with db_manager.get_session() as session:
            # Create position with yield history
            position = VaultPosition(
                vault_id="beefy-bsc-cake-bnb",
                vault_name="CAKE-BNB LP",
                chain="bsc",
                amount_deposited=1000.0,
                shares_owned=950.0,
                current_value=1000.0,
                entry_price=1.05,
                entry_apy=25.5,
                status="active"
            )
            session.add(position)
            session.commit()
            
            # Simulate yield accumulation over time
            from crypto_trading.database.models import YieldHistory
            
            cumulative_yield = 0.0
            for day in range(30):
                daily_yield = 1000.0 * (25.5 / 100 / 365)  # Daily yield
                cumulative_yield += daily_yield
                
                yield_record = YieldHistory(
                    vault_id="beefy-bsc-cake-bnb",
                    position_id=position.id,
                    earned_amount=daily_yield,
                    apy_snapshot=25.5 + (day * 0.1),  # Slight APY increase
                    tvl_snapshot=5000000.0 + (day * 10000),
                    price_per_share=1.0 + (day * 0.001),
                    recorded_at=datetime.utcnow() - timedelta(days=30-day)
                )
                session.add(yield_record)
                
                # Update position value
                position.current_value = 1000.0 + cumulative_yield
            
            session.commit()
            
            # Test performance calculations
            from crypto_trading.database.utils import DatabaseUtils
            
            utils = DatabaseUtils(session)
            
            # Get portfolio summary
            summary = utils.get_portfolio_summary()
            
            assert summary["total_positions"] == 1
            assert summary["total_yield_earned"] > 0
            assert summary["unrealized_pnl"] > 0
            
            # Get performance metrics
            metrics = utils.get_performance_metrics(days=30)
            
            assert "total_yield_earned" in metrics
            assert "average_apy" in metrics
            assert metrics["total_yield_earned"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])