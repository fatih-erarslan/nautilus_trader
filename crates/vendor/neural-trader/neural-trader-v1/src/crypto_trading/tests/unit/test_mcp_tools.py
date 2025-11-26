"""
Comprehensive tests for MCP tools and handlers

Tests all MCP tool endpoints, handlers, and integration functionality.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from crypto_trading.mcp_tools.beefy_tools import BeefyMCPTools
from crypto_trading.mcp_tools.handlers.vault_handler import VaultHandler
from crypto_trading.mcp_tools.handlers.investment_handler import InvestmentHandler
from crypto_trading.mcp_tools.handlers.portfolio_handler import PortfolioHandler
from crypto_trading.mcp_tools.handlers.analytics_handler import AnalyticsHandler
from crypto_trading.mcp_tools.integration import MCPIntegration
from crypto_trading.mcp_tools.schemas import (
    VaultSearchParams, InvestmentParams, PortfolioAnalysisParams
)


class TestBeefyMCPTools:
    """Test main MCP tools interface"""

    @pytest.fixture
    def mcp_tools(self):
        """Create MCP tools instance"""
        return BeefyMCPTools()

    @pytest.fixture
    def mock_beefy_client(self):
        """Mock Beefy client for testing"""
        mock_client = AsyncMock()
        
        # Mock vault data
        mock_client.get_vaults.return_value = [
            Mock(
                id="beefy-bsc-cake-bnb",
                name="CAKE-BNB LP",
                chain="bsc",
                platform="pancakeswap",
                apy=25.5,
                tvl=5000000.0,
                status="active",
                assets=["CAKE", "BNB"],
                risks=["COMPLEXITY_LOW"],
                earnContractAddress="0x123"
            )
        ]
        
        # Mock APY data
        mock_client.get_apy.return_value = {
            "beefy-bsc-cake-bnb": Mock(
                vaultId="beefy-bsc-cake-bnb",
                totalApy=25.5,
                vaultApy=24.0,
                beefyPerformanceFee=0.045
            )
        }
        
        # Mock prices
        mock_client.get_prices.return_value = {
            "CAKE": Mock(symbol="CAKE", price=2.15),
            "BNB": Mock(symbol="BNB", price=245.67)
        }
        
        return mock_client

    def test_mcp_tools_initialization(self, mcp_tools):
        """Test MCP tools initialization"""
        assert mcp_tools.vault_handler is not None
        assert mcp_tools.investment_handler is not None
        assert mcp_tools.portfolio_handler is not None
        assert mcp_tools.analytics_handler is not None

    @pytest.mark.asyncio
    async def test_list_available_tools(self, mcp_tools):
        """Test listing available MCP tools"""
        tools = await mcp_tools.list_available_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check for expected tools
        tool_names = [tool["name"] for tool in tools]
        assert "search_vaults" in tool_names
        assert "analyze_vault" in tool_names
        assert "invest_in_vault" in tool_names
        assert "get_portfolio_summary" in tool_names

    @pytest.mark.asyncio
    async def test_search_vaults_tool(self, mcp_tools, mock_beefy_client):
        """Test vault search MCP tool"""
        with patch.object(mcp_tools.vault_handler, 'beefy_client', mock_beefy_client):
            params = {
                "query": "CAKE",
                "chain": "bsc",
                "min_apy": 20.0,
                "max_results": 10
            }
            
            result = await mcp_tools.search_vaults(params)
            
            assert "vaults" in result
            assert isinstance(result["vaults"], list)
            if result["vaults"]:
                vault = result["vaults"][0]
                assert "id" in vault
                assert "name" in vault
                assert "apy" in vault

    @pytest.mark.asyncio
    async def test_analyze_vault_tool(self, mcp_tools, mock_beefy_client):
        """Test vault analysis MCP tool"""
        with patch.object(mcp_tools.vault_handler, 'beefy_client', mock_beefy_client):
            params = {
                "vault_id": "beefy-bsc-cake-bnb",
                "include_risk_analysis": True,
                "include_historical_data": True
            }
            
            result = await mcp_tools.analyze_vault(params)
            
            assert "vault_info" in result
            assert "risk_analysis" in result
            assert "recommendations" in result

    @pytest.mark.asyncio
    async def test_invest_in_vault_tool(self, mcp_tools, mock_beefy_client):
        """Test investment MCP tool"""
        with patch.object(mcp_tools.investment_handler, 'beefy_client', mock_beefy_client):
            params = {
                "vault_id": "beefy-bsc-cake-bnb",
                "amount": 1000.0,
                "strategy": "yield_chaser",
                "dry_run": True
            }
            
            result = await mcp_tools.invest_in_vault(params)
            
            assert "transaction_plan" in result
            assert "estimated_costs" in result
            assert "expected_returns" in result

    @pytest.mark.asyncio
    async def test_portfolio_summary_tool(self, mcp_tools):
        """Test portfolio summary MCP tool"""
        with patch.object(mcp_tools.portfolio_handler, 'get_portfolio_summary') as mock_summary:
            mock_summary.return_value = {
                "total_value": 15000.0,
                "total_yield_earned": 750.0,
                "active_positions": 5,
                "chains": ["bsc", "ethereum"],
                "average_apy": 18.5
            }
            
            result = await mcp_tools.get_portfolio_summary({})
            
            assert "total_value" in result
            assert "active_positions" in result
            assert result["active_positions"] == 5

    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_tools):
        """Test error handling in MCP tools"""
        # Test with invalid parameters
        with pytest.raises(Exception):
            await mcp_tools.search_vaults({"invalid_param": "value"})
        
        # Test with missing required parameters
        with pytest.raises(Exception):
            await mcp_tools.analyze_vault({})  # Missing vault_id


class TestVaultHandler:
    """Test vault handler functionality"""

    @pytest.fixture
    def vault_handler(self):
        """Create vault handler"""
        return VaultHandler()

    @pytest.fixture
    def mock_beefy_client(self):
        """Mock Beefy client"""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_search_vaults(self, vault_handler, mock_beefy_client):
        """Test vault search functionality"""
        mock_beefy_client.search_vaults.return_value = [
            {
                "id": "vault-1",
                "name": "Test Vault 1",
                "apy": 25.0,
                "tvl": "1000000",
                "chain": "bsc"
            }
        ]
        
        vault_handler.beefy_client = mock_beefy_client
        
        params = VaultSearchParams(
            query="test",
            chain="bsc",
            min_apy=20.0,
            max_results=10
        )
        
        result = await vault_handler.search_vaults(params)
        
        assert len(result) == 1
        assert result[0]["name"] == "Test Vault 1"

    @pytest.mark.asyncio
    async def test_get_vault_details(self, vault_handler, mock_beefy_client):
        """Test vault details retrieval"""
        mock_vault = Mock(
            id="test-vault",
            name="Test Vault",
            chain="bsc",
            apy=20.0,
            tvl=1000000,
            status="active"
        )
        
        mock_beefy_client.get_vaults.return_value = [mock_vault]
        mock_beefy_client.get_apy.return_value = {
            "test-vault": Mock(totalApy=20.0, vaultApy=19.0)
        }
        
        vault_handler.beefy_client = mock_beefy_client
        
        result = await vault_handler.get_vault_details("test-vault")
        
        assert result["id"] == "test-vault"
        assert result["name"] == "Test Vault"

    @pytest.mark.asyncio
    async def test_calculate_vault_metrics(self, vault_handler):
        """Test vault metrics calculation"""
        vault_data = {
            "id": "test-vault",
            "apy": 25.0,
            "tvl": 5000000,
            "platform_fee": 0.045,
            "risks": ["COMPLEXITY_LOW"]
        }
        
        metrics = await vault_handler.calculate_vault_metrics(vault_data)
        
        assert "risk_score" in metrics
        assert "efficiency_ratio" in metrics
        assert "liquidity_score" in metrics
        assert 0 <= metrics["risk_score"] <= 100

    @pytest.mark.asyncio
    async def test_vault_comparison(self, vault_handler):
        """Test vault comparison functionality"""
        vaults = [
            {"id": "vault-1", "apy": 25.0, "risk_score": 60},
            {"id": "vault-2", "apy": 15.0, "risk_score": 30},
            {"id": "vault-3", "apy": 35.0, "risk_score": 80}
        ]
        
        comparison = await vault_handler.compare_vaults(vaults)
        
        assert "rankings" in comparison
        assert "risk_return_analysis" in comparison
        assert len(comparison["rankings"]) == 3


class TestInvestmentHandler:
    """Test investment handler functionality"""

    @pytest.fixture
    def investment_handler(self):
        """Create investment handler"""
        return InvestmentHandler()

    @pytest.mark.asyncio
    async def test_prepare_investment(self, investment_handler):
        """Test investment preparation"""
        params = InvestmentParams(
            vault_id="test-vault",
            amount=1000.0,
            strategy="yield_chaser",
            max_slippage=0.01
        )
        
        with patch.object(investment_handler, '_get_vault_info') as mock_vault:
            mock_vault.return_value = {
                "id": "test-vault",
                "name": "Test Vault",
                "apy": 25.0,
                "chain": "bsc"
            }
            
            result = await investment_handler.prepare_investment(params)
            
            assert "investment_plan" in result
            assert "estimated_gas" in result
            assert "expected_apy" in result

    @pytest.mark.asyncio
    async def test_calculate_investment_returns(self, investment_handler):
        """Test investment return calculations"""
        investment_data = {
            "amount": 1000.0,
            "apy": 25.0,
            "compound_frequency": 365,
            "platform_fee": 0.045
        }
        
        returns = await investment_handler.calculate_expected_returns(
            investment_data, 
            time_horizon_days=365
        )
        
        assert "gross_return" in returns
        assert "net_return" in returns
        assert "yield_breakdown" in returns
        assert returns["net_return"] < returns["gross_return"]

    @pytest.mark.asyncio
    async def test_risk_assessment(self, investment_handler):
        """Test investment risk assessment"""
        vault_info = {
            "id": "test-vault",
            "chain": "bsc",
            "protocol": "pancakeswap",
            "risks": ["COMPLEXITY_LOW", "BATTLE_TESTED"],
            "tvl": 5000000,
            "age_days": 365
        }
        
        risk_assessment = await investment_handler.assess_investment_risk(vault_info, 1000.0)
        
        assert "overall_risk_score" in risk_assessment
        assert "risk_factors" in risk_assessment
        assert "recommendations" in risk_assessment
        assert 0 <= risk_assessment["overall_risk_score"] <= 100

    @pytest.mark.asyncio
    async def test_simulate_investment(self, investment_handler):
        """Test investment simulation"""
        params = {
            "vault_id": "test-vault",
            "amount": 1000.0,
            "apy": 25.0,
            "simulation_days": 30
        }
        
        simulation = await investment_handler.simulate_investment(params)
        
        assert "daily_values" in simulation
        assert "final_value" in simulation
        assert "total_yield" in simulation
        assert len(simulation["daily_values"]) == 30


class TestPortfolioHandler:
    """Test portfolio handler functionality"""

    @pytest.fixture
    def portfolio_handler(self):
        """Create portfolio handler"""
        return PortfolioHandler()

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        return Mock()

    @pytest.mark.asyncio
    async def test_get_portfolio_summary(self, portfolio_handler, mock_db_session):
        """Test portfolio summary generation"""
        with patch.object(portfolio_handler, 'db_session', mock_db_session):
            mock_db_session.query.return_value.filter.return_value.all.return_value = [
                Mock(
                    vault_id="vault-1",
                    current_value=1100.0,
                    amount_deposited=1000.0,
                    entry_apy=20.0,
                    chain="bsc"
                ),
                Mock(
                    vault_id="vault-2",
                    current_value=2200.0,
                    amount_deposited=2000.0,
                    entry_apy=15.0,
                    chain="ethereum"
                )
            ]
            
            summary = await portfolio_handler.get_portfolio_summary()
            
            assert "total_value" in summary
            assert "unrealized_pnl" in summary
            assert "chain_allocation" in summary
            assert "position_count" in summary

    @pytest.mark.asyncio
    async def test_analyze_portfolio_performance(self, portfolio_handler, mock_db_session):
        """Test portfolio performance analysis"""
        params = PortfolioAnalysisParams(
            time_period=30,
            include_yield_history=True,
            benchmark="market"
        )
        
        with patch.object(portfolio_handler, '_get_historical_data') as mock_history:
            mock_history.return_value = [
                {"date": "2024-01-01", "value": 1000.0},
                {"date": "2024-01-02", "value": 1010.0},
                {"date": "2024-01-03", "value": 1025.0}
            ]
            
            analysis = await portfolio_handler.analyze_performance(params)
            
            assert "total_return" in analysis
            assert "sharpe_ratio" in analysis
            assert "max_drawdown" in analysis
            assert "volatility" in analysis

    @pytest.mark.asyncio
    async def test_rebalancing_suggestions(self, portfolio_handler):
        """Test portfolio rebalancing suggestions"""
        current_allocation = {
            "bsc": 60.0,
            "ethereum": 40.0
        }
        
        target_allocation = {
            "bsc": 50.0,
            "ethereum": 40.0,
            "polygon": 10.0
        }
        
        suggestions = await portfolio_handler.get_rebalancing_suggestions(
            current_allocation, 
            target_allocation
        )
        
        assert "rebalancing_trades" in suggestions
        assert "estimated_costs" in suggestions
        assert "impact_analysis" in suggestions

    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, portfolio_handler):
        """Test portfolio risk metrics"""
        positions = [
            {
                "vault_id": "vault-1",
                "amount": 1000.0,
                "risk_score": 30.0,
                "correlation": 0.5
            },
            {
                "vault_id": "vault-2",
                "amount": 2000.0,
                "risk_score": 50.0,
                "correlation": 0.3
            }
        ]
        
        risk_metrics = await portfolio_handler.calculate_portfolio_risk(positions)
        
        assert "portfolio_risk_score" in risk_metrics
        assert "diversification_ratio" in risk_metrics
        assert "concentration_risk" in risk_metrics
        assert "correlation_matrix" in risk_metrics


class TestAnalyticsHandler:
    """Test analytics handler functionality"""

    @pytest.fixture
    def analytics_handler(self):
        """Create analytics handler"""
        return AnalyticsHandler()

    @pytest.mark.asyncio
    async def test_market_analysis(self, analytics_handler):
        """Test market analysis functionality"""
        with patch.object(analytics_handler, '_fetch_market_data') as mock_data:
            mock_data.return_value = {
                "total_tvl": 10000000000,
                "avg_apy": 15.5,
                "chain_distribution": {"bsc": 40, "ethereum": 35, "polygon": 25},
                "protocol_distribution": {"uniswap": 30, "pancakeswap": 25, "aave": 20}
            }
            
            analysis = await analytics_handler.analyze_market_trends()
            
            assert "market_summary" in analysis
            assert "trending_protocols" in analysis
            assert "yield_opportunities" in analysis

    @pytest.mark.asyncio
    async def test_yield_forecasting(self, analytics_handler):
        """Test yield forecasting"""
        historical_data = [
            {"date": "2024-01-01", "apy": 20.0},
            {"date": "2024-01-02", "apy": 20.5},
            {"date": "2024-01-03", "apy": 19.8},
            {"date": "2024-01-04", "apy": 21.2}
        ]
        
        forecast = await analytics_handler.forecast_yields(
            vault_id="test-vault",
            historical_data=historical_data,
            forecast_days=7
        )
        
        assert "predictions" in forecast
        assert "confidence_intervals" in forecast
        assert "model_accuracy" in forecast
        assert len(forecast["predictions"]) == 7

    @pytest.mark.asyncio
    async def test_arbitrage_detection(self, analytics_handler):
        """Test arbitrage opportunity detection"""
        with patch.object(analytics_handler, '_scan_cross_chain_opportunities') as mock_scan:
            mock_scan.return_value = [
                {
                    "token_pair": ("USDC", "USDT"),
                    "chain_1": "ethereum",
                    "chain_2": "bsc",
                    "apy_difference": 5.2,
                    "estimated_profit": 52.0,
                    "risk_level": "medium"
                }
            ]
            
            opportunities = await analytics_handler.detect_arbitrage_opportunities()
            
            assert len(opportunities) == 1
            assert opportunities[0]["apy_difference"] == 5.2

    @pytest.mark.asyncio
    async def test_performance_attribution(self, analytics_handler):
        """Test performance attribution analysis"""
        portfolio_returns = {
            "vault-1": 0.15,  # 15% return
            "vault-2": 0.08,  # 8% return
            "vault-3": -0.02  # -2% return
        }
        
        benchmark_return = 0.10  # 10% benchmark
        
        attribution = await analytics_handler.calculate_performance_attribution(
            portfolio_returns,
            benchmark_return
        )
        
        assert "total_excess_return" in attribution
        assert "vault_contributions" in attribution
        assert "risk_adjusted_performance" in attribution

    @pytest.mark.asyncio
    async def test_correlation_analysis(self, analytics_handler):
        """Test correlation analysis between vaults"""
        vault_ids = ["vault-1", "vault-2", "vault-3"]
        
        with patch.object(analytics_handler, '_get_price_history') as mock_prices:
            mock_prices.return_value = {
                "vault-1": [100, 102, 101, 105, 103],
                "vault-2": [200, 198, 201, 205, 202],
                "vault-3": [150, 155, 152, 158, 156]
            }
            
            correlation_matrix = await analytics_handler.calculate_correlations(vault_ids)
            
            assert "correlation_matrix" in correlation_matrix
            assert "diversification_score" in correlation_matrix
            assert len(correlation_matrix["correlation_matrix"]) == 3


class TestMCPIntegration:
    """Test MCP integration functionality"""

    @pytest.fixture
    def mcp_integration(self):
        """Create MCP integration instance"""
        return MCPIntegration()

    @pytest.mark.asyncio
    async def test_register_tools(self, mcp_integration):
        """Test tool registration with MCP server"""
        tools = await mcp_integration.register_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Verify tool structure
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    @pytest.mark.asyncio
    async def test_handle_tool_call(self, mcp_integration):
        """Test handling MCP tool calls"""
        tool_request = {
            "name": "search_vaults",
            "arguments": {
                "query": "CAKE",
                "chain": "bsc",
                "min_apy": 20.0
            }
        }
        
        with patch.object(mcp_integration.beefy_tools, 'search_vaults') as mock_search:
            mock_search.return_value = {"vaults": []}
            
            result = await mcp_integration.handle_tool_call(tool_request)
            
            assert "content" in result
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_in_integration(self, mcp_integration):
        """Test error handling in MCP integration"""
        invalid_request = {
            "name": "nonexistent_tool",
            "arguments": {}
        }
        
        result = await mcp_integration.handle_tool_call(invalid_request)
        
        assert "error" in result["content"][0]

    def test_schema_validation(self, mcp_integration):
        """Test schema validation for tool parameters"""
        # Valid schema
        valid_params = {
            "vault_id": "beefy-bsc-cake",
            "amount": 1000.0,
            "strategy": "yield_chaser"
        }
        
        is_valid = mcp_integration.validate_parameters("invest_in_vault", valid_params)
        assert is_valid is True
        
        # Invalid schema
        invalid_params = {
            "vault_id": "beefy-bsc-cake",
            "amount": "invalid_amount"  # Should be float
        }
        
        is_valid = mcp_integration.validate_parameters("invest_in_vault", invalid_params)
        assert is_valid is False


class TestMCPToolsEndToEnd:
    """End-to-end tests for MCP tools"""

    @pytest.mark.asyncio
    async def test_complete_investment_workflow(self):
        """Test complete investment workflow through MCP tools"""
        mcp_tools = BeefyMCPTools()
        
        # Step 1: Search for vaults
        search_params = {
            "query": "CAKE",
            "chain": "bsc",
            "min_apy": 15.0
        }
        
        with patch.object(mcp_tools.vault_handler, 'beefy_client') as mock_client:
            # Mock search results
            mock_client.search_vaults.return_value = [
                {
                    "id": "beefy-bsc-cake-bnb",
                    "name": "CAKE-BNB LP",
                    "apy": 25.5,
                    "chain": "bsc"
                }
            ]
            
            search_result = await mcp_tools.search_vaults(search_params)
            assert len(search_result["vaults"]) > 0
            
            vault_id = search_result["vaults"][0]["id"]
            
            # Step 2: Analyze selected vault
            analyze_params = {
                "vault_id": vault_id,
                "include_risk_analysis": True
            }
            
            analysis_result = await mcp_tools.analyze_vault(analyze_params)
            assert "risk_analysis" in analysis_result
            
            # Step 3: Prepare investment
            invest_params = {
                "vault_id": vault_id,
                "amount": 1000.0,
                "strategy": "yield_chaser",
                "dry_run": True
            }
            
            investment_result = await mcp_tools.invest_in_vault(invest_params)
            assert "transaction_plan" in investment_result
            
            # Step 4: Get portfolio summary
            portfolio_result = await mcp_tools.get_portfolio_summary({})
            assert "total_value" in portfolio_result

    @pytest.mark.asyncio
    async def test_portfolio_management_workflow(self):
        """Test portfolio management workflow"""
        mcp_tools = BeefyMCPTools()
        
        # Mock portfolio data
        with patch.object(mcp_tools.portfolio_handler, 'get_portfolio_summary') as mock_summary:
            mock_summary.return_value = {
                "total_value": 15000.0,
                "active_positions": 5,
                "chains": ["bsc", "ethereum", "polygon"]
            }
            
            # Get portfolio summary
            summary = await mcp_tools.get_portfolio_summary({})
            assert summary["total_value"] == 15000.0
            
            # Analyze performance
            analysis_params = {
                "time_period": 30,
                "include_yield_history": True
            }
            
            with patch.object(mcp_tools.portfolio_handler, 'analyze_performance') as mock_analysis:
                mock_analysis.return_value = {
                    "total_return": 12.5,
                    "sharpe_ratio": 1.8,
                    "max_drawdown": -5.2
                }
                
                performance = await mcp_tools.analyze_portfolio_performance(analysis_params)
                assert performance["sharpe_ratio"] == 1.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])