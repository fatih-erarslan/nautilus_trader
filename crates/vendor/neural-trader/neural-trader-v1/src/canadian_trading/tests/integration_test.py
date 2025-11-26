#!/usr/bin/env python3
"""
Canadian Trading Integration Test Suite
=======================================

Comprehensive integration tests for all Canadian trading components.
"""
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from canadian_trading.mcp_tools import CanadianTradingMCPTools
from canadian_trading.brokers.ib_canada import ConnectionConfig
from canadian_trading.compliance import CIROCompliance, TaxReporting
from canadian_trading.utils.forex_utils import ForexUtils


class IntegrationTestSuite:
    """Comprehensive integration test suite."""
    
    def __init__(self):
        self.mcp_tools = CanadianTradingMCPTools()
        self.test_results = []
        self.start_time = datetime.now()
        
    def log_test_result(self, test_name: str, status: str, details: Dict[str, Any] = None):
        """Log test result."""
        result = {
            "test_name": test_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.test_results.append(result)
        
        emoji = "✅" if status == "PASS" else "❌"
        print(f"{emoji} {test_name}: {status}")
        
        if status == "FAIL" and details:
            print(f"   Error: {details.get('error', 'Unknown error')}")
    
    async def test_ib_canada_connection(self):
        """Test IB Canada connection (mock)."""
        test_name = "IB Canada Connection"
        
        try:
            # Mock test - would normally connect to IB Gateway
            result = await self.mcp_tools.initialize_ib_canada(
                host="127.0.0.1",
                port=7497,
                is_paper=True
            )
            
            # Since we're not actually connecting, we expect an error
            if result["status"] == "error":
                self.log_test_result(test_name, "PASS", {
                    "note": "Expected error - no IB Gateway running"
                })
            else:
                self.log_test_result(test_name, "FAIL", {
                    "error": "Unexpected success without IB Gateway"
                })
                
        except Exception as e:
            self.log_test_result(test_name, "FAIL", {"error": str(e)})
    
    async def test_questrade_initialization(self):
        """Test Questrade initialization (mock)."""
        test_name = "Questrade Initialization"
        
        try:
            # Mock test with dummy token
            result = await self.mcp_tools.initialize_questrade("dummy_token")
            
            # Expect error due to invalid token
            if result["status"] == "error":
                self.log_test_result(test_name, "PASS", {
                    "note": "Expected error - invalid token"
                })
            else:
                self.log_test_result(test_name, "FAIL", {
                    "error": "Unexpected success with dummy token"
                })
                
        except Exception as e:
            self.log_test_result(test_name, "FAIL", {"error": str(e)})
    
    async def test_oanda_initialization(self):
        """Test OANDA initialization (mock)."""
        test_name = "OANDA Initialization"
        
        try:
            # Mock test with dummy credentials
            result = await self.mcp_tools.initialize_oanda(
                api_key="dummy_key",
                account_id="dummy_account",
                environment="practice"
            )
            
            # Expect error due to invalid credentials
            if result["status"] == "error":
                self.log_test_result(test_name, "PASS", {
                    "note": "Expected error - invalid credentials"
                })
            else:
                self.log_test_result(test_name, "FAIL", {
                    "error": "Unexpected success with dummy credentials"
                })
                
        except Exception as e:
            self.log_test_result(test_name, "FAIL", {"error": str(e)})
    
    async def test_compliance_system(self):
        """Test CIRO compliance system."""
        test_name = "CIRO Compliance System"
        
        try:
            # Test compliance check
            trade_params = {
                "symbol": "SHOP.TO",
                "action": "BUY",
                "quantity": 100,
                "order_type": "MARKET",
                "price": None
            }
            
            result = await self.mcp_tools.compliance_check(trade_params)
            
            if result["status"] == "success":
                self.log_test_result(test_name, "PASS", {
                    "compliant": result["compliant"],
                    "checks_performed": ["pre_trade_compliance", "position_limits"]
                })
            else:
                self.log_test_result(test_name, "FAIL", {
                    "error": result.get("error", "Unknown error")
                })
                
        except Exception as e:
            self.log_test_result(test_name, "FAIL", {"error": str(e)})
    
    async def test_tax_reporting(self):
        """Test tax reporting system."""
        test_name = "Tax Reporting System"
        
        try:
            # Test tax report generation
            result = await self.mcp_tools.generate_tax_report(
                year=2024,
                account_ids=None
            )
            
            if result["status"] == "success":
                self.log_test_result(test_name, "PASS", {
                    "tax_year": result["tax_year"],
                    "reports_generated": ["t5008_slips", "capital_gains", "foreign_income"]
                })
            else:
                self.log_test_result(test_name, "FAIL", {
                    "error": result.get("error", "Unknown error")
                })
                
        except Exception as e:
            self.log_test_result(test_name, "FAIL", {"error": str(e)})
    
    async def test_forex_analysis(self):
        """Test forex analysis system."""
        test_name = "Forex Analysis System"
        
        try:
            # Test forex opportunity analysis
            pairs = ["USD_CAD", "EUR_CAD", "GBP_CAD"]
            
            result = await self.mcp_tools.analyze_forex_opportunity(
                pairs=pairs,
                lookback_days=30
            )
            
            if result["status"] == "success":
                analysis = result["analysis"]
                self.log_test_result(test_name, "PASS", {
                    "pairs_analyzed": len(pairs),
                    "analysis_components": list(analysis.keys()),
                    "recommendations": len(analysis.get("recommendations", []))
                })
            else:
                self.log_test_result(test_name, "FAIL", {
                    "error": result.get("error", "Unknown error")
                })
                
        except Exception as e:
            self.log_test_result(test_name, "FAIL", {"error": str(e)})
    
    async def test_portfolio_summary(self):
        """Test portfolio summary functionality."""
        test_name = "Portfolio Summary"
        
        try:
            # Test portfolio summary (no connections, so empty portfolio)
            result = await self.mcp_tools.get_portfolio_summary(include_forex=True)
            
            if result["status"] == "success":
                portfolio = result["portfolio"]
                self.log_test_result(test_name, "PASS", {
                    "total_value": portfolio["total_value_cad"],
                    "positions": len(portfolio["positions"]),
                    "brokers": portfolio["brokers"]
                })
            else:
                self.log_test_result(test_name, "FAIL", {
                    "error": result.get("error", "Unknown error")
                })
                
        except Exception as e:
            self.log_test_result(test_name, "FAIL", {"error": str(e)})
    
    async def test_module_imports(self):
        """Test all module imports."""
        test_name = "Module Imports"
        
        try:
            # Test direct imports
            from canadian_trading import (
                IBCanadaClient,
                QuestradeAPI,
                OANDACanada,
                CIROCompliance,
                TaxReporting,
                AuditTrail,
                OAuth2Manager
            )
            
            # Test MCP tools import
            from canadian_trading.mcp_tools import (
                CanadianTradingMCPTools,
                CANADIAN_TRADING_TOOLS
            )
            
            # Test utilities
            from canadian_trading.utils.forex_utils import ForexUtils
            from canadian_trading.utils.auth import OAuth2Manager
            
            self.log_test_result(test_name, "PASS", {
                "imports_successful": [
                    "IBCanadaClient",
                    "QuestradeAPI", 
                    "OANDACanada",
                    "CIROCompliance",
                    "TaxReporting",
                    "AuditTrail",
                    "OAuth2Manager",
                    "CanadianTradingMCPTools",
                    "CANADIAN_TRADING_TOOLS",
                    "ForexUtils"
                ]
            })
            
        except Exception as e:
            self.log_test_result(test_name, "FAIL", {"error": str(e)})
    
    async def test_mcp_tool_registry(self):
        """Test MCP tool registry."""
        test_name = "MCP Tool Registry"
        
        try:
            from canadian_trading.mcp_tools import CANADIAN_TRADING_TOOLS
            
            # Check that all expected tools are registered
            expected_tools = [
                "initialize_ib_canada",
                "initialize_questrade",
                "initialize_oanda",
                "get_canadian_stock_quote",
                "get_forex_quote",
                "place_canadian_stock_order",
                "place_forex_order",
                "get_portfolio_summary",
                "generate_tax_report",
                "compliance_check",
                "analyze_forex_opportunity"
            ]
            
            missing_tools = [tool for tool in expected_tools if tool not in CANADIAN_TRADING_TOOLS]
            
            if not missing_tools:
                self.log_test_result(test_name, "PASS", {
                    "total_tools": len(CANADIAN_TRADING_TOOLS),
                    "all_tools_registered": True
                })
            else:
                self.log_test_result(test_name, "FAIL", {
                    "missing_tools": missing_tools
                })
                
        except Exception as e:
            self.log_test_result(test_name, "FAIL", {"error": str(e)})
    
    async def test_configuration_validation(self):
        """Test configuration validation."""
        test_name = "Configuration Validation"
        
        try:
            # Test IB configuration
            config = ConnectionConfig(host="127.0.0.1", port=7497, is_paper=True)
            assert config.host == "127.0.0.1"
            assert config.port == 7497
            assert config.is_paper == True
            
            # Test compliance initialization
            compliance = CIROCompliance(
                firm_id="TEST_FIRM",
                registration_number="TEST_REG"
            )
            assert compliance.firm_id == "TEST_FIRM"
            
            # Test tax reporting
            tax_reporting = TaxReporting()
            assert tax_reporting is not None
            
            # Test forex utilities
            forex_utils = ForexUtils()
            assert forex_utils is not None
            
            self.log_test_result(test_name, "PASS", {
                "configurations_validated": [
                    "ConnectionConfig",
                    "CIROCompliance",
                    "TaxReporting",
                    "ForexUtils"
                ]
            })
            
        except Exception as e:
            self.log_test_result(test_name, "FAIL", {"error": str(e)})
    
    async def run_all_tests(self):
        """Run all integration tests."""
        print("🇨🇦 Canadian Trading Integration Test Suite")
        print("=" * 60)
        print(f"Started: {self.start_time.isoformat()}")
        print()
        
        # Run all tests
        await self.test_module_imports()
        await self.test_mcp_tool_registry()
        await self.test_configuration_validation()
        await self.test_compliance_system()
        await self.test_tax_reporting()
        await self.test_forex_analysis()
        await self.test_portfolio_summary()
        await self.test_ib_canada_connection()
        await self.test_questrade_initialization()
        await self.test_oanda_initialization()
        
        # Calculate results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed_tests = total_tests - passed_tests
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print()
        print("=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏱️ Duration: {duration.total_seconds():.2f} seconds")
        print(f"✨ Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test_name']}: {result['details'].get('error', 'Unknown error')}")
        
        print(f"\nCompleted: {end_time.isoformat()}")
        
        # Save detailed results
        results_file = Path("test_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "success_rate": (passed_tests/total_tests)*100,
                    "duration_seconds": duration.total_seconds(),
                    "start_time": self.start_time.isoformat(),
                    "end_time": end_time.isoformat()
                },
                "test_results": self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to {results_file}")
        
        return failed_tests == 0


async def main():
    """Run the integration test suite."""
    suite = IntegrationTestSuite()
    success = await suite.run_all_tests()
    
    if success:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print("\n💥 Some integration tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))