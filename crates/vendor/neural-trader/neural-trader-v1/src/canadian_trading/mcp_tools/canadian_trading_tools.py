"""
MCP Tools for Canadian Trading APIs
==================================

This module provides MCP (Model Context Protocol) tools for Canadian trading integration.
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from ..brokers import IBCanadaClient, QuestradeAPI, OANDACanada
from ..compliance import CIROCompliance, TaxReporting, AuditTrail, ComplianceMonitor
from ..utils.auth import OAuth2Manager
from ..utils.forex_utils import ForexUtils


class CanadianTradingMCPTools:
    """MCP tools for Canadian trading operations."""
    
    def __init__(self):
        self.ib_client = None
        self.questrade_client = None
        self.oanda_client = None
        self.compliance = None
        self.tax_reporting = TaxReporting()
        self.audit_trail = AuditTrail()
        self.forex_utils = ForexUtils()
        
    async def initialize_ib_canada(self, host: str = "127.0.0.1", port: int = 7497, 
                                  is_paper: bool = True) -> Dict[str, Any]:
        """
        Initialize Interactive Brokers Canada connection.
        
        Args:
            host: IB Gateway host
            port: IB Gateway port
            is_paper: Whether to use paper trading
            
        Returns:
            Connection status and account info
        """
        try:
            from ..brokers.ib_canada import ConnectionConfig
            
            config = ConnectionConfig(host=host, port=port, is_paper=is_paper)
            self.ib_client = IBCanadaClient(config)
            await self.ib_client.connect()
            
            account_info = await self.ib_client.get_account_summary()
            
            return {
                "status": "success",
                "broker": "Interactive Brokers Canada",
                "connected": True,
                "account_info": account_info,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "broker": "Interactive Brokers Canada",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def initialize_questrade(self, refresh_token: str) -> Dict[str, Any]:
        """
        Initialize Questrade API connection.
        
        Args:
            refresh_token: Questrade refresh token
            
        Returns:
            Connection status and account info
        """
        try:
            self.questrade_client = QuestradeAPI(refresh_token=refresh_token)
            await self.questrade_client.initialize()
            
            accounts = await self.questrade_client.get_accounts()
            
            return {
                "status": "success",
                "broker": "Questrade",
                "connected": True,
                "accounts": accounts,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "broker": "Questrade",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def initialize_oanda(self, api_key: str, account_id: str, 
                              environment: str = "practice") -> Dict[str, Any]:
        """
        Initialize OANDA Canada connection.
        
        Args:
            api_key: OANDA API key
            account_id: OANDA account ID
            environment: Trading environment (practice/live)
            
        Returns:
            Connection status and account info
        """
        try:
            self.oanda_client = OANDACanada(
                api_key=api_key,
                account_id=account_id,
                environment=environment
            )
            await self.oanda_client.initialize()
            
            account_info = await self.oanda_client.get_account_summary()
            
            return {
                "status": "success",
                "broker": "OANDA Canada",
                "connected": True,
                "account_info": account_info,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "broker": "OANDA Canada",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_canadian_stock_quote(self, symbol: str, broker: str = "ib") -> Dict[str, Any]:
        """
        Get real-time quote for Canadian stock.
        
        Args:
            symbol: Stock symbol (e.g., "SHOP.TO")
            broker: Which broker to use (ib/questrade)
            
        Returns:
            Real-time quote data
        """
        try:
            if broker == "ib" and self.ib_client:
                contract = self.ib_client.create_canadian_stock(symbol.replace(".TO", ""), "CAD")
                quote = await self.ib_client.get_market_data(contract)
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "broker": "Interactive Brokers",
                    "quote": quote,
                    "timestamp": datetime.now().isoformat()
                }
                
            elif broker == "questrade" and self.questrade_client:
                from ..brokers.questrade import QuestradeDataFeed
                data_feed = QuestradeDataFeed(self.questrade_client)
                quote = await data_feed.get_quote(symbol)
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "broker": "Questrade",
                    "quote": quote,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "error": f"Broker {broker} not initialized",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_forex_quote(self, pair: str) -> Dict[str, Any]:
        """
        Get real-time forex quote.
        
        Args:
            pair: Currency pair (e.g., "USD_CAD")
            
        Returns:
            Forex quote with spread analysis
        """
        try:
            if not self.oanda_client:
                return {
                    "status": "error",
                    "error": "OANDA not initialized",
                    "timestamp": datetime.now().isoformat()
                }
            
            price = await self.oanda_client.get_current_price(pair)
            spread_analysis = await self.oanda_client.analyze_spread(pair)
            
            return {
                "status": "success",
                "pair": pair,
                "price": price,
                "spread_analysis": spread_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "pair": pair,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def place_canadian_stock_order(self, symbol: str, action: str, quantity: int,
                                       order_type: str = "MARKET", price: Optional[float] = None,
                                       broker: str = "ib") -> Dict[str, Any]:
        """
        Place order for Canadian stock.
        
        Args:
            symbol: Stock symbol
            action: BUY or SELL
            quantity: Number of shares
            order_type: Order type (MARKET/LIMIT)
            price: Limit price (required for LIMIT orders)
            broker: Which broker to use
            
        Returns:
            Order execution result
        """
        try:
            # Compliance check
            if self.compliance:
                compliance_check = await self.compliance.pre_trade_compliance_check({
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "order_type": order_type,
                    "price": price
                })
                
                if not compliance_check["compliant"]:
                    return {
                        "status": "error",
                        "error": "Compliance check failed",
                        "violations": compliance_check["violations"],
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Place order based on broker
            if broker == "ib" and self.ib_client:
                from ..brokers.ib_canada import OrderType
                
                contract = self.ib_client.create_canadian_stock(symbol.replace(".TO", ""), "CAD")
                order_result = await self.ib_client.place_order(
                    contract=contract,
                    order_type=getattr(OrderType, order_type),
                    action=action,
                    quantity=quantity,
                    price=price
                )
                
                # Audit trail
                await self.audit_trail.log_trade_event({
                    "broker": "Interactive Brokers",
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "order_type": order_type,
                    "price": price,
                    "result": order_result
                })
                
                # Tax reporting
                if order_result.get("status") == "FILLED":
                    self.tax_reporting.process_trade_for_tax({
                        "symbol": symbol,
                        "action": action,
                        "quantity": quantity,
                        "price": order_result.get("fill_price"),
                        "commission": order_result.get("commission", 0),
                        "currency": "CAD"
                    })
                
                return {
                    "status": "success",
                    "broker": "Interactive Brokers",
                    "order_result": order_result,
                    "timestamp": datetime.now().isoformat()
                }
                
            elif broker == "questrade" and self.questrade_client:
                order_result = await self.questrade_client.place_order(
                    symbol=symbol,
                    action=action.lower(),
                    quantity=quantity,
                    order_type=order_type.lower(),
                    price=price
                )
                
                # Audit trail
                await self.audit_trail.log_trade_event({
                    "broker": "Questrade",
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "order_type": order_type,
                    "price": price,
                    "result": order_result
                })
                
                return {
                    "status": "success",
                    "broker": "Questrade",
                    "order_result": order_result,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "error": f"Broker {broker} not initialized",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def place_forex_order(self, pair: str, units: int, side: str,
                               order_type: str = "MARKET", price: Optional[float] = None,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Place forex order.
        
        Args:
            pair: Currency pair
            units: Position size
            side: BUY or SELL
            order_type: Order type
            price: Limit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order execution result
        """
        try:
            if not self.oanda_client:
                return {
                    "status": "error",
                    "error": "OANDA not initialized",
                    "timestamp": datetime.now().isoformat()
                }
            
            order_result = await self.oanda_client.place_order(
                instrument=pair,
                units=units if side == "BUY" else -units,
                order_type=order_type.lower(),
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Audit trail
            await self.audit_trail.log_trade_event({
                "broker": "OANDA",
                "instrument": pair,
                "units": units,
                "side": side,
                "order_type": order_type,
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "result": order_result
            })
            
            return {
                "status": "success",
                "broker": "OANDA Canada",
                "order_result": order_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_portfolio_summary(self, include_forex: bool = True) -> Dict[str, Any]:
        """
        Get combined portfolio summary across all brokers.
        
        Args:
            include_forex: Whether to include forex positions
            
        Returns:
            Comprehensive portfolio summary
        """
        try:
            portfolio = {
                "total_value_cad": 0,
                "positions": [],
                "cash_balances": {},
                "brokers": []
            }
            
            # IB Canada positions
            if self.ib_client:
                ib_positions = await self.ib_client.get_positions()
                ib_account = await self.ib_client.get_account_summary()
                
                portfolio["positions"].extend([{
                    "broker": "Interactive Brokers",
                    **pos
                } for pos in ib_positions])
                
                portfolio["cash_balances"]["IB"] = ib_account.get("cash_balance", {})
                portfolio["brokers"].append("Interactive Brokers")
            
            # Questrade positions
            if self.questrade_client:
                qt_positions = await self.questrade_client.get_positions()
                qt_balances = await self.questrade_client.get_balances()
                
                portfolio["positions"].extend([{
                    "broker": "Questrade",
                    **pos
                } for pos in qt_positions])
                
                portfolio["cash_balances"]["Questrade"] = qt_balances
                portfolio["brokers"].append("Questrade")
            
            # OANDA forex positions
            if include_forex and self.oanda_client:
                forex_positions = await self.oanda_client.get_open_positions()
                forex_account = await self.oanda_client.get_account_summary()
                
                portfolio["positions"].extend([{
                    "broker": "OANDA",
                    "asset_class": "Forex",
                    **pos
                } for pos in forex_positions])
                
                portfolio["cash_balances"]["OANDA"] = {
                    "CAD": forex_account.get("balance", 0)
                }
                portfolio["brokers"].append("OANDA")
            
            # Calculate total portfolio value
            total_cad = sum(
                balance.get("CAD", 0) 
                for balance in portfolio["cash_balances"].values()
            )
            
            for position in portfolio["positions"]:
                if position.get("market_value_cad"):
                    total_cad += position["market_value_cad"]
            
            portfolio["total_value_cad"] = total_cad
            
            return {
                "status": "success",
                "portfolio": portfolio,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_tax_report(self, year: int, account_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate Canadian tax reports.
        
        Args:
            year: Tax year
            account_ids: Specific account IDs to include
            
        Returns:
            Tax report data including T5008 slips
        """
        try:
            # Get all transactions for the year
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            transactions = await self.audit_trail.get_transactions(
                start_date=start_date,
                end_date=end_date,
                event_types=["trade_execution", "dividend", "interest"]
            )
            
            # Generate T5008 slips
            t5008_slips = self.tax_reporting.generate_t5008_slips(year)
            
            # Calculate capital gains/losses
            capital_gains = self.tax_reporting.calculate_capital_gains(year)
            
            # Get foreign income
            foreign_income = self.tax_reporting.get_foreign_income_summary(year)
            
            return {
                "status": "success",
                "tax_year": year,
                "t5008_slips": t5008_slips,
                "capital_gains_summary": capital_gains,
                "foreign_income": foreign_income,
                "total_transactions": len(transactions),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def compliance_check(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run pre-trade compliance check.
        
        Args:
            trade_params: Trade parameters to check
            
        Returns:
            Compliance check results
        """
        try:
            if not self.compliance:
                self.compliance = CIROCompliance(
                    firm_id="DEMO_FIRM",
                    registration_number="DEMO_REG"
                )
            
            # Run compliance checks
            compliance_result = await self.compliance.pre_trade_compliance_check(trade_params)
            
            # Check position limits
            monitor_config = {
                "position_limits": {
                    "single_stock_limit": 0.10,
                    "sector_limit": 0.25
                }
            }
            monitor = ComplianceMonitor(monitor_config)
            monitoring_result = monitor.check_position_limits(trade_params)
            
            return {
                "status": "success",
                "compliant": compliance_result["compliant"] and monitoring_result["compliant"],
                "compliance_check": compliance_result,
                "position_check": monitoring_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_forex_opportunity(self, pairs: List[str], 
                                      lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze forex trading opportunities.
        
        Args:
            pairs: List of currency pairs
            lookback_days: Historical data period
            
        Returns:
            Forex analysis with recommendations
        """
        try:
            # Currency strength analysis
            strength_analysis = self.forex_utils.calculate_currency_strength(
                pairs, lookback_days
            )
            
            # Correlation analysis
            correlations = self.forex_utils.calculate_correlations(
                pairs, lookback_days
            )
            
            # Pattern detection
            patterns = {}
            for pair in pairs:
                patterns[pair] = self.forex_utils.detect_patterns(
                    pair, lookback_days
                )
            
            # Optimal trading times
            trading_times = self.forex_utils.get_optimal_trading_times(pairs[0])
            
            # Generate recommendations
            recommendations = []
            for pair in pairs:
                if strength_analysis.get(pair, {}).get("trend") == "strong_up":
                    recommendations.append({
                        "pair": pair,
                        "action": "BUY",
                        "reason": "Strong upward trend with positive momentum"
                    })
                elif strength_analysis.get(pair, {}).get("trend") == "strong_down":
                    recommendations.append({
                        "pair": pair,
                        "action": "SELL",
                        "reason": "Strong downward trend with negative momentum"
                    })
            
            return {
                "status": "success",
                "analysis": {
                    "currency_strength": strength_analysis,
                    "correlations": correlations,
                    "patterns": patterns,
                    "optimal_trading_times": trading_times,
                    "recommendations": recommendations
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# MCP Tool Registry
CANADIAN_TRADING_TOOLS = {
    "initialize_ib_canada": {
        "description": "Initialize Interactive Brokers Canada connection",
        "parameters": {
            "host": "IB Gateway host (default: 127.0.0.1)",
            "port": "IB Gateway port (default: 7497)",
            "is_paper": "Use paper trading (default: true)"
        }
    },
    "initialize_questrade": {
        "description": "Initialize Questrade API connection",
        "parameters": {
            "refresh_token": "Questrade OAuth2 refresh token (required)"
        }
    },
    "initialize_oanda": {
        "description": "Initialize OANDA Canada forex connection",
        "parameters": {
            "api_key": "OANDA API key (required)",
            "account_id": "OANDA account ID (required)",
            "environment": "Trading environment: practice/live (default: practice)"
        }
    },
    "get_canadian_stock_quote": {
        "description": "Get real-time quote for Canadian stock",
        "parameters": {
            "symbol": "Stock symbol (e.g., SHOP.TO)",
            "broker": "Which broker to use: ib/questrade (default: ib)"
        }
    },
    "get_forex_quote": {
        "description": "Get real-time forex quote with spread analysis",
        "parameters": {
            "pair": "Currency pair (e.g., USD_CAD)"
        }
    },
    "place_canadian_stock_order": {
        "description": "Place order for Canadian stock with compliance checks",
        "parameters": {
            "symbol": "Stock symbol",
            "action": "BUY or SELL",
            "quantity": "Number of shares",
            "order_type": "MARKET or LIMIT (default: MARKET)",
            "price": "Limit price (required for LIMIT orders)",
            "broker": "Which broker to use: ib/questrade (default: ib)"
        }
    },
    "place_forex_order": {
        "description": "Place forex order with risk management",
        "parameters": {
            "pair": "Currency pair",
            "units": "Position size",
            "side": "BUY or SELL",
            "order_type": "MARKET or LIMIT (default: MARKET)",
            "price": "Limit price (optional)",
            "stop_loss": "Stop loss price (optional)",
            "take_profit": "Take profit price (optional)"
        }
    },
    "get_portfolio_summary": {
        "description": "Get combined portfolio summary across all Canadian brokers",
        "parameters": {
            "include_forex": "Include forex positions (default: true)"
        }
    },
    "generate_tax_report": {
        "description": "Generate Canadian tax reports including T5008 slips",
        "parameters": {
            "year": "Tax year (required)",
            "account_ids": "Specific account IDs to include (optional)"
        }
    },
    "compliance_check": {
        "description": "Run CIRO pre-trade compliance check",
        "parameters": {
            "trade_params": "Trade parameters dict with symbol, action, quantity, etc."
        }
    },
    "analyze_forex_opportunity": {
        "description": "Analyze forex trading opportunities with AI",
        "parameters": {
            "pairs": "List of currency pairs to analyze",
            "lookback_days": "Historical data period (default: 30)"
        }
    }
}