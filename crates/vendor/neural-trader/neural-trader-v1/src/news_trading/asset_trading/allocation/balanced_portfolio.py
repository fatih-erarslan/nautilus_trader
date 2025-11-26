"""Balanced portfolio management for 60/40 and similar strategies."""

from typing import Dict, List, Tuple
import math


class BalancedPortfolioManager:
    """Manages balanced portfolios with fixed target allocations."""
    
    def __init__(
        self,
        target_stock: float = 0.6,
        target_bond: float = 0.4,
        rebalance_threshold: float = 0.05
    ):
        """Initialize the balanced portfolio manager.
        
        Args:
            target_stock: Target stock allocation
            target_bond: Target bond allocation
            rebalance_threshold: Threshold for rebalancing
        """
        self.target_stock = target_stock
        self.target_bond = target_bond
        self.rebalance_threshold = rebalance_threshold
        
        # Validate allocations
        if abs(target_stock + target_bond - 1.0) > 0.001:
            raise ValueError("Target allocations must sum to 1.0")
    
    def generate_rebalance_trades(
        self,
        current_allocation: Dict[str, float],
        portfolio_value: float
    ) -> List[Dict]:
        """Generate trades to rebalance portfolio.
        
        Args:
            current_allocation: Current allocation percentages
            portfolio_value: Total portfolio value
            
        Returns:
            List of rebalancing trades
        """
        trades = []
        
        current_stock = current_allocation.get("stocks", 0)
        current_bond = current_allocation.get("bonds", 0)
        
        # Calculate deviations
        stock_deviation = current_stock - self.target_stock
        bond_deviation = current_bond - self.target_bond
        
        # Check if rebalancing needed
        if (abs(stock_deviation) > self.rebalance_threshold or
            abs(bond_deviation) > self.rebalance_threshold):
            
            # Calculate dollar amounts
            stock_target_value = portfolio_value * self.target_stock
            bond_target_value = portfolio_value * self.target_bond
            
            stock_current_value = portfolio_value * current_stock
            bond_current_value = portfolio_value * current_bond
            
            stock_trade_amount = stock_target_value - stock_current_value
            bond_trade_amount = bond_target_value - bond_current_value
            
            # Generate trades
            if stock_trade_amount < 0:
                trades.append({
                    "action": "sell",
                    "asset_class": "stocks",
                    "amount": abs(stock_trade_amount),
                    "reason": "Rebalance - reduce overweight",
                })
            elif stock_trade_amount > 0:
                trades.append({
                    "action": "buy",
                    "asset_class": "stocks",
                    "amount": stock_trade_amount,
                    "reason": "Rebalance - increase underweight",
                })
            
            if bond_trade_amount < 0:
                trades.append({
                    "action": "sell",
                    "asset_class": "bonds",
                    "amount": abs(bond_trade_amount),
                    "reason": "Rebalance - reduce overweight",
                })
            elif bond_trade_amount > 0:
                trades.append({
                    "action": "buy",
                    "asset_class": "bonds",
                    "amount": bond_trade_amount,
                    "reason": "Rebalance - increase underweight",
                })
        
        return trades
    
    def calculate_rebalance_frequency(
        self,
        volatility: float,
        transaction_costs: float
    ) -> str:
        """Calculate optimal rebalancing frequency.
        
        Args:
            volatility: Portfolio volatility
            transaction_costs: Transaction cost percentage
            
        Returns:
            Recommended frequency
        """
        # Higher volatility -> more frequent rebalancing
        # Higher costs -> less frequent rebalancing
        
        if transaction_costs > 0.01:  # High costs
            if volatility > 20:
                return "quarterly"
            else:
                return "semi-annually"
        else:  # Low costs
            if volatility > 25:
                return "monthly"
            elif volatility > 15:
                return "quarterly"
            else:
                return "semi-annually"
    
    def calculate_drift_tolerance(
        self,
        asset_volatility: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate drift tolerance bands for each asset.
        
        Args:
            asset_volatility: Volatility by asset class
            
        Returns:
            Tolerance bands
        """
        tolerances = {}
        
        # Base tolerance is 5%, adjust for volatility
        for asset, vol in asset_volatility.items():
            if vol < 10:
                tolerances[asset] = 0.03  # 3% for low vol
            elif vol < 20:
                tolerances[asset] = 0.05  # 5% for medium vol
            else:
                tolerances[asset] = 0.10  # 10% for high vol
        
        return tolerances
    
    def analyze_rebalance_impact(
        self,
        trades: List[Dict],
        market_impact: float = 0.001,
        commission: float = 0
    ) -> Dict[str, float]:
        """Analyze the cost and impact of rebalancing.
        
        Args:
            trades: List of rebalancing trades
            market_impact: Expected market impact
            commission: Fixed commission per trade
            
        Returns:
            Impact analysis
        """
        total_traded = sum(trade["amount"] for trade in trades)
        num_trades = len(trades)
        
        # Calculate costs
        market_impact_cost = total_traded * market_impact
        commission_cost = num_trades * commission
        total_cost = market_impact_cost + commission_cost
        
        # Calculate turnover
        portfolio_value = sum(trade["amount"] for trade in trades if trade["action"] == "sell") * 2
        turnover = total_traded / portfolio_value if portfolio_value > 0 else 0
        
        return {
            "total_traded": total_traded,
            "market_impact_cost": market_impact_cost,
            "commission_cost": commission_cost,
            "total_cost": total_cost,
            "turnover": turnover,
            "num_trades": num_trades,
        }
    
    def get_tactical_overlay(
        self,
        market_conditions: Dict[str, any]
    ) -> Dict[str, float]:
        """Get tactical adjustments to strategic allocation.
        
        Args:
            market_conditions: Current market conditions
            
        Returns:
            Tactical tilts
        """
        stock_tilt = 0
        bond_tilt = 0
        
        # Value tilt
        pe_ratio = market_conditions.get("pe_ratio", 20)
        if pe_ratio > 25:
            stock_tilt -= 0.05  # Reduce stocks if expensive
            bond_tilt += 0.05
        elif pe_ratio < 15:
            stock_tilt += 0.05  # Increase stocks if cheap
            bond_tilt -= 0.05
        
        # Momentum tilt
        stock_momentum = market_conditions.get("stock_momentum", 0)
        if stock_momentum > 0.1:  # 10% positive momentum
            stock_tilt += 0.03
            bond_tilt -= 0.03
        elif stock_momentum < -0.1:
            stock_tilt -= 0.03
            bond_tilt += 0.03
        
        # Rate environment
        rate_trend = market_conditions.get("rate_trend", "stable")
        if rate_trend == "rising":
            bond_tilt -= 0.05  # Reduce bonds in rising rates
            stock_tilt += 0.05
        elif rate_trend == "falling":
            bond_tilt += 0.05
            stock_tilt -= 0.05
        
        # Cap tilts at +/- 10%
        stock_tilt = max(-0.10, min(0.10, stock_tilt))
        bond_tilt = max(-0.10, min(0.10, bond_tilt))
        
        return {
            "stock_target": self.target_stock + stock_tilt,
            "bond_target": self.target_bond + bond_tilt,
            "stock_tilt": stock_tilt,
            "bond_tilt": bond_tilt,
        }
    
    def calculate_glide_path(
        self,
        current_age: int,
        retirement_age: int,
        risk_tolerance: str = "moderate"
    ) -> Tuple[float, float]:
        """Calculate target allocation based on age (glide path).
        
        Args:
            current_age: Current age
            retirement_age: Target retirement age
            risk_tolerance: Risk tolerance level
            
        Returns:
            Tuple of (stock_allocation, bond_allocation)
        """
        years_to_retirement = max(0, retirement_age - current_age)
        
        # Base glide path
        if risk_tolerance == "conservative":
            stock_pct = max(20, min(70, 100 - current_age))
        elif risk_tolerance == "moderate":
            stock_pct = max(30, min(80, 110 - current_age))
        else:  # aggressive
            stock_pct = max(40, min(90, 120 - current_age))
        
        # Adjust for years to retirement
        if years_to_retirement < 5:
            stock_pct = min(stock_pct, 40)  # Reduce risk near retirement
        
        bond_pct = 100 - stock_pct
        
        return stock_pct / 100, bond_pct / 100