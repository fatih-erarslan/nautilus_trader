"""
Risk Balanced Strategy

Portfolio strategy that maintains optimal risk-return balance through diversification
across chains, protocols, and risk levels.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.optimize import minimize
from .base_strategy import (
    BaseStrategy, VaultOpportunity, PortfolioState, Position,
    RiskLevel, ChainType
)


class RiskBalancedStrategy(BaseStrategy):
    """Strategy that optimizes risk-adjusted returns through diversification"""
    
    def __init__(self,
                 target_risk_level: RiskLevel = RiskLevel.MEDIUM,
                 min_apy_threshold: float = 10.0,
                 max_position_size: float = 0.20,
                 min_positions: int = 5,
                 max_positions: int = 15,
                 risk_budget: float = 50.0,
                 correlation_threshold: float = 0.7):
        """
        Initialize Risk Balanced strategy
        
        Args:
            target_risk_level: Target portfolio risk level
            min_apy_threshold: Minimum APY to consider
            max_position_size: Maximum single position size
            min_positions: Minimum number of positions for diversification
            max_positions: Maximum number of positions
            risk_budget: Total risk budget (0-100)
            correlation_threshold: Max correlation between positions
        """
        super().__init__(
            name="Risk Balanced",
            risk_level=target_risk_level,
            min_apy_threshold=min_apy_threshold,
            max_position_size=max_position_size,
            rebalance_threshold=0.1
        )
        self.min_positions = min_positions
        self.max_positions = max_positions
        self.risk_budget = risk_budget
        self.correlation_threshold = correlation_threshold
        
        # Risk limits per category
        self.risk_limits = {
            RiskLevel.LOW: {'min': 0, 'max': 30, 'target': 15},
            RiskLevel.MEDIUM: {'min': 20, 'max': 60, 'target': 40},
            RiskLevel.HIGH: {'min': 40, 'max': 80, 'target': 60},
            RiskLevel.EXTREME: {'min': 60, 'max': 100, 'target': 80}
        }
        
    def evaluate_opportunities(self,
                             opportunities: List[VaultOpportunity],
                             portfolio: PortfolioState) -> List[Tuple[VaultOpportunity, float]]:
        """
        Evaluate opportunities using portfolio optimization
        """
        # Filter viable opportunities
        viable_opps = []
        for opp in opportunities:
            if (opp.total_apy >= self.min_apy_threshold and
                not opp.is_paused and
                opp.tvl >= 250000):  # Min $250k TVL
                
                risk_score = self.calculate_risk_score(opp)
                if risk_score <= self.risk_limits[self.risk_level]['max']:
                    viable_opps.append(opp)
        
        if not viable_opps:
            return []
            
        # Calculate expected returns and risks
        returns = np.array([opp.net_apy for opp in viable_opps])
        risks = np.array([self.calculate_risk_score(opp) for opp in viable_opps])
        
        # Estimate correlation matrix (simplified - would use historical data)
        correlation_matrix = self._estimate_correlation_matrix(viable_opps)
        
        # Run portfolio optimization
        optimal_weights = self._optimize_portfolio(
            returns, risks, correlation_matrix, len(portfolio.positions)
        )
        
        # Convert weights to allocations
        allocations = []
        total_available = portfolio.available_capital
        
        for i, (opp, weight) in enumerate(zip(viable_opps, optimal_weights)):
            if weight > 0.01:  # Minimum 1% weight
                allocation = total_available * weight
                
                # Apply constraints
                allocation = self.validate_position_size(allocation, portfolio, opp)
                
                if allocation >= 100:  # Minimum position size
                    allocations.append((opp, allocation))
        
        # Sort by weight descending
        allocations.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure minimum diversification
        if len(portfolio.positions) + len(allocations) < self.min_positions:
            # Add more positions if needed
            remaining_opps = [o for o in viable_opps if o not in [a[0] for a in allocations]]
            remaining_opps.sort(key=lambda o: o.net_apy / (1 + self.calculate_risk_score(o)/100), reverse=True)
            
            for opp in remaining_opps[:self.min_positions - len(portfolio.positions) - len(allocations)]:
                min_allocation = min(total_available * 0.05, 1000)  # 5% or $1000
                if min_allocation >= 100:
                    allocations.append((opp, min_allocation))
                    
        return allocations[:self.max_positions - len(portfolio.positions)]
    
    def calculate_risk_score(self, opportunity: VaultOpportunity) -> float:
        """
        Comprehensive risk scoring for balanced portfolio
        """
        risk_score = 0.0
        
        # TVL risk with graduated scale
        tvl_millions = opportunity.tvl / 1_000_000
        if tvl_millions < 0.5:
            risk_score += 25
        elif tvl_millions < 1:
            risk_score += 18
        elif tvl_millions < 5:
            risk_score += 12
        elif tvl_millions < 10:
            risk_score += 8
        elif tvl_millions < 50:
            risk_score += 5
        else:
            risk_score += 2
            
        # APY risk (extreme APYs are risky)
        if opportunity.total_apy > 200:
            risk_score += 20
        elif opportunity.total_apy > 100:
            risk_score += 12
        elif opportunity.total_apy > 50:
            risk_score += 8
        elif opportunity.total_apy < 5:
            risk_score += 10  # Too low might indicate issues
            
        # Protocol maturity
        age_days = (datetime.now() - opportunity.created_at).days
        if age_days < 14:
            risk_score += 18
        elif age_days < 30:
            risk_score += 12
        elif age_days < 90:
            risk_score += 8
        elif age_days < 180:
            risk_score += 4
            
        # Risk factors from opportunity
        platform_risk = opportunity.risk_factors.get('platform_risk', 0)
        impermanent_loss = opportunity.risk_factors.get('impermanent_loss', 0)
        smart_contract = opportunity.risk_factors.get('smart_contract', 0)
        
        risk_score += platform_risk * 0.25
        risk_score += impermanent_loss * 0.25
        risk_score += smart_contract * 0.30
        
        # Chain risk assessment
        chain_scores = {
            ChainType.ETHEREUM: 3,
            ChainType.POLYGON: 8,
            ChainType.ARBITRUM: 7,
            ChainType.OPTIMISM: 7,
            ChainType.BSC: 12,
            ChainType.AVALANCHE: 10,
            ChainType.FANTOM: 15
        }
        risk_score += chain_scores.get(opportunity.chain, 20)
        
        # Token pair risk
        token1, token2 = opportunity.token_pair
        stablecoins = {'USDC', 'USDT', 'DAI', 'BUSD', 'FRAX'}
        blue_chips = {'ETH', 'WETH', 'BTC', 'WBTC', 'BNB', 'MATIC', 'AVAX'}
        
        if token1 in stablecoins and token2 in stablecoins:
            risk_score += 2  # Lowest risk
        elif (token1 in stablecoins or token2 in stablecoins) and \
             (token1 in blue_chips or token2 in blue_chips):
            risk_score += 5  # Low-medium risk
        elif token1 in blue_chips and token2 in blue_chips:
            risk_score += 8  # Medium risk
        else:
            risk_score += 15  # Higher risk for other pairs
            
        # Fee risk
        total_fees = opportunity.platform_fee + opportunity.withdraw_fee
        if total_fees > 0.03:  # >3% total fees
            risk_score += 10
        elif total_fees > 0.02:
            risk_score += 5
            
        return min(risk_score, 100)
    
    def should_rebalance(self, portfolio: PortfolioState) -> bool:
        """
        Check if portfolio needs rebalancing
        """
        if not portfolio.positions:
            return False
            
        # Check portfolio risk level
        current_risk = self._calculate_portfolio_risk(portfolio)
        risk_limits = self.risk_limits[self.risk_level]
        
        if current_risk < risk_limits['min'] or current_risk > risk_limits['max']:
            return True
            
        # Check diversification
        if len(portfolio.positions) < self.min_positions:
            return True
        elif len(portfolio.positions) > self.max_positions:
            return True
            
        # Check concentration
        diversification = self.diversification_score(portfolio)
        if diversification < 60:  # Less than 60% diversified
            return True
            
        # Check for position imbalances
        position_values = [pos.amount for pos in portfolio.positions]
        if position_values:
            max_position = max(position_values)
            avg_position = np.mean(position_values)
            if max_position > avg_position * 3:  # One position is 3x average
                return True
                
        # Regular rebalancing (bi-weekly)
        oldest_position = min(portfolio.positions, key=lambda p: p.entry_time)
        if (datetime.now() - oldest_position.entry_time).days >= 14:
            return True
            
        return False
    
    def generate_rebalance_trades(self,
                                portfolio: PortfolioState,
                                opportunities: List[VaultOpportunity]) -> List[Dict[str, Any]]:
        """
        Generate trades to rebalance portfolio
        """
        trades = []
        current_risk = self._calculate_portfolio_risk(portfolio)
        target_risk = self.risk_limits[self.risk_level]['target']
        
        # Categorize positions by risk
        position_risks = []
        for pos in portfolio.positions:
            opp = next((o for o in opportunities if o.vault_id == pos.vault_id), None)
            if opp:
                risk = self.calculate_risk_score(opp)
                position_risks.append({
                    'position': pos,
                    'opportunity': opp,
                    'risk': risk,
                    'risk_adjusted_return': opp.net_apy / (1 + risk/100)
                })
        
        # Sort by risk-adjusted return
        position_risks.sort(key=lambda x: x['risk_adjusted_return'], reverse=True)
        
        # If over-risk, reduce high-risk positions
        if current_risk > target_risk + 10:
            high_risk = [p for p in position_risks if p['risk'] > 70]
            for p in high_risk[:len(high_risk)//2]:  # Reduce half of high-risk
                trades.append({
                    'action': 'exit',
                    'vault_id': p['position'].vault_id,
                    'chain': p['position'].chain.value,
                    'amount': p['position'].amount,
                    'reason': f"Risk reduction: position risk {p['risk']:.0f} > 70"
                })
                
        # If under-risk, can take on more risk
        elif current_risk < target_risk - 10:
            # Look for medium-high risk opportunities with good returns
            new_opps = self.evaluate_opportunities(opportunities, portfolio)
            for opp, allocation in new_opps[:3]:  # Add up to 3 new positions
                risk = self.calculate_risk_score(opp)
                if risk > current_risk:  # Higher risk than current
                    trades.append({
                        'action': 'enter',
                        'vault_id': opp.vault_id,
                        'chain': opp.chain.value,
                        'amount': allocation,
                        'expected_apy': opp.net_apy,
                        'risk_score': risk,
                        'reason': f"Risk increase: target {target_risk:.0f}, current {current_risk:.0f}"
                    })
                    
        # Rebalance oversized positions
        total_value = sum(p['position'].amount for p in position_risks)
        target_size = total_value / len(position_risks) if position_risks else 0
        
        for p in position_risks:
            if p['position'].amount > target_size * 2:  # More than 2x target
                reduction = p['position'].amount - target_size * 1.5
                trades.append({
                    'action': 'reduce',
                    'vault_id': p['position'].vault_id,
                    'chain': p['position'].chain.value,
                    'amount': reduction,
                    'reason': f"Position rebalancing: {p['position'].amount:.0f} > {target_size*2:.0f}"
                })
                
        # Ensure minimum diversification
        if len(portfolio.positions) - len([t for t in trades if t['action'] == 'exit']) < self.min_positions:
            new_opps = self.evaluate_opportunities(opportunities, portfolio)
            needed = self.min_positions - len(portfolio.positions) + len([t for t in trades if t['action'] == 'exit'])
            
            for opp, allocation in new_opps[:needed]:
                trades.append({
                    'action': 'enter',
                    'vault_id': opp.vault_id,
                    'chain': opp.chain.value,
                    'amount': allocation,
                    'expected_apy': opp.net_apy,
                    'risk_score': self.calculate_risk_score(opp),
                    'reason': f"Diversification: need {self.min_positions} positions"
                })
                
        return trades
    
    def _optimize_portfolio(self,
                          returns: np.ndarray,
                          risks: np.ndarray,
                          correlation: np.ndarray,
                          current_positions: int) -> np.ndarray:
        """
        Optimize portfolio using mean-variance optimization
        """
        n_assets = len(returns)
        
        # Adjust returns for risk
        risk_adjusted_returns = returns / (1 + risks/100)
        
        # Covariance matrix (simplified - using risk and correlation)
        risk_matrix = np.outer(risks, risks)
        covariance = correlation * risk_matrix / 10000
        
        # Objective: maximize risk-adjusted returns minus variance
        def objective(weights):
            portfolio_return = np.dot(weights, risk_adjusted_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance, weights))
            return -(portfolio_return - 0.5 * portfolio_variance)  # Risk aversion = 0.5
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds (0 to max_position_size)
        bounds = [(0, self.max_position_size) for _ in range(n_assets)]
        
        # Add constraint for minimum positions if needed
        if current_positions < self.min_positions:
            min_weight = 1 / (self.min_positions * 2)  # Ensure some diversification
            bounds = [(min_weight, self.max_position_size) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 200}
        )
        
        if result.success:
            return result.x
        else:
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets
    
    def _estimate_correlation_matrix(self, opportunities: List[VaultOpportunity]) -> np.ndarray:
        """
        Estimate correlation matrix between opportunities
        """
        n = len(opportunities)
        correlation = np.eye(n)  # Start with identity matrix
        
        for i in range(n):
            for j in range(i+1, n):
                corr = 0.0
                
                # Same chain increases correlation
                if opportunities[i].chain == opportunities[j].chain:
                    corr += 0.3
                    
                # Same protocol increases correlation significantly
                if opportunities[i].protocol == opportunities[j].protocol:
                    corr += 0.5
                    
                # Similar token pairs increase correlation
                tokens_i = set(opportunities[i].token_pair)
                tokens_j = set(opportunities[j].token_pair)
                overlap = len(tokens_i.intersection(tokens_j))
                corr += overlap * 0.2
                
                # Cap correlation
                corr = min(corr, self.correlation_threshold)
                
                correlation[i, j] = corr
                correlation[j, i] = corr
                
        return correlation
    
    def _calculate_portfolio_risk(self, portfolio: PortfolioState) -> float:
        """
        Calculate overall portfolio risk score
        """
        if not portfolio.positions:
            return 0.0
            
        # Weighted average of position risks
        total_value = sum(pos.amount for pos in portfolio.positions)
        weighted_risk = sum(pos.risk_score * pos.amount / total_value 
                           for pos in portfolio.positions)
        
        # Adjust for diversification (lower risk with better diversification)
        diversification = self.diversification_score(portfolio)
        risk_reduction = diversification / 100 * 0.2  # Up to 20% reduction
        
        return weighted_risk * (1 - risk_reduction)
    
    def get_strategy_metrics(self, portfolio: PortfolioState) -> Dict[str, float]:
        """
        Get risk-balanced strategy metrics
        """
        if not portfolio.positions:
            return {
                'portfolio_risk': 0.0,
                'diversification_score': 0.0,
                'risk_adjusted_return': 0.0,
                'position_count': 0,
                'avg_position_risk': 0.0
            }
            
        # Calculate metrics
        portfolio_risk = self._calculate_portfolio_risk(portfolio)
        diversification = self.diversification_score(portfolio)
        
        # Risk-adjusted return
        total_return = sum(pos.apy * pos.amount for pos in portfolio.positions)
        total_value = sum(pos.amount for pos in portfolio.positions)
        avg_return = total_return / total_value if total_value > 0 else 0
        risk_adjusted_return = avg_return / (1 + portfolio_risk/100)
        
        # Position risks
        position_risks = [pos.risk_score for pos in portfolio.positions]
        
        return {
            'portfolio_risk': portfolio_risk,
            'diversification_score': diversification,
            'risk_adjusted_return': risk_adjusted_return,
            'position_count': len(portfolio.positions),
            'avg_position_risk': np.mean(position_risks),
            'min_position_risk': np.min(position_risks),
            'max_position_risk': np.max(position_risks),
            'risk_budget_used': portfolio_risk / self.risk_budget * 100
        }