"""
Yield Chaser Strategy

Aggressively pursues the highest APY opportunities across chains while managing risk.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from .base_strategy import (
    BaseStrategy, VaultOpportunity, PortfolioState, Position,
    RiskLevel, ChainType
)


class YieldChaserStrategy(BaseStrategy):
    """Strategy that chases highest yields with dynamic risk management"""
    
    def __init__(self,
                 min_apy_threshold: float = 20.0,
                 max_position_size: float = 0.30,
                 apy_decay_factor: float = 0.95,
                 min_tvl: float = 100000,
                 max_chain_exposure: float = 0.40):
        """
        Initialize Yield Chaser strategy
        
        Args:
            min_apy_threshold: Minimum APY to consider (default 20%)
            max_position_size: Maximum position size (default 30% of portfolio)
            apy_decay_factor: Factor to discount APY over time (default 0.95)
            min_tvl: Minimum TVL required (default $100k)
            max_chain_exposure: Maximum exposure to single chain (default 40%)
        """
        super().__init__(
            name="Yield Chaser",
            risk_level=RiskLevel.HIGH,
            min_apy_threshold=min_apy_threshold,
            max_position_size=max_position_size,
            rebalance_threshold=0.15
        )
        self.apy_decay_factor = apy_decay_factor
        self.min_tvl = min_tvl
        self.max_chain_exposure = max_chain_exposure
        
    def evaluate_opportunities(self,
                             opportunities: List[VaultOpportunity],
                             portfolio: PortfolioState) -> List[Tuple[VaultOpportunity, float]]:
        """
        Evaluate and rank opportunities by risk-adjusted APY
        """
        evaluated = []
        
        for opp in opportunities:
            # Filter out low TVL and paused vaults
            if opp.tvl < self.min_tvl or opp.is_paused:
                continue
                
            # Filter by minimum APY threshold
            if opp.total_apy < self.min_apy_threshold:
                continue
                
            # Calculate risk score
            risk_score = self.calculate_risk_score(opp)
            
            # Skip extreme risk unless we're already exposed
            if risk_score > 80 and not self._has_position(opp.vault_id, portfolio):
                continue
            
            # Calculate risk-adjusted APY with time decay
            age_days = (datetime.now() - opp.created_at).days
            time_factor = self.apy_decay_factor ** (age_days / 30)  # Decay per month
            risk_factor = 1 - (risk_score / 100) * 0.7  # Max 70% reduction for risk
            
            adjusted_apy = opp.net_apy * risk_factor * time_factor
            
            # Calculate allocation based on APY and available capital
            base_allocation = self._calculate_base_allocation(
                adjusted_apy, portfolio.available_capital
            )
            
            # Adjust for chain exposure limits
            chain_allocation = portfolio.get_chain_exposure(opp.chain)
            if chain_allocation >= self.max_chain_exposure * 100:
                base_allocation *= 0.1  # Drastically reduce if over-exposed
            
            # Validate position size
            final_allocation = self.validate_position_size(
                base_allocation, portfolio, opp
            )
            
            if final_allocation > 0:
                evaluated.append((opp, final_allocation))
                
        # Sort by adjusted APY (descending)
        evaluated.sort(key=lambda x: x[0].net_apy * (1 - self.calculate_risk_score(x[0])/100), 
                      reverse=True)
        
        # Take top opportunities that fit within capital constraints
        selected = []
        remaining_capital = portfolio.available_capital
        
        for opp, allocation in evaluated:
            if allocation <= remaining_capital:
                selected.append((opp, allocation))
                remaining_capital -= allocation
                
        return selected
    
    def calculate_risk_score(self, opportunity: VaultOpportunity) -> float:
        """
        Calculate comprehensive risk score for yield chasing
        """
        risk_score = 0.0
        
        # TVL risk (lower TVL = higher risk)
        if opportunity.tvl < 500000:
            risk_score += 25
        elif opportunity.tvl < 1000000:
            risk_score += 15
        elif opportunity.tvl < 5000000:
            risk_score += 10
        else:
            risk_score += 5
            
        # APY sustainability risk (extremely high APY often unsustainable)
        if opportunity.total_apy > 1000:
            risk_score += 30
        elif opportunity.total_apy > 500:
            risk_score += 20
        elif opportunity.total_apy > 200:
            risk_score += 10
        elif opportunity.total_apy > 100:
            risk_score += 5
            
        # Age risk (newer = higher risk)
        age_days = (datetime.now() - opportunity.created_at).days
        if age_days < 7:
            risk_score += 20
        elif age_days < 30:
            risk_score += 10
        elif age_days < 90:
            risk_score += 5
            
        # Platform risk from risk factors
        platform_risk = opportunity.risk_factors.get('platform_risk', 0)
        impermanent_loss_risk = opportunity.risk_factors.get('impermanent_loss', 0)
        smart_contract_risk = opportunity.risk_factors.get('smart_contract', 0)
        
        risk_score += platform_risk * 0.2
        risk_score += impermanent_loss_risk * 0.3
        risk_score += smart_contract_risk * 0.2
        
        # Chain risk
        chain_risks = {
            ChainType.ETHEREUM: 5,
            ChainType.BSC: 15,
            ChainType.POLYGON: 10,
            ChainType.ARBITRUM: 8,
            ChainType.OPTIMISM: 8,
            ChainType.AVALANCHE: 12,
            ChainType.FANTOM: 20
        }
        risk_score += chain_risks.get(opportunity.chain, 25)
        
        # Fee risk
        if opportunity.withdraw_fee > 0.01:  # >1% withdraw fee
            risk_score += 10
            
        return min(risk_score, 100)
    
    def should_rebalance(self, portfolio: PortfolioState) -> bool:
        """
        Determine if portfolio needs rebalancing
        """
        if not portfolio.positions:
            return False
            
        # Check if any position has dropped significantly in APY
        for position in portfolio.positions:
            current_apy = position.apy  # Would need real-time data
            if position.metadata.get('initial_apy', current_apy) > current_apy * 1.5:
                return True
                
        # Check if utilization is too low
        if portfolio.utilization_rate < 70 and portfolio.available_capital > 1000:
            return True
            
        # Check chain concentration
        for chain, allocation in portfolio.chain_allocations.items():
            if allocation > self.max_chain_exposure * 100:
                return True
                
        # Check time-based rebalancing (weekly for yield chasing)
        oldest_position = min(portfolio.positions, key=lambda p: p.entry_time)
        if (datetime.now() - oldest_position.entry_time).days >= 7:
            return True
            
        return False
    
    def generate_rebalance_trades(self,
                                portfolio: PortfolioState,
                                opportunities: List[VaultOpportunity]) -> List[Dict[str, Any]]:
        """
        Generate trades to rebalance portfolio for maximum yield
        """
        trades = []
        
        # Identify underperforming positions
        position_performance = []
        for pos in portfolio.positions:
            # Find current opportunity data
            current_opp = next(
                (o for o in opportunities if o.vault_id == pos.vault_id),
                None
            )
            if current_opp:
                current_apy = current_opp.net_apy
                risk_score = self.calculate_risk_score(current_opp)
                risk_adjusted_apy = current_apy * (1 - risk_score/100)
                
                position_performance.append({
                    'position': pos,
                    'current_apy': current_apy,
                    'risk_adjusted_apy': risk_adjusted_apy,
                    'opportunity': current_opp
                })
        
        # Sort by risk-adjusted APY
        position_performance.sort(key=lambda x: x['risk_adjusted_apy'])
        
        # Evaluate new opportunities
        new_opportunities = self.evaluate_opportunities(opportunities, portfolio)
        
        # Exit underperforming positions if better opportunities exist
        freed_capital = 0
        for perf in position_performance[:len(position_performance)//3]:  # Bottom third
            if new_opportunities and new_opportunities[0][0].net_apy > perf['current_apy'] * 1.5:
                trades.append({
                    'action': 'exit',
                    'vault_id': perf['position'].vault_id,
                    'chain': perf['position'].chain.value,
                    'amount': perf['position'].amount,
                    'reason': f"Low APY: {perf['current_apy']:.2f}% vs new opportunity {new_opportunities[0][0].net_apy:.2f}%"
                })
                freed_capital += perf['position'].amount
        
        # Enter new high-yield positions
        available_capital = portfolio.available_capital + freed_capital
        for opp, allocation in new_opportunities:
            if allocation <= available_capital:
                trades.append({
                    'action': 'enter',
                    'vault_id': opp.vault_id,
                    'chain': opp.chain.value,
                    'amount': allocation,
                    'expected_apy': opp.net_apy,
                    'risk_score': self.calculate_risk_score(opp),
                    'reason': f"High yield opportunity: {opp.net_apy:.2f}% APY"
                })
                available_capital -= allocation
                
        # Rebalance overweight positions
        for chain, allocation in portfolio.chain_allocations.items():
            if allocation > self.max_chain_exposure * 100:
                # Find positions on this chain to reduce
                chain_positions = [
                    p for p in position_performance 
                    if p['position'].chain == chain
                ]
                chain_positions.sort(key=lambda x: x['risk_adjusted_apy'])
                
                target_reduction = allocation - (self.max_chain_exposure * 100)
                for pos_data in chain_positions:
                    reduction = min(
                        pos_data['position'].amount,
                        target_reduction * portfolio.total_value / 100
                    )
                    trades.append({
                        'action': 'reduce',
                        'vault_id': pos_data['position'].vault_id,
                        'chain': chain.value,
                        'amount': reduction,
                        'reason': f"Chain overexposure: {allocation:.1f}% > {self.max_chain_exposure*100:.1f}%"
                    })
                    target_reduction -= reduction * 100 / portfolio.total_value
                    if target_reduction <= 0:
                        break
                        
        return trades
    
    def _calculate_base_allocation(self, adjusted_apy: float, available_capital: float) -> float:
        """
        Calculate base allocation amount based on APY
        """
        # Higher APY gets larger allocation, with diminishing returns
        apy_factor = np.log(1 + adjusted_apy/100) / np.log(2)  # Log base 2
        allocation_pct = min(apy_factor * 0.1, self.max_position_size)  # 10% per doubling
        
        return available_capital * allocation_pct
    
    def _has_position(self, vault_id: str, portfolio: PortfolioState) -> bool:
        """Check if portfolio already has position in vault"""
        return any(pos.vault_id == vault_id for pos in portfolio.positions)
    
    def get_strategy_metrics(self, portfolio: PortfolioState) -> Dict[str, float]:
        """
        Get strategy-specific metrics
        """
        if not portfolio.positions:
            return {
                'avg_apy': 0.0,
                'avg_risk_score': 0.0,
                'chain_concentration': 0.0,
                'position_count': 0
            }
            
        apys = [pos.apy for pos in portfolio.positions]
        risk_scores = [pos.risk_score for pos in portfolio.positions]
        
        # Calculate chain concentration (HHI)
        chain_allocations = list(portfolio.chain_allocations.values())
        chain_hhi = sum([(v/100)**2 for v in chain_allocations]) if chain_allocations else 0
        
        return {
            'avg_apy': np.mean(apys),
            'max_apy': np.max(apys),
            'min_apy': np.min(apys),
            'avg_risk_score': np.mean(risk_scores),
            'chain_concentration': chain_hhi,
            'position_count': len(portfolio.positions),
            'utilization_rate': portfolio.utilization_rate
        }