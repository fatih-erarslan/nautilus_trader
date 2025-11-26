"""
Stable Farmer Strategy

Conservative strategy focused on stablecoin pairs with consistent yields and minimal impermanent loss.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from datetime import datetime, timedelta
from .base_strategy import (
    BaseStrategy, VaultOpportunity, PortfolioState, Position,
    RiskLevel, ChainType
)


class StableFarmerStrategy(BaseStrategy):
    """Strategy that farms stable yields from stablecoin pairs"""
    
    # Recognized stablecoins
    STABLECOINS = {
        'USDC', 'USDT', 'DAI', 'BUSD', 'FRAX', 'TUSD', 'USDP', 'GUSD',
        'LUSD', 'USDD', 'CUSD', 'MAI', 'MIM', 'UST', 'DOLA', 'USDN'
    }
    
    # Preferred stablecoins (lower risk)
    PREFERRED_STABLES = {'USDC', 'USDT', 'DAI', 'FRAX'}
    
    def __init__(self,
                 min_apy_threshold: float = 8.0,
                 max_position_size: float = 0.40,
                 min_tvl: float = 500000,
                 max_protocol_exposure: float = 0.35,
                 stability_premium: float = 1.2):
        """
        Initialize Stable Farmer strategy
        
        Args:
            min_apy_threshold: Minimum APY for stablecoin pairs (default 8%)
            max_position_size: Maximum position size (default 40% - higher for stable)
            min_tvl: Minimum TVL required (default $500k)
            max_protocol_exposure: Maximum exposure to single protocol (default 35%)
            stability_premium: Multiplier for stable-stable pairs (default 1.2x)
        """
        super().__init__(
            name="Stable Farmer",
            risk_level=RiskLevel.LOW,
            min_apy_threshold=min_apy_threshold,
            max_position_size=max_position_size,
            rebalance_threshold=0.05  # Lower threshold for stable strategy
        )
        self.min_tvl = min_tvl
        self.max_protocol_exposure = max_protocol_exposure
        self.stability_premium = stability_premium
        
    def evaluate_opportunities(self,
                             opportunities: List[VaultOpportunity],
                             portfolio: PortfolioState) -> List[Tuple[VaultOpportunity, float]]:
        """
        Evaluate opportunities focusing on stable pairs
        """
        evaluated = []
        
        for opp in opportunities:
            # Only consider stablecoin pairs
            if not self._is_stable_pair(opp.token_pair):
                continue
                
            # Filter basic criteria
            if opp.tvl < self.min_tvl or opp.is_paused:
                continue
                
            # Filter by minimum APY
            if opp.total_apy < self.min_apy_threshold:
                continue
                
            # Calculate risk score
            risk_score = self.calculate_risk_score(opp)
            
            # Skip high risk even for stables
            if risk_score > 60:
                continue
            
            # Calculate allocation score
            allocation_score = self._calculate_allocation_score(opp, portfolio)
            
            # Calculate suggested allocation
            base_allocation = self._calculate_stable_allocation(
                opp, portfolio, allocation_score
            )
            
            # Validate allocation
            final_allocation = self.validate_position_size(
                base_allocation, portfolio, opp
            )
            
            if final_allocation > 0:
                evaluated.append((opp, final_allocation))
        
        # Sort by allocation score (considers APY, risk, and stability)
        evaluated.sort(
            key=lambda x: self._calculate_allocation_score(x[0], portfolio),
            reverse=True
        )
        
        # Select positions that fit within constraints
        selected = []
        remaining_capital = portfolio.available_capital
        protocol_allocations = portfolio.protocol_allocations.copy()
        
        for opp, allocation in evaluated:
            # Check protocol exposure
            current_protocol_exposure = protocol_allocations.get(opp.protocol, 0)
            new_exposure = current_protocol_exposure + (allocation / portfolio.total_value * 100)
            
            if new_exposure <= self.max_protocol_exposure * 100:
                if allocation <= remaining_capital:
                    selected.append((opp, allocation))
                    remaining_capital -= allocation
                    protocol_allocations[opp.protocol] = new_exposure
                    
        return selected
    
    def calculate_risk_score(self, opportunity: VaultOpportunity) -> float:
        """
        Calculate risk score with emphasis on stability
        """
        risk_score = 0.0
        
        # TVL risk (higher threshold for stables)
        if opportunity.tvl < 1000000:
            risk_score += 20
        elif opportunity.tvl < 5000000:
            risk_score += 10
        elif opportunity.tvl < 10000000:
            risk_score += 5
        else:
            risk_score += 2
            
        # Stability of tokens
        token1, token2 = opportunity.token_pair
        if token1 not in self.PREFERRED_STABLES:
            risk_score += 10
        if token2 not in self.PREFERRED_STABLES:
            risk_score += 10
            
        # APY sustainability (moderate APY preferred)
        if opportunity.total_apy > 50:
            risk_score += 15
        elif opportunity.total_apy > 30:
            risk_score += 8
        elif opportunity.total_apy < 5:
            risk_score += 10  # Too low might indicate issues
            
        # Age preference (older = more stable)
        age_days = (datetime.now() - opportunity.created_at).days
        if age_days < 30:
            risk_score += 15
        elif age_days < 90:
            risk_score += 8
        elif age_days < 180:
            risk_score += 4
            
        # Platform risks
        platform_risk = opportunity.risk_factors.get('platform_risk', 0)
        smart_contract_risk = opportunity.risk_factors.get('smart_contract', 0)
        
        risk_score += platform_risk * 0.3
        risk_score += smart_contract_risk * 0.3
        
        # Chain risk (prefer established chains)
        chain_risks = {
            ChainType.ETHEREUM: 2,
            ChainType.POLYGON: 5,
            ChainType.ARBITRUM: 6,
            ChainType.OPTIMISM: 6,
            ChainType.BSC: 10,
            ChainType.AVALANCHE: 8,
            ChainType.FANTOM: 12
        }
        risk_score += chain_risks.get(opportunity.chain, 15)
        
        # Fee risk
        if opportunity.withdraw_fee > 0.005:  # >0.5% withdraw fee
            risk_score += 8
            
        # Boost risk (boosted vaults might have additional risks)
        if opportunity.has_boost:
            risk_score += 5
            
        return min(risk_score, 100)
    
    def should_rebalance(self, portfolio: PortfolioState) -> bool:
        """
        Conservative rebalancing for stable positions
        """
        if not portfolio.positions:
            return False
            
        # Check if any stable position has issues
        for position in portfolio.positions:
            # Would need real-time data to check if vault still exists/is stable
            if position.risk_score > 70:
                return True
                
        # Check utilization (stable farmers should stay invested)
        if portfolio.utilization_rate < 80 and portfolio.available_capital > 5000:
            return True
            
        # Check protocol concentration
        for protocol, allocation in portfolio.protocol_allocations.items():
            if allocation > self.max_protocol_exposure * 100:
                return True
                
        # Monthly rebalancing for stable positions
        oldest_position = min(portfolio.positions, key=lambda p: p.entry_time)
        if (datetime.now() - oldest_position.entry_time).days >= 30:
            return True
            
        return False
    
    def generate_rebalance_trades(self,
                                portfolio: PortfolioState,
                                opportunities: List[VaultOpportunity]) -> List[Dict[str, Any]]:
        """
        Generate conservative rebalancing trades
        """
        trades = []
        
        # Check existing positions health
        unhealthy_positions = []
        healthy_positions = []
        
        for pos in portfolio.positions:
            current_opp = next(
                (o for o in opportunities if o.vault_id == pos.vault_id),
                None
            )
            
            if not current_opp or current_opp.is_paused:
                unhealthy_positions.append(pos)
            elif not self._is_stable_pair(current_opp.token_pair):
                # Position no longer stable
                unhealthy_positions.append(pos)
            elif current_opp.net_apy < self.min_apy_threshold * 0.8:
                # APY dropped too much
                unhealthy_positions.append(pos)
            else:
                healthy_positions.append({
                    'position': pos,
                    'opportunity': current_opp,
                    'current_apy': current_opp.net_apy
                })
        
        # Exit unhealthy positions
        freed_capital = 0
        for pos in unhealthy_positions:
            trades.append({
                'action': 'exit',
                'vault_id': pos.vault_id,
                'chain': pos.chain.value,
                'amount': pos.amount,
                'reason': 'Position no longer meets stable criteria'
            })
            freed_capital += pos.amount
        
        # Find new stable opportunities
        new_portfolio = PortfolioState(
            positions=[p['position'] for p in healthy_positions],
            total_value=portfolio.total_value,
            available_capital=portfolio.available_capital + freed_capital,
            timestamp=portfolio.timestamp,
            chain_allocations=portfolio.chain_allocations,
            protocol_allocations=portfolio.protocol_allocations
        )
        
        new_opportunities = self.evaluate_opportunities(opportunities, new_portfolio)
        
        # Enter new stable positions
        for opp, allocation in new_opportunities:
            trades.append({
                'action': 'enter',
                'vault_id': opp.vault_id,
                'chain': opp.chain.value,
                'amount': allocation,
                'expected_apy': opp.net_apy,
                'token_pair': opp.token_pair,
                'reason': f'Stable opportunity: {opp.net_apy:.2f}% APY'
            })
        
        # Rebalance protocol overexposure
        for protocol, allocation in portfolio.protocol_allocations.items():
            if allocation > self.max_protocol_exposure * 100:
                protocol_positions = [
                    p for p in healthy_positions
                    if p['opportunity'].protocol == protocol
                ]
                protocol_positions.sort(key=lambda x: x['current_apy'])
                
                target_reduction = allocation - (self.max_protocol_exposure * 100)
                for pos_data in protocol_positions:
                    reduction = min(
                        pos_data['position'].amount * 0.5,  # Max 50% reduction
                        target_reduction * portfolio.total_value / 100
                    )
                    trades.append({
                        'action': 'reduce',
                        'vault_id': pos_data['position'].vault_id,
                        'chain': pos_data['position'].chain.value,
                        'amount': reduction,
                        'reason': f'Protocol overexposure: {allocation:.1f}%'
                    })
                    target_reduction -= reduction * 100 / portfolio.total_value
                    if target_reduction <= 0:
                        break
                        
        return trades
    
    def _is_stable_pair(self, token_pair: Tuple[str, str]) -> bool:
        """Check if token pair consists of stablecoins"""
        token1, token2 = token_pair
        return token1 in self.STABLECOINS and token2 in self.STABLECOINS
    
    def _is_premium_stable_pair(self, token_pair: Tuple[str, str]) -> bool:
        """Check if both tokens are preferred stablecoins"""
        token1, token2 = token_pair
        return token1 in self.PREFERRED_STABLES and token2 in self.PREFERRED_STABLES
    
    def _calculate_allocation_score(self, 
                                  opportunity: VaultOpportunity,
                                  portfolio: PortfolioState) -> float:
        """
        Calculate allocation score for stable opportunities
        """
        # Base score from APY
        apy_score = opportunity.net_apy / 100 * 40  # Max 40 points from APY
        
        # Risk adjustment
        risk_score = self.calculate_risk_score(opportunity)
        risk_adjustment = (100 - risk_score) / 100 * 30  # Max 30 points from low risk
        
        # TVL score (preference for higher TVL)
        tvl_score = min(opportunity.tvl / 10000000, 1.0) * 20  # Max 20 points at $10M TVL
        
        # Stability premium
        stability_score = 0
        if self._is_premium_stable_pair(opportunity.token_pair):
            stability_score = 10  # 10 points for premium pairs
        elif self._is_stable_pair(opportunity.token_pair):
            stability_score = 5   # 5 points for regular stable pairs
            
        # Age bonus (older = better for stables)
        age_days = (datetime.now() - opportunity.created_at).days
        age_score = min(age_days / 180, 1.0) * 10  # Max 10 points at 6 months
        
        total_score = apy_score + risk_adjustment + tvl_score + stability_score + age_score
        
        # Apply stability premium multiplier for premium pairs
        if self._is_premium_stable_pair(opportunity.token_pair):
            total_score *= self.stability_premium
            
        return total_score
    
    def _calculate_stable_allocation(self,
                                   opportunity: VaultOpportunity,
                                   portfolio: PortfolioState,
                                   allocation_score: float) -> float:
        """
        Calculate allocation for stable opportunities
        """
        # Base allocation as percentage of available capital
        score_normalized = allocation_score / 100  # Normalize to 0-1
        base_allocation_pct = score_normalized * 0.3  # Max 30% per position
        
        # Increase allocation for premium stable pairs
        if self._is_premium_stable_pair(opportunity.token_pair):
            base_allocation_pct *= 1.5
            
        # Cap at max position size
        base_allocation_pct = min(base_allocation_pct, self.max_position_size)
        
        return portfolio.available_capital * base_allocation_pct
    
    def get_strategy_metrics(self, portfolio: PortfolioState) -> Dict[str, float]:
        """
        Get stable farming specific metrics
        """
        if not portfolio.positions:
            return {
                'avg_stable_apy': 0.0,
                'stable_position_count': 0,
                'premium_stable_ratio': 0.0,
                'avg_position_age_days': 0,
                'protocol_concentration': 0.0
            }
            
        stable_positions = []
        premium_positions = []
        
        for pos in portfolio.positions:
            if self._is_stable_pair(pos.token_pair):
                stable_positions.append(pos)
                if self._is_premium_stable_pair(pos.token_pair):
                    premium_positions.append(pos)
                    
        # Calculate metrics
        avg_apy = np.mean([p.apy for p in stable_positions]) if stable_positions else 0
        premium_ratio = len(premium_positions) / len(stable_positions) if stable_positions else 0
        
        # Average position age
        if stable_positions:
            ages = [(datetime.now() - p.entry_time).days for p in stable_positions]
            avg_age = np.mean(ages)
        else:
            avg_age = 0
            
        # Protocol concentration (HHI)
        protocol_values = list(portfolio.protocol_allocations.values())
        protocol_hhi = sum([(v/100)**2 for v in protocol_values]) if protocol_values else 0
        
        return {
            'avg_stable_apy': avg_apy,
            'stable_position_count': len(stable_positions),
            'premium_stable_ratio': premium_ratio,
            'avg_position_age_days': avg_age,
            'protocol_concentration': protocol_hhi,
            'total_stable_value': sum(p.amount for p in stable_positions)
        }