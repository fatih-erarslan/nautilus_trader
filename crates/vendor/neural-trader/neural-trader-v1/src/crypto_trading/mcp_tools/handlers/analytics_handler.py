"""
Analytics and performance tracking handler for Beefy Finance
"""

import sqlite3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import json

logger = logging.getLogger(__name__)

class AnalyticsHandler:
    """Handle analytics, risk calculations, and performance tracking"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._init_database()
        
        # Risk thresholds
        self.risk_thresholds = {
            "liquidity": {"low": 1000000, "medium": 500000},  # TVL thresholds
            "volatility": {"low": 0.1, "medium": 0.25},      # APY volatility
            "concentration": {"low": 0.2, "medium": 0.4}      # Portfolio concentration
        }
    
    def _init_database(self):
        """Initialize database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info("Analytics database connection established")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    async def calculate_vault_risk(self, vault_id: str) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for a vault"""
        
        try:
            # Get vault data
            vault_data = await self._get_vault_analytics_data(vault_id)
            
            # Calculate individual risk scores
            liquidity_risk = self._calculate_liquidity_risk(vault_data['tvl'])
            smart_contract_risk = self._calculate_smart_contract_risk(vault_data)
            protocol_risk = self._calculate_protocol_risk(vault_data)
            market_risk = self._calculate_market_risk(vault_data)
            
            # Calculate overall risk score (0-100)
            overall_risk = (
                liquidity_risk * 0.25 +
                smart_contract_risk * 0.35 +
                protocol_risk * 0.25 +
                market_risk * 0.15
            )
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(
                liquidity_risk,
                smart_contract_risk,
                protocol_risk,
                market_risk
            )
            
            return {
                "overall_risk_score": round(overall_risk, 2),
                "liquidity_risk": round(liquidity_risk, 2),
                "smart_contract_risk": round(smart_contract_risk, 2),
                "protocol_risk": round(protocol_risk, 2),
                "market_risk": round(market_risk, 2),
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error calculating vault risk: {e}")
            raise
    
    async def calculate_rebalance(
        self,
        current_positions: List[Dict[str, Any]],
        target_allocations: Optional[Dict[str, float]] = None,
        strategy: str = "equal_weight"
    ) -> Dict[str, Any]:
        """Calculate optimal portfolio rebalancing actions"""
        
        try:
            # Calculate total portfolio value
            total_value = sum(p['current_value'] for p in current_positions)
            
            # Current allocations
            current_allocations = {
                p['vault_id']: p['current_value'] / total_value 
                for p in current_positions
            }
            
            # Determine target allocations based on strategy
            if strategy == "equal_weight":
                num_vaults = len(current_positions)
                target_allocations = {
                    p['vault_id']: 1.0 / num_vaults 
                    for p in current_positions
                }
            elif strategy == "risk_parity":
                target_allocations = await self._calculate_risk_parity_allocation(
                    current_positions
                )
            elif strategy == "max_apy":
                target_allocations = await self._calculate_max_apy_allocation(
                    current_positions
                )
            # else: use provided custom allocations
            
            # Calculate rebalancing actions
            actions = []
            
            for vault_id, target_alloc in target_allocations.items():
                current_alloc = current_allocations.get(vault_id, 0)
                diff = target_alloc - current_alloc
                
                if abs(diff) > 0.01:  # 1% threshold
                    amount = abs(diff * total_value)
                    
                    # Find position data
                    position = next(
                        (p for p in current_positions if p['vault_id'] == vault_id),
                        None
                    )
                    
                    if diff < 0:  # Need to withdraw
                        actions.append({
                            "type": "withdraw",
                            "vault_id": vault_id,
                            "amount": amount,
                            "shares": amount / position['current_value'] * position['shares_owned'],
                            "reason": f"Reduce allocation from {current_alloc:.1%} to {target_alloc:.1%}"
                        })
                    else:  # Need to deposit
                        actions.append({
                            "type": "deposit",
                            "vault_id": vault_id,
                            "amount": amount,
                            "shares": amount,  # Will be calculated during execution
                            "reason": f"Increase allocation from {current_alloc:.1%} to {target_alloc:.1%}"
                        })
            
            return {
                "actions": actions,
                "current_allocations": current_allocations,
                "new_allocations": target_allocations,
                "total_value": total_value,
                "rebalance_cost_estimate": len(actions) * 5  # $5 per transaction estimate
            }
            
        except Exception as e:
            logger.error(f"Error calculating rebalance: {e}")
            raise
    
    async def process_apy_update(
        self,
        vault_id: str,
        apy: float,
        timestamp: datetime
    ) -> None:
        """Process real-time APY update from WebSocket"""
        
        try:
            # Store APY update in database
            cursor = self.conn.cursor()
            
            # Get active position for this vault
            cursor.execute("""
                SELECT id FROM vault_positions 
                WHERE vault_id = ? AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
            """, (vault_id,))
            
            position = cursor.fetchone()
            if position:
                # Record in yield history for tracking
                cursor.execute("""
                    INSERT INTO yield_history
                    (vault_id, position_id, earned_amount, apy_snapshot, 
                     price_per_share, recorded_at)
                    VALUES (?, ?, 0, ?, 1.0, ?)
                """, (
                    vault_id,
                    position['id'],
                    apy,
                    timestamp
                ))
                
                self.conn.commit()
                
                # Check for significant APY changes
                await self._check_apy_alerts(vault_id, apy)
                
        except Exception as e:
            logger.error(f"Error processing APY update: {e}")
    
    def _calculate_liquidity_risk(self, tvl: float) -> float:
        """Calculate liquidity risk based on TVL"""
        if tvl > self.risk_thresholds["liquidity"]["low"]:
            return 10  # Very low risk
        elif tvl > self.risk_thresholds["liquidity"]["medium"]:
            return 30  # Low risk
        else:
            return 60  # Medium to high risk
    
    def _calculate_smart_contract_risk(self, vault_data: Dict) -> float:
        """Calculate smart contract risk"""
        risk = 50  # Base risk
        
        # Reduce risk for audited contracts
        if vault_data.get('has_audit', False):
            risk -= 20
        
        # Reduce risk for established protocols
        if vault_data.get('days_active', 0) > 180:
            risk -= 10
        
        # Increase risk for complex strategies
        if 'leveraged' in vault_data.get('strategy_type', '').lower():
            risk += 20
        
        return max(0, min(100, risk))
    
    def _calculate_protocol_risk(self, vault_data: Dict) -> float:
        """Calculate protocol risk"""
        risk = 40  # Base risk
        
        # Known safe protocols
        safe_protocols = ['pancakeswap', 'curve', 'aave', 'compound']
        if any(p in vault_data.get('platform', '').lower() for p in safe_protocols):
            risk -= 15
        
        # Multi-strategy vaults have higher complexity
        if vault_data.get('strategy_count', 1) > 1:
            risk += 10
        
        return max(0, min(100, risk))
    
    def _calculate_market_risk(self, vault_data: Dict) -> float:
        """Calculate market risk based on volatility"""
        # Use APY volatility as proxy for market risk
        volatility = vault_data.get('apy_volatility', 0.15)
        
        if volatility < self.risk_thresholds["volatility"]["low"]:
            return 20
        elif volatility < self.risk_thresholds["volatility"]["medium"]:
            return 40
        else:
            return 70
    
    def _generate_risk_recommendations(
        self,
        liquidity_risk: float,
        smart_contract_risk: float,
        protocol_risk: float,
        market_risk: float
    ) -> List[str]:
        """Generate risk-based recommendations"""
        recommendations = []
        
        if liquidity_risk > 50:
            recommendations.append("Consider vaults with higher TVL for better liquidity")
        
        if smart_contract_risk > 60:
            recommendations.append("Prefer audited vaults with established track records")
        
        if protocol_risk > 50:
            recommendations.append("Diversify across multiple protocols to reduce risk")
        
        if market_risk > 60:
            recommendations.append("High volatility detected - consider stable asset vaults")
        
        # Overall portfolio recommendations
        overall_risk = (liquidity_risk + smart_contract_risk + protocol_risk + market_risk) / 4
        if overall_risk > 60:
            recommendations.append("Overall risk is high - consider more conservative allocations")
        elif overall_risk < 30:
            recommendations.append("Risk profile is conservative - consider higher yield opportunities")
        
        return recommendations
    
    async def _calculate_risk_parity_allocation(
        self,
        positions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate risk parity allocation"""
        # Simplified risk parity - inverse volatility weighting
        risk_weights = {}
        total_inverse_vol = 0
        
        for position in positions:
            # Use APY as proxy for volatility (simplified)
            volatility = position['current_apy'] / 100 * 0.5  # Rough approximation
            inverse_vol = 1 / max(volatility, 0.05)
            risk_weights[position['vault_id']] = inverse_vol
            total_inverse_vol += inverse_vol
        
        # Normalize to get allocations
        return {
            vault_id: weight / total_inverse_vol 
            for vault_id, weight in risk_weights.items()
        }
    
    async def _calculate_max_apy_allocation(
        self,
        positions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate max APY allocation with risk constraints"""
        # Sort by APY
        sorted_positions = sorted(
            positions, 
            key=lambda p: p['current_apy'], 
            reverse=True
        )
        
        allocations = {}
        remaining = 1.0
        max_single_allocation = 0.4  # Max 40% in any single vault
        
        for position in sorted_positions:
            allocation = min(remaining, max_single_allocation)
            allocations[position['vault_id']] = allocation
            remaining -= allocation
            
            if remaining <= 0:
                break
        
        # Distribute any remaining equally
        if remaining > 0:
            equal_share = remaining / len(allocations)
            for vault_id in allocations:
                allocations[vault_id] += equal_share
        
        return allocations
    
    async def _get_vault_analytics_data(self, vault_id: str) -> Dict[str, Any]:
        """Get comprehensive vault data for analytics"""
        # Simulate fetching vault analytics data
        # In production, this would aggregate from multiple sources
        return {
            "vault_id": vault_id,
            "tvl": 5000000,  # $5M TVL
            "apy": 15.5,
            "apy_volatility": 0.12,
            "has_audit": True,
            "days_active": 250,
            "strategy_type": "auto-compound",
            "platform": "PancakeSwap",
            "strategy_count": 1
        }
    
    async def _check_apy_alerts(self, vault_id: str, new_apy: float) -> None:
        """Check for significant APY changes and generate alerts"""
        # Get recent APY history
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT AVG(apy_snapshot) as avg_apy
            FROM yield_history
            WHERE vault_id = ? 
            AND recorded_at > datetime('now', '-24 hours')
        """, (vault_id,))
        
        result = cursor.fetchone()
        if result and result['avg_apy']:
            avg_apy = result['avg_apy']
            change_pct = abs(new_apy - avg_apy) / avg_apy
            
            if change_pct > 0.2:  # 20% change
                logger.warning(
                    f"Significant APY change for vault {vault_id}: "
                    f"{avg_apy:.2f}% -> {new_apy:.2f}% ({change_pct:.1%} change)"
                )
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()