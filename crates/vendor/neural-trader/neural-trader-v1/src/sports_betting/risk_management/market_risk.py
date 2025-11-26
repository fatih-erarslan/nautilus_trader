"""
Market Risk Analysis for Sports Betting

Implements odds movement tracking, liquidity risk assessment,
counterparty risk evaluation, and regulatory risk monitoring.
"""

import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from collections import defaultdict, deque


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BookmakerTier(Enum):
    """Bookmaker tier classifications"""
    TIER_1 = "tier_1"  # Major regulated bookmakers
    TIER_2 = "tier_2"  # Established but smaller bookmakers
    TIER_3 = "tier_3"  # Newer or less established
    UNRATED = "unrated"


@dataclass
class OddsMovement:
    """Tracks odds movement for a selection"""
    timestamp: datetime.datetime
    odds: float
    volume: Optional[float] = None
    bookmaker: Optional[str] = None


@dataclass
class MarketLiquidity:
    """Market liquidity assessment"""
    market_id: str
    total_volume: float
    bid_ask_spread: float
    depth: Dict[float, float]  # price -> volume
    last_updated: datetime.datetime
    liquidity_score: float = 0.0


@dataclass
class BookmakerProfile:
    """Bookmaker risk profile"""
    name: str
    tier: BookmakerTier
    credit_rating: Optional[str] = None
    regulatory_status: Dict[str, str] = field(default_factory=dict)  # jurisdiction -> status
    max_exposure_limit: float = 100000
    current_exposure: float = 0
    payment_history: List[Dict] = field(default_factory=list)
    risk_score: float = 0.0
    last_evaluated: Optional[datetime.datetime] = None


@dataclass
class RegulatoryAlert:
    """Regulatory risk alert"""
    jurisdiction: str
    alert_type: str  # 'restriction', 'ban', 'tax_change', 'license_issue'
    severity: RiskLevel
    description: str
    effective_date: datetime.datetime
    sports_affected: List[str] = field(default_factory=list)


@dataclass
class MarketRiskAssessment:
    """Comprehensive market risk assessment"""
    market_id: str
    timestamp: datetime.datetime
    odds_volatility: float
    liquidity_risk: RiskLevel
    counterparty_risk: RiskLevel
    regulatory_risk: RiskLevel
    overall_risk: RiskLevel
    risk_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MarketRiskAnalyzer:
    """
    Analyzes market risks including odds movements, liquidity,
    counterparty risk, and regulatory compliance.
    """
    
    def __init__(self,
                 max_odds_volatility: float = 0.10,
                 min_liquidity_score: float = 0.7,
                 max_bookmaker_exposure: float = 50000):
        """
        Initialize Market Risk Analyzer
        
        Args:
            max_odds_volatility: Maximum acceptable odds volatility
            min_liquidity_score: Minimum acceptable liquidity score
            max_bookmaker_exposure: Maximum exposure per bookmaker
        """
        self.max_odds_volatility = max_odds_volatility
        self.min_liquidity_score = min_liquidity_score
        self.max_bookmaker_exposure = max_bookmaker_exposure
        
        # Odds tracking
        self.odds_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.odds_alerts: Dict[str, List[Dict]] = defaultdict(list)
        
        # Liquidity tracking
        self.liquidity_data: Dict[str, MarketLiquidity] = {}
        self.liquidity_thresholds = {
            'min_volume': 10000,
            'max_spread': 0.05,
            'min_depth': 5
        }
        
        # Bookmaker profiles
        self.bookmaker_profiles: Dict[str, BookmakerProfile] = {}
        self._initialize_bookmaker_profiles()
        
        # Regulatory tracking
        self.regulatory_alerts: List[RegulatoryAlert] = []
        self.blocked_jurisdictions: Set[str] = set()
        self.restricted_sports: Dict[str, Set[str]] = defaultdict(set)
        
    def _initialize_bookmaker_profiles(self):
        """Initialize default bookmaker profiles"""
        # Tier 1 bookmakers
        tier1_bookmakers = [
            ('Pinnacle', 500000),
            ('Bet365', 400000),
            ('William Hill', 350000),
            ('Betfair', 450000)
        ]
        
        for name, limit in tier1_bookmakers:
            self.bookmaker_profiles[name] = BookmakerProfile(
                name=name,
                tier=BookmakerTier.TIER_1,
                credit_rating='A',
                max_exposure_limit=limit,
                regulatory_status={'US': 'licensed', 'UK': 'licensed', 'EU': 'licensed'}
            )
            
        # Tier 2 bookmakers
        tier2_bookmakers = [
            ('Unibet', 200000),
            ('888Sport', 150000),
            ('Betway', 180000)
        ]
        
        for name, limit in tier2_bookmakers:
            self.bookmaker_profiles[name] = BookmakerProfile(
                name=name,
                tier=BookmakerTier.TIER_2,
                credit_rating='BBB',
                max_exposure_limit=limit,
                regulatory_status={'US': 'partial', 'UK': 'licensed', 'EU': 'licensed'}
            )
            
    def track_odds_movement(self,
                            market_id: str,
                            odds: float,
                            bookmaker: Optional[str] = None,
                            volume: Optional[float] = None):
        """
        Track odds movement for a market
        
        Args:
            market_id: Unique market identifier
            odds: Current odds
            bookmaker: Bookmaker name
            volume: Volume at these odds
        """
        movement = OddsMovement(
            timestamp=datetime.datetime.now(),
            odds=odds,
            volume=volume,
            bookmaker=bookmaker
        )
        
        self.odds_history[market_id].append(movement)
        
        # Check for significant movements
        if self._detect_significant_movement(market_id):
            alert = {
                'timestamp': datetime.datetime.now(),
                'type': 'significant_movement',
                'market_id': market_id,
                'details': self._analyze_recent_movement(market_id)
            }
            self.odds_alerts[market_id].append(alert)
            logger.warning(f"Significant odds movement detected for {market_id}")
            
    def _detect_significant_movement(self, market_id: str) -> bool:
        """Detect if there's been significant odds movement"""
        history = list(self.odds_history[market_id])
        
        if len(history) < 5:
            return False
            
        # Check recent movements
        recent_odds = [m.odds for m in history[-10:]]
        
        if len(recent_odds) >= 2:
            # Calculate percentage change
            pct_change = abs(recent_odds[-1] - recent_odds[0]) / recent_odds[0]
            
            # Check volatility
            if len(recent_odds) >= 5:
                volatility = np.std(recent_odds) / np.mean(recent_odds)
                return pct_change > 0.05 or volatility > self.max_odds_volatility
                
            return pct_change > 0.10
            
        return False
    
    def _analyze_recent_movement(self, market_id: str) -> Dict:
        """Analyze recent odds movement patterns"""
        history = list(self.odds_history[market_id])[-20:]
        
        if not history:
            return {}
            
        odds = [m.odds for m in history]
        
        analysis = {
            'start_odds': odds[0],
            'current_odds': odds[-1],
            'pct_change': ((odds[-1] - odds[0]) / odds[0] * 100),
            'volatility': np.std(odds) / np.mean(odds) if len(odds) > 1 else 0,
            'trend': 'shortening' if odds[-1] < odds[0] else 'drifting',
            'num_changes': len(odds)
        }
        
        # Detect steam moves (rapid shortening)
        if len(odds) >= 5:
            recent_change = (odds[-1] - odds[-5]) / odds[-5]
            if recent_change < -0.10:  # 10% shortening
                analysis['steam_move'] = True
                
        return analysis
    
    def assess_liquidity_risk(self,
                              market_id: str,
                              total_volume: float,
                              bid_ask_spread: float,
                              depth_data: Dict[float, float]) -> MarketLiquidity:
        """
        Assess liquidity risk for a market
        
        Args:
            market_id: Market identifier
            total_volume: Total matched volume
            bid_ask_spread: Current bid-ask spread
            depth_data: Order book depth
            
        Returns:
            MarketLiquidity assessment
        """
        liquidity = MarketLiquidity(
            market_id=market_id,
            total_volume=total_volume,
            bid_ask_spread=bid_ask_spread,
            depth=depth_data,
            last_updated=datetime.datetime.now()
        )
        
        # Calculate liquidity score (0-1)
        scores = []
        
        # Volume score
        volume_score = min(total_volume / self.liquidity_thresholds['min_volume'], 1.0)
        scores.append(volume_score * 0.4)  # 40% weight
        
        # Spread score
        spread_score = 1 - min(bid_ask_spread / self.liquidity_thresholds['max_spread'], 1.0)
        scores.append(spread_score * 0.3)  # 30% weight
        
        # Depth score
        depth_levels = len(depth_data)
        depth_score = min(depth_levels / self.liquidity_thresholds['min_depth'], 1.0)
        scores.append(depth_score * 0.3)  # 30% weight
        
        liquidity.liquidity_score = sum(scores)
        
        # Store liquidity data
        self.liquidity_data[market_id] = liquidity
        
        return liquidity
    
    def evaluate_counterparty_risk(self,
                                   bookmaker: str,
                                   proposed_exposure: float
                                   ) -> Tuple[RiskLevel, List[str]]:
        """
        Evaluate counterparty risk for a bookmaker
        
        Args:
            bookmaker: Bookmaker name
            proposed_exposure: Proposed additional exposure
            
        Returns:
            Tuple of (risk_level, risk_factors)
        """
        if bookmaker not in self.bookmaker_profiles:
            return RiskLevel.HIGH, ["Unknown bookmaker"]
            
        profile = self.bookmaker_profiles[bookmaker]
        risk_factors = []
        
        # Check exposure limits
        new_exposure = profile.current_exposure + proposed_exposure
        exposure_ratio = new_exposure / profile.max_exposure_limit
        
        if exposure_ratio > 0.9:
            risk_factors.append(f"Near exposure limit ({exposure_ratio:.1%})")
        elif exposure_ratio > 0.7:
            risk_factors.append(f"High exposure ({exposure_ratio:.1%})")
            
        # Check tier
        if profile.tier == BookmakerTier.TIER_3:
            risk_factors.append("Lower tier bookmaker")
        elif profile.tier == BookmakerTier.UNRATED:
            risk_factors.append("Unrated bookmaker")
            
        # Check credit rating
        if profile.credit_rating and profile.credit_rating not in ['A', 'AA', 'AAA']:
            risk_factors.append(f"Lower credit rating: {profile.credit_rating}")
            
        # Check payment history
        if profile.payment_history:
            late_payments = sum(1 for p in profile.payment_history[-10:] if p.get('late', False))
            if late_payments > 2:
                risk_factors.append(f"Payment issues: {late_payments} late payments")
                
        # Calculate risk score
        risk_score = len(risk_factors) * 0.2 + exposure_ratio * 0.3
        
        if profile.tier == BookmakerTier.TIER_1:
            risk_score *= 0.5  # Lower risk for tier 1
        elif profile.tier == BookmakerTier.TIER_3:
            risk_score *= 1.5  # Higher risk for tier 3
            
        profile.risk_score = risk_score
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = RiskLevel.LOW
        elif risk_score < 0.6:
            risk_level = RiskLevel.MEDIUM
        elif risk_score < 0.8:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
            
        return risk_level, risk_factors
    
    def check_regulatory_compliance(self,
                                    jurisdiction: str,
                                    sport: str,
                                    bet_type: str
                                    ) -> Tuple[bool, List[str]]:
        """
        Check regulatory compliance for a bet
        
        Args:
            jurisdiction: Betting jurisdiction
            sport: Sport type
            bet_type: Type of bet
            
        Returns:
            Tuple of (is_compliant, compliance_issues)
        """
        compliance_issues = []
        
        # Check blocked jurisdictions
        if jurisdiction in self.blocked_jurisdictions:
            compliance_issues.append(f"Betting blocked in {jurisdiction}")
            
        # Check sport restrictions
        if sport in self.restricted_sports.get(jurisdiction, set()):
            compliance_issues.append(f"{sport} betting restricted in {jurisdiction}")
            
        # Check active regulatory alerts
        for alert in self.regulatory_alerts:
            if alert.jurisdiction == jurisdiction:
                if not alert.sports_affected or sport in alert.sports_affected:
                    if alert.effective_date <= datetime.datetime.now():
                        compliance_issues.append(
                            f"{alert.alert_type}: {alert.description}"
                        )
                        
        # Check bet type restrictions (could be expanded)
        restricted_bet_types = {
            'US': ['live_props', 'player_props'],
            'UK': ['credit_betting'],
            'AU': ['in_play_betting']
        }
        
        if bet_type in restricted_bet_types.get(jurisdiction, []):
            compliance_issues.append(f"{bet_type} restricted in {jurisdiction}")
            
        return len(compliance_issues) == 0, compliance_issues
    
    def add_regulatory_alert(self, alert: RegulatoryAlert):
        """Add a regulatory alert"""
        self.regulatory_alerts.append(alert)
        
        # Update blocked jurisdictions if needed
        if alert.alert_type == 'ban' and alert.severity == RiskLevel.CRITICAL:
            self.blocked_jurisdictions.add(alert.jurisdiction)
            
        # Update sport restrictions
        if alert.alert_type == 'restriction' and alert.sports_affected:
            for sport in alert.sports_affected:
                self.restricted_sports[alert.jurisdiction].add(sport)
                
        logger.warning(
            f"Regulatory alert added: {alert.alert_type} in {alert.jurisdiction} "
            f"- {alert.description}"
        )
        
    def perform_comprehensive_risk_assessment(self,
                                              market_id: str,
                                              bookmaker: str,
                                              jurisdiction: str,
                                              sport: str,
                                              proposed_stake: float
                                              ) -> MarketRiskAssessment:
        """
        Perform comprehensive risk assessment for a betting opportunity
        
        Args:
            market_id: Market identifier
            bookmaker: Bookmaker name
            jurisdiction: Betting jurisdiction
            sport: Sport type
            proposed_stake: Proposed bet amount
            
        Returns:
            Comprehensive risk assessment
        """
        assessment = MarketRiskAssessment(
            market_id=market_id,
            timestamp=datetime.datetime.now(),
            odds_volatility=0.0,
            liquidity_risk=RiskLevel.LOW,
            counterparty_risk=RiskLevel.LOW,
            regulatory_risk=RiskLevel.LOW,
            overall_risk=RiskLevel.LOW
        )
        
        # Assess odds volatility
        if market_id in self.odds_history:
            history = list(self.odds_history[market_id])
            if len(history) >= 5:
                odds = [m.odds for m in history[-20:]]
                assessment.odds_volatility = np.std(odds) / np.mean(odds)
                
                if assessment.odds_volatility > self.max_odds_volatility:
                    assessment.risk_factors.append(
                        f"High odds volatility: {assessment.odds_volatility:.2%}"
                    )
                    
        # Assess liquidity risk
        if market_id in self.liquidity_data:
            liquidity = self.liquidity_data[market_id]
            if liquidity.liquidity_score < self.min_liquidity_score:
                assessment.liquidity_risk = RiskLevel.HIGH
                assessment.risk_factors.append(
                    f"Low liquidity score: {liquidity.liquidity_score:.2f}"
                )
            elif liquidity.liquidity_score < 0.85:
                assessment.liquidity_risk = RiskLevel.MEDIUM
                
        # Assess counterparty risk
        cp_risk, cp_factors = self.evaluate_counterparty_risk(bookmaker, proposed_stake)
        assessment.counterparty_risk = cp_risk
        assessment.risk_factors.extend(cp_factors)
        
        # Assess regulatory risk
        is_compliant, compliance_issues = self.check_regulatory_compliance(
            jurisdiction, sport, 'standard'
        )
        if not is_compliant:
            assessment.regulatory_risk = RiskLevel.HIGH
            assessment.risk_factors.extend(compliance_issues)
            
        # Calculate overall risk
        risk_scores = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3
        }
        
        avg_risk_score = np.mean([
            risk_scores[assessment.liquidity_risk],
            risk_scores[assessment.counterparty_risk],
            risk_scores[assessment.regulatory_risk]
        ])
        
        # Add volatility impact
        if assessment.odds_volatility > self.max_odds_volatility * 1.5:
            avg_risk_score += 1
            
        # Determine overall risk
        if avg_risk_score < 0.5:
            assessment.overall_risk = RiskLevel.LOW
        elif avg_risk_score < 1.5:
            assessment.overall_risk = RiskLevel.MEDIUM
        elif avg_risk_score < 2.5:
            assessment.overall_risk = RiskLevel.HIGH
        else:
            assessment.overall_risk = RiskLevel.CRITICAL
            
        # Generate recommendations
        if assessment.overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            assessment.recommendations.append("Consider reducing stake or avoiding bet")
            
        if assessment.liquidity_risk == RiskLevel.HIGH:
            assessment.recommendations.append("Place bet early or in smaller chunks")
            
        if assessment.counterparty_risk == RiskLevel.HIGH:
            assessment.recommendations.append("Diversify across multiple bookmakers")
            
        if assessment.regulatory_risk == RiskLevel.HIGH:
            assessment.recommendations.append("Ensure full regulatory compliance")
            
        return assessment
    
    def update_bookmaker_exposure(self, bookmaker: str, amount: float):
        """Update bookmaker exposure"""
        if bookmaker in self.bookmaker_profiles:
            self.bookmaker_profiles[bookmaker].current_exposure += amount
            
    def get_market_risk_summary(self) -> Dict:
        """Get summary of current market risks"""
        return {
            'odds_alerts': {
                market: len(alerts) for market, alerts in self.odds_alerts.items()
            },
            'liquidity_warnings': [
                {
                    'market': market,
                    'score': liquidity.liquidity_score,
                    'volume': liquidity.total_volume
                }
                for market, liquidity in self.liquidity_data.items()
                if liquidity.liquidity_score < self.min_liquidity_score
            ],
            'bookmaker_exposures': {
                name: {
                    'current': profile.current_exposure,
                    'limit': profile.max_exposure_limit,
                    'utilization': f"{profile.current_exposure / profile.max_exposure_limit:.1%}"
                }
                for name, profile in self.bookmaker_profiles.items()
                if profile.current_exposure > 0
            },
            'regulatory_alerts': len(self.regulatory_alerts),
            'blocked_jurisdictions': list(self.blocked_jurisdictions)
        }