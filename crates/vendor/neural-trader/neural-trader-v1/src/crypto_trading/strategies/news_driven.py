"""
News Driven Strategy

Trading strategy that adjusts positions based on market sentiment and news events.
Integrates with existing news analysis infrastructure.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from .base_strategy import (
    BaseStrategy, VaultOpportunity, PortfolioState, Position,
    RiskLevel, ChainType
)


class SentimentLevel(Enum):
    """Market sentiment levels"""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


class EventType(Enum):
    """Types of news events"""
    PROTOCOL_HACK = "hack"
    CHAIN_ISSUE = "chain_issue"
    REGULATORY = "regulatory"
    PARTNERSHIP = "partnership"
    UPGRADE = "upgrade"
    MARKET_CRASH = "market_crash"
    MARKET_RALLY = "market_rally"
    TOKEN_LISTING = "listing"
    AUDIT_REPORT = "audit"


@dataclass
class NewsEvent:
    """Represents a news event"""
    event_type: EventType
    severity: float  # 0-1 scale
    chains_affected: List[ChainType]
    protocols_affected: List[str]
    tokens_affected: List[str]
    timestamp: datetime
    sentiment_impact: SentimentLevel
    confidence: float  # 0-1 confidence in the news
    source: str
    description: str


@dataclass
class MarketSentiment:
    """Current market sentiment analysis"""
    overall_sentiment: SentimentLevel
    sentiment_score: float  # -100 to +100
    chain_sentiments: Dict[ChainType, float]
    protocol_sentiments: Dict[str, float]
    token_sentiments: Dict[str, float]
    trending_topics: List[str]
    fear_greed_index: float  # 0-100
    social_volume: float  # Normalized social media activity
    timestamp: datetime


class NewsDrivenStrategy(BaseStrategy):
    """Strategy that reacts to news and market sentiment"""
    
    def __init__(self,
                 min_apy_threshold: float = 15.0,
                 max_position_size: float = 0.25,
                 sentiment_weight: float = 0.4,
                 news_decay_hours: float = 24.0,
                 panic_exit_threshold: float = -70.0,
                 fomo_entry_threshold: float = 70.0):
        """
        Initialize News Driven strategy
        
        Args:
            min_apy_threshold: Minimum APY in neutral market
            max_position_size: Maximum position size
            sentiment_weight: Weight of sentiment in decisions (0-1)
            news_decay_hours: Hours before news impact decays
            panic_exit_threshold: Sentiment score for panic exit
            fomo_entry_threshold: Sentiment score for FOMO entry
        """
        super().__init__(
            name="News Driven",
            risk_level=RiskLevel.MEDIUM,
            min_apy_threshold=min_apy_threshold,
            max_position_size=max_position_size,
            rebalance_threshold=0.08
        )
        self.sentiment_weight = sentiment_weight
        self.news_decay_hours = news_decay_hours
        self.panic_exit_threshold = panic_exit_threshold
        self.fomo_entry_threshold = fomo_entry_threshold
        
        # Recent news cache
        self.recent_news: List[NewsEvent] = []
        self.current_sentiment: Optional[MarketSentiment] = None
        
    def update_market_data(self, 
                          news_events: List[NewsEvent],
                          sentiment: MarketSentiment):
        """Update strategy with latest news and sentiment"""
        # Filter recent news
        cutoff_time = datetime.now() - timedelta(hours=self.news_decay_hours * 2)
        self.recent_news = [n for n in news_events if n.timestamp > cutoff_time]
        self.current_sentiment = sentiment
        
    def evaluate_opportunities(self,
                             opportunities: List[VaultOpportunity],
                             portfolio: PortfolioState) -> List[Tuple[VaultOpportunity, float]]:
        """
        Evaluate opportunities based on news and sentiment
        """
        if not self.current_sentiment:
            # Fallback to base evaluation without sentiment
            return self._base_evaluation(opportunities, portfolio)
            
        evaluated = []
        
        for opp in opportunities:
            # Basic filters
            if opp.is_paused or opp.tvl < 200000:
                continue
                
            # Calculate sentiment-adjusted metrics
            sentiment_multiplier = self._calculate_sentiment_multiplier(opp)
            
            # Adjust minimum APY based on sentiment
            adjusted_min_apy = self.min_apy_threshold / sentiment_multiplier
            if opp.total_apy < adjusted_min_apy:
                continue
                
            # Calculate risk score with news impact
            base_risk = self.calculate_risk_score(opp)
            news_risk_adjustment = self._calculate_news_risk_adjustment(opp)
            adjusted_risk = min(base_risk + news_risk_adjustment, 100)
            
            # Skip if too risky given current news
            if adjusted_risk > 80:
                continue
                
            # Calculate allocation with sentiment adjustment
            base_allocation = self._calculate_sentiment_based_allocation(
                opp, portfolio, sentiment_multiplier, adjusted_risk
            )
            
            # Validate allocation
            final_allocation = self.validate_position_size(
                base_allocation, portfolio, opp
            )
            
            if final_allocation > 0:
                evaluated.append((opp, final_allocation))
                
        # Sort by sentiment-adjusted score
        evaluated.sort(
            key=lambda x: self._calculate_opportunity_score(x[0]),
            reverse=True
        )
        
        return evaluated
    
    def calculate_risk_score(self, opportunity: VaultOpportunity) -> float:
        """
        Calculate risk score with news considerations
        """
        base_risk = 0.0
        
        # Standard risk factors
        if opportunity.tvl < 1000000:
            base_risk += 15
        elif opportunity.tvl < 5000000:
            base_risk += 8
        else:
            base_risk += 4
            
        # APY risk
        if opportunity.total_apy > 100:
            base_risk += 15
        elif opportunity.total_apy > 50:
            base_risk += 8
            
        # Age risk
        age_days = (datetime.now() - opportunity.created_at).days
        if age_days < 30:
            base_risk += 12
        elif age_days < 90:
            base_risk += 6
            
        # Platform risks
        platform_risk = opportunity.risk_factors.get('platform_risk', 0)
        smart_contract_risk = opportunity.risk_factors.get('smart_contract', 0)
        base_risk += platform_risk * 0.3
        base_risk += smart_contract_risk * 0.3
        
        # Chain risk with sentiment adjustment
        chain_sentiment = self.current_sentiment.chain_sentiments.get(
            opportunity.chain, 0
        ) if self.current_sentiment else 0
        
        chain_base_risks = {
            ChainType.ETHEREUM: 5,
            ChainType.BSC: 12,
            ChainType.POLYGON: 8,
            ChainType.ARBITRUM: 7,
            ChainType.OPTIMISM: 7,
            ChainType.AVALANCHE: 10,
            ChainType.FANTOM: 15
        }
        
        chain_risk = chain_base_risks.get(opportunity.chain, 20)
        # Negative sentiment increases risk
        if chain_sentiment < -20:
            chain_risk *= 1.5
        elif chain_sentiment > 20:
            chain_risk *= 0.8
            
        base_risk += chain_risk
        
        # Check for specific news events affecting this opportunity
        news_impact = self._get_news_impact(opportunity)
        base_risk += news_impact
        
        return min(base_risk, 100)
    
    def should_rebalance(self, portfolio: PortfolioState) -> bool:
        """
        Check if rebalancing needed based on news/sentiment
        """
        if not portfolio.positions or not self.current_sentiment:
            return False
            
        # Emergency rebalance on extreme sentiment
        if self.current_sentiment.sentiment_score <= self.panic_exit_threshold:
            return True
        elif self.current_sentiment.sentiment_score >= self.fomo_entry_threshold:
            return True
            
        # Check for critical news events
        critical_news = self._get_critical_news_events()
        if critical_news:
            # Check if any position is affected
            for pos in portfolio.positions:
                if self._is_position_affected_by_news(pos, critical_news):
                    return True
                    
        # Check sentiment shift
        for pos in portfolio.positions:
            position_sentiment = self._get_position_sentiment(pos)
            if position_sentiment < -50:  # Very negative sentiment
                return True
                
        # Regular rebalancing check
        return super().should_rebalance(portfolio)
    
    def generate_rebalance_trades(self,
                                portfolio: PortfolioState,
                                opportunities: List[VaultOpportunity]) -> List[Dict[str, Any]]:
        """
        Generate trades based on news and sentiment
        """
        trades = []
        
        if not self.current_sentiment:
            return trades
            
        # Handle panic situations
        if self.current_sentiment.sentiment_score <= self.panic_exit_threshold:
            return self._generate_panic_trades(portfolio)
            
        # Check each position against current sentiment
        for pos in portfolio.positions:
            position_sentiment = self._get_position_sentiment(pos)
            news_events = self._get_affecting_news(pos)
            
            # Exit positions with very negative sentiment or bad news
            if position_sentiment < -60 or any(e.severity > 0.7 for e in news_events):
                trades.append({
                    'action': 'exit',
                    'vault_id': pos.vault_id,
                    'chain': pos.chain.value,
                    'amount': pos.amount,
                    'reason': f"Negative sentiment: {position_sentiment:.0f} or critical news"
                })
                continue
                
            # Reduce positions with moderate negative sentiment
            if position_sentiment < -30:
                reduction = pos.amount * 0.5
                trades.append({
                    'action': 'reduce',
                    'vault_id': pos.vault_id,
                    'chain': pos.chain.value,
                    'amount': reduction,
                    'reason': f"Moderate negative sentiment: {position_sentiment:.0f}"
                })
                
        # Look for opportunities in positive sentiment areas
        if self.current_sentiment.sentiment_score > 30:
            # Find chains/protocols with positive sentiment
            positive_chains = [
                chain for chain, sent in self.current_sentiment.chain_sentiments.items()
                if sent > 40
            ]
            
            # Evaluate opportunities in positive sentiment areas
            filtered_opps = [
                o for o in opportunities 
                if o.chain in positive_chains or 
                self.current_sentiment.protocol_sentiments.get(o.protocol, 0) > 40
            ]
            
            new_opportunities = self.evaluate_opportunities(filtered_opps, portfolio)
            
            for opp, allocation in new_opportunities[:5]:  # Top 5 opportunities
                trades.append({
                    'action': 'enter',
                    'vault_id': opp.vault_id,
                    'chain': opp.chain.value,
                    'amount': allocation,
                    'expected_apy': opp.net_apy,
                    'sentiment_score': self._get_opportunity_sentiment(opp),
                    'reason': f"Positive sentiment opportunity"
                })
                
        return trades
    
    def _calculate_sentiment_multiplier(self, opportunity: VaultOpportunity) -> float:
        """
        Calculate sentiment multiplier for opportunity evaluation
        """
        if not self.current_sentiment:
            return 1.0
            
        # Get relevant sentiment scores
        chain_sentiment = self.current_sentiment.chain_sentiments.get(
            opportunity.chain, 0
        )
        protocol_sentiment = self.current_sentiment.protocol_sentiments.get(
            opportunity.protocol, 0
        )
        
        # Weight sentiments
        combined_sentiment = (chain_sentiment * 0.4 + protocol_sentiment * 0.6)
        
        # Check token sentiments
        for token in opportunity.token_pair:
            token_sentiment = self.current_sentiment.token_sentiments.get(token, 0)
            combined_sentiment = (combined_sentiment * 0.7 + token_sentiment * 0.3)
            
        # Convert to multiplier (sentiment from -100 to +100)
        # At -100 sentiment: 0.3x multiplier
        # At 0 sentiment: 1.0x multiplier  
        # At +100 sentiment: 2.0x multiplier
        multiplier = 1.0 + (combined_sentiment / 100) * 0.7
        
        # Apply sentiment weight
        weighted_multiplier = 1.0 + (multiplier - 1.0) * self.sentiment_weight
        
        return max(0.3, min(2.0, weighted_multiplier))
    
    def _calculate_news_risk_adjustment(self, opportunity: VaultOpportunity) -> float:
        """
        Calculate risk adjustment based on recent news
        """
        risk_adjustment = 0.0
        
        for news in self.recent_news:
            # Check if news affects this opportunity
            affects_chain = opportunity.chain in news.chains_affected
            affects_protocol = opportunity.protocol in news.protocols_affected
            affects_token = any(t in news.tokens_affected for t in opportunity.token_pair)
            
            if not (affects_chain or affects_protocol or affects_token):
                continue
                
            # Calculate time decay
            hours_old = (datetime.now() - news.timestamp).total_seconds() / 3600
            decay_factor = max(0, 1 - hours_old / self.news_decay_hours)
            
            # Calculate impact based on event type and severity
            impact = 0.0
            if news.event_type in [EventType.PROTOCOL_HACK, EventType.CHAIN_ISSUE]:
                impact = 30 * news.severity
            elif news.event_type == EventType.REGULATORY:
                impact = 20 * news.severity
            elif news.event_type == EventType.MARKET_CRASH:
                impact = 25 * news.severity
            elif news.event_type in [EventType.PARTNERSHIP, EventType.UPGRADE]:
                impact = -10 * news.severity  # Positive news reduces risk
            elif news.event_type == EventType.AUDIT_REPORT:
                impact = -15 * news.severity if news.sentiment_impact.value > 0 else 20
                
            # Apply confidence and decay
            impact *= news.confidence * decay_factor
            
            # Higher impact if directly affects protocol
            if affects_protocol:
                impact *= 1.5
                
            risk_adjustment += impact
            
        return risk_adjustment
    
    def _calculate_sentiment_based_allocation(self,
                                            opportunity: VaultOpportunity,
                                            portfolio: PortfolioState,
                                            sentiment_multiplier: float,
                                            risk_score: float) -> float:
        """
        Calculate allocation based on sentiment and risk
        """
        # Base allocation from APY and risk
        risk_factor = 1 - (risk_score / 100) * 0.8
        base_score = opportunity.net_apy * risk_factor
        
        # Apply sentiment multiplier
        sentiment_adjusted_score = base_score * sentiment_multiplier
        
        # Calculate allocation percentage
        if sentiment_adjusted_score > 100:
            alloc_pct = 0.20  # 20% for very high scores
        elif sentiment_adjusted_score > 50:
            alloc_pct = 0.15
        elif sentiment_adjusted_score > 30:
            alloc_pct = 0.10
        elif sentiment_adjusted_score > 20:
            alloc_pct = 0.05
        else:
            alloc_pct = 0.03
            
        # Adjust for fear & greed
        if self.current_sentiment:
            if self.current_sentiment.fear_greed_index < 20:  # Extreme fear
                alloc_pct *= 1.5  # Contrarian: increase allocation
            elif self.current_sentiment.fear_greed_index > 80:  # Extreme greed
                alloc_pct *= 0.7  # Cautious: reduce allocation
                
        return portfolio.available_capital * alloc_pct
    
    def _base_evaluation(self,
                        opportunities: List[VaultOpportunity],
                        portfolio: PortfolioState) -> List[Tuple[VaultOpportunity, float]]:
        """
        Fallback evaluation without sentiment data
        """
        evaluated = []
        
        for opp in opportunities:
            if (opp.total_apy >= self.min_apy_threshold and
                not opp.is_paused and
                opp.tvl >= 200000):
                
                risk_score = self.calculate_risk_score(opp)
                if risk_score <= 70:
                    allocation = portfolio.available_capital * 0.1
                    allocation = self.validate_position_size(allocation, portfolio, opp)
                    if allocation > 0:
                        evaluated.append((opp, allocation))
                        
        return sorted(evaluated, key=lambda x: x[0].net_apy, reverse=True)
    
    def _get_news_impact(self, opportunity: VaultOpportunity) -> float:
        """Get total news impact score for opportunity"""
        return self._calculate_news_risk_adjustment(opportunity)
    
    def _get_critical_news_events(self) -> List[NewsEvent]:
        """Get critical news events from recent news"""
        critical_types = {
            EventType.PROTOCOL_HACK, EventType.CHAIN_ISSUE,
            EventType.REGULATORY, EventType.MARKET_CRASH
        }
        return [
            news for news in self.recent_news
            if news.event_type in critical_types and news.severity > 0.6
        ]
    
    def _is_position_affected_by_news(self, 
                                    position: Position,
                                    news_events: List[NewsEvent]) -> bool:
        """Check if position is affected by news events"""
        for news in news_events:
            if position.chain in news.chains_affected:
                return True
            if position.protocol in news.protocols_affected:
                return True
            if any(t in news.tokens_affected for t in position.token_pair):
                return True
        return False
    
    def _get_position_sentiment(self, position: Position) -> float:
        """Get sentiment score for a position"""
        if not self.current_sentiment:
            return 0.0
            
        chain_sent = self.current_sentiment.chain_sentiments.get(position.chain, 0)
        protocol_sent = self.current_sentiment.protocol_sentiments.get(position.protocol, 0)
        
        # Average of chain and protocol sentiment
        return (chain_sent + protocol_sent) / 2
    
    def _get_affecting_news(self, position: Position) -> List[NewsEvent]:
        """Get news events affecting a position"""
        affecting = []
        for news in self.recent_news:
            if (position.chain in news.chains_affected or
                position.protocol in news.protocols_affected or
                any(t in news.tokens_affected for t in position.token_pair)):
                affecting.append(news)
        return affecting
    
    def _generate_panic_trades(self, portfolio: PortfolioState) -> List[Dict[str, Any]]:
        """Generate trades for panic market conditions"""
        trades = []
        
        # Exit all high-risk positions
        for pos in portfolio.positions:
            if pos.risk_score > 50:
                trades.append({
                    'action': 'exit',
                    'vault_id': pos.vault_id,
                    'chain': pos.chain.value,
                    'amount': pos.amount,
                    'reason': 'Panic market conditions - exiting high risk'
                })
            else:
                # Reduce medium risk positions by 50%
                trades.append({
                    'action': 'reduce',
                    'vault_id': pos.vault_id,
                    'chain': pos.chain.value,
                    'amount': pos.amount * 0.5,
                    'reason': 'Panic market conditions - reducing exposure'
                })
                
        return trades
    
    def _calculate_opportunity_score(self, opportunity: VaultOpportunity) -> float:
        """Calculate overall score for opportunity ranking"""
        base_score = opportunity.net_apy
        risk_score = self.calculate_risk_score(opportunity)
        sentiment_mult = self._calculate_sentiment_multiplier(opportunity)
        
        return base_score * sentiment_mult / (1 + risk_score/100)
    
    def _get_opportunity_sentiment(self, opportunity: VaultOpportunity) -> float:
        """Get sentiment score for an opportunity"""
        if not self.current_sentiment:
            return 0.0
            
        chain_sent = self.current_sentiment.chain_sentiments.get(opportunity.chain, 0)
        protocol_sent = self.current_sentiment.protocol_sentiments.get(opportunity.protocol, 0)
        
        return (chain_sent + protocol_sent) / 2
    
    def get_strategy_metrics(self, portfolio: PortfolioState) -> Dict[str, float]:
        """Get news-driven strategy metrics"""
        base_metrics = {
            'avg_position_sentiment': 0.0,
            'news_events_count': len(self.recent_news),
            'market_sentiment': 0.0,
            'fear_greed_index': 50.0,
            'sentiment_influence': self.sentiment_weight
        }
        
        if self.current_sentiment:
            base_metrics['market_sentiment'] = self.current_sentiment.sentiment_score
            base_metrics['fear_greed_index'] = self.current_sentiment.fear_greed_index
            
            if portfolio.positions:
                sentiments = [self._get_position_sentiment(pos) for pos in portfolio.positions]
                base_metrics['avg_position_sentiment'] = np.mean(sentiments)
                
        return base_metrics