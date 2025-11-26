"""Mock news feed generator for testing news-driven trading.

Generates realistic news events with sentiment scores and market impact.
"""

import asyncio
import random
import time
import uuid
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class NewsCategory(Enum):
    """News categories."""
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY = "regulatory"
    ECONOMIC_DATA = "economic_data"
    ANALYST_RATING = "analyst_rating"
    MANAGEMENT_CHANGE = "management_change"
    LEGAL = "legal"
    PARTNERSHIP = "partnership"
    MARKET_COMMENTARY = "market_commentary"


class NewsSentiment(Enum):
    """News sentiment levels."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class NewsEvent:
    """A news event."""
    id: str
    timestamp: float
    symbol: str
    headline: str
    summary: str
    category: NewsCategory
    sentiment: NewsSentiment
    impact_score: float  # 0-1, expected market impact
    source: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockNewsConfig:
    """Configuration for mock news generator."""
    symbols: List[str]
    events_per_hour: float = 10.0
    enable_correlated_events: bool = True  # Multiple related news
    enable_scheduled_events: bool = True  # Earnings, data releases
    sentiment_bias: float = 0.0  # -1 to 1, 0 is neutral
    high_impact_probability: float = 0.1


class NewsTemplates:
    """Templates for generating realistic news."""
    
    EARNINGS_TEMPLATES = {
        NewsSentiment.VERY_POSITIVE: [
            "{symbol} Reports Blowout Q{quarter} Earnings, Beats Estimates by {beat}%",
            "{symbol} Crushes Earnings Expectations with {beat}% Revenue Growth",
            "Breaking: {symbol} Posts Record Quarterly Profit, Raises Full-Year Guidance",
        ],
        NewsSentiment.POSITIVE: [
            "{symbol} Beats Q{quarter} Earnings Estimates by ${eps}",
            "{symbol} Reports Solid Quarter, Revenue Up {growth}% YoY",
            "{symbol} Exceeds Analyst Expectations in Q{quarter} Results",
        ],
        NewsSentiment.NEUTRAL: [
            "{symbol} Reports Q{quarter} Earnings In Line with Estimates",
            "{symbol} Posts Mixed Q{quarter} Results, Maintains Guidance",
            "{symbol} Q{quarter} Revenue Meets Expectations at ${revenue}B",
        ],
        NewsSentiment.NEGATIVE: [
            "{symbol} Misses Q{quarter} Revenue Estimates by {miss}%",
            "{symbol} Reports Disappointing Earnings, Lowers Guidance",
            "{symbol} Q{quarter} Profit Falls Short of Analyst Expectations",
        ],
        NewsSentiment.VERY_NEGATIVE: [
            "{symbol} Posts Shocking Q{quarter} Loss, Slashes Dividend",
            "Breaking: {symbol} Revenue Plunges {drop}%, CEO to Step Down",
            "{symbol} Warns of Significant Headwinds After Dismal Q{quarter}",
        ],
    }
    
    REGULATORY_TEMPLATES = {
        NewsSentiment.VERY_POSITIVE: [
            "{symbol} Receives FDA Approval for Breakthrough Drug",
            "EU Approves {symbol}'s Major Acquisition, Clearing Final Hurdle",
            "{symbol} Wins Landmark Patent Case Worth ${value}B",
        ],
        NewsSentiment.NEGATIVE: [
            "{symbol} Faces SEC Investigation Over Accounting Practices",
            "DOJ Opens Antitrust Probe into {symbol}'s Market Practices",
            "{symbol} Hit with ${fine}M Fine for Regulatory Violations",
        ],
    }
    
    ANALYST_TEMPLATES = {
        NewsSentiment.VERY_POSITIVE: [
            "Goldman Sachs Upgrades {symbol} to Buy, Sees {upside}% Upside",
            "{symbol} Added to {bank}'s Conviction Buy List",
            "Analysts Raise {symbol} Price Target to ${target} After Strong Data",
        ],
        NewsSentiment.NEGATIVE: [
            "Morgan Stanley Downgrades {symbol} to Sell on Growth Concerns",
            "{symbol} Cut to Hold at {bank}, Cites Competitive Pressures",
            "Analyst Slashes {symbol} Price Target by {cut}% on Weak Outlook",
        ],
    }


class MockNewsFeedGenerator:
    """Generate realistic mock news events."""
    
    def __init__(self, config: MockNewsConfig):
        self.config = config
        self._running = False
        self._news_task = None
        self._callbacks: List[Callable] = []
        
        # Track recent news to avoid duplicates
        self._recent_news: List[NewsEvent] = []
        self._total_events = 0
        
        # Scheduled events (earnings calendar, etc.)
        self._scheduled_events: List[Dict] = self._generate_scheduled_events()
        
    def _generate_scheduled_events(self) -> List[Dict]:
        """Generate scheduled events like earnings."""
        events = []
        
        if self.config.enable_scheduled_events:
            # Generate earnings calendar
            for symbol in self.config.symbols[:10]:  # Top 10 symbols
                # Random earnings date in next 30 days
                days_ahead = random.randint(1, 30)
                earnings_time = time.time() + (days_ahead * 86400)
                
                events.append({
                    "symbol": symbol,
                    "time": earnings_time,
                    "type": "earnings",
                    "pre_announced": True,
                })
        
        return sorted(events, key=lambda x: x["time"])
    
    async def start(self):
        """Start generating news events."""
        if self._running:
            return
        
        self._running = True
        self._news_task = asyncio.create_task(self._news_generation_loop())
    
    async def stop(self):
        """Stop generating news events."""
        self._running = False
        
        if self._news_task:
            self._news_task.cancel()
            try:
                await self._news_task
            except asyncio.CancelledError:
                pass
    
    def add_callback(self, callback: Callable[[NewsEvent], None]):
        """Add callback for news events."""
        self._callbacks.append(callback)
    
    async def _news_generation_loop(self):
        """Main news generation loop."""
        interval = 3600.0 / self.config.events_per_hour  # Seconds between events
        
        while self._running:
            try:
                # Check for scheduled events
                current_time = time.time()
                while self._scheduled_events and self._scheduled_events[0]["time"] <= current_time:
                    event_data = self._scheduled_events.pop(0)
                    news = self._generate_scheduled_news(event_data)
                    await self._publish_news(news)
                
                # Generate random news
                if random.random() < 0.7:  # 70% chance of news in each interval
                    news = self._generate_random_news()
                    await self._publish_news(news)
                    
                    # Generate correlated news
                    if self.config.enable_correlated_events and news.impact_score > 0.7:
                        await self._generate_correlated_news(news)
                
                # Sleep with some randomness
                sleep_time = interval * random.uniform(0.5, 1.5)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"News generation error: {e}")
                await asyncio.sleep(10)
    
    def _generate_random_news(self) -> NewsEvent:
        """Generate a random news event."""
        symbol = random.choice(self.config.symbols)
        category = random.choice(list(NewsCategory))
        
        # Determine sentiment with bias
        sentiment_value = random.gauss(self.config.sentiment_bias, 1.0)
        sentiment_value = max(-2, min(2, sentiment_value))  # Clamp to valid range
        
        # Map to sentiment enum
        if sentiment_value < -1.5:
            sentiment = NewsSentiment.VERY_NEGATIVE
        elif sentiment_value < -0.5:
            sentiment = NewsSentiment.NEGATIVE
        elif sentiment_value < 0.5:
            sentiment = NewsSentiment.NEUTRAL
        elif sentiment_value < 1.5:
            sentiment = NewsSentiment.POSITIVE
        else:
            sentiment = NewsSentiment.VERY_POSITIVE
        
        # Determine impact
        is_high_impact = random.random() < self.config.high_impact_probability
        if is_high_impact:
            impact_score = random.uniform(0.7, 1.0)
        else:
            impact_score = random.uniform(0.1, 0.6)
        
        # Adjust impact based on sentiment extremity
        if sentiment in [NewsSentiment.VERY_NEGATIVE, NewsSentiment.VERY_POSITIVE]:
            impact_score = min(1.0, impact_score * 1.3)
        
        # Generate content
        headline, summary = self._generate_content(symbol, category, sentiment)
        
        # Create news event
        return NewsEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            symbol=symbol,
            headline=headline,
            summary=summary,
            category=category,
            sentiment=sentiment,
            impact_score=impact_score,
            source=random.choice(["Reuters", "Bloomberg", "CNBC", "WSJ", "FT"]),
            tags=self._generate_tags(category, sentiment),
            metadata={
                "is_breaking": is_high_impact,
                "read_time": random.randint(30, 180),  # Seconds to read
            }
        )
    
    def _generate_scheduled_news(self, event_data: Dict) -> NewsEvent:
        """Generate news for a scheduled event."""
        symbol = event_data["symbol"]
        
        # Earnings events tend to have higher impact
        impact_score = random.uniform(0.6, 1.0)
        
        # Random sentiment for earnings
        sentiment = random.choice(list(NewsSentiment))
        
        headline, summary = self._generate_content(
            symbol, 
            NewsCategory.EARNINGS, 
            sentiment
        )
        
        return NewsEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            symbol=symbol,
            headline=headline,
            summary=summary,
            category=NewsCategory.EARNINGS,
            sentiment=sentiment,
            impact_score=impact_score,
            source="Company Statement",
            tags=["earnings", "scheduled", "quarterly-results"],
            metadata={
                "is_scheduled": True,
                "pre_announced": event_data.get("pre_announced", False),
                "fiscal_quarter": f"Q{(datetime.now().month - 1) // 3 + 1}",
            }
        )
    
    def _generate_content(self, symbol: str, category: NewsCategory, 
                         sentiment: NewsSentiment) -> tuple[str, str]:
        """Generate headline and summary for news event."""
        # Get templates
        templates = getattr(NewsTemplates, f"{category.value.upper()}_TEMPLATES", None)
        
        if not templates or sentiment not in templates:
            # Fallback generic template
            headline = f"{symbol} {category.value.replace('_', ' ').title()} News"
        else:
            template = random.choice(templates[sentiment])
            
            # Fill in template variables
            headline = template.format(
                symbol=symbol,
                quarter=random.randint(1, 4),
                beat=random.randint(5, 30),
                miss=random.randint(5, 20),
                eps=f"{random.uniform(0.1, 2.0):.2f}",
                revenue=f"{random.uniform(1, 50):.1f}",
                growth=random.randint(5, 50),
                drop=random.randint(10, 40),
                value=random.randint(1, 10),
                fine=random.randint(10, 500),
                bank=random.choice(["Goldman Sachs", "Morgan Stanley", "JPMorgan", "Barclays"]),
                target=random.randint(50, 500),
                upside=random.randint(10, 50),
                cut=random.randint(10, 30),
            )
        
        # Generate summary
        summary = self._generate_summary(symbol, category, sentiment, headline)
        
        return headline, summary
    
    def _generate_summary(self, symbol: str, category: NewsCategory, 
                         sentiment: NewsSentiment, headline: str) -> str:
        """Generate a news summary."""
        summaries = {
            NewsCategory.EARNINGS: f"{symbol} reported quarterly results that {self._sentiment_to_performance(sentiment)} analyst expectations. The company's performance reflects {self._sentiment_to_outlook(sentiment)} market conditions.",
            NewsCategory.REGULATORY: f"Regulatory developments affecting {symbol} indicate {self._sentiment_to_regulatory(sentiment)} for the company's operations.",
            NewsCategory.ANALYST_RATING: f"Analyst actions on {symbol} reflect {self._sentiment_to_view(sentiment)} on the stock's near-term prospects.",
        }
        
        return summaries.get(category, f"Market news affecting {symbol} indicates {self._sentiment_to_impact(sentiment)} for investors.")
    
    def _sentiment_to_performance(self, sentiment: NewsSentiment) -> str:
        """Convert sentiment to performance description."""
        return {
            NewsSentiment.VERY_POSITIVE: "significantly exceeded",
            NewsSentiment.POSITIVE: "beat",
            NewsSentiment.NEUTRAL: "met",
            NewsSentiment.NEGATIVE: "missed",
            NewsSentiment.VERY_NEGATIVE: "badly missed",
        }[sentiment]
    
    def _sentiment_to_outlook(self, sentiment: NewsSentiment) -> str:
        """Convert sentiment to outlook description."""
        return {
            NewsSentiment.VERY_POSITIVE: "exceptionally strong",
            NewsSentiment.POSITIVE: "favorable",
            NewsSentiment.NEUTRAL: "mixed",
            NewsSentiment.NEGATIVE: "challenging",
            NewsSentiment.VERY_NEGATIVE: "severely deteriorating",
        }[sentiment]
    
    def _sentiment_to_regulatory(self, sentiment: NewsSentiment) -> str:
        """Convert sentiment to regulatory description."""
        return {
            NewsSentiment.VERY_POSITIVE: "major positive developments",
            NewsSentiment.POSITIVE: "favorable outcomes",
            NewsSentiment.NEUTRAL: "ongoing proceedings",
            NewsSentiment.NEGATIVE: "increased scrutiny",
            NewsSentiment.VERY_NEGATIVE: "serious enforcement actions",
        }[sentiment]
    
    def _sentiment_to_view(self, sentiment: NewsSentiment) -> str:
        """Convert sentiment to analyst view."""
        return {
            NewsSentiment.VERY_POSITIVE: "very bullish sentiment",
            NewsSentiment.POSITIVE: "positive outlook",
            NewsSentiment.NEUTRAL: "mixed views",
            NewsSentiment.NEGATIVE: "growing concerns",
            NewsSentiment.VERY_NEGATIVE: "severe downgrades",
        }[sentiment]
    
    def _sentiment_to_impact(self, sentiment: NewsSentiment) -> str:
        """Convert sentiment to impact description."""
        return {
            NewsSentiment.VERY_POSITIVE: "significant positive implications",
            NewsSentiment.POSITIVE: "favorable developments",
            NewsSentiment.NEUTRAL: "limited immediate impact",
            NewsSentiment.NEGATIVE: "negative pressure",
            NewsSentiment.VERY_NEGATIVE: "severe adverse effects",
        }[sentiment]
    
    def _generate_tags(self, category: NewsCategory, sentiment: NewsSentiment) -> List[str]:
        """Generate relevant tags for the news event."""
        tags = [category.value]
        
        # Add sentiment tags
        if sentiment.value <= -1:
            tags.extend(["bearish", "negative"])
        elif sentiment.value >= 1:
            tags.extend(["bullish", "positive"])
        
        # Add category-specific tags
        category_tags = {
            NewsCategory.EARNINGS: ["quarterly-results", "financial-performance"],
            NewsCategory.MERGER_ACQUISITION: ["m&a", "corporate-action"],
            NewsCategory.REGULATORY: ["compliance", "legal"],
            NewsCategory.ANALYST_RATING: ["analyst-action", "price-target"],
        }
        
        tags.extend(category_tags.get(category, []))
        
        return tags
    
    async def _generate_correlated_news(self, original: NewsEvent):
        """Generate correlated news events (sector impact, competitor news, etc.)."""
        if not self.config.enable_correlated_events:
            return
        
        # Generate 1-3 correlated events
        num_correlated = random.randint(1, 3)
        
        for _ in range(num_correlated):
            # Pick a related symbol (in real implementation, would use sector mapping)
            related_symbols = [s for s in self.config.symbols if s != original.symbol]
            if not related_symbols:
                continue
            
            symbol = random.choice(related_symbols)
            
            # Correlated sentiment (usually similar but not always)
            if random.random() < 0.7:  # 70% same direction
                sentiment = original.sentiment
            else:
                # Opposite or neutral
                if original.sentiment.value > 0:
                    sentiment = random.choice([NewsSentiment.NEGATIVE, NewsSentiment.NEUTRAL])
                else:
                    sentiment = random.choice([NewsSentiment.POSITIVE, NewsSentiment.NEUTRAL])
            
            # Lower impact for correlated news
            impact_score = original.impact_score * random.uniform(0.3, 0.7)
            
            headline = f"{symbol} Moves on {original.symbol} {original.category.value.title()} News"
            summary = f"Shares of {symbol} are reacting to developments at {original.symbol}. " + \
                     f"Sector correlation and market sentiment are driving {self._sentiment_to_impact(sentiment)}."
            
            correlated_news = NewsEvent(
                id=str(uuid.uuid4()),
                timestamp=time.time() + random.uniform(30, 300),  # Delayed reaction
                symbol=symbol,
                headline=headline,
                summary=summary,
                category=NewsCategory.MARKET_COMMENTARY,
                sentiment=sentiment,
                impact_score=impact_score,
                source=original.source,
                tags=["correlated", "sector-news"] + original.tags,
                metadata={
                    "correlated_with": original.id,
                    "original_symbol": original.symbol,
                }
            )
            
            # Schedule the correlated news
            await asyncio.sleep(random.uniform(0.5, 5.0))
            await self._publish_news(correlated_news)
    
    async def _publish_news(self, news: NewsEvent):
        """Publish news event to callbacks."""
        self._total_events += 1
        self._recent_news.append(news)
        
        # Keep only last 100 news items
        if len(self._recent_news) > 100:
            self._recent_news = self._recent_news[-100:]
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                await callback(news)
            except Exception as e:
                print(f"News callback error: {e}")
    
    def inject_breaking_news(self, symbol: str, category: NewsCategory, 
                           sentiment: NewsSentiment, impact: float = 0.9):
        """Manually inject breaking news for testing."""
        headline, summary = self._generate_content(symbol, category, sentiment)
        
        news = NewsEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            symbol=symbol,
            headline=f"BREAKING: {headline}",
            summary=summary,
            category=category,
            sentiment=sentiment,
            impact_score=impact,
            source="Breaking News Desk",
            tags=["breaking", "high-impact"] + self._generate_tags(category, sentiment),
            metadata={
                "is_breaking": True,
                "manually_injected": True,
            }
        )
        
        # Publish immediately
        asyncio.create_task(self._publish_news(news))
    
    def get_recent_news(self, symbol: Optional[str] = None, 
                       limit: int = 10) -> List[NewsEvent]:
        """Get recent news events."""
        news = self._recent_news.copy()
        
        if symbol:
            news = [n for n in news if n.symbol == symbol]
        
        # Sort by timestamp descending
        news.sort(key=lambda x: x.timestamp, reverse=True)
        
        return news[:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get generator metrics."""
        sentiment_distribution = {}
        category_distribution = {}
        
        for news in self._recent_news:
            # Sentiment
            sent_name = news.sentiment.name
            sentiment_distribution[sent_name] = sentiment_distribution.get(sent_name, 0) + 1
            
            # Category
            cat_name = news.category.value
            category_distribution[cat_name] = category_distribution.get(cat_name, 0) + 1
        
        return {
            "total_events": self._total_events,
            "recent_events": len(self._recent_news),
            "events_per_hour": self.config.events_per_hour,
            "sentiment_distribution": sentiment_distribution,
            "category_distribution": category_distribution,
            "scheduled_events_remaining": len(self._scheduled_events),
        }