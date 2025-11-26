"""
News-related factories for testing.

This module provides factory classes for generating test news data
including articles, sentiment analysis, and news sources.
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import factory
from factory import Factory, LazyAttribute, LazyFunction, SubFactory, Trait
from faker import Faker

fake = Faker()


class NewsSourceFactory(Factory):
    """Factory for creating news sources."""
    
    class Meta:
        model = dict
    
    source_id = factory.LazyFunction(lambda: str(uuid4()))
    name = factory.LazyFunction(lambda: random.choice([
        "Reuters", "Bloomberg", "CoinDesk", "CoinTelegraph", "The Block",
        "WSJ", "Financial Times", "CNBC", "Yahoo Finance", "MarketWatch"
    ]))
    
    @factory.lazy_attribute
    def url(self):
        """Generate URL based on source name."""
        domains = {
            "Reuters": "reuters.com",
            "Bloomberg": "bloomberg.com",
            "CoinDesk": "coindesk.com",
            "CoinTelegraph": "cointelegraph.com",
            "The Block": "theblockcrypto.com",
            "WSJ": "wsj.com",
            "Financial Times": "ft.com",
            "CNBC": "cnbc.com",
            "Yahoo Finance": "finance.yahoo.com",
            "MarketWatch": "marketwatch.com"
        }
        return f"https://{domains.get(self.name, 'example.com')}"
    
    category = factory.LazyFunction(lambda: random.choice([
        "crypto", "stocks", "forex", "commodities", "economy", "general"
    ]))
    
    reliability_score = factory.LazyFunction(lambda: round(random.uniform(0.7, 1.0), 2))
    bias_rating = factory.LazyFunction(lambda: random.choice(["left", "center", "right", "neutral"]))
    
    language = "en"
    country = factory.LazyFunction(lambda: random.choice(["US", "UK", "EU", "Asia", "Global"]))
    
    is_premium = factory.LazyFunction(lambda: random.choice([True, False]))
    requires_auth = factory.LazyAttribute(lambda obj: obj.is_premium)
    
    metadata = factory.LazyFunction(lambda: {
        "update_frequency": random.choice(["realtime", "hourly", "daily"]),
        "specialization": random.choice(["breaking_news", "analysis", "opinion", "data"]),
        "established_year": random.randint(1980, 2020)
    })


class NewsArticleFactory(Factory):
    """Factory for creating news articles."""
    
    class Meta:
        model = dict
    
    article_id = factory.LazyFunction(lambda: str(uuid4()))
    source = SubFactory(NewsSourceFactory)
    
    @factory.lazy_attribute
    def title(self):
        """Generate realistic news title."""
        templates = [
            "{company} Reports {change}% {direction} in Q{quarter} Earnings",
            "Breaking: {crypto} Surges to New {timeframe} High of ${price}",
            "{regulator} Announces New {asset_class} Trading Regulations",
            "Market Analysis: Why {asset} Could {action} {percent}% This {timeframe}",
            "{company} CEO {person} {action} on {topic} Outlook",
            "Technical Analysis: {asset} Forms {pattern} Pattern at ${price}",
            "{country} Central Bank {action} Interest Rates by {percent}%",
            "Institutional Investors {action} ${amount}M in {asset}",
            "{asset} {direction} as {catalyst} {impact} Market Sentiment",
            "Exclusive: {company} Plans {action} for {asset_class} Trading"
        ]
        
        template = random.choice(templates)
        return template.format(
            company=fake.company(),
            change=round(random.uniform(-20, 30), 1),
            direction=random.choice(["Gain", "Loss", "Jump", "Drop"]),
            quarter=random.randint(1, 4),
            crypto=random.choice(["Bitcoin", "Ethereum", "XRP", "Cardano"]),
            timeframe=random.choice(["Daily", "Weekly", "Monthly", "All-Time"]),
            price=round(random.uniform(1000, 60000), 0),
            regulator=random.choice(["SEC", "CFTC", "Fed", "ECB", "BoE"]),
            asset_class=random.choice(["Crypto", "Stock", "Forex", "Commodity"]),
            asset=random.choice(["BTC", "ETH", "Gold", "S&P 500", "EUR/USD"]),
            action=random.choice(["Rise", "Fall", "Consolidate", "Break Out"]),
            percent=round(random.uniform(5, 25), 1),
            person=fake.name(),
            topic=random.choice(["Market", "Economic", "Regulatory", "Technology"]),
            pattern=random.choice(["Bull Flag", "Head and Shoulders", "Triangle", "Wedge"]),
            country=random.choice(["US", "European", "Chinese", "Japanese"]),
            amount=round(random.uniform(10, 500), 0),
            catalyst=random.choice(["Fed Decision", "Earnings Report", "Economic Data"]),
            impact=random.choice(["Boosts", "Dampens", "Shifts", "Stabilizes"])
        )
    
    @factory.lazy_attribute
    def summary(self):
        """Generate article summary."""
        return fake.paragraph(nb_sentences=3)
    
    @factory.lazy_attribute
    def content(self):
        """Generate full article content."""
        paragraphs = []
        for _ in range(random.randint(5, 10)):
            paragraphs.append(fake.paragraph(nb_sentences=random.randint(3, 8)))
        return "\n\n".join(paragraphs)
    
    @factory.lazy_attribute
    def url(self):
        """Generate article URL."""
        slug = "-".join(self.title.lower().split()[:6])
        return f"{self.source['url']}/articles/{slug}-{fake.random_number(digits=6)}"
    
    published_at = factory.LazyFunction(
        lambda: datetime.now(timezone.utc) - timedelta(hours=random.randint(0, 72))
    )
    
    updated_at = factory.LazyAttribute(
        lambda obj: obj.published_at + timedelta(minutes=random.randint(0, 120))
    )
    
    # Article metadata
    author = factory.LazyFunction(fake.name)
    
    tags = factory.LazyFunction(lambda: fake.words(nb=random.randint(3, 8)))
    
    @factory.lazy_attribute
    def categories(self):
        """Generate article categories."""
        all_categories = ["crypto", "stocks", "forex", "commodities", "economy", "analysis", "breaking"]
        return random.sample(all_categories, k=random.randint(1, 3))
    
    @factory.lazy_attribute
    def mentioned_assets(self):
        """Generate list of mentioned assets."""
        assets = ["BTC", "ETH", "XRP", "ADA", "AAPL", "MSFT", "GOOGL", "TSLA", "Gold", "Oil"]
        return random.sample(assets, k=random.randint(1, 4))
    
    @factory.lazy_attribute
    def entities(self):
        """Generate named entities found in article."""
        return [
            {
                "text": fake.company(),
                "type": "ORG",
                "relevance": round(random.uniform(0.5, 1.0), 2)
            },
            {
                "text": fake.name(),
                "type": "PERSON",
                "relevance": round(random.uniform(0.3, 0.8), 2)
            },
            {
                "text": random.choice(["United States", "Europe", "China", "Japan"]),
                "type": "LOC",
                "relevance": round(random.uniform(0.4, 0.9), 2)
            }
        ]
    
    relevance_score = factory.LazyFunction(lambda: round(random.uniform(0.5, 1.0), 2))
    credibility_score = factory.LazyFunction(lambda: round(random.uniform(0.6, 1.0), 2))
    
    # Engagement metrics
    views = factory.LazyFunction(lambda: random.randint(100, 100000))
    shares = factory.LazyAttribute(lambda obj: random.randint(0, obj.views // 100))
    comments = factory.LazyAttribute(lambda obj: random.randint(0, obj.views // 200))
    
    class Params:
        """Trait parameters for different article types."""
        breaking_news = Trait(
            categories=["breaking"],
            published_at=factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(minutes=random.randint(1, 30))),
            relevance_score=factory.LazyFunction(lambda: round(random.uniform(0.9, 1.0), 2))
        )
        analysis = Trait(
            categories=["analysis"],
            content=factory.LazyFunction(lambda: "\n\n".join([fake.paragraph(nb_sentences=random.randint(5, 10)) for _ in range(10)])),
            credibility_score=factory.LazyFunction(lambda: round(random.uniform(0.8, 1.0), 2))
        )
        fud = Trait(
            title=factory.LazyFunction(lambda: random.choice([
                "URGENT: Crypto Market Faces Imminent Crash, Experts Warn",
                "Breaking: Major Exchange Hack Could Trigger Market Collapse",
                "Regulatory Crackdown: Is This the End for Cryptocurrency?"
            ])),
            relevance_score=factory.LazyFunction(lambda: round(random.uniform(0.3, 0.6), 2))
        )


class SentimentFactory(Factory):
    """Factory for creating sentiment analysis results."""
    
    class Meta:
        model = dict
    
    article_id = factory.LazyFunction(lambda: str(uuid4()))
    
    # Overall sentiment
    sentiment = factory.LazyFunction(lambda: random.choice(["bullish", "bearish", "neutral"]))
    confidence = factory.LazyFunction(lambda: round(random.uniform(0.6, 0.95), 2))
    
    # Detailed scores
    @factory.lazy_attribute
    def scores(self):
        """Generate sentiment scores that sum to 1."""
        if self.sentiment == "bullish":
            positive = random.uniform(0.6, 0.9)
            remaining = 1 - positive
            negative = random.uniform(0, remaining * 0.7)
            neutral = 1 - positive - negative
        elif self.sentiment == "bearish":
            negative = random.uniform(0.6, 0.9)
            remaining = 1 - negative
            positive = random.uniform(0, remaining * 0.7)
            neutral = 1 - negative - positive
        else:  # neutral
            neutral = random.uniform(0.5, 0.7)
            remaining = 1 - neutral
            positive = random.uniform(remaining * 0.4, remaining * 0.6)
            negative = 1 - neutral - positive
        
        return {
            "positive": round(positive, 3),
            "negative": round(negative, 3),
            "neutral": round(neutral, 3)
        }
    
    # Market impact assessment
    market_impact = factory.LazyFunction(lambda: random.choice(["high", "medium", "low", "negligible"]))
    
    @factory.lazy_attribute
    def impact_score(self):
        """Generate numeric impact score based on impact level."""
        impact_map = {
            "high": random.uniform(0.8, 1.0),
            "medium": random.uniform(0.5, 0.8),
            "low": random.uniform(0.2, 0.5),
            "negligible": random.uniform(0, 0.2)
        }
        return round(impact_map.get(self.market_impact, 0.5), 2)
    
    # Asset-specific sentiment
    @factory.lazy_attribute
    def asset_sentiments(self):
        """Generate sentiment for specific assets."""
        assets = ["BTC", "ETH", "XRP", "stocks", "forex", "commodities"]
        sentiments = {}
        
        for _ in range(random.randint(1, 4)):
            asset = random.choice(assets)
            sentiments[asset] = {
                "sentiment": random.choice(["bullish", "bearish", "neutral"]),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "mentions": random.randint(1, 10)
            }
        
        return sentiments
    
    # Key phrases and topics
    @factory.lazy_attribute
    def key_phrases(self):
        """Extract key phrases that indicate sentiment."""
        bullish_phrases = [
            "strong growth", "positive outlook", "breaking resistance",
            "institutional adoption", "bullish momentum", "recovery signs"
        ]
        bearish_phrases = [
            "downward pressure", "support broken", "regulatory concerns",
            "selling pressure", "bearish sentiment", "market correction"
        ]
        neutral_phrases = [
            "consolidation phase", "sideways movement", "waiting for direction",
            "mixed signals", "market uncertainty", "range-bound trading"
        ]
        
        if self.sentiment == "bullish":
            return random.sample(bullish_phrases, k=random.randint(2, 4))
        elif self.sentiment == "bearish":
            return random.sample(bearish_phrases, k=random.randint(2, 4))
        else:
            return random.sample(neutral_phrases, k=random.randint(2, 4))
    
    topics = factory.LazyFunction(lambda: random.sample([
        "regulation", "adoption", "technology", "market_analysis",
        "institutional", "retail", "defi", "nft", "economics", "geopolitics"
    ], k=random.randint(2, 5)))
    
    # Time-based sentiment
    @factory.lazy_attribute
    def time_horizon(self):
        """Sentiment time horizon."""
        return random.choice(["short_term", "medium_term", "long_term"])
    
    analyzed_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    
    # Model metadata
    model_name = factory.LazyFunction(lambda: random.choice([
        "FinBERT", "VADER", "TextBlob", "Custom_LSTM", "GPT_Sentiment"
    ]))
    model_version = factory.LazyFunction(lambda: f"v{random.randint(1, 3)}.{random.randint(0, 9)}")
    
    class Params:
        """Trait parameters for different sentiment scenarios."""
        strong_bullish = Trait(
            sentiment="bullish",
            confidence=factory.LazyFunction(lambda: round(random.uniform(0.85, 0.95), 2)),
            market_impact="high",
            scores={
                "positive": 0.85,
                "negative": 0.05,
                "neutral": 0.10
            }
        )
        strong_bearish = Trait(
            sentiment="bearish",
            confidence=factory.LazyFunction(lambda: round(random.uniform(0.85, 0.95), 2)),
            market_impact="high",
            scores={
                "positive": 0.05,
                "negative": 0.85,
                "neutral": 0.10
            }
        )
        uncertain = Trait(
            sentiment="neutral",
            confidence=factory.LazyFunction(lambda: round(random.uniform(0.4, 0.6), 2)),
            market_impact="low"
        )


def create_news_feed(
    count: int = 20,
    hours_back: int = 24,
    asset_focus: Optional[str] = None,
    sentiment_bias: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Create a realistic news feed with articles and sentiment.
    
    Args:
        count: Number of articles to generate
        hours_back: How many hours back to generate articles
        asset_focus: Specific asset to focus on (e.g., "BTC", "ETH")
        sentiment_bias: Overall sentiment bias ("bullish", "bearish", "neutral")
        
    Returns:
        List of news articles with sentiment analysis
    """
    news_feed = []
    current_time = datetime.now(timezone.utc)
    
    for i in range(count):
        # Generate article timing
        minutes_back = random.uniform(0, hours_back * 60)
        article_time = current_time - timedelta(minutes=minutes_back)
        
        # Create article
        article = NewsArticleFactory(published_at=article_time)
        
        # Add asset focus if specified
        if asset_focus:
            article["mentioned_assets"] = [asset_focus] + article["mentioned_assets"][:2]
            article["title"] = article["title"].replace(
                random.choice(["Bitcoin", "Ethereum", "asset", "crypto"]),
                asset_focus
            )
        
        # Create sentiment analysis
        sentiment = SentimentFactory(article_id=article["article_id"])
        
        # Apply sentiment bias if specified
        if sentiment_bias:
            sentiment["sentiment"] = sentiment_bias
            if sentiment_bias == "bullish":
                sentiment = SentimentFactory(
                    article_id=article["article_id"],
                    strong_bullish=True
                )
            elif sentiment_bias == "bearish":
                sentiment = SentimentFactory(
                    article_id=article["article_id"],
                    strong_bearish=True
                )
        
        # Combine article and sentiment
        news_item = {
            **article,
            "sentiment_analysis": sentiment
        }
        
        news_feed.append(news_item)
    
    # Sort by publication time (newest first)
    news_feed.sort(key=lambda x: x["published_at"], reverse=True)
    
    return news_feed


def create_market_event(
    event_type: str = "random",
    impact_level: str = "medium"
) -> Dict[str, Any]:
    """
    Create a market-moving news event.
    
    Args:
        event_type: Type of event ("earnings", "regulation", "hack", "adoption", "random")
        impact_level: Impact level ("high", "medium", "low")
        
    Returns:
        Dictionary representing a market event with full details
    """
    event_templates = {
        "earnings": {
            "title": "{company} Reports {beat_miss} Earnings, Stock {direction} {percent}%",
            "summary": "{company} reported Q{quarter} earnings of ${eps} per share, {beat_miss} analyst estimates.",
            "assets": ["stocks"],
            "sentiment": lambda: "bullish" if random.random() > 0.5 else "bearish"
        },
        "regulation": {
            "title": "{regulator} Announces {positive_negative} Crypto Regulation Framework",
            "summary": "The {regulator} unveiled new {asset_class} trading rules that could {impact} the market.",
            "assets": ["BTC", "ETH", "crypto"],
            "sentiment": lambda: "bearish" if random.random() > 0.3 else "bullish"
        },
        "hack": {
            "title": "Breaking: {exchange} Exchange Suffers ${amount}M Security Breach",
            "summary": "Major cryptocurrency exchange {exchange} reported unauthorized access resulting in ${amount}M loss.",
            "assets": ["BTC", "ETH", "crypto"],
            "sentiment": lambda: "bearish"
        },
        "adoption": {
            "title": "{company} Announces {adoption_type} for {asset}",
            "summary": "{company} will begin {adoption_action} starting {timeframe}.",
            "assets": ["BTC", "ETH"],
            "sentiment": lambda: "bullish"
        }
    }
    
    if event_type == "random":
        event_type = random.choice(list(event_templates.keys()))
    
    template = event_templates.get(event_type, event_templates["earnings"])
    
    # Generate event details
    event_data = {
        "company": fake.company(),
        "beat_miss": random.choice(["Beat", "Missed"]),
        "direction": random.choice(["Jumps", "Falls"]),
        "percent": round(random.uniform(5, 20), 1),
        "quarter": random.randint(1, 4),
        "eps": round(random.uniform(1, 5), 2),
        "regulator": random.choice(["SEC", "CFTC", "EU", "UK FCA"]),
        "positive_negative": random.choice(["Favorable", "Restrictive"]),
        "asset_class": random.choice(["Cryptocurrency", "Digital Asset", "DeFi"]),
        "impact": random.choice(["boost", "dampen", "reshape"]),
        "exchange": random.choice(["Binance", "Coinbase", "Kraken", "FTX"]),
        "amount": round(random.uniform(10, 500), 0),
        "adoption_type": random.choice(["Bitcoin Treasury", "Crypto Payments", "Blockchain Integration"]),
        "asset": random.choice(["Bitcoin", "Ethereum", "Cryptocurrency"]),
        "adoption_action": random.choice(["accepting crypto payments", "holding BTC on balance sheet", "launching blockchain services"]),
        "timeframe": random.choice(["next month", "Q1 2024", "immediately"])
    }
    
    # Create the event article
    article = NewsArticleFactory(
        title=template["title"].format(**event_data),
        summary=template["summary"].format(**event_data),
        mentioned_assets=template["assets"],
        breaking_news=True if impact_level == "high" else False
    )
    
    # Create high-impact sentiment
    sentiment = SentimentFactory(
        article_id=article["article_id"],
        sentiment=template["sentiment"](),
        market_impact=impact_level,
        confidence=round(random.uniform(0.8, 0.95), 2)
    )
    
    return {
        "event_type": event_type,
        "impact_level": impact_level,
        "article": article,
        "sentiment": sentiment,
        "expected_market_reaction": {
            "direction": sentiment["sentiment"],
            "magnitude": impact_level,
            "affected_assets": template["assets"],
            "time_to_impact": "immediate" if impact_level == "high" else "gradual"
        }
    }