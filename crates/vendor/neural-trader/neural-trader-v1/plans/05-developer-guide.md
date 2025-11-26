# AI News Trading Platform: Developer Onboarding Guide

## Welcome to the Team! 🚀

Welcome to the AI News Trading Platform development team! This guide will help you get up and running quickly, understand our development practices, and start contributing effectively. We follow Test-Driven Development (TDD) principles and maintain a zero-cost architecture using only free and open-source technologies.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Development Environment](#development-environment)
4. [TDD Workflow](#tdd-workflow)
5. [Architecture Deep Dive](#architecture-deep-dive)
6. [Testing Guidelines](#testing-guidelines)
7. [Code Standards](#code-standards)
8. [Common Tasks](#common-tasks)
9. [Troubleshooting](#troubleshooting)
10. [Resources & Support](#resources--support)

## Quick Start

### 15-Minute Setup
```bash
# 1. Clone the repository
git clone <repository-url>
cd ai-news-trader

# 2. Set up Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up pre-commit hooks
pre-commit install

# 5. Run initial tests
pytest tests/

# 6. Start the development server
python src/main.py --dev
```

### First Day Checklist
- [ ] Development environment set up
- [ ] All tests passing locally
- [ ] Read this guide completely
- [ ] Review the TDD Master Plan
- [ ] Join team communication channels
- [ ] Complete your first test-driven feature

## Project Overview

### What We're Building
The AI News Trading Platform is a comprehensive financial intelligence system that:
- **Monitors** 15+ free financial news sources in real-time
- **Analyzes** market impact using AI/ML models (FinBERT, spaCy)
- **Provides** conversational interface for market insights
- **Runs** entirely on free services and open-source software

### Key Technologies
```yaml
Backend:
  - Python 3.x
  - Flask (Web Framework)
  - SQLite (Database)
  - pandas/numpy (Data Processing)

AI/ML:
  - spaCy (NLP)
  - Hugging Face Transformers (FinBERT)
  - Ollama (Local LLM)

Frontend:
  - Vanilla JavaScript (ES6+)
  - HTML5/CSS3
  - WebSockets

Infrastructure:
  - Docker
  - GitHub Actions (CI/CD)
```

### Project Structure
```
ai-news-trader/
├── src/
│   ├── ingestion/        # News data collection
│   ├── analysis/         # Market impact analysis
│   ├── processing/       # Real-time processing
│   ├── interface/        # Web UI and API
│   └── utils/           # Shared utilities
├── tests/
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── e2e/            # End-to-end tests
├── docs/                # Documentation
├── scripts/             # Utility scripts
└── docker/             # Docker configurations
```

## Development Environment

### Required Tools
```bash
# Check your setup
python --version  # Should be 3.8+
docker --version  # Should be 20.10+
node --version    # Should be 14+ (for frontend)
```

### Environment Variables
Create a `.env` file in the project root:
```env
# Development Settings
FLASK_ENV=development
FLASK_DEBUG=1
LOG_LEVEL=DEBUG

# API Keys (all free tier)
OPENROUTER_API_KEY=your_free_key_here

# Database
DATABASE_URL=sqlite:///dev.db

# Testing
TEST_DATABASE_URL=sqlite:///test.db
PYTEST_COVERAGE_THRESHOLD=90
```

### VS Code Setup (Recommended)
```json
// .vscode/settings.json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.linting.mypyEnabled": true
}
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
```

## TDD Workflow

### The Red-Green-Refactor Cycle

#### 1. Red Phase - Write a Failing Test
```python
# tests/unit/test_news_parser.py
def test_extract_ticker_from_headline():
    """Parser should extract stock tickers from headlines"""
    parser = NewsParser()
    headline = "Apple (AAPL) Reports Record Q4 Earnings"
    
    result = parser.extract_tickers(headline)
    
    assert result == ['AAPL']  # This will fail initially
```

#### 2. Green Phase - Make the Test Pass
```python
# src/ingestion/parser.py
class NewsParser:
    def extract_tickers(self, text):
        """Extract stock tickers from text"""
        import re
        pattern = r'\b([A-Z]{1,5})\b'
        tickers = re.findall(pattern, text)
        return [t for t in tickers if self._is_valid_ticker(t)]
    
    def _is_valid_ticker(self, ticker):
        # Simple validation for now
        return len(ticker) >= 2 and ticker.isupper()
```

#### 3. Refactor Phase - Improve the Code
```python
# src/ingestion/parser.py
import re
from typing import List

class NewsParser:
    """Parser for extracting structured data from news content"""
    
    TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')
    COMMON_WORDS = {'CEO', 'FDA', 'IPO', 'SEC', 'API'}
    
    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract valid stock tickers from text.
        
        Args:
            text: Input text to parse
            
        Returns:
            List of validated ticker symbols
        """
        potential_tickers = self.TICKER_PATTERN.findall(text)
        return [
            ticker for ticker in potential_tickers 
            if self._is_valid_ticker(ticker)
        ]
    
    def _is_valid_ticker(self, ticker: str) -> bool:
        """Validate if string is likely a stock ticker"""
        return (
            ticker not in self.COMMON_WORDS and
            2 <= len(ticker) <= 5 and
            ticker.isupper()
        )
```

### TDD Best Practices

#### 1. Write Descriptive Test Names
```python
# Good
def test_sentiment_analyzer_returns_negative_score_for_bankruptcy_news():
    pass

# Bad
def test_sentiment():
    pass
```

#### 2. Follow AAA Pattern
```python
def test_impact_scorer_weights_multiple_factors():
    # Arrange
    article = create_test_article(
        keywords=['merger', 'acquisition'],
        sentiment=0.8,
        source_reliability=0.9
    )
    scorer = ImpactScorer()
    
    # Act
    score = scorer.calculate(article)
    
    # Assert
    assert 0.7 <= score <= 0.9
```

#### 3. Use Fixtures for Common Setup
```python
@pytest.fixture
def mock_news_feed():
    """Provide mock news feed data for testing"""
    return [
        {
            'headline': 'Test Article 1',
            'content': 'Test content...',
            'source': 'TestSource',
            'timestamp': datetime.now()
        }
    ]

def test_feed_processor(mock_news_feed):
    processor = FeedProcessor()
    results = processor.process(mock_news_feed)
    assert len(results) == 1
```

## Architecture Deep Dive

### Component Architecture

#### 1. News Ingestion Layer
```python
# src/ingestion/feed_manager.py
class FeedManager:
    """Manages multiple news feed sources"""
    
    def __init__(self):
        self.feeds = [
            YahooFinanceFeed(),
            ReutersFeed(),
            MarketWatchFeed(),
            # ... more feeds
        ]
        
    async def fetch_all(self):
        """Fetch from all feeds concurrently"""
        tasks = [feed.fetch() for feed in self.feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._merge_results(results)
```

#### 2. Analysis Pipeline
```python
# src/analysis/pipeline.py
class AnalysisPipeline:
    """Main analysis pipeline for news articles"""
    
    def __init__(self):
        self.sentiment_analyzer = FinBERTAnalyzer()
        self.entity_extractor = SpacyExtractor()
        self.impact_scorer = ImpactScorer()
        
    def analyze(self, article):
        """Run full analysis pipeline"""
        # Extract entities
        entities = self.entity_extractor.extract(article)
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze(article)
        
        # Calculate impact
        impact = self.impact_scorer.calculate(
            article, entities, sentiment
        )
        
        return AnalysisResult(
            entities=entities,
            sentiment=sentiment,
            impact=impact
        )
```

#### 3. Real-time Processing
```python
# src/processing/stream_processor.py
class StreamProcessor:
    """Processes news stream in real-time"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.workers = []
        
    async def start(self, num_workers=4):
        """Start processing workers"""
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f'worker-{i}'))
            self.workers.append(worker)
            
    async def _worker(self, name):
        """Worker coroutine"""
        while True:
            article = await self.queue.get()
            try:
                await self._process_article(article)
            except Exception as e:
                logger.error(f"{name} error: {e}")
            finally:
                self.queue.task_done()
```

### Data Flow Diagram
```
[RSS Feeds] ─┐
             ├─> [Feed Aggregator] ─> [Standardizer] ─> [Queue]
[Web Scraper]┘                                             │
                                                          ▼
[Market Data] ─> [Context Engine] ─┐                [Processor]
                                   │                      │
                                   └─> [Analyzer] <───────┘
                                           │
                                           ▼
                                    [Impact Scorer]
                                           │
                                           ▼
                                   [Alert Generator]
                                           │
                                           ▼
                                    [Web Interface]
```

## Testing Guidelines

### Test Organization
```
tests/
├── unit/              # Fast, isolated tests
│   ├── test_parser.py
│   ├── test_analyzer.py
│   └── test_scorer.py
├── integration/       # Component interaction tests
│   ├── test_feed_pipeline.py
│   └── test_analysis_flow.py
├── e2e/              # Full system tests
│   └── test_user_workflows.py
├── fixtures/         # Test data and mocks
└── conftest.py      # Shared pytest configuration
```

### Writing Effective Tests

#### Unit Tests
```python
# tests/unit/test_sentiment.py
class TestSentimentAnalyzer:
    """Test sentiment analysis functionality"""
    
    def test_positive_sentiment_detection(self):
        analyzer = SentimentAnalyzer()
        text = "Company reports record profits and growth"
        
        score = analyzer.analyze(text)
        
        assert score > 0.5  # Positive sentiment
        
    def test_negative_sentiment_detection(self):
        analyzer = SentimentAnalyzer()
        text = "Company files for bankruptcy protection"
        
        score = analyzer.analyze(text)
        
        assert score < -0.5  # Negative sentiment
        
    @pytest.mark.parametrize("text,expected_range", [
        ("neutral financial report", (-0.2, 0.2)),
        ("major acquisition announced", (0.3, 0.8)),
        ("severe losses reported", (-0.8, -0.3)),
    ])
    def test_sentiment_ranges(self, text, expected_range):
        analyzer = SentimentAnalyzer()
        score = analyzer.analyze(text)
        assert expected_range[0] <= score <= expected_range[1]
```

#### Integration Tests
```python
# tests/integration/test_news_pipeline.py
@pytest.mark.integration
class TestNewsPipeline:
    """Test the complete news processing pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        return NewsPipeline(test_mode=True)
        
    async def test_end_to_end_processing(self, pipeline):
        # Arrange
        test_article = create_test_article()
        
        # Act
        result = await pipeline.process(test_article)
        
        # Assert
        assert result.status == 'processed'
        assert result.sentiment is not None
        assert result.impact_score > 0
        assert result.entities  # Not empty
```

#### Performance Tests
```python
# tests/performance/test_throughput.py
@pytest.mark.performance
def test_processing_throughput():
    """Ensure system meets throughput requirements"""
    processor = NewsProcessor()
    articles = generate_test_articles(1000)
    
    start_time = time.time()
    processor.process_batch(articles)
    duration = time.time() - start_time
    
    articles_per_second = 1000 / duration
    assert articles_per_second >= 100  # Requirement: 100+ articles/sec
```

### Mocking External Services
```python
# tests/fixtures/mock_services.py
@pytest.fixture
def mock_yahoo_finance(monkeypatch):
    """Mock Yahoo Finance RSS feed"""
    def mock_fetch():
        return [{
            'title': 'Test Article',
            'link': 'http://example.com',
            'published': datetime.now(),
            'summary': 'Test summary'
        }]
    
    monkeypatch.setattr(
        'src.ingestion.feeds.YahooFinanceFeed.fetch',
        mock_fetch
    )
```

### Test Coverage Requirements
```yaml
# pytest.ini
[pytest]
minversion = 6.0
addopts = 
    --cov=src 
    --cov-report=html 
    --cov-report=term 
    --cov-fail-under=90
    -v
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

## Code Standards

### Python Style Guide
```python
# src/example_module.py
"""
Module docstring explaining purpose and usage.

This module handles the core business logic for news analysis.
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class NewsAnalyzer:
    """
    Analyzes news articles for market impact.
    
    This class provides methods to extract entities, analyze sentiment,
    and calculate potential market impact scores.
    
    Attributes:
        model: The ML model used for analysis
        config: Configuration dictionary
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        self.model = self._load_model()
        
    def analyze(self, article: Dict) -> Dict:
        """
        Analyze a single article.
        
        Args:
            article: Dictionary containing article data
            
        Returns:
            Dictionary with analysis results
            
        Raises:
            ValueError: If article format is invalid
        """
        self._validate_article(article)
        
        try:
            entities = self._extract_entities(article)
            sentiment = self._analyze_sentiment(article)
            impact = self._calculate_impact(entities, sentiment)
            
            return {
                'entities': entities,
                'sentiment': sentiment,
                'impact': impact,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
```

### Naming Conventions
```python
# Constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Classes: PascalCase
class NewsProcessor:
    pass

# Functions/Variables: snake_case
def process_article(article_data):
    processed_count = 0
    return processed_count

# Private methods: leading underscore
def _internal_helper():
    pass

# Test names: descriptive
def test_processor_handles_empty_articles():
    pass
```

### Documentation Standards
```python
def calculate_impact_score(
    article: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate the market impact score for an article.
    
    Uses a weighted combination of keyword presence, sentiment score,
    and source reliability to determine potential market impact.
    
    Args:
        article: Dictionary containing:
            - content (str): Article text
            - sentiment (float): Sentiment score (-1 to 1)
            - source (str): News source name
        weights: Optional weight overrides for scoring factors
        
    Returns:
        Float between 0 and 1 representing impact magnitude
        
    Raises:
        KeyError: If required article fields are missing
        ValueError: If sentiment score is out of range
        
    Example:
        >>> article = {
        ...     'content': 'Apple announces record profits',
        ...     'sentiment': 0.8,
        ...     'source': 'Reuters'
        ... }
        >>> score = calculate_impact_score(article)
        >>> print(f"Impact: {score:.2f}")
        Impact: 0.75
    """
    # Implementation here
    pass
```

## Common Tasks

### Adding a New News Source

#### 1. Create the feed class
```python
# src/ingestion/feeds/bloomberg_feed.py
from .base_feed import BaseFeed

class BloombergFeed(BaseFeed):
    """Bloomberg news RSS feed parser"""
    
    URL = "https://feeds.bloomberg.com/markets/news.rss"
    SOURCE_NAME = "Bloomberg"
    
    def parse_item(self, item):
        """Parse a single RSS item"""
        return {
            'headline': item.get('title'),
            'url': item.get('link'),
            'published': self.parse_date(item.get('pubDate')),
            'summary': item.get('description'),
            'source': self.SOURCE_NAME
        }
```

#### 2. Write tests first
```python
# tests/unit/test_bloomberg_feed.py
def test_bloomberg_feed_parsing():
    feed = BloombergFeed()
    mock_item = {
        'title': 'Test News',
        'link': 'http://example.com',
        'pubDate': 'Mon, 20 Jan 2024 10:00:00 GMT',
        'description': 'Test description'
    }
    
    result = feed.parse_item(mock_item)
    
    assert result['headline'] == 'Test News'
    assert result['source'] == 'Bloomberg'
```

#### 3. Register the feed
```python
# src/ingestion/feed_registry.py
AVAILABLE_FEEDS = [
    YahooFinanceFeed,
    ReutersFeed,
    BloombergFeed,  # Add new feed
    # ...
]
```

### Implementing a New Analysis Feature

#### 1. Design the interface
```python
# Write the test first
def test_volatility_predictor():
    predictor = VolatilityPredictor()
    article = create_high_impact_article()
    
    prediction = predictor.predict(article)
    
    assert prediction.volatility_score > 0.7
    assert prediction.confidence > 0.5
```

#### 2. Implement the feature
```python
# src/analysis/volatility.py
class VolatilityPredictor:
    """Predicts market volatility from news content"""
    
    def __init__(self):
        self.model = self._load_model()
        
    def predict(self, article):
        features = self._extract_features(article)
        score = self.model.predict(features)
        confidence = self._calculate_confidence(features)
        
        return VolatilityPrediction(
            volatility_score=score,
            confidence=confidence
        )
```

### Debugging Production Issues

#### 1. Enable debug logging
```python
# src/utils/debug.py
import logging

def enable_debug_mode():
    """Enable verbose logging for debugging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add performance profiling
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler
```

#### 2. Use debugging decorators
```python
# src/utils/decorators.py
def debug_trace(func):
    """Decorator to trace function calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {e}")
            raise
    return wrapper
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Tests Failing Locally
```bash
# Clear test cache
pytest --cache-clear

# Run with verbose output
pytest -vvs tests/unit/test_failing.py

# Check for missing dependencies
pip install -r requirements-dev.txt
```

#### 2. Import Errors
```python
# Add to conftest.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

#### 3. Database Issues
```bash
# Reset test database
rm test.db
python -m src.database.init_db --test

# Check migrations
alembic current
alembic upgrade head
```

#### 4. Async Test Issues
```python
# Use pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None

# Or use pytest-trio for more complex scenarios
```

### Performance Troubleshooting

#### Memory Profiling
```python
# src/utils/profiling.py
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function code here
    pass

# Run with: python -m memory_profiler src/module.py
```

#### CPU Profiling
```python
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code to profile
    result = expensive_operation()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

## Resources & Support

### Internal Resources
- **Team Wiki**: Internal documentation and decisions
- **Slack Channel**: #ai-news-trader-dev
- **Code Reviews**: All PRs require 2 approvals
- **Daily Standups**: 9:30 AM EST

### External Resources
- [Flask Documentation](https://flask.palletsprojects.com/)
- [pytest Documentation](https://docs.pytest.org/)
- [spaCy Guide](https://spacy.io/usage)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### Learning Path
1. **Week 1**: Environment setup, first PR
2. **Week 2**: Complete a full feature with TDD
3. **Week 3**: Contribute to architecture decisions
4. **Week 4**: Lead a feature implementation

### Getting Help
```python
# When stuck, use our help system
from src.utils.help import get_help

# In code
help_text = get_help('news_processing')

# In terminal
python -m src.help news_processing

# Or ask in Slack with context
```

### Code Review Checklist
Before submitting a PR:
- [ ] All tests pass locally
- [ ] Test coverage >= 90%
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance benchmarks met
- [ ] PR description is clear

## Final Tips

### Do's
- ✅ Write tests first (always!)
- ✅ Ask questions early and often
- ✅ Document your decisions
- ✅ Profile before optimizing
- ✅ Use type hints everywhere
- ✅ Keep functions small and focused

### Don'ts
- ❌ Skip writing tests
- ❌ Commit directly to main
- ❌ Ignore failing tests
- ❌ Use production data in tests
- ❌ Hardcode configuration values
- ❌ Suppress exceptions silently

### Your First Week Goals
1. Get all tests passing locally
2. Submit your first PR (even if small)
3. Review someone else's code
4. Add one new test case
5. Improve existing documentation

Welcome aboard! We're excited to have you on the team. Remember, good tests lead to good code, and good code leads to a great product. Happy coding! 🎉