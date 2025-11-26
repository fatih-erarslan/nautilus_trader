# AI News Trading Platform - Requirements Analysis

## Executive Summary

This document provides a comprehensive requirements analysis for the AI News Trading platform, comparing the stated objectives in the project README with the current implementation. The analysis reveals significant architectural divergence and identifies critical gaps that must be addressed to achieve the original vision.

## 1. Project Vision Analysis

### 1.1 Original Vision (From README.md)

The AI News Trading Agent was envisioned as:
- A **zero-cost** financial intelligence platform
- **Real-time news monitoring** from 15+ free sources
- **AI-powered impact analysis** using NLP and sentiment analysis
- **Conversational interface** for natural language queries
- **Local-first deployment** with complete privacy
- **Human-oriented** (not HFT) trading insights

### 1.2 Current Implementation

The existing codebase implements:
- A **symbolic mathematics** trading platform
- **Mathematical expression** analysis and transformation
- **Cryptocurrency** focus with narrative forecasting
- **OpenRouter LLM** integration (requires paid API)
- **Docker-based** distributed architecture
- **No news ingestion** capabilities

### 1.3 Critical Gap Analysis

**Major Deviations:**
1. **Core Purpose**: News analysis vs. Mathematical trading
2. **Cost Model**: Zero-cost requirement violated (OpenRouter API)
3. **Data Sources**: No news feeds implemented
4. **User Interface**: No conversational chat interface
5. **Market Focus**: Crypto-specific vs. general market

## 2. Functional Requirements Matrix

### 2.1 Phase 1: News Ingestion Engine

| Requirement | Priority | Status | Gap Analysis |
|------------|----------|---------|--------------|
| RSS Feed Aggregator | P0 | ❌ Not Implemented | Core feature missing |
| Web Scraping Module | P0 | ❌ Not Implemented | No BeautifulSoup integration |
| Data Standardization | P0 | ❌ Not Implemented | No news data schema |
| Deduplication Logic | P1 | ❌ Not Implemented | No duplicate detection |

**Required Components:**
```python
# News item schema
{
    "event_id": str,
    "headline": str,
    "content_excerpt": str,
    "source": str,
    "pub_timestamp": datetime,
    "discovery_timestamp": datetime,
    "extracted_tickers": List[str],
    "category": str,
    "quality_score": float,
    "reliability_score": float
}
```

### 2.2 Phase 2: Market Impact Analysis

| Requirement | Priority | Status | Gap Analysis |
|------------|----------|---------|--------------|
| Financial NER (spaCy) | P0 | ❌ Not Implemented | No entity recognition |
| FinBERT Integration | P0 | ❌ Not Implemented | No sentiment analysis |
| Keyword Impact Scoring | P0 | ❌ Not Implemented | No rule engine |
| Market Context (yfinance) | P1 | ❌ Not Implemented | No market data integration |

**Required Keywords Dictionary:**
- High-Impact: merger, acquisition, bankruptcy, FDA approval, earnings
- Medium-Impact: partnership, contract, expansion
- Macro-Impact: interest rate, FOMC, CPI report

### 2.3 Phase 3: Real-Time Processing

| Requirement | Priority | Status | Gap Analysis |
|------------|----------|---------|--------------|
| Async Processing Queue | P0 | ⚠️ Partial | Has async but not for news |
| SQLite Caching | P0 | ❌ Not Implemented | Uses SQLAlchemy but no cache |
| Event Prioritization | P0 | ❌ Not Implemented | No priority logic |
| Alert Generation | P1 | ❌ Not Implemented | No alert system |

### 2.4 Phase 4: Conversational Interface

| Requirement | Priority | Status | Gap Analysis |
|------------|----------|---------|--------------|
| Local LLM (Ollama) | P0 | ❌ Not Implemented | Uses paid OpenRouter |
| Intent Recognition | P0 | ❌ Not Implemented | No query parsing |
| Web Chat Interface | P0 | ❌ Not Implemented | No frontend |
| Conversation History | P1 | ❌ Not Implemented | No context management |

### 2.5 Phase 5: System Integration

| Requirement | Priority | Status | Gap Analysis |
|------------|----------|---------|--------------|
| Flask API Server | P0 | ❌ Not Implemented | No web framework |
| Configuration Management | P0 | ✅ Implemented | Has .env support |
| Docker Deployment | P0 | ✅ Implemented | Has Dockerfile |
| Logging System | P0 | ⚠️ Partial | Basic logging only |

## 3. Non-Functional Requirements

### 3.1 Performance Requirements

| Requirement | Target | Current | Status |
|------------|--------|---------|---------|
| News Processing Time | <5s per article | N/A | ❌ Not Measured |
| API Response Time | <2s | N/A | ❌ Not Implemented |
| Concurrent Users | 10+ | N/A | ❌ No Multi-user |
| News Update Frequency | Real-time (1-5 min) | N/A | ❌ No Updates |

### 3.2 Security Requirements

| Requirement | Priority | Status | Notes |
|------------|----------|---------|--------|
| No API Keys Required | P0 | ❌ Failed | Requires OpenRouter API |
| Local Data Storage | P0 | ✅ Met | SQLite capable |
| HTTPS for Scraping | P1 | ❌ Not Implemented | No scraping |
| Input Sanitization | P0 | ❌ Not Implemented | No user input handling |

### 3.3 Scalability Requirements

| Requirement | Target | Status |
|------------|--------|---------|
| News Sources | 15+ simultaneous | ❌ 0 sources |
| Historical Data | 30 days retention | ❌ No storage |
| Database Size | <1GB for 30 days | ❌ No implementation |
| Docker Memory | <2GB RAM | ❓ Unknown |

## 4. User Stories and Acceptance Criteria

### 4.1 Core User Stories

**US-001: Real-Time News Monitoring**
```
As a trader
I want to monitor multiple news sources in real-time
So that I can stay informed about market-moving events

Acceptance Criteria:
- System fetches news from 15+ sources
- Updates occur every 1-5 minutes
- Duplicates are automatically filtered
- News is categorized by relevance
```

**US-002: AI-Powered Analysis**
```
As a trader
I want AI to analyze news impact automatically
So that I can focus on high-impact events

Acceptance Criteria:
- Each article receives impact score (0-100)
- Sentiment analysis identifies bullish/bearish tone
- Relevant tickers are extracted
- Source reliability affects scoring
```

**US-003: Conversational Queries**
```
As a trader
I want to ask questions in natural language
So that I can get specific insights quickly

Acceptance Criteria:
- System understands queries like "What's new on TSLA?"
- Responses include summarized news and impact
- Context is maintained for follow-up questions
- Works without internet (local LLM)
```

## 5. Technical Constraints and Dependencies

### 5.1 Required Technology Stack

| Component | Required | Current | Action Needed |
|-----------|----------|---------|---------------|
| Python 3.x | ✅ | ✅ 3.12 | Align version |
| Flask | ✅ | ❌ | Implement |
| BeautifulSoup | ✅ | ❌ | Add to requirements |
| feedparser | ✅ | ❌ | Add to requirements |
| spaCy | ✅ | ❌ | Add to requirements |
| FinBERT | ✅ | ❌ | Add via Transformers |
| Ollama | ✅ | ❌ | Replace OpenRouter |
| yfinance | ✅ | ❌ | Add to requirements |

### 5.2 Architecture Alignment

**Required Architecture:**
```
ai-news-trader/
├── src/
│   ├── ingestion/          # News collection
│   ├── analysis/           # NLP and scoring
│   ├── processing/         # Queue and alerts
│   ├── interface/          # Chat and API
│   └── storage/            # SQLite management
├── web/                    # Frontend assets
├── config/                 # Configuration
└── tests/                  # Test suite
```

## 6. Implementation Priority Matrix

### 6.1 P0 - Critical Path (Weeks 1-2)

1. **News Ingestion Module**
   - RSS feed parser for Yahoo Finance, Reuters
   - Basic data standardization
   - SQLite schema creation

2. **Basic Analysis Engine**
   - Keyword-based impact scoring
   - Simple ticker extraction
   - Source reliability weights

3. **Flask API Server**
   - Basic endpoints for news retrieval
   - Simple JSON responses
   - Error handling

### 6.2 P1 - Core Features (Weeks 3-4)

1. **Enhanced Analysis**
   - spaCy NER integration
   - FinBERT sentiment analysis
   - Market context from yfinance

2. **Processing Pipeline**
   - Async queue implementation
   - Deduplication logic
   - Alert generation

3. **Web Interface**
   - Basic HTML/CSS/JS chat
   - WebSocket for real-time updates
   - Local storage for history

### 6.3 P2 - Advanced Features (Weeks 5-6)

1. **Local LLM Integration**
   - Ollama setup and configuration
   - Fallback templates
   - Context management

2. **Extended Sources**
   - SEC EDGAR integration
   - Federal Reserve feeds
   - Additional RSS sources

3. **Performance Optimization**
   - Caching strategies
   - Database indexing
   - Resource monitoring

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Web scraping breakage | High | High | Multiple sources, fallbacks |
| LLM resource usage | High | Medium | Optimize models, caching |
| Rate limiting | Medium | High | Respectful delays, rotation |
| Data quality | High | Medium | Multi-source validation |

### 7.2 Project Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scope creep | High | Strict phase adherence |
| Integration complexity | Medium | Incremental testing |
| Performance issues | Medium | Early benchmarking |

## 8. TDD Implementation Strategy

### 8.1 Test Categories

1. **Unit Tests** (pytest)
   - News parser functions
   - Impact scoring algorithms
   - Data transformation logic

2. **Integration Tests**
   - API endpoint responses
   - Database operations
   - External API mocking

3. **System Tests**
   - End-to-end news flow
   - Chat interaction scenarios
   - Performance benchmarks

### 8.2 Test-First Development Order

1. **Week 1**: Data models and parsers
2. **Week 2**: Analysis algorithms
3. **Week 3**: API endpoints
4. **Week 4**: Frontend integration
5. **Week 5**: LLM interactions
6. **Week 6**: Performance tests

## 9. Migration Strategy

Given the significant divergence between current implementation and requirements:

### 9.1 Recommended Approach

1. **Preserve Valuable Components**
   - Docker configuration
   - Configuration management
   - Testing framework

2. **Refactor Architecture**
   - Create news-focused modules
   - Remove cryptocurrency-specific code
   - Replace paid APIs with free alternatives

3. **Incremental Migration**
   - Start with news ingestion
   - Add analysis layer
   - Build interface last

### 9.2 Coexistence Strategy

- Maintain symbolic trading in separate namespace
- Share common utilities (logging, config)
- Gradual feature parity before deprecation

## 10. Success Metrics

### 10.1 Functional Metrics

- News sources active: Target 15+
- Articles processed/hour: Target 1000+
- Average analysis time: Target <5s
- Query response time: Target <2s

### 10.2 Quality Metrics

- Test coverage: Target >80%
- Error rate: Target <1%
- Duplicate rate: Target <5%
- User satisfaction: Target >90%

## Conclusion

The current implementation has diverged significantly from the original AI News Trading vision. A comprehensive rebuild focusing on news ingestion, analysis, and conversational interface is required. The existing symbolic trading platform should be preserved but separated from the news trading functionality. Success depends on strict adherence to the zero-cost principle and phased implementation approach.

## Next Steps

1. **Immediate Actions**
   - Set up project structure for news modules
   - Create initial test suite structure
   - Begin RSS feed implementation

2. **Week 1 Deliverables**
   - Working RSS aggregator
   - Basic data models
   - Initial test coverage

3. **Stakeholder Communication**
   - Review requirements with team
   - Confirm priority adjustments
   - Establish progress tracking