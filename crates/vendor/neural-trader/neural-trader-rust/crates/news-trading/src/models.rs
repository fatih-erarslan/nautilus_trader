use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsArticle {
    pub id: String,
    pub title: String,
    pub content: String,
    pub source: String,
    pub published_at: DateTime<Utc>,
    pub symbols: Vec<String>,
    pub sentiment: Option<Sentiment>,
    pub relevance: f64,
    pub url: Option<String>,
    pub author: Option<String>,
    pub tags: Vec<String>,
}

impl NewsArticle {
    pub fn new(id: String, title: String, content: String, source: String) -> Self {
        Self {
            id,
            title,
            content,
            source,
            published_at: Utc::now(),
            symbols: Vec::new(),
            sentiment: None,
            relevance: 0.0,
            url: None,
            author: None,
            tags: Vec::new(),
        }
    }

    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = symbols;
        self
    }

    pub fn with_sentiment(mut self, sentiment: Sentiment) -> Self {
        self.sentiment = Some(sentiment);
        self
    }

    pub fn with_relevance(mut self, relevance: f64) -> Self {
        self.relevance = relevance;
        self
    }

    pub fn has_symbol(&self, symbol: &str) -> bool {
        self.symbols.iter().any(|s| s.eq_ignore_ascii_case(symbol))
    }
}

impl Hash for NewsArticle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for NewsArticle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for NewsArticle {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Sentiment {
    /// Sentiment score from -1.0 (very negative) to 1.0 (very positive)
    pub score: f64,
    /// Magnitude/confidence from 0.0 to 1.0
    pub magnitude: f64,
    /// Categorical label
    pub label: SentimentLabel,
}

impl Sentiment {
    pub fn new(score: f64, magnitude: f64) -> Self {
        let label = SentimentLabel::from_score(score);
        Self {
            score: score.clamp(-1.0, 1.0),
            magnitude: magnitude.clamp(0.0, 1.0),
            label,
        }
    }

    pub fn is_positive(&self) -> bool {
        self.score > 0.1
    }

    pub fn is_negative(&self) -> bool {
        self.score < -0.1
    }

    pub fn is_neutral(&self) -> bool {
        !self.is_positive() && !self.is_negative()
    }

    pub fn is_strong(&self) -> bool {
        self.magnitude > 0.6
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SentimentLabel {
    VeryNegative,
    Negative,
    Neutral,
    Positive,
    VeryPositive,
}

impl SentimentLabel {
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s <= -0.6 => Self::VeryNegative,
            s if s <= -0.2 => Self::Negative,
            s if s >= 0.6 => Self::VeryPositive,
            s if s >= 0.2 => Self::Positive,
            _ => Self::Neutral,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::VeryNegative => "very_negative",
            Self::Negative => "negative",
            Self::Neutral => "neutral",
            Self::Positive => "positive",
            Self::VeryPositive => "very_positive",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub symbol: String,
    pub direction: Direction,
    pub confidence: f64,
    pub reason: String,
    pub news_id: String,
    pub timestamp: DateTime<Utc>,
    pub sentiment_score: f64,
    pub impact_score: f64,
}

impl TradingSignal {
    pub fn new(
        symbol: String,
        direction: Direction,
        confidence: f64,
        reason: String,
        news_id: String,
    ) -> Self {
        Self {
            symbol,
            direction,
            confidence: confidence.clamp(0.0, 1.0),
            reason,
            news_id,
            timestamp: Utc::now(),
            sentiment_score: 0.0,
            impact_score: 0.0,
        }
    }

    pub fn with_scores(mut self, sentiment: f64, impact: f64) -> Self {
        self.sentiment_score = sentiment;
        self.impact_score = impact;
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Direction {
    Long,
    Short,
    Neutral,
}

impl Direction {
    pub fn from_sentiment(sentiment: &Sentiment) -> Self {
        if sentiment.is_positive() && sentiment.is_strong() {
            Self::Long
        } else if sentiment.is_negative() && sentiment.is_strong() {
            Self::Short
        } else {
            Self::Neutral
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsQuery {
    pub symbols: Option<Vec<String>>,
    pub sources: Option<Vec<String>>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub min_relevance: Option<f64>,
    pub sentiment_filter: Option<SentimentLabel>,
    pub limit: Option<usize>,
}

impl Default for NewsQuery {
    fn default() -> Self {
        Self {
            symbols: None,
            sources: None,
            start_date: None,
            end_date: None,
            min_relevance: None,
            sentiment_filter: None,
            limit: Some(100),
        }
    }
}

impl NewsQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = Some(symbols);
        self
    }

    pub fn with_sources(mut self, sources: Vec<String>) -> Self {
        self.sources = Some(sources);
        self
    }

    pub fn with_date_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_date = Some(start);
        self.end_date = Some(end);
        self
    }

    pub fn with_min_relevance(mut self, min: f64) -> Self {
        self.min_relevance = Some(min);
        self
    }

    pub fn matches(&self, article: &NewsArticle) -> bool {
        // Check symbols
        if let Some(ref symbols) = self.symbols {
            if !article.symbols.iter().any(|s| symbols.contains(s)) {
                return false;
            }
        }

        // Check sources
        if let Some(ref sources) = self.sources {
            if !sources.contains(&article.source) {
                return false;
            }
        }

        // Check date range
        if let Some(start) = self.start_date {
            if article.published_at < start {
                return false;
            }
        }
        if let Some(end) = self.end_date {
            if article.published_at > end {
                return false;
            }
        }

        // Check relevance
        if let Some(min_rel) = self.min_relevance {
            if article.relevance < min_rel {
                return false;
            }
        }

        // Check sentiment
        if let Some(sentiment_filter) = self.sentiment_filter {
            if let Some(sentiment) = &article.sentiment {
                if sentiment.label != sentiment_filter {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventType {
    pub name: String,
    pub category: EventCategory,
    pub impact_weight: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EventCategory {
    Earnings,
    MergersAcquisitions,
    Regulatory,
    ProductLaunch,
    Leadership,
    Economic,
    Other,
}

impl EventCategory {
    pub fn base_impact(&self) -> f64 {
        match self {
            Self::Earnings => 0.8,
            Self::MergersAcquisitions => 0.9,
            Self::Regulatory => 0.7,
            Self::ProductLaunch => 0.6,
            Self::Leadership => 0.5,
            Self::Economic => 0.7,
            Self::Other => 0.3,
        }
    }
}
