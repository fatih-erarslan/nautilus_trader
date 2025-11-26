use crate::aggregator::NewsAggregator;
use crate::error::Result;
use crate::models::{Direction, EventCategory, NewsArticle, Sentiment, TradingSignal};
use crate::sentiment::SentimentAnalyzer;
use std::collections::HashMap;
use std::sync::Arc;

pub struct NewsTradingStrategy {
    aggregator: Arc<NewsAggregator>,
    analyzer: Arc<SentimentAnalyzer>,
    config: StrategyConfig,
    event_weights: HashMap<String, f64>,
}

impl NewsTradingStrategy {
    pub fn new(
        aggregator: Arc<NewsAggregator>,
        analyzer: Arc<SentimentAnalyzer>,
        config: StrategyConfig,
    ) -> Self {
        let event_weights = Self::default_event_weights();

        Self {
            aggregator,
            analyzer,
            config,
            event_weights,
        }
    }

    pub fn default_with_aggregator(aggregator: Arc<NewsAggregator>) -> Self {
        Self::new(
            aggregator,
            Arc::new(SentimentAnalyzer::default()),
            StrategyConfig::default(),
        )
    }

    /// Process a news article and generate trading signal if appropriate
    pub async fn on_news(&mut self, article: NewsArticle) -> Result<Option<TradingSignal>> {
        // Analyze sentiment if not already done
        let sentiment = if let Some(s) = article.sentiment {
            s
        } else {
            let text = format!("{} {}", article.title, article.content);
            self.analyzer.analyze(&text)
        };

        // Calculate impact score
        let impact = self.calculate_impact(&article, &sentiment);

        // Generate signal if criteria are met
        if impact >= self.config.min_impact_threshold
            && sentiment.magnitude >= self.config.min_sentiment_magnitude
            && sentiment.score.abs() > 0.05  // Must have some directional sentiment
        {
            for symbol in &article.symbols {
                let signal = self.generate_signal(symbol, &sentiment, impact, &article);
                let confidence = self.calculate_confidence(&sentiment, impact);

                // Only return if confidence is high enough
                if confidence >= self.config.min_confidence {
                    return Ok(Some(signal));
                }
            }
        }

        Ok(None)
    }

    /// Calculate the impact score of a news article
    pub fn calculate_impact(&self, article: &NewsArticle, sentiment: &Sentiment) -> f64 {
        let mut impact = 0.0;

        // Base impact from sentiment magnitude
        impact += sentiment.magnitude * 0.4;

        // Relevance score
        impact += article.relevance * 0.3;

        // Event detection bonus
        let event_bonus = self.detect_events(&article.title, &article.content);
        impact += event_bonus * 0.3;

        impact.clamp(0.0, 1.0)
    }

    /// Generate a trading signal
    pub fn generate_signal(
        &self,
        symbol: &str,
        sentiment: &Sentiment,
        impact: f64,
        article: &NewsArticle,
    ) -> TradingSignal {
        let direction = Direction::from_sentiment(sentiment);

        let confidence = self.calculate_confidence(sentiment, impact);

        let reason = format!(
            "News-based signal: {} sentiment (score: {:.2}, magnitude: {:.2}), impact: {:.2}",
            sentiment.label.as_str(),
            sentiment.score,
            sentiment.magnitude,
            impact
        );

        TradingSignal::new(
            symbol.to_string(),
            direction,
            confidence,
            reason,
            article.id.clone(),
        )
        .with_scores(sentiment.score, impact)
    }

    fn calculate_confidence(&self, sentiment: &Sentiment, impact: f64) -> f64 {
        let sentiment_component = sentiment.magnitude * 0.5;
        let impact_component = impact * 0.5;

        (sentiment_component + impact_component).clamp(0.0, 1.0)
    }

    fn detect_events(&self, title: &str, content: &str) -> f64 {
        let text = format!("{} {}", title, content).to_lowercase();
        let mut max_weight: f64 = 0.0;

        for (keyword, weight) in &self.event_weights {
            if text.contains(keyword) {
                max_weight = max_weight.max(*weight);
            }
        }

        max_weight
    }

    fn default_event_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();

        // Earnings events
        weights.insert("earnings".to_string(), EventCategory::Earnings.base_impact());
        weights.insert("quarterly results".to_string(), 0.85);
        weights.insert("guidance".to_string(), 0.75);

        // M&A events
        weights.insert("merger".to_string(), EventCategory::MergersAcquisitions.base_impact());
        weights.insert("acquisition".to_string(), 0.9);
        weights.insert("takeover".to_string(), 0.85);

        // Regulatory events
        weights.insert("fda approval".to_string(), 0.9);
        weights.insert("regulatory".to_string(), EventCategory::Regulatory.base_impact());
        weights.insert("investigation".to_string(), 0.75);

        // Product events
        weights.insert("product launch".to_string(), EventCategory::ProductLaunch.base_impact());
        weights.insert("new product".to_string(), 0.65);

        // Leadership events
        weights.insert("ceo".to_string(), EventCategory::Leadership.base_impact());
        weights.insert("resignation".to_string(), 0.6);

        weights
    }

    pub async fn backtest(
        &mut self,
        symbols: &[String],
        days: u32,
    ) -> Result<BacktestResults> {
        let mut results = BacktestResults::new();

        // Fetch historical news
        let articles = self.aggregator.fetch_news(symbols).await?;

        for article in articles {
            if let Some(signal) = self.on_news(article).await? {
                results.add_signal(signal);
            }
        }

        Ok(results)
    }
}

#[derive(Debug, Clone)]
pub struct StrategyConfig {
    pub min_impact_threshold: f64,
    pub min_sentiment_magnitude: f64,
    pub min_confidence: f64,
    pub max_signals_per_day: usize,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            min_impact_threshold: 0.3,
            min_sentiment_magnitude: 0.2,
            min_confidence: 0.4,
            max_signals_per_day: 10,
        }
    }
}

#[derive(Debug)]
pub struct BacktestResults {
    signals: Vec<TradingSignal>,
    total_signals: usize,
    positive_signals: usize,
    negative_signals: usize,
}

impl BacktestResults {
    pub fn new() -> Self {
        Self {
            signals: Vec::new(),
            total_signals: 0,
            positive_signals: 0,
            negative_signals: 0,
        }
    }

    pub fn add_signal(&mut self, signal: TradingSignal) {
        self.total_signals += 1;

        match signal.direction {
            Direction::Long => self.positive_signals += 1,
            Direction::Short => self.negative_signals += 1,
            Direction::Neutral => {}
        }

        self.signals.push(signal);
    }

    pub fn summary(&self) -> String {
        format!(
            "Backtest Results:\n  Total signals: {}\n  Long signals: {}\n  Short signals: {}",
            self.total_signals, self.positive_signals, self.negative_signals
        )
    }

    pub fn signals(&self) -> &[TradingSignal] {
        &self.signals
    }
}

impl Default for BacktestResults {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_detection() {
        let aggregator = Arc::new(NewsAggregator::new());
        let strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

        let impact = strategy.detect_events(
            "Company announces quarterly earnings beat",
            "The company exceeded expectations",
        );

        assert!(impact > 0.7);
    }

    #[test]
    fn test_confidence_calculation() {
        let aggregator = Arc::new(NewsAggregator::new());
        let strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

        let sentiment = Sentiment::new(0.8, 0.9);
        let confidence = strategy.calculate_confidence(&sentiment, 0.7);

        assert!(confidence > 0.6);
        assert!(confidence <= 1.0);
    }
}
