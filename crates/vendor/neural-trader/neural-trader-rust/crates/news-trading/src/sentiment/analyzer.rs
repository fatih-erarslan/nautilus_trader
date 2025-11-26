use crate::error::{NewsError, Result};
use crate::models::{Sentiment, SentimentLabel};
use crate::sentiment::models::{SentimentConfig, WordScore};
use std::collections::HashMap;

pub struct SentimentAnalyzer {
    config: SentimentConfig,
    lexicon: HashMap<String, f64>,
}

impl SentimentAnalyzer {
    pub fn new(config: SentimentConfig) -> Self {
        let lexicon = Self::load_financial_lexicon();
        Self { config, lexicon }
    }

    pub fn default() -> Self {
        Self::new(SentimentConfig::default())
    }

    /// Analyze sentiment of a single text
    pub fn analyze(&self, text: &str) -> Sentiment {
        let words = self.tokenize(text);
        let scores = self.score_words(&words);

        let (total_score, total_magnitude) = scores.iter().fold((0.0, 0.0), |(s, m), word_score| {
            (s + word_score.score, m + word_score.score.abs())
        });

        let word_count = words.len() as f64;
        let normalized_score = if word_count > 0.0 {
            (total_score / word_count).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let normalized_magnitude = if word_count > 0.0 {
            (total_magnitude / word_count).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Sentiment::new(normalized_score, normalized_magnitude)
    }

    /// Analyze sentiment of multiple texts in batch
    pub fn analyze_batch(&self, texts: &[&str]) -> Vec<Sentiment> {
        texts.iter().map(|text| self.analyze(text)).collect()
    }

    /// Analyze with detailed breakdown
    pub fn analyze_detailed(&self, text: &str) -> DetailedSentiment {
        let words = self.tokenize(text);
        let word_scores = self.score_words(&words);

        let sentiment = self.analyze(text);

        let positive_words: Vec<String> = word_scores
            .iter()
            .filter(|ws| ws.score > 0.0)
            .map(|ws| ws.word.clone())
            .collect();

        let negative_words: Vec<String> = word_scores
            .iter()
            .filter(|ws| ws.score < 0.0)
            .map(|ws| ws.word.clone())
            .collect();

        DetailedSentiment {
            sentiment,
            positive_words,
            negative_words,
            word_count: words.len(),
            total_score: word_scores.iter().map(|ws| ws.score).sum(),
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    fn score_words(&self, words: &[String]) -> Vec<WordScore> {
        words
            .iter()
            .filter_map(|word| {
                self.lexicon.get(word).map(|&score| WordScore {
                    word: word.clone(),
                    score,
                })
            })
            .collect()
    }

    fn load_financial_lexicon() -> HashMap<String, f64> {
        let mut lexicon = HashMap::new();

        // Positive financial terms - increased weights for better detection
        let positive = vec![
            ("bullish", 1.5),
            ("profit", 1.2),
            ("profits", 1.2),
            ("growth", 1.2),
            ("surge", 1.4),
            ("surged", 1.4),
            ("rally", 1.2),
            ("gain", 1.0),
            ("gains", 1.0),
            ("beat", 1.3),
            ("beats", 1.3),
            ("exceed", 1.2),
            ("exceeded", 1.2),
            ("exceeds", 1.2),
            ("strong", 1.0),
            ("upgrade", 1.3),
            ("outperform", 1.4),
            ("positive", 0.8),
            ("optimistic", 1.0),
            ("success", 1.2),
            ("successful", 1.2),
            ("boom", 1.4),
            ("soar", 1.4),
            ("soaring", 1.4),
            ("jump", 1.0),
            ("jumped", 1.0),
            ("rise", 0.8),
            ("rising", 0.8),
            ("increase", 0.7),
            ("expand", 0.8),
            ("expanding", 0.8),
            ("momentum", 1.0),
            ("breakthrough", 1.4),
            ("record", 1.2),
            ("robust", 1.0),
            ("recovery", 1.0),
            ("exceptional", 1.3),
            ("excellent", 1.2),
            ("outstanding", 1.3),
            ("impressive", 1.1),
        ];

        // Negative financial terms - increased weights for better detection
        let negative = vec![
            ("bearish", -1.5),
            ("loss", -1.2),
            ("losses", -1.2),
            ("decline", -1.0),
            ("declining", -1.0),
            ("crash", -1.8),
            ("plunge", -1.5),
            ("plunges", -1.5),
            ("plunging", -1.5),
            ("fall", -1.0),
            ("falling", -1.0),
            ("miss", -1.2),
            ("missed", -1.2),
            ("weak", -1.0),
            ("downgrade", -1.3),
            ("underperform", -1.4),
            ("negative", -0.8),
            ("pessimistic", -1.0),
            ("failure", -1.2),
            ("recession", -1.5),
            ("crisis", -1.6),
            ("collapse", -1.7),
            ("drop", -1.0),
            ("dropping", -1.0),
            ("decrease", -0.7),
            ("shrink", -0.8),
            ("concern", -0.8),
            ("concerns", -0.8),
            ("warning", -1.0),
            ("risk", -0.7),
            ("risks", -0.7),
            ("volatile", -0.6),
            ("uncertain", -0.8),
            ("uncertainty", -0.8),
            ("bankruptcy", -1.8),
            ("disappointing", -1.2),
            ("disappoints", -1.2),
            ("investigation", -1.0),
        ];

        for (word, score) in positive {
            lexicon.insert(word.to_string(), score);
        }

        for (word, score) in negative {
            lexicon.insert(word.to_string(), score);
        }

        lexicon
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::default()
    }
}

pub struct DetailedSentiment {
    pub sentiment: Sentiment,
    pub positive_words: Vec<String>,
    pub negative_words: Vec<String>,
    pub word_count: usize,
    pub total_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_sentiment() {
        let analyzer = SentimentAnalyzer::default();
        let sentiment = analyzer.analyze("The stock is showing strong growth and bullish momentum with record profits");

        assert!(sentiment.is_positive(), "Expected positive sentiment, got score: {}", sentiment.score);
        assert!(sentiment.score > 0.2, "Expected score > 0.2, got {}", sentiment.score);
    }

    #[test]
    fn test_negative_sentiment() {
        let analyzer = SentimentAnalyzer::default();
        let sentiment = analyzer.analyze("The company faces a major crisis with declining profits and severe losses bankruptcy crash plunge");

        assert!(sentiment.is_negative(), "Expected negative sentiment, got score: {}", sentiment.score);
        assert!(sentiment.score < 0.0, "Expected score < 0.0, got {}", sentiment.score);
    }

    #[test]
    fn test_neutral_sentiment() {
        let analyzer = SentimentAnalyzer::default();
        let sentiment = analyzer.analyze("The company held a meeting today");

        assert!(sentiment.is_neutral());
    }

    #[test]
    fn test_batch_analysis() {
        let analyzer = SentimentAnalyzer::default();
        let texts = vec![
            "Bullish rally continues",
            "Market crash expected",
            "Normal trading day",
        ];

        let results = analyzer.analyze_batch(&texts);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_positive());
        assert!(results[1].is_negative());
        assert!(results[2].is_neutral());
    }
}
