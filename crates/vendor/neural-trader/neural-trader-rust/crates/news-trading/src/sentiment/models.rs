use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentConfig {
    pub use_financial_lexicon: bool,
    pub min_word_length: usize,
    pub case_sensitive: bool,
    pub sentiment_threshold: f64,
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            use_financial_lexicon: true,
            min_word_length: 3,
            case_sensitive: false,
            sentiment_threshold: 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WordScore {
    pub word: String,
    pub score: f64,
}
