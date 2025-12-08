use crate::NarrativeError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SentimentDimension {
    Polarity,    // Positive vs negative
    Confidence,  // Confidence vs uncertainty  
    Fear,        // Fear vs greed
    Volatility,  // Stable vs volatile expectations
    Momentum,    // Future trend expectations
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    pub overall_sentiment: f64,
    pub confidence: f64,
    pub dimensions: HashMap<String, f64>,
    pub key_phrases: Vec<String>,
}

impl Default for SentimentResult {
    fn default() -> Self {
        Self {
            overall_sentiment: 0.5,
            confidence: 0.5,
            dimensions: HashMap::new(),
            key_phrases: vec![],
        }
    }
}

pub struct SentimentAnalyzer {
    lexicons: HashMap<SentimentDimension, HashMap<String, f64>>,
}

impl SentimentAnalyzer {
    pub fn new() -> Self {
        let mut lexicons = HashMap::new();
        
        // Polarity lexicon
        let mut polarity_lexicon = HashMap::new();
        polarity_lexicon.insert("bullish".to_string(), 0.8);
        polarity_lexicon.insert("positive".to_string(), 0.7);
        polarity_lexicon.insert("optimistic".to_string(), 0.7);
        polarity_lexicon.insert("bearish".to_string(), -0.8);
        polarity_lexicon.insert("negative".to_string(), -0.7);
        polarity_lexicon.insert("pessimistic".to_string(), -0.7);
        lexicons.insert(SentimentDimension::Polarity, polarity_lexicon);
        
        // Confidence lexicon
        let mut confidence_lexicon = HashMap::new();
        confidence_lexicon.insert("confident".to_string(), 0.8);
        confidence_lexicon.insert("certain".to_string(), 0.8);
        confidence_lexicon.insert("uncertain".to_string(), -0.7);
        confidence_lexicon.insert("unsure".to_string(), -0.7);
        lexicons.insert(SentimentDimension::Confidence, confidence_lexicon);
        
        // Fear lexicon
        let mut fear_lexicon = HashMap::new();
        fear_lexicon.insert("fear".to_string(), 0.8);
        fear_lexicon.insert("panic".to_string(), 0.9);
        fear_lexicon.insert("greed".to_string(), -0.8);
        fear_lexicon.insert("fomo".to_string(), -0.9);
        lexicons.insert(SentimentDimension::Fear, fear_lexicon);
        
        // Volatility lexicon
        let mut volatility_lexicon = HashMap::new();
        volatility_lexicon.insert("volatile".to_string(), 0.8);
        volatility_lexicon.insert("stable".to_string(), -0.8);
        volatility_lexicon.insert("turbulent".to_string(), 0.7);
        volatility_lexicon.insert("calm".to_string(), -0.7);
        lexicons.insert(SentimentDimension::Volatility, volatility_lexicon);
        
        // Momentum lexicon
        let mut momentum_lexicon = HashMap::new();
        momentum_lexicon.insert("rising".to_string(), 0.7);
        momentum_lexicon.insert("falling".to_string(), -0.7);
        momentum_lexicon.insert("surging".to_string(), 0.9);
        momentum_lexicon.insert("crashing".to_string(), -0.9);
        lexicons.insert(SentimentDimension::Momentum, momentum_lexicon);
        
        Self { lexicons }
    }
    
    pub async fn analyze_comprehensive_sentiment(
        &self,
        text: &str,
        _symbol: &str,
    ) -> Result<SentimentResult, NarrativeError> {
        let text_lower = text.to_lowercase();
        
        let mut dimensions = HashMap::new();
        let mut overall_scores = vec![];
        let mut key_phrases = vec![];
        
        // Analyze each dimension
        for (dimension, lexicon) in &self.lexicons {
            let score = self.calculate_dimension_score(&text_lower, lexicon);
            
            let dimension_name = match dimension {
                SentimentDimension::Polarity => "polarity",
                SentimentDimension::Confidence => "confidence",
                SentimentDimension::Fear => "fear",
                SentimentDimension::Volatility => "volatility",
                SentimentDimension::Momentum => "momentum",
            };
            
            dimensions.insert(dimension_name.to_string(), score);
            overall_scores.push(score);
            
            // Extract key phrases for this dimension
            for (phrase, _) in lexicon {
                if text_lower.contains(phrase) {
                    key_phrases.push(phrase.clone());
                }
            }
        }
        
        // Calculate overall sentiment
        let overall_sentiment = if overall_scores.is_empty() {
            0.5
        } else {
            let sum: f64 = overall_scores.iter().sum();
            (sum / overall_scores.len() as f64 + 1.0) / 2.0 // Normalize to [0,1]
        };
        
        // Calculate confidence based on number of matched terms
        let confidence = (key_phrases.len() as f64 / 10.0).min(1.0);
        
        // Limit key phrases
        key_phrases.truncate(5);
        
        Ok(SentimentResult {
            overall_sentiment,
            confidence,
            dimensions,
            key_phrases,
        })
    }
    
    fn calculate_dimension_score(&self, text: &str, lexicon: &HashMap<String, f64>) -> f64 {
        let mut total_score = 0.0;
        let mut match_count = 0;
        
        for (term, score) in lexicon {
            if text.contains(term) {
                total_score += score;
                match_count += 1;
            }
        }
        
        if match_count > 0 {
            total_score / match_count as f64
        } else {
            0.0 // Neutral if no matches
        }
    }
}