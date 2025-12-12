use crate::NarrativeError;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionData {
    pub price_prediction: f64,
    pub confidence_score: f64,
    pub timeframe: String,
}

pub struct PredictionExtractor {
    price_regex: Regex,
    confidence_regex: Regex,
}

impl PredictionExtractor {
    pub fn new() -> Self {
        Self {
            // Enhanced regex patterns for Sonnet 4's more detailed output
            price_regex: Regex::new(r"(?i)price\s+prediction:\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)").unwrap(),
            confidence_regex: Regex::new(r"(?i)confidence:\s*(\d+(?:\.\d+)?)").unwrap(),
        }
    }
    
    pub fn extract_prediction_data(
        &self,
        narrative: &str,
        fallback_price: f64,
    ) -> Result<PredictionData, NarrativeError> {
        // Extract price prediction with enhanced parsing for Sonnet 4 output
        let price_prediction = if let Some(captures) = self.price_regex.captures(narrative) {
            captures.get(1)
                .and_then(|m| {
                    // Remove commas for parsing (e.g., "50,000" -> "50000")
                    let price_str = m.as_str().replace(",", "");
                    price_str.parse::<f64>().ok()
                })
                .unwrap_or(fallback_price)
        } else {
            // Try alternative patterns optimized for Sonnet 4's detailed reasoning
            self.extract_price_alternative(narrative, fallback_price)
        };
        
        // Extract confidence score
        let confidence_score = if let Some(captures) = self.confidence_regex.captures(narrative) {
            captures.get(1)
                .and_then(|m| m.as_str().parse::<f64>().ok())
                .unwrap_or(0.5)
        } else {
            self.extract_confidence_alternative(narrative)
        };
        
        // Extract timeframe (default to 24h)
        let timeframe = self.extract_timeframe(narrative);
        
        Ok(PredictionData {
            price_prediction,
            confidence_score: confidence_score.max(0.0).min(1.0),
            timeframe,
        })
    }
    
    fn extract_price_alternative(&self, narrative: &str, fallback: f64) -> f64 {
        // Enhanced patterns for Sonnet 4's sophisticated output
        let patterns = vec![
            r"(?i)\$(\d+(?:,\d{3})*(?:\.\d+)?)", // Formatted prices with commas
            r"(?i)price.*?(\d+(?:,\d{3})*(?:\.\d+)?)",
            r"(?i)target.*?(\d+(?:,\d{3})*(?:\.\d+)?)",
            r"(?i)reach.*?(\d+(?:,\d{3})*(?:\.\d+)?)",
            r"(?i)level.*?(\d+(?:,\d{3})*(?:\.\d+)?)",
            r"(?i)trading.*?(\d+(?:,\d{3})*(?:\.\d+)?)",
            r"(?i)statistical.*?(\d+(?:,\d{3})*(?:\.\d+)?)", // For statistical confidence mentions
        ];
        
        for pattern in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if let Some(captures) = regex.captures(narrative) {
                    if let Some(price_str) = captures.get(1) {
                        // Remove commas for parsing
                        let clean_price_str = price_str.as_str().replace(",", "");
                        if let Ok(price) = clean_price_str.parse::<f64>() {
                            // Enhanced sanity check for crypto prices
                            if price > 0.0 && price < 10_000_000.0 {
                                return price;
                            }
                        }
                    }
                }
            }
        }
        
        fallback
    }
    
    fn extract_confidence_alternative(&self, narrative: &str) -> f64 {
        // Enhanced confidence indicators for Sonnet 4's nuanced reasoning
        let high_confidence_words = ["certain", "confident", "strong", "definite", "statistical", "robust", "validated", "conclusive"];
        let medium_confidence_words = ["likely", "probable", "moderate", "reasonable", "supported", "indicative"];
        let low_confidence_words = ["uncertain", "unclear", "weak", "doubtful", "speculative", "tentative", "limited"];
        
        let narrative_lower = narrative.to_lowercase();
        
        let high_count = high_confidence_words.iter()
            .filter(|&word| narrative_lower.contains(word))
            .count();
        
        let medium_count = medium_confidence_words.iter()
            .filter(|&word| narrative_lower.contains(word))
            .count();
        
        let low_count = low_confidence_words.iter()
            .filter(|&word| narrative_lower.contains(word))
            .count();
        
        if high_count > 0 {
            0.8
        } else if medium_count > 0 {
            0.6
        } else if low_count > 0 {
            0.3
        } else {
            0.5 // Default neutral confidence
        }
    }
    
    fn extract_timeframe(&self, narrative: &str) -> String {
        let timeframe_patterns = vec![
            (r"(?i)24\s*hours?", "24h"),
            (r"(?i)1\s*day", "24h"),
            (r"(?i)next\s+day", "24h"),
            (r"(?i)tomorrow", "24h"),
            (r"(?i)48\s*hours?", "48h"),
            (r"(?i)2\s*days?", "48h"),
            (r"(?i)week", "1w"),
            (r"(?i)7\s*days?", "1w"),
        ];
        
        for (pattern, timeframe) in timeframe_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(narrative) {
                    return timeframe.to_string();
                }
            }
        }
        
        "24h".to_string() // Default timeframe
    }
}