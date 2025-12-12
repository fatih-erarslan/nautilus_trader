use crate::*;

pub struct CorrelationAnalyzer {
    correlation_matrix: HashMap<String, HashMap<String, f64>>,
    clusters: HashMap<String, String>,
}

impl CorrelationAnalyzer {
    pub fn new() -> Self {
        Self {
            correlation_matrix: HashMap::new(),
            clusters: HashMap::new(),
        }
    }
    
    pub async fn get_correlation_cluster(&self, symbol: &str) -> Result<String, IntelligenceError> {
        Ok(self.clusters.get(symbol).cloned().unwrap_or_else(|| {
            if symbol.contains("BTC") {
                "large-cap".to_string()
            } else if symbol.contains("ETH") || symbol.contains("BNB") {
                "large-cap".to_string()  
            } else {
                "alt-coins".to_string()
            }
        }))
    }
}