use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

#[derive(Debug, Clone)]
pub struct EmergenceSystem {
    patterns: Arc<RwLock<HashMap<String, EmergentPattern>>>,
}

#[derive(Debug, Clone)]
pub struct EmergentPattern {
    pub id: String,
    pub complexity: f64,
    pub stability: f64,
}

impl EmergenceSystem {
    pub fn new() -> Self {
        Self {
            patterns: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn detect_patterns(&self) -> Vec<EmergentPattern> {
        self.patterns.read().values().cloned().collect()
    }
    
    pub fn add_pattern(&self, pattern: EmergentPattern) {
        self.patterns.write().insert(pattern.id.clone(), pattern);
    }
}

impl Default for EmergenceSystem {
    fn default() -> Self {
        Self::new()
    }
}