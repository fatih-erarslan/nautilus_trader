/// Order book configuration

/// Configuration for order book processing
#[derive(Debug, Clone)]
pub struct OrderBookConfig {
    /// Maximum number of price levels to maintain
    pub max_depth: usize,

    /// Minimum order size to process
    pub min_order_size: f64,

    /// Enable order book validation
    pub validate: bool,
}

impl Default for OrderBookConfig {
    fn default() -> Self {
        Self {
            max_depth: 20,
            min_order_size: 0.0,
            validate: true,
        }
    }
}

impl OrderBookConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set minimum order size
    pub fn with_min_order_size(mut self, size: f64) -> Self {
        self.min_order_size = size;
        self
    }

    /// Enable/disable validation
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OrderBookConfig::default();
        assert_eq!(config.max_depth, 20);
        assert_eq!(config.min_order_size, 0.0);
        assert!(config.validate);
    }

    #[test]
    fn test_builder_pattern() {
        let config = OrderBookConfig::new()
            .with_max_depth(10)
            .with_min_order_size(0.01)
            .with_validation(false);

        assert_eq!(config.max_depth, 10);
        assert_eq!(config.min_order_size, 0.01);
        assert!(!config.validate);
    }
}
