//! Whale detection and tracking module

use crate::{MacchiavelianConfig, MarketData, TalebianRiskError, WhaleDetection, WhaleDirection};
use serde::{Deserialize, Serialize};

/// Whale detection engine
pub struct WhaleDetectionEngine {
    config: MacchiavelianConfig,
    detection_history: Vec<WhaleDetection>,
}

/// Whale activity summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleActivitySummary {
    pub total_detections: usize,
    pub recent_activity_level: f64,
    pub confidence_avg: f64,
    pub dominant_direction: WhaleDirection,
}

impl WhaleDetectionEngine {
    pub fn new(config: MacchiavelianConfig) -> Self {
        Self {
            config,
            detection_history: Vec::new(),
        }
    }

    pub fn detect_whale_activity(
        &mut self,
        market_data: &MarketData,
    ) -> Result<WhaleDetection, TalebianRiskError> {
        // Simple whale detection based on volume spikes
        let volume_threshold = self.config.whale_volume_threshold;
        let avg_volume = market_data.volume_history.iter().sum::<f64>()
            / market_data.volume_history.len().max(1) as f64;

        let volume_spike = market_data.volume / avg_volume;
        let detected = volume_spike > volume_threshold;

        let direction = if market_data.bid_volume > market_data.ask_volume * 1.2 {
            WhaleDirection::Buying
        } else if market_data.ask_volume > market_data.bid_volume * 1.2 {
            WhaleDirection::Selling
        } else {
            WhaleDirection::Neutral
        };

        let confidence = if detected {
            (volume_spike - volume_threshold) / volume_threshold
        } else {
            0.1
        }
        .min(0.95);

        let whale_detection = WhaleDetection {
            timestamp: market_data.timestamp_unix,
            detected,
            volume_spike,
            direction,
            confidence,
            whale_size: market_data.volume,
            impact: market_data.spread() / market_data.mid_price(),
            is_whale_detected: detected,
            order_book_imbalance: (market_data.bid_volume - market_data.ask_volume)
                / (market_data.bid_volume + market_data.ask_volume),
            price_impact: market_data.spread() / market_data.mid_price(),
        };

        self.detection_history.push(whale_detection.clone());

        // Keep only recent history
        if self.detection_history.len() > 1000 {
            self.detection_history.drain(0..500);
        }

        Ok(whale_detection)
    }

    pub fn get_whale_activity_summary(&self) -> WhaleActivitySummary {
        let total_detections = self.detection_history.iter().filter(|w| w.detected).count();
        let recent_activity_level = if self.detection_history.len() > 10 {
            self.detection_history
                .iter()
                .rev()
                .take(10)
                .filter(|w| w.detected)
                .count() as f64
                / 10.0
        } else {
            0.0
        };

        let confidence_avg = if !self.detection_history.is_empty() {
            self.detection_history
                .iter()
                .map(|w| w.confidence)
                .sum::<f64>()
                / self.detection_history.len() as f64
        } else {
            0.0
        };

        // Determine dominant direction
        let buying_count = self
            .detection_history
            .iter()
            .filter(|w| matches!(w.direction, WhaleDirection::Buying))
            .count();
        let selling_count = self
            .detection_history
            .iter()
            .filter(|w| matches!(w.direction, WhaleDirection::Selling))
            .count();

        let dominant_direction = if buying_count > selling_count {
            WhaleDirection::Buying
        } else if selling_count > buying_count {
            WhaleDirection::Selling
        } else {
            WhaleDirection::Neutral
        };

        WhaleActivitySummary {
            total_detections,
            recent_activity_level,
            confidence_avg,
            dominant_direction,
        }
    }
}
