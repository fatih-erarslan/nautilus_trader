//! Volume observer implementation

use crate::prelude::*;

/// Observer for volume analysis
#[derive(Debug, Clone)]
pub struct VolumeObserver {
    pub window: usize,
    pub profile_bins: usize,
}

impl VolumeObserver {
    pub fn new(window: usize, profile_bins: usize) -> Self {
        Self { window, profile_bins }
    }
    
    pub fn observe(&self, prices: &[f64], volumes: &[f64]) -> VolumeMetrics {
        if prices.len() != volumes.len() || prices.len() < self.window {
            return VolumeMetrics::default();
        }
        
        let recent_prices = &prices[prices.len() - self.window..];
        let recent_volumes = &volumes[volumes.len() - self.window..];
        
        VolumeMetrics {
            total_volume: recent_volumes.iter().sum(),
            average_volume: recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64,
            volume_weighted_price: self.calculate_vwap(recent_prices, recent_volumes),
            volume_profile: self.calculate_volume_profile(recent_prices, recent_volumes),
            on_balance_volume: self.calculate_obv(recent_prices, recent_volumes),
        }
    }
    
    fn calculate_vwap(&self, prices: &[f64], volumes: &[f64]) -> f64 {
        let total_volume: f64 = volumes.iter().sum();
        if total_volume == 0.0 {
            return prices.iter().sum::<f64>() / prices.len() as f64;
        }
        
        prices.iter()
            .zip(volumes.iter())
            .map(|(p, v)| p * v)
            .sum::<f64>() / total_volume
    }
    
    fn calculate_volume_profile(&self, prices: &[f64], volumes: &[f64]) -> Vec<(f64, f64)> {
        let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let bin_size = (max_price - min_price) / self.profile_bins as f64;
        
        let mut profile = vec![(0.0, 0.0); self.profile_bins];
        
        for (i, (&price, &volume)) in prices.iter().zip(volumes.iter()).enumerate() {
            let bin_index = ((price - min_price) / bin_size).floor() as usize;
            if bin_index < self.profile_bins {
                profile[bin_index].0 = min_price + (bin_index as f64 + 0.5) * bin_size;
                profile[bin_index].1 += volume;
            }
        }
        
        profile
    }
    
    fn calculate_obv(&self, prices: &[f64], volumes: &[f64]) -> f64 {
        let mut obv = 0.0;
        
        for i in 1..prices.len() {
            if prices[i] > prices[i-1] {
                obv += volumes[i];
            } else if prices[i] < prices[i-1] {
                obv -= volumes[i];
            }
        }
        
        obv
    }
}

#[derive(Debug, Clone, Default)]
pub struct VolumeMetrics {
    pub total_volume: f64,
    pub average_volume: f64,
    pub volume_weighted_price: f64,
    pub volume_profile: Vec<(f64, f64)>, // (price_level, volume)
    pub on_balance_volume: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_volume_observer() {
        let observer = VolumeObserver::new(5, 10);
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0];
        let volumes = vec![1000.0, 1200.0, 800.0, 1500.0, 900.0];
        
        let metrics = observer.observe(&prices, &volumes);
        assert!(metrics.total_volume > 0.0);
        assert!(metrics.volume_weighted_price > 0.0);
    }
}