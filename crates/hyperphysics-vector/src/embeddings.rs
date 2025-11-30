//! Embedding generation utilities for market data

use crate::error::{Result, VectorError};
use ndarray::{Array1, Array2};
use std::f32::consts::PI;

/// Embedding generator for market data
pub struct EmbeddingGenerator {
    dimensions: usize,
    normalization: Normalization,
}

/// Normalization method for embeddings
#[derive(Debug, Clone, Copy, Default)]
pub enum Normalization {
    /// L2 normalization (unit vector)
    #[default]
    L2,
    /// No normalization
    None,
    /// Min-max to [0, 1]
    MinMax,
    /// Z-score normalization
    ZScore,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            normalization: Normalization::L2,
        }
    }

    /// Set normalization method
    pub fn with_normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Generate embedding from price series
    pub fn embed_price_series(&self, prices: &[f64]) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimensions];

        if prices.is_empty() {
            return embedding;
        }

        // Basic statistical features
        let n = prices.len() as f64;
        let mean = prices.iter().sum::<f64>() / n;
        let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // Returns
        let returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0].max(1e-10))
            .collect();

        let return_mean = if !returns.is_empty() {
            returns.iter().sum::<f64>() / returns.len() as f64
        } else {
            0.0
        };

        let return_std = if returns.len() > 1 {
            let var = returns.iter().map(|r| (r - return_mean).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            var.sqrt()
        } else {
            0.0
        };

        // Trend features
        let trend = if prices.len() >= 2 {
            (prices.last().unwrap() - prices.first().unwrap())
                / prices.first().unwrap().max(1e-10)
        } else {
            0.0
        };

        // Fill basic features
        let num_basic = 8.min(self.dimensions);
        if num_basic >= 8 {
            embedding[0] = (mean / 10000.0) as f32; // Normalized price level
            embedding[1] = (std_dev / mean.max(1e-10)) as f32; // Coefficient of variation
            embedding[2] = (return_mean * 100.0) as f32; // Return in percent
            embedding[3] = (return_std * 100.0) as f32; // Volatility in percent
            embedding[4] = trend as f32;
            embedding[5] = (prices.len() as f64 / 1000.0) as f32; // Sequence length feature

            // Skewness approximation
            if !returns.is_empty() && return_std > 1e-10 {
                let skew = returns.iter()
                    .map(|r| ((r - return_mean) / return_std).powi(3))
                    .sum::<f64>() / returns.len() as f64;
                embedding[6] = skew as f32;
            }

            // Kurtosis approximation
            if !returns.is_empty() && return_std > 1e-10 {
                let kurt = returns.iter()
                    .map(|r| ((r - return_mean) / return_std).powi(4))
                    .sum::<f64>() / returns.len() as f64 - 3.0;
                embedding[7] = kurt as f32;
            }
        }

        // Autocorrelation features
        let autocorr_start = 8.min(self.dimensions);
        let num_autocorr = (self.dimensions - autocorr_start).min(8);
        for lag in 1..=num_autocorr {
            if autocorr_start + lag - 1 < self.dimensions && returns.len() > lag {
                let autocorr = self.autocorrelation(&returns, lag);
                embedding[autocorr_start + lag - 1] = autocorr as f32;
            }
        }

        // Fourier features for periodicity
        let fourier_start = autocorr_start + num_autocorr;
        let num_fourier = (self.dimensions - fourier_start).min(16);
        for k in 0..num_fourier / 2 {
            if fourier_start + k * 2 + 1 < self.dimensions {
                let (real, imag) = self.dft_component(prices, k + 1);
                embedding[fourier_start + k * 2] = real as f32;
                embedding[fourier_start + k * 2 + 1] = imag as f32;
            }
        }

        self.normalize(&mut embedding);
        embedding
    }

    /// Generate embedding from OHLCV candles
    pub fn embed_ohlcv(
        &self,
        opens: &[f64],
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
        volumes: &[f64],
    ) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimensions];

        if closes.is_empty() {
            return embedding;
        }

        let n = closes.len();

        // Price-based features
        let close_mean = closes.iter().sum::<f64>() / n as f64;
        let volume_mean = volumes.iter().sum::<f64>() / n as f64;

        // Candle body and shadow features
        let mut body_sizes = Vec::with_capacity(n);
        let mut upper_shadows = Vec::with_capacity(n);
        let mut lower_shadows = Vec::with_capacity(n);

        for i in 0..n {
            let body = (closes[i] - opens[i]).abs();
            let range = highs[i] - lows[i];
            body_sizes.push(body / range.max(1e-10));
            upper_shadows.push((highs[i] - closes[i].max(opens[i])) / range.max(1e-10));
            lower_shadows.push((closes[i].min(opens[i]) - lows[i]) / range.max(1e-10));
        }

        // VWAP
        let vwap = if volumes.iter().sum::<f64>() > 0.0 {
            closes.iter().zip(volumes.iter())
                .map(|(c, v)| c * v)
                .sum::<f64>() / volumes.iter().sum::<f64>()
        } else {
            close_mean
        };

        // Fill features
        let num_basic = 16.min(self.dimensions);
        if num_basic >= 16 {
            embedding[0] = (close_mean / 10000.0) as f32;
            embedding[1] = (vwap / close_mean.max(1e-10)) as f32;
            embedding[2] = (volume_mean.ln().max(0.0) / 20.0) as f32;

            // Body statistics
            embedding[3] = (body_sizes.iter().sum::<f64>() / n as f64) as f32;
            embedding[4] = (upper_shadows.iter().sum::<f64>() / n as f64) as f32;
            embedding[5] = (lower_shadows.iter().sum::<f64>() / n as f64) as f32;

            // Volume trend
            if n >= 2 {
                let vol_trend = (volumes.last().unwrap() - volumes.first().unwrap())
                    / volumes.first().unwrap().max(1e-10);
                embedding[6] = vol_trend as f32;
            }

            // Price momentum
            let returns: Vec<f64> = closes.windows(2)
                .map(|w| (w[1] - w[0]) / w[0].max(1e-10))
                .collect();

            if !returns.is_empty() {
                embedding[7] = (returns.iter().sum::<f64>() / returns.len() as f64 * 100.0) as f32;
            }

            // Range features
            let avg_range = highs.iter().zip(lows.iter())
                .map(|(h, l)| h - l)
                .sum::<f64>() / n as f64;
            embedding[8] = (avg_range / close_mean.max(1e-10) * 100.0) as f32;

            // Direction counts
            let up_days = returns.iter().filter(|&&r| r > 0.0).count();
            embedding[9] = (up_days as f64 / returns.len().max(1) as f64) as f32;

            // Volume profile
            if n >= 5 {
                for i in 0..5.min(n) {
                    let idx = n - 1 - i;
                    if 10 + i < num_basic {
                        embedding[10 + i] = (volumes[idx].ln().max(0.0) / 20.0) as f32;
                    }
                }
            }
        }

        // Add close price embedding
        let close_embedding = self.embed_price_series(closes);
        let remaining = self.dimensions.saturating_sub(num_basic);
        for i in 0..remaining.min(close_embedding.len()) {
            if num_basic + i < self.dimensions {
                embedding[num_basic + i] = close_embedding[i];
            }
        }

        self.normalize(&mut embedding);
        embedding
    }

    /// Calculate autocorrelation at given lag
    fn autocorrelation(&self, series: &[f64], lag: usize) -> f64 {
        if series.len() <= lag {
            return 0.0;
        }

        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let var: f64 = series.iter().map(|x| (x - mean).powi(2)).sum();

        if var.abs() < 1e-10 {
            return 0.0;
        }

        let cov: f64 = series[..series.len() - lag]
            .iter()
            .zip(&series[lag..])
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum();

        cov / var
    }

    /// Calculate single DFT component
    fn dft_component(&self, series: &[f64], k: usize) -> (f64, f64) {
        let n = series.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        let mut real = 0.0;
        let mut imag = 0.0;

        for (i, &x) in series.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            real += x * angle.cos();
            imag -= x * angle.sin();
        }

        // Normalize by length and scale
        let scale = 1.0 / (n as f64).sqrt();
        (real * scale / 1000.0, imag * scale / 1000.0)
    }

    /// Normalize embedding vector
    fn normalize(&self, embedding: &mut [f32]) {
        match self.normalization {
            Normalization::L2 => {
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for val in embedding.iter_mut() {
                        *val /= norm;
                    }
                }
            }
            Normalization::MinMax => {
                let min = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;
                if range > 1e-10 {
                    for val in embedding.iter_mut() {
                        *val = (*val - min) / range;
                    }
                }
            }
            Normalization::ZScore => {
                let mean = embedding.iter().sum::<f32>() / embedding.len() as f32;
                let var = embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / embedding.len() as f32;
                let std = var.sqrt();
                if std > 1e-10 {
                    for val in embedding.iter_mut() {
                        *val = (*val - mean) / std;
                    }
                }
            }
            Normalization::None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_embedding() {
        let generator = EmbeddingGenerator::new(64);
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();

        let embedding = generator.embed_price_series(&prices);

        assert_eq!(embedding.len(), 64);

        // Check L2 normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01 || embedding.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ohlcv_embedding() {
        let generator = EmbeddingGenerator::new(128);

        let n = 50;
        let opens: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();
        let highs: Vec<f64> = opens.iter().map(|o| o + 1.0).collect();
        let lows: Vec<f64> = opens.iter().map(|o| o - 0.5).collect();
        let closes: Vec<f64> = opens.iter().map(|o| o + 0.5).collect();
        let volumes: Vec<f64> = (0..n).map(|_| 1000.0).collect();

        let embedding = generator.embed_ohlcv(&opens, &highs, &lows, &closes, &volumes);

        assert_eq!(embedding.len(), 128);
    }

    #[test]
    fn test_empty_input() {
        let generator = EmbeddingGenerator::new(32);

        let empty: Vec<f64> = vec![];
        let embedding = generator.embed_price_series(&empty);

        assert_eq!(embedding.len(), 32);
        assert!(embedding.iter().all(|&x| x == 0.0));
    }
}
