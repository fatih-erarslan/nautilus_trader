//! Time series analysis utilities

use crate::error::{TalebianResult as Result, TalebianError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value
    pub value: f64,
}

/// Time series data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Series identifier
    pub id: String,
    /// Data points
    pub data: Vec<TimeSeriesPoint>,
    /// Metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl TimeSeries {
    /// Create a new time series
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            data: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Add a data point
    pub fn add_point(&mut self, timestamp: DateTime<Utc>, value: f64) {
        self.data.push(TimeSeriesPoint { timestamp, value });
        
        // Keep data sorted by timestamp
        self.data.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    }
    
    /// Get values as a vector
    pub fn values(&self) -> Vec<f64> {
        self.data.iter().map(|p| p.value).collect()
    }
    
    /// Get timestamps
    pub fn timestamps(&self) -> Vec<DateTime<Utc>> {
        self.data.iter().map(|p| p.timestamp).collect()
    }
    
    /// Calculate simple statistics
    pub fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            None
        } else {
            Some(self.data.iter().map(|p| p.value).sum::<f64>() / self.data.len() as f64)
        }
    }
    
    /// Calculate standard deviation
    pub fn std_dev(&self) -> Option<f64> {
        if self.data.len() < 2 {
            None
        } else {
            let mean = self.mean()?;
            let variance = self.data.iter()
                .map(|p| (p.value - mean).powi(2))
                .sum::<f64>() / (self.data.len() - 1) as f64;
            Some(variance.sqrt())
        }
    }
    
    /// Calculate returns
    pub fn returns(&self) -> Result<Vec<f64>> {
        if self.data.len() < 2 {
            return Err(TalebianError::insufficient_data(2, self.data.len()));
        }
        
        let mut returns = Vec::new();
        for i in 1..self.data.len() {
            let prev_value = self.data[i-1].value;
            let curr_value = self.data[i].value;
            
            if prev_value <= 0.0 {
                return Err(TalebianError::mathematical("Cannot calculate returns with non-positive values"));
            }
            
            returns.push((curr_value - prev_value) / prev_value);
        }
        
        Ok(returns)
    }
    
    /// Get data within a time range
    pub fn slice(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> TimeSeries {
        let filtered_data: Vec<TimeSeriesPoint> = self.data.iter()
            .filter(|p| p.timestamp >= start && p.timestamp <= end)
            .cloned()
            .collect();
        
        TimeSeries {
            id: format!("{}_slice", self.id),
            data: filtered_data,
            metadata: self.metadata.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    
    #[test]
    fn test_time_series_creation() {
        let ts = TimeSeries::new("test_series");
        assert_eq!(ts.id, "test_series");
        assert!(ts.data.is_empty());
    }
    
    #[test]
    fn test_add_points() {
        let mut ts = TimeSeries::new("test");
        let now = Utc::now();
        
        ts.add_point(now, 100.0);
        ts.add_point(now + Duration::days(1), 105.0);
        
        assert_eq!(ts.data.len(), 2);
        assert_eq!(ts.values(), vec![100.0, 105.0]);
    }
    
    #[test]
    fn test_statistics() {
        let mut ts = TimeSeries::new("test");
        let now = Utc::now();
        
        ts.add_point(now, 100.0);
        ts.add_point(now + Duration::days(1), 102.0);
        ts.add_point(now + Duration::days(2), 98.0);
        
        assert_eq!(ts.mean(), Some(100.0));
        assert!(ts.std_dev().is_some());
        assert!(ts.std_dev().unwrap() > 0.0);
    }
    
    #[test]
    fn test_returns_calculation() {
        let mut ts = TimeSeries::new("test");
        let now = Utc::now();
        
        ts.add_point(now, 100.0);
        ts.add_point(now + Duration::days(1), 105.0);
        ts.add_point(now + Duration::days(2), 102.0);
        
        let returns = ts.returns().unwrap();
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.05).abs() < 1e-10); // 5% return
        assert!((returns[1] - (-105.0 + 102.0) / 105.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_time_slice() {
        let mut ts = TimeSeries::new("test");
        let start = Utc::now();
        
        for i in 0..10 {
            ts.add_point(start + Duration::days(i), i as f64);
        }
        
        let slice = ts.slice(start + Duration::days(2), start + Duration::days(5));
        assert_eq!(slice.data.len(), 4); // Days 2, 3, 4, 5
        assert_eq!(slice.values(), vec![2.0, 3.0, 4.0, 5.0]);
    }
}