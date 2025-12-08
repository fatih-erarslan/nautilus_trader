use crate::*;

pub struct RiskAssessor {
    volatility_threshold: f64,
    volume_threshold: f64,
    correlation_threshold: f64,
}

impl RiskAssessor {
    pub fn new() -> Self {
        Self {
            volatility_threshold: 0.15,
            volume_threshold: 0.5,
            correlation_threshold: 0.8,
        }
    }
    
    pub async fn calculate_risk_score(
        &self,
        _symbol: &str,
        trend: &TrendScore,
        context: &MarketContext,
    ) -> Result<f64, IntelligenceError> {
        let mut risk_components = vec![];
        
        // Volatility risk
        let vol_risk = self.calculate_volatility_risk(trend.volatility);
        risk_components.push(vol_risk * 0.3);
        
        // Liquidity risk
        let liquidity_risk = self.calculate_liquidity_risk(context.liquidity_score);
        risk_components.push(liquidity_risk * 0.25);
        
        // Market regime risk
        let regime_risk = self.calculate_regime_risk(&context.market_regime);
        risk_components.push(regime_risk * 0.2);
        
        // Trend quality risk
        let trend_risk = self.calculate_trend_risk(trend);
        risk_components.push(trend_risk * 0.15);
        
        // Correlation risk
        let correlation_risk = self.calculate_correlation_risk(&context.correlation_cluster);
        risk_components.push(correlation_risk * 0.1);
        
        Ok(risk_components.iter().sum::<f64>().min(1.0))
    }
    
    fn calculate_volatility_risk(&self, volatility: f64) -> f64 {
        if volatility < 0.02 {
            0.2 // Low volatility can indicate low liquidity
        } else if volatility < 0.05 {
            0.1 // Optimal range
        } else if volatility < 0.1 {
            0.3
        } else if volatility < 0.2 {
            0.6
        } else {
            0.9 // Very high volatility = high risk
        }
    }
    
    fn calculate_liquidity_risk(&self, liquidity_score: f64) -> f64 {
        1.0 - liquidity_score // Higher liquidity = lower risk
    }
    
    fn calculate_regime_risk(&self, regime: &str) -> f64 {
        match regime {
            "trending" => 0.2,
            "ranging" => 0.4,
            "volatile" => 0.8,
            "breakout" => 0.6,
            _ => 0.5,
        }
    }
    
    fn calculate_trend_risk(&self, trend: &TrendScore) -> f64 {
        let confidence_risk = 1.0 - trend.confidence;
        let momentum_risk = if trend.momentum_score.abs() < 0.1 { 0.7 } else { 0.3 };
        
        (confidence_risk + momentum_risk) / 2.0
    }
    
    fn calculate_correlation_risk(&self, _cluster: &str) -> f64 {
        0.5 // Simplified correlation risk
    }
}