//! Cryptocurrency market forecasting using consciousness-aware NHITS
//! 
//! This module provides specialized forecasting capabilities for cryptocurrency
//! markets, accounting for their unique characteristics such as 24/7 trading,
//! extreme volatility, and sentiment-driven price movements.

use super::*;
use ndarray::{Array1, Array2, s};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Cryptocurrency forecasting engine
#[derive(Debug)]
pub struct CryptoForecaster {
    pub base_predictor: super::price_prediction::PricePredictor,
    pub volatility_predictor: super::volatility_modeling::VolatilityPredictor,
    pub market_regime_detector: super::market_regime::MarketRegimeDetector,
    pub crypto_specific_features: CryptoFeatureExtractor,
    pub consciousness_crypto_adaptation: f32,
    pub sentiment_analyzer: CryptoSentimentAnalyzer,
    pub on_chain_analyzer: OnChainAnalyzer,
}

#[derive(Debug)]
pub struct CryptoFeatureExtractor {
    pub fear_greed_index: f32,
    pub social_sentiment: f32,
    pub institutional_flow: f32,
    pub defi_tvl_change: f32,
    pub network_activity: f32,
    pub mining_difficulty: f32,
    pub exchange_flows: f32,
    pub stablecoin_dominance: f32,
}

#[derive(Debug)]
pub struct CryptoSentimentAnalyzer {
    pub twitter_sentiment: f32,
    pub reddit_sentiment: f32,
    pub telegram_sentiment: f32,
    pub news_sentiment: f32,
    pub influencer_sentiment: f32,
    pub whale_sentiment: f32,
}

#[derive(Debug)]
pub struct OnChainAnalyzer {
    pub active_addresses: f32,
    pub transaction_volume: f32,
    pub network_hash_rate: f32,
    pub exchange_reserves: f32,
    pub whale_movements: f32,
    pub long_term_holder_behavior: f32,
    pub realized_cap: f32,
    pub mvrv_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoForecast {
    pub symbol: String,
    pub price_predictions: Vec<f32>,
    pub volatility_predictions: Vec<f32>,
    pub sentiment_trend: f32,
    pub fear_greed_forecast: f32,
    pub market_regime: super::market_regime::MarketRegime,
    pub on_chain_signals: Vec<String>,
    pub consciousness_crypto_factor: f32,
    pub prediction_confidence: f32,
    pub support_levels: Vec<f32>,
    pub resistance_levels: Vec<f32>,
    pub forecast_timestamp: i64,
    pub forecast_horizon_hours: u32,
}

#[derive(Debug, Clone)]
pub struct CryptoTradingSignal {
    pub base_signal: TradingSignal,
    pub crypto_specific_factors: CryptoPriceFactors,
    pub social_momentum: f32,
    pub on_chain_strength: f32,
    pub fear_greed_adjustment: f32,
    pub defi_correlation: f32,
}

#[derive(Debug, Clone)]
pub struct CryptoPriceFactors {
    pub btc_dominance_impact: f32,
    pub eth_correlation: f32,
    pub stablecoin_flow: f32,
    pub institutional_adoption: f32,
    pub regulatory_sentiment: f32,
    pub technical_momentum: f32,
    pub mining_economics: f32,
}

impl CryptoForecaster {
    pub fn new(lookback_window: usize, forecast_horizon: usize) -> Self {
        Self {
            base_predictor: super::price_prediction::PricePredictor::new(lookback_window, forecast_horizon),
            volatility_predictor: super::volatility_modeling::VolatilityPredictor::new(
                15, lookback_window, forecast_horizon, 
                super::volatility_modeling::VolatilityType::GARCH
            ),
            market_regime_detector: super::market_regime::MarketRegimeDetector::new(15, lookback_window, forecast_horizon),
            crypto_specific_features: CryptoFeatureExtractor {
                fear_greed_index: 50.0,
                social_sentiment: 0.0,
                institutional_flow: 0.0,
                defi_tvl_change: 0.0,
                network_activity: 1.0,
                mining_difficulty: 1.0,
                exchange_flows: 0.0,
                stablecoin_dominance: 0.1,
            },
            consciousness_crypto_adaptation: 1.2,  // Crypto markets are more consciousness-sensitive
            sentiment_analyzer: CryptoSentimentAnalyzer {
                twitter_sentiment: 0.0,
                reddit_sentiment: 0.0,
                telegram_sentiment: 0.0,
                news_sentiment: 0.0,
                influencer_sentiment: 0.0,
                whale_sentiment: 0.0,
            },
            on_chain_analyzer: OnChainAnalyzer {
                active_addresses: 1.0,
                transaction_volume: 1.0,
                network_hash_rate: 1.0,
                exchange_reserves: 1.0,
                whale_movements: 0.0,
                long_term_holder_behavior: 1.0,
                realized_cap: 1.0,
                mvrv_ratio: 1.0,
            },
        }
    }
    
    /// Add cryptocurrency for forecasting
    pub fn add_crypto(&mut self, symbol: String) {
        // Crypto-specific feature dimension (extended from traditional assets)
        let feature_dim = 20;  // More features for crypto
        self.base_predictor.add_asset(symbol, feature_dim);
    }
    
    /// Generate comprehensive crypto forecast
    pub fn forecast_crypto(
        &mut self,
        symbol: &str,
        market_data: &Array2<f32>,
        social_data: Option<&SocialMediaData>,
        on_chain_data: Option<&OnChainData>,
    ) -> Result<CryptoForecast, String> {
        // Update crypto-specific features
        if let Some(social) = social_data {
            self.update_sentiment_features(social);
        }
        
        if let Some(on_chain) = on_chain_data {
            self.update_on_chain_features(on_chain);
        }
        
        // Enhance market data with crypto-specific features
        let enhanced_data = self.enhance_market_data_with_crypto_features(market_data)?;
        
        // Get base predictions
        let price_prediction_result = self.base_predictor.predict_single(symbol, &enhanced_data)?;
        let volatility_forecast = self.volatility_predictor.predict_volatility(&utils::calculate_returns(
            &enhanced_data.slice(s![.., 3]).to_vec()
        ))?;
        let regime_observation = self.market_regime_detector.detect_current_regime(&enhanced_data)?;
        
        // Calculate crypto-specific consciousness factor
        let crypto_consciousness = self.calculate_crypto_consciousness(&enhanced_data, social_data, on_chain_data);
        
        // Adjust predictions for crypto market characteristics
        let adjusted_price_predictions = self.adjust_predictions_for_crypto(
            &price_prediction_result.predicted_prices,
            crypto_consciousness
        );
        
        // Calculate support and resistance levels
        let (support_levels, resistance_levels) = self.calculate_crypto_support_resistance(&enhanced_data);
        
        // Analyze on-chain signals
        let on_chain_signals = self.generate_on_chain_signals(on_chain_data);
        
        // Calculate sentiment trend
        let sentiment_trend = self.calculate_sentiment_trend(social_data);
        
        // Forecast Fear & Greed Index
        let fear_greed_forecast = self.forecast_fear_greed_index();
        
        Ok(CryptoForecast {
            symbol: symbol.to_string(),
            price_predictions: adjusted_price_predictions,
            volatility_predictions: volatility_forecast.volatility_predictions,
            sentiment_trend,
            fear_greed_forecast,
            market_regime: regime_observation.current_regime,
            on_chain_signals,
            consciousness_crypto_factor: crypto_consciousness,
            prediction_confidence: price_prediction_result.consciousness_state * crypto_consciousness,
            support_levels,
            resistance_levels,
            forecast_timestamp: chrono::Utc::now().timestamp(),
            forecast_horizon_hours: 24,  // 24-hour forecast typical for crypto
        })
    }
    
    /// Generate crypto-specific trading signals
    pub fn generate_crypto_trading_signals(
        &mut self,
        market_data: &HashMap<String, Array2<f32>>,
        social_data: &HashMap<String, SocialMediaData>,
        on_chain_data: &HashMap<String, OnChainData>,
    ) -> Vec<CryptoTradingSignal> {
        let mut crypto_signals = Vec::new();
        
        for (symbol, data) in market_data {
            let social = social_data.get(symbol);
            let on_chain = on_chain_data.get(symbol);
            
            if let Ok(forecast) = self.forecast_crypto(symbol, data, social, on_chain) {
                if let Some(trading_signal) = self.convert_forecast_to_trading_signal(&forecast, data) {
                    let crypto_factors = self.calculate_crypto_price_factors(symbol, social, on_chain);
                    
                    let crypto_signal = CryptoTradingSignal {
                        base_signal: trading_signal,
                        crypto_specific_factors: crypto_factors,
                        social_momentum: self.calculate_social_momentum(social),
                        on_chain_strength: self.calculate_on_chain_strength(on_chain),
                        fear_greed_adjustment: self.calculate_fear_greed_adjustment(),
                        defi_correlation: self.calculate_defi_correlation(symbol),
                    };
                    
                    crypto_signals.push(crypto_signal);
                }
            }
        }
        
        crypto_signals
    }
    
    /// Analyze crypto market cycles
    pub fn analyze_crypto_cycle(&self, btc_data: &Array2<f32>) -> CryptoCycleAnalysis {
        let prices = btc_data.slice(s![.., 3]).to_vec();
        
        // Calculate moving averages for cycle analysis
        let ma_200 = self.calculate_moving_average(&prices, 200);
        let ma_50 = self.calculate_moving_average(&prices, 50);
        let ma_20 = self.calculate_moving_average(&prices, 20);
        
        let current_price = prices[prices.len() - 1];
        let ma_200_current = ma_200.last().copied().unwrap_or(current_price);
        let ma_50_current = ma_50.last().copied().unwrap_or(current_price);
        
        // Determine cycle phase
        let cycle_phase = if current_price > ma_200_current && ma_50_current > ma_200_current {
            if self.crypto_specific_features.fear_greed_index > 75.0 {
                CryptoCyclePhase::LateUserophia
            } else {
                CryptoCyclePhase::Bull
            }
        } else if current_price < ma_200_current && ma_50_current < ma_200_current {
            if self.crypto_specific_features.fear_greed_index < 25.0 {
                CryptoCyclePhase::Capitulation
            } else {
                CryptoCyclePhase::Bear
            }
        } else {
            CryptoCyclePhase::Transition
        };
        
        // Calculate cycle metrics
        let cycle_length_estimate = self.estimate_cycle_length(&prices);
        let cycle_progress = self.calculate_cycle_progress(&cycle_phase, &prices);
        
        CryptoCycleAnalysis {
            current_phase: cycle_phase,
            cycle_progress,
            estimated_cycle_length: cycle_length_estimate,
            bull_market_probability: self.calculate_bull_market_probability(&prices),
            bear_market_probability: self.calculate_bear_market_probability(&prices),
            cycle_top_indicator: self.calculate_cycle_top_indicator(&prices),
            cycle_bottom_indicator: self.calculate_cycle_bottom_indicator(&prices),
        }
    }
    
    /// DeFi-specific forecasting
    pub fn forecast_defi_token(
        &mut self,
        symbol: &str,
        market_data: &Array2<f32>,
        defi_metrics: &DeFiMetrics,
    ) -> Result<DeFiForecast, String> {
        // Base crypto forecast
        let base_forecast = self.forecast_crypto(symbol, market_data, None, None)?;
        
        // DeFi-specific adjustments
        let tvl_impact = self.calculate_tvl_impact(defi_metrics);
        let yield_farming_impact = self.calculate_yield_farming_impact(defi_metrics);
        let governance_impact = self.calculate_governance_impact(defi_metrics);
        
        // Adjust predictions
        let defi_adjusted_prices = base_forecast.price_predictions.iter()
            .map(|&price| {
                let defi_multiplier = 1.0 + (tvl_impact + yield_farming_impact + governance_impact) / 3.0;
                price * defi_multiplier
            })
            .collect();
        
        Ok(DeFiForecast {
            base_forecast,
            defi_adjusted_prices,
            tvl_impact_factor: tvl_impact,
            yield_farming_appeal: yield_farming_impact,
            governance_activity: governance_impact,
            protocol_health_score: self.calculate_protocol_health(defi_metrics),
            impermanent_loss_risk: self.calculate_impermanent_loss_risk(defi_metrics),
        })
    }
    
    /// NFT market forecasting
    pub fn forecast_nft_market(&self, nft_data: &NFTMarketData) -> NFTMarketForecast {
        // Analyze NFT market trends
        let volume_trend = self.calculate_nft_volume_trend(&nft_data.daily_volumes);
        let floor_price_trend = self.calculate_nft_floor_price_trend(&nft_data.floor_prices);
        let rarity_premium = self.calculate_rarity_premium(nft_data);
        
        // Celebrity/influencer impact
        let celebrity_impact = self.calculate_celebrity_impact(&nft_data.celebrity_activities);
        
        // Market consciousness for NFTs (highly sentiment-driven)
        let nft_consciousness = self.calculate_nft_market_consciousness(nft_data);
        
        NFTMarketForecast {
            volume_trend_7d: volume_trend,
            floor_price_trend_7d: floor_price_trend,
            rarity_premium_forecast: rarity_premium,
            celebrity_impact_score: celebrity_impact,
            market_consciousness: nft_consciousness,
            recommended_collection_allocation: self.recommend_nft_allocation(nft_data),
        }
    }
    
    // Private helper methods
    
    fn enhance_market_data_with_crypto_features(&self, market_data: &Array2<f32>) -> Result<Array2<f32>, String> {
        let (rows, cols) = market_data.dim();
        let mut enhanced_data = Array2::zeros((rows, cols + 10));  // Add 10 crypto-specific features
        
        // Copy original data
        for i in 0..rows {
            for j in 0..cols {
                enhanced_data[[i, j]] = market_data[[i, j]];
            }
        }
        
        // Add crypto-specific features
        for i in 0..rows {
            enhanced_data[[i, cols]] = self.crypto_specific_features.fear_greed_index / 100.0;
            enhanced_data[[i, cols + 1]] = self.crypto_specific_features.social_sentiment;
            enhanced_data[[i, cols + 2]] = self.crypto_specific_features.institutional_flow;
            enhanced_data[[i, cols + 3]] = self.crypto_specific_features.defi_tvl_change;
            enhanced_data[[i, cols + 4]] = self.crypto_specific_features.network_activity;
            enhanced_data[[i, cols + 5]] = self.crypto_specific_features.mining_difficulty;
            enhanced_data[[i, cols + 6]] = self.crypto_specific_features.exchange_flows;
            enhanced_data[[i, cols + 7]] = self.crypto_specific_features.stablecoin_dominance;
            enhanced_data[[i, cols + 8]] = self.sentiment_analyzer.twitter_sentiment;
            enhanced_data[[i, cols + 9]] = self.on_chain_analyzer.whale_movements;
        }
        
        Ok(enhanced_data)
    }
    
    fn calculate_crypto_consciousness(
        &self,
        market_data: &Array2<f32>,
        social_data: Option<&SocialMediaData>,
        on_chain_data: Option<&OnChainData>,
    ) -> f32 {
        // Base market consciousness
        let returns = market_data.slice(s![.., 5]).to_vec();
        let base_consciousness = self.calculate_market_consciousness(&returns);
        
        // Social media consciousness factor
        let social_consciousness = if let Some(social) = social_data {
            let sentiment_coherence = (social.twitter_sentiment_score.abs() + 
                                     social.reddit_sentiment_score.abs()) / 2.0;
            sentiment_coherence
        } else {
            0.5
        };
        
        // On-chain consciousness factor
        let on_chain_consciousness = if let Some(on_chain) = on_chain_data {
            // Higher on-chain activity = higher consciousness
            let activity_coherence = (on_chain.active_addresses_change + 
                                    on_chain.transaction_volume_change.abs()) / 2.0;
            activity_coherence.min(1.0).max(0.0)
        } else {
            0.5
        };
        
        // Combine all consciousness factors
        let combined_consciousness = (base_consciousness + social_consciousness + on_chain_consciousness) / 3.0;
        
        // Apply crypto-specific amplification
        combined_consciousness * self.consciousness_crypto_adaptation
    }
    
    fn calculate_market_consciousness(&self, returns: &[f32]) -> f32 {
        if returns.len() < 10 {
            return 0.5;
        }
        
        let volatility = {
            let mean = returns.iter().sum::<f32>() / returns.len() as f32;
            let variance = returns.iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f32>() / returns.len() as f32;
            variance.sqrt()
        };
        
        let trend_consistency = self.calculate_trend_consistency(returns);
        
        // For crypto: higher volatility can indicate higher consciousness due to emotional trading
        let volatility_consciousness = (volatility * 5.0).min(1.0);
        (trend_consistency + volatility_consciousness) / 2.0
    }
    
    fn calculate_trend_consistency(&self, returns: &[f32]) -> f32 {
        if returns.len() < 2 {
            return 0.5;
        }
        
        let mut consistent_periods = 0;
        for i in 1..returns.len() {
            if (returns[i] > 0.0) == (returns[i-1] > 0.0) {
                consistent_periods += 1;
            }
        }
        
        consistent_periods as f32 / (returns.len() - 1) as f32
    }
    
    fn adjust_predictions_for_crypto(&self, predictions: &[f32], crypto_consciousness: f32) -> Vec<f32> {
        predictions.iter()
            .enumerate()
            .map(|(i, &pred)| {
                // Crypto markets are more volatile and consciousness-sensitive
                let time_decay = 1.0 - (i as f32 * 0.05);  // Prediction quality decreases over time
                let crypto_volatility_factor = 1.0 + crypto_consciousness * 0.3;  // Higher consciousness = more volatility
                
                pred * crypto_volatility_factor * time_decay
            })
            .collect()
    }
    
    fn calculate_crypto_support_resistance(&self, data: &Array2<f32>) -> (Vec<f32>, Vec<f32>) {
        let prices = data.slice(s![.., 3]).to_vec();
        let highs = data.slice(s![.., 1]).to_vec();
        let lows = data.slice(s![.., 2]).to_vec();
        
        // Calculate support levels (recent lows)
        let mut support_levels = Vec::new();
        let mut resistance_levels = Vec::new();
        
        if prices.len() >= 20 {
            // Recent lows as support
            let recent_lows = &lows[lows.len() - 20..];
            let mut sorted_lows = recent_lows.to_vec();
            sorted_lows.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            support_levels.push(sorted_lows[0]);  // Strongest support
            support_levels.push(sorted_lows[sorted_lows.len() / 4]);  // Secondary support
            
            // Recent highs as resistance
            let recent_highs = &highs[highs.len() - 20..];
            let mut sorted_highs = recent_highs.to_vec();
            sorted_highs.sort_by(|a, b| b.partial_cmp(a).unwrap());
            
            resistance_levels.push(sorted_highs[0]);  // Strongest resistance
            resistance_levels.push(sorted_highs[sorted_highs.len() / 4]);  // Secondary resistance
        }
        
        (support_levels, resistance_levels)
    }
    
    fn generate_on_chain_signals(&self, on_chain_data: Option<&OnChainData>) -> Vec<String> {
        let mut signals = Vec::new();
        
        if let Some(data) = on_chain_data {
            if data.exchange_inflow > 1.5 {
                signals.push("High exchange inflow - potential sell pressure".to_string());
            }
            
            if data.exchange_outflow > 1.5 {
                signals.push("High exchange outflow - potential accumulation".to_string());
            }
            
            if data.whale_transactions > 2.0 {
                signals.push("Increased whale activity detected".to_string());
            }
            
            if data.active_addresses_change > 0.2 {
                signals.push("Growing network adoption".to_string());
            }
            
            if data.long_term_holder_supply_change < -0.1 {
                signals.push("Long-term holders distributing".to_string());
            }
        }
        
        signals
    }
    
    fn calculate_sentiment_trend(&self, social_data: Option<&SocialMediaData>) -> f32 {
        if let Some(data) = social_data {
            let combined_sentiment = (data.twitter_sentiment_score + 
                                    data.reddit_sentiment_score + 
                                    data.telegram_sentiment_score) / 3.0;
            
            // Calculate trend from sentiment history
            if data.sentiment_history.len() >= 2 {
                let recent_avg = data.sentiment_history[data.sentiment_history.len() - 3..]
                    .iter().sum::<f32>() / 3.0;
                let older_avg = data.sentiment_history[data.sentiment_history.len() - 7..data.sentiment_history.len() - 3]
                    .iter().sum::<f32>() / 4.0;
                
                recent_avg - older_avg
            } else {
                combined_sentiment
            }
        } else {
            0.0
        }
    }
    
    fn forecast_fear_greed_index(&self) -> f32 {
        // Simple momentum-based forecast
        let current_fgi = self.crypto_specific_features.fear_greed_index;
        let momentum = self.crypto_specific_features.social_sentiment * 10.0;
        
        (current_fgi + momentum).max(0.0).min(100.0)
    }
    
    fn update_sentiment_features(&mut self, social_data: &SocialMediaData) {
        self.sentiment_analyzer.twitter_sentiment = social_data.twitter_sentiment_score;
        self.sentiment_analyzer.reddit_sentiment = social_data.reddit_sentiment_score;
        self.sentiment_analyzer.telegram_sentiment = social_data.telegram_sentiment_score;
        self.sentiment_analyzer.news_sentiment = social_data.news_sentiment_score;
        
        // Update aggregated social sentiment
        self.crypto_specific_features.social_sentiment = (
            social_data.twitter_sentiment_score + 
            social_data.reddit_sentiment_score + 
            social_data.telegram_sentiment_score
        ) / 3.0;
    }
    
    fn update_on_chain_features(&mut self, on_chain_data: &OnChainData) {
        self.on_chain_analyzer.active_addresses = on_chain_data.active_addresses_change;
        self.on_chain_analyzer.transaction_volume = on_chain_data.transaction_volume_change;
        self.on_chain_analyzer.whale_movements = on_chain_data.whale_transactions;
        self.on_chain_analyzer.exchange_reserves = on_chain_data.exchange_reserves_change;
        
        // Update network activity indicator
        self.crypto_specific_features.network_activity = (
            on_chain_data.active_addresses_change + 
            on_chain_data.transaction_volume_change
        ) / 2.0;
    }
    
    fn convert_forecast_to_trading_signal(&self, forecast: &CryptoForecast, market_data: &Array2<f32>) -> Option<TradingSignal> {
        let current_price = market_data.slice(s![market_data.nrows() - 1, 3])[0];
        let predicted_price = forecast.price_predictions.get(0).copied()?;
        
        let price_change = (predicted_price - current_price) / current_price;
        
        if price_change.abs() > 0.05 {  // 5% threshold for crypto signals
            let signal_type = if price_change > 0.0 {
                if price_change > 0.15 { SignalType::StrongBuy } else { SignalType::Buy }
            } else {
                if price_change < -0.15 { SignalType::StrongSell } else { SignalType::Sell }
            };
            
            Some(TradingSignal {
                symbol: forecast.symbol.clone(),
                signal_type,
                strength: (price_change.abs() / 0.05).min(3.0),
                confidence: forecast.prediction_confidence,
                target_price: Some(predicted_price),
                stop_loss: Some(current_price * if price_change > 0.0 { 0.95 } else { 1.05 }),
                take_profit: Some(predicted_price),
                consciousness_factor: forecast.consciousness_crypto_factor,
                strategy_name: "CryptoForecast".to_string(),
                timestamp: chrono::Utc::now().timestamp(),
                risk_score: 1.0 - forecast.prediction_confidence,
            })
        } else {
            None
        }
    }
    
    fn calculate_crypto_price_factors(&self, symbol: &str, social_data: Option<&SocialMediaData>, on_chain_data: Option<&OnChainData>) -> CryptoPriceFactors {
        CryptoPriceFactors {
            btc_dominance_impact: self.calculate_btc_dominance_impact(symbol),
            eth_correlation: self.calculate_eth_correlation(symbol),
            stablecoin_flow: self.crypto_specific_features.stablecoin_dominance,
            institutional_adoption: self.crypto_specific_features.institutional_flow,
            regulatory_sentiment: self.sentiment_analyzer.news_sentiment * 0.5,  // News often regulatory
            technical_momentum: self.calculate_technical_momentum(symbol),
            mining_economics: self.crypto_specific_features.mining_difficulty,
        }
    }
    
    fn calculate_social_momentum(&self, social_data: Option<&SocialMediaData>) -> f32 {
        if let Some(data) = social_data {
            let volume_momentum = data.social_volume_change;
            let sentiment_momentum = data.twitter_sentiment_score * 0.5 + data.reddit_sentiment_score * 0.3 + data.telegram_sentiment_score * 0.2;
            
            (volume_momentum + sentiment_momentum) / 2.0
        } else {
            0.0
        }
    }
    
    fn calculate_on_chain_strength(&self, on_chain_data: Option<&OnChainData>) -> f32 {
        if let Some(data) = on_chain_data {
            let network_strength = (data.active_addresses_change + data.transaction_volume_change) / 2.0;
            let holder_strength = data.long_term_holder_supply_change * -1.0;  // Negative change is positive (accumulation)
            let whale_influence = data.whale_transactions.min(1.0);  // Cap at 1.0
            
            (network_strength + holder_strength + whale_influence) / 3.0
        } else {
            0.0
        }
    }
    
    fn calculate_fear_greed_adjustment(&self) -> f32 {
        let fgi = self.crypto_specific_features.fear_greed_index;
        
        // Contrarian adjustment: extreme fear = bullish, extreme greed = bearish
        if fgi < 25.0 {
            (25.0 - fgi) / 25.0  // Positive adjustment for extreme fear
        } else if fgi > 75.0 {
            (75.0 - fgi) / 25.0  // Negative adjustment for extreme greed
        } else {
            0.0
        }
    }
    
    fn calculate_defi_correlation(&self, symbol: &str) -> f32 {
        // Simplified DeFi correlation based on token type
        if symbol.contains("UNI") || symbol.contains("SUSHI") || symbol.contains("COMP") {
            0.8  // High DeFi correlation
        } else if symbol.contains("ETH") {
            0.6  // Medium DeFi correlation
        } else {
            0.2  // Low DeFi correlation
        }
    }
    
    // Additional helper methods for specific calculations...
    
    fn calculate_moving_average(&self, prices: &[f32], period: usize) -> Vec<f32> {
        if prices.len() < period {
            return vec![prices.iter().sum::<f32>() / prices.len() as f32];
        }
        
        prices.windows(period)
            .map(|window| window.iter().sum::<f32>() / period as f32)
            .collect()
    }
    
    fn estimate_cycle_length(&self, prices: &[f32]) -> u32 {
        // Simplified cycle length estimation based on historical patterns
        // Bitcoin cycles historically ~4 years
        1460  // ~4 years in days
    }
    
    fn calculate_cycle_progress(&self, phase: &CryptoCyclePhase, prices: &[f32]) -> f32 {
        match phase {
            CryptoCyclePhase::Bull => 0.3,
            CryptoCyclePhase::LateUserophia => 0.8,
            CryptoCyclePhase::Bear => 0.9,
            CryptoCyclePhase::Capitulation => 0.95,
            CryptoCyclePhase::Transition => 0.1,
        }
    }
    
    fn calculate_bull_market_probability(&self, prices: &[f32]) -> f32 {
        if prices.len() < 200 {
            return 0.5;
        }
        
        let current_price = prices[prices.len() - 1];
        let ma_200 = prices[prices.len() - 200..].iter().sum::<f32>() / 200.0;
        
        if current_price > ma_200 {
            let distance_ratio = (current_price - ma_200) / ma_200;
            (0.6 + distance_ratio * 0.4).min(0.95)
        } else {
            let distance_ratio = (ma_200 - current_price) / ma_200;
            (0.4 - distance_ratio * 0.4).max(0.05)
        }
    }
    
    fn calculate_bear_market_probability(&self, prices: &[f32]) -> f32 {
        1.0 - self.calculate_bull_market_probability(prices)
    }
    
    fn calculate_cycle_top_indicator(&self, prices: &[f32]) -> f32 {
        // Combine multiple top indicators
        let fgi_signal = if self.crypto_specific_features.fear_greed_index > 80.0 { 0.8 } else { 0.2 };
        let social_signal = if self.crypto_specific_features.social_sentiment > 0.7 { 0.7 } else { 0.3 };
        
        (fgi_signal + social_signal) / 2.0
    }
    
    fn calculate_cycle_bottom_indicator(&self, prices: &[f32]) -> f32 {
        // Combine multiple bottom indicators
        let fgi_signal = if self.crypto_specific_features.fear_greed_index < 20.0 { 0.8 } else { 0.2 };
        let social_signal = if self.crypto_specific_features.social_sentiment < -0.7 { 0.7 } else { 0.3 };
        
        (fgi_signal + social_signal) / 2.0
    }
    
    fn calculate_btc_dominance_impact(&self, symbol: &str) -> f32 {
        if symbol == "BTC" {
            0.0  // BTC doesn't impact itself
        } else {
            -0.5  // Most alts negatively correlated with BTC dominance
        }
    }
    
    fn calculate_eth_correlation(&self, symbol: &str) -> f32 {
        if symbol == "ETH" {
            1.0
        } else if symbol.contains("ETH") || symbol.ends_with("ETH") {
            0.8  // ETH-based tokens
        } else {
            0.4  // General crypto correlation with ETH
        }
    }
    
    fn calculate_technical_momentum(&self, symbol: &str) -> f32 {
        // Placeholder for technical momentum calculation
        0.5
    }
    
    // Methods for DeFi and NFT forecasting...
    
    fn calculate_tvl_impact(&self, defi_metrics: &DeFiMetrics) -> f32 {
        defi_metrics.tvl_change_7d
    }
    
    fn calculate_yield_farming_impact(&self, defi_metrics: &DeFiMetrics) -> f32 {
        defi_metrics.yield_rate_change
    }
    
    fn calculate_governance_impact(&self, defi_metrics: &DeFiMetrics) -> f32 {
        defi_metrics.governance_activity
    }
    
    fn calculate_protocol_health(&self, defi_metrics: &DeFiMetrics) -> f32 {
        (defi_metrics.tvl_change_7d + defi_metrics.active_users_change + defi_metrics.transaction_count_change) / 3.0
    }
    
    fn calculate_impermanent_loss_risk(&self, defi_metrics: &DeFiMetrics) -> f32 {
        defi_metrics.volatility_ratio
    }
    
    fn calculate_nft_volume_trend(&self, volumes: &[f32]) -> f32 {
        if volumes.len() < 7 {
            return 0.0;
        }
        
        let recent_avg = volumes[volumes.len() - 3..].iter().sum::<f32>() / 3.0;
        let older_avg = volumes[volumes.len() - 7..volumes.len() - 3].iter().sum::<f32>() / 4.0;
        
        if older_avg > 0.0 {
            (recent_avg - older_avg) / older_avg
        } else {
            0.0
        }
    }
    
    fn calculate_nft_floor_price_trend(&self, floor_prices: &[f32]) -> f32 {
        if floor_prices.len() < 7 {
            return 0.0;
        }
        
        let recent_avg = floor_prices[floor_prices.len() - 3..].iter().sum::<f32>() / 3.0;
        let older_avg = floor_prices[floor_prices.len() - 7..floor_prices.len() - 3].iter().sum::<f32>() / 4.0;
        
        if older_avg > 0.0 {
            (recent_avg - older_avg) / older_avg
        } else {
            0.0
        }
    }
    
    fn calculate_rarity_premium(&self, nft_data: &NFTMarketData) -> f32 {
        nft_data.rarity_premium_trend
    }
    
    fn calculate_celebrity_impact(&self, celebrity_activities: &[CelebrityActivity]) -> f32 {
        celebrity_activities.iter()
            .map(|activity| activity.influence_score * activity.sentiment_score)
            .sum::<f32>() / celebrity_activities.len() as f32
    }
    
    fn calculate_nft_market_consciousness(&self, nft_data: &NFTMarketData) -> f32 {
        // NFT consciousness based on social activity and volume patterns
        let volume_consistency = self.calculate_volume_consistency(&nft_data.daily_volumes);
        let social_coherence = nft_data.social_mentions_trend.abs();
        
        (volume_consistency + social_coherence) / 2.0
    }
    
    fn calculate_volume_consistency(&self, volumes: &[f32]) -> f32 {
        if volumes.len() < 2 {
            return 0.5;
        }
        
        let mean = volumes.iter().sum::<f32>() / volumes.len() as f32;
        let variance = volumes.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>() / volumes.len() as f32;
        let std_dev = variance.sqrt();
        
        // Lower relative volatility = higher consistency
        if mean > 0.0 {
            1.0 - (std_dev / mean).min(1.0)
        } else {
            0.0
        }
    }
    
    fn recommend_nft_allocation(&self, nft_data: &NFTMarketData) -> HashMap<String, f32> {
        let mut allocations = HashMap::new();
        
        // Simple allocation based on volume and price trends
        for (i, collection) in nft_data.collection_names.iter().enumerate() {
            let volume_score = nft_data.daily_volumes.get(i).copied().unwrap_or(0.0);
            let price_score = nft_data.floor_prices.get(i).copied().unwrap_or(0.0);
            
            let allocation = (volume_score + price_score) / 2.0;
            allocations.insert(collection.clone(), allocation);
        }
        
        allocations
    }
}

// Supporting data structures
#[derive(Debug, Clone)]
pub struct SocialMediaData {
    pub twitter_sentiment_score: f32,
    pub reddit_sentiment_score: f32,
    pub telegram_sentiment_score: f32,
    pub news_sentiment_score: f32,
    pub social_volume_change: f32,
    pub sentiment_history: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct OnChainData {
    pub active_addresses_change: f32,
    pub transaction_volume_change: f32,
    pub exchange_inflow: f32,
    pub exchange_outflow: f32,
    pub exchange_reserves_change: f32,
    pub whale_transactions: f32,
    pub long_term_holder_supply_change: f32,
    pub miner_selling_pressure: f32,
}

#[derive(Debug, Clone)]
pub enum CryptoCyclePhase {
    Bull,
    LateUserophia,
    Bear,
    Capitulation,
    Transition,
}

#[derive(Debug, Clone)]
pub struct CryptoCycleAnalysis {
    pub current_phase: CryptoCyclePhase,
    pub cycle_progress: f32,
    pub estimated_cycle_length: u32,
    pub bull_market_probability: f32,
    pub bear_market_probability: f32,
    pub cycle_top_indicator: f32,
    pub cycle_bottom_indicator: f32,
}

#[derive(Debug, Clone)]
pub struct DeFiMetrics {
    pub tvl_change_7d: f32,
    pub yield_rate_change: f32,
    pub active_users_change: f32,
    pub transaction_count_change: f32,
    pub governance_activity: f32,
    pub volatility_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct DeFiForecast {
    pub base_forecast: CryptoForecast,
    pub defi_adjusted_prices: Vec<f32>,
    pub tvl_impact_factor: f32,
    pub yield_farming_appeal: f32,
    pub governance_activity: f32,
    pub protocol_health_score: f32,
    pub impermanent_loss_risk: f32,
}

#[derive(Debug, Clone)]
pub struct NFTMarketData {
    pub collection_names: Vec<String>,
    pub daily_volumes: Vec<f32>,
    pub floor_prices: Vec<f32>,
    pub rarity_premium_trend: f32,
    pub celebrity_activities: Vec<CelebrityActivity>,
    pub social_mentions_trend: f32,
}

#[derive(Debug, Clone)]
pub struct CelebrityActivity {
    pub celebrity_name: String,
    pub activity_type: String,
    pub influence_score: f32,
    pub sentiment_score: f32,
}

#[derive(Debug, Clone)]
pub struct NFTMarketForecast {
    pub volume_trend_7d: f32,
    pub floor_price_trend_7d: f32,
    pub rarity_premium_forecast: f32,
    pub celebrity_impact_score: f32,
    pub market_consciousness: f32,
    pub recommended_collection_allocation: HashMap<String, f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crypto_forecaster_creation() {
        let forecaster = CryptoForecaster::new(60, 24);
        assert_eq!(forecaster.consciousness_crypto_adaptation, 1.2);
    }
    
    #[test]
    fn test_crypto_consciousness_calculation() {
        let forecaster = CryptoForecaster::new(60, 24);
        let test_data = Array2::zeros((100, 15));
        
        let consciousness = forecaster.calculate_crypto_consciousness(&test_data, None, None);
        assert!(consciousness >= 0.0 && consciousness <= 2.0);  // Can exceed 1.0 due to crypto amplification
    }
    
    #[test]
    fn test_fear_greed_forecast() {
        let forecaster = CryptoForecaster::new(60, 24);
        let fgi_forecast = forecaster.forecast_fear_greed_index();
        
        assert!(fgi_forecast >= 0.0 && fgi_forecast <= 100.0);
    }
    
    #[test]
    fn test_cycle_analysis() {
        let forecaster = CryptoForecaster::new(60, 24);
        let btc_data = Array2::zeros((500, 10));  // Simulate 500 days of BTC data
        
        let cycle_analysis = forecaster.analyze_crypto_cycle(&btc_data);
        assert!(cycle_analysis.bull_market_probability >= 0.0);
        assert!(cycle_analysis.bear_market_probability >= 0.0);
        assert!((cycle_analysis.bull_market_probability + cycle_analysis.bear_market_probability - 1.0).abs() < 0.1);
    }
    
    #[test]
    fn test_support_resistance_calculation() {
        let forecaster = CryptoForecaster::new(60, 24);
        let mut test_data = Array2::zeros((50, 10));
        
        // Add some price data
        for i in 0..50 {
            test_data[[i, 1]] = 110.0 + i as f32;  // Highs
            test_data[[i, 2]] = 90.0 + i as f32;   // Lows
            test_data[[i, 3]] = 100.0 + i as f32;  // Close
        }
        
        let (support_levels, resistance_levels) = forecaster.calculate_crypto_support_resistance(&test_data);
        
        assert!(!support_levels.is_empty());
        assert!(!resistance_levels.is_empty());
        assert!(support_levels[0] < resistance_levels[0]);
    }
}