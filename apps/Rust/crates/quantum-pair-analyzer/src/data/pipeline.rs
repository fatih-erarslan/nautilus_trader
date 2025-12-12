// Data Pipeline Implementation
// Copyright (c) 2025 TENGRI Trading Swarm

use anyhow::Result;
use async_trait::async_trait;
use tokio_stream::Stream;
use crate::{TradingPair, PairId, MarketUpdate, AnalyzerError};

#[async_trait]
pub trait ExchangeConnector: Send + Sync {
    async fn test_connection(&self) -> Result<bool, AnalyzerError>;
    async fn is_real_connection(&self) -> Result<bool, AnalyzerError>;
    async fn last_data_update(&self) -> Result<chrono::DateTime<chrono::Utc>, AnalyzerError>;
    async fn supports_pair(&self, pair: &PairId) -> Result<bool, AnalyzerError>;
    async fn get_pair_data(&self, pair: &PairId) -> Result<Option<crate::data::MarketData>, AnalyzerError>;
    async fn fetch_trading_pairs(&self) -> Result<Vec<TradingPair>, AnalyzerError>;
    async fn subscribe_market_data(&self) -> Box<dyn Stream<Item = MarketUpdate> + Send + Unpin>;
}