// Mock infrastructure for testing
//
// This module provides mock implementations of all external dependencies
// for comprehensive testing without real API calls.

pub mod mock_broker;
pub mod mock_market_data;

pub use mock_broker::MockBrokerClient;
pub use mock_market_data::MockMarketDataProvider;
