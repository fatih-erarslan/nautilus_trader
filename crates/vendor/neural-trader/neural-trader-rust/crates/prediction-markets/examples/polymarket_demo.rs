//! Comprehensive Polymarket demonstration
//!
//! This example demonstrates all major features of the Polymarket integration:
//! - REST API client usage
//! - WebSocket streaming
//! - Market making
//! - Arbitrage detection
//!
//! Run with: cargo run --example polymarket_demo

use nt_prediction_markets::error::Result;
use nt_prediction_markets::models::*;
use nt_prediction_markets::polymarket::*;
use nt_prediction_markets::polymarket::arbitrage::ArbitrageOutcome;
use rust_decimal_macros::dec;
use std::env;
use tokio::time::Duration;
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting Polymarket demonstration");

    // Get API key from environment
    let api_key = env::var("POLYMARKET_API_KEY")
        .unwrap_or_else(|_| "demo_key".to_string());

    // === 1. CLIENT SETUP ===
    info!("=== Setting up Polymarket client ===");
    let config = ClientConfig::new(api_key)
        .with_timeout(Duration::from_secs(30))
        .with_max_retries(3);

    let client = PolymarketClient::new(config)?;
    info!("Client initialized successfully");

    // === 2. FETCH MARKETS ===
    info!("\n=== Fetching markets ===");
    demo_fetch_markets(&client).await?;

    // === 3. ORDERBOOK ANALYSIS ===
    info!("\n=== Analyzing orderbooks ===");
    demo_orderbook_analysis(&client).await?;

    // === 4. ORDER MANAGEMENT ===
    info!("\n=== Order management demo ===");
    demo_order_management(&client).await?;

    // === 5. POSITION TRACKING ===
    info!("\n=== Position tracking ===");
    demo_positions(&client).await?;

    // === 6. MARKET MAKING ===
    info!("\n=== Market making strategy ===");
    demo_market_making(client.clone()).await?;

    // === 7. ARBITRAGE DETECTION ===
    info!("\n=== Arbitrage detection ===");
    demo_arbitrage(client.clone()).await?;

    // === 8. WEBSOCKET STREAMING ===
    info!("\n=== WebSocket streaming ===");
    demo_websocket_streaming().await?;

    info!("\nDemonstration complete!");
    Ok(())
}

async fn demo_fetch_markets(client: &PolymarketClient) -> Result<()> {
    // Fetch all markets
    match client.get_markets().await {
        Ok(markets) => {
            info!("Found {} active markets", markets.len());

            // Display first 3 markets
            for market in markets.iter().take(3) {
                info!("Market: {}", market.question);
                info!("  ID: {}", market.id);
                info!("  Volume: ${}", market.volume);
                info!("  Liquidity: ${}", market.liquidity);
                info!("  Outcomes: {}", market.outcomes.len());

                for outcome in &market.outcomes {
                    info!("    - {}: {:.2}%", outcome.title, outcome.probability() * dec!(100));
                }
            }
        }
        Err(e) => {
            info!("Note: Market fetch failed (expected in demo mode): {}", e);
        }
    }

    // Search for specific markets
    match client.search_markets("election", Some(5)).await {
        Ok(markets) => {
            info!("Found {} election markets", markets.len());
        }
        Err(e) => {
            info!("Note: Search failed (expected in demo mode): {}", e);
        }
    }

    Ok(())
}

async fn demo_orderbook_analysis(_client: &PolymarketClient) -> Result<()> {
    // Create a sample orderbook for demonstration
    let sample_orderbook = OrderBook {
        market_id: "demo_market".to_string(),
        outcome_id: "yes".to_string(),
        bids: vec![
            OrderBookLevel { price: dec!(0.60), size: dec!(500) },
            OrderBookLevel { price: dec!(0.59), size: dec!(750) },
            OrderBookLevel { price: dec!(0.58), size: dec!(1000) },
        ],
        asks: vec![
            OrderBookLevel { price: dec!(0.62), size: dec!(600) },
            OrderBookLevel { price: dec!(0.63), size: dec!(800) },
            OrderBookLevel { price: dec!(0.64), size: dec!(1200) },
        ],
        timestamp: chrono::Utc::now(),
    };

    info!("Sample Orderbook Analysis:");
    info!("  Best Bid: {:?}", sample_orderbook.best_bid());
    info!("  Best Ask: {:?}", sample_orderbook.best_ask());
    info!("  Spread: {:?}", sample_orderbook.spread());
    info!("  Mid Price: {:?}", sample_orderbook.mid_price());
    info!("  Total Bid Size: {}", sample_orderbook.total_bid_size());
    info!("  Total Ask Size: {}", sample_orderbook.total_ask_size());

    // Calculate price impact
    let impact_100 = sample_orderbook.calculate_price_impact(OrderSide::Buy, dec!(100));
    let impact_500 = sample_orderbook.calculate_price_impact(OrderSide::Buy, dec!(500));
    info!("  Price Impact (100 shares): {:?}", impact_100);
    info!("  Price Impact (500 shares): {:?}", impact_500);

    // Show depth
    let (bids, asks) = sample_orderbook.get_depth(3);
    info!("  Orderbook Depth (3 levels):");
    info!("    Bids: {} levels", bids.len());
    info!("    Asks: {} levels", asks.len());

    Ok(())
}

async fn demo_order_management(_client: &PolymarketClient) -> Result<()> {
    // Create sample order request
    let order_request = OrderRequest {
        market_id: "demo_market".to_string(),
        outcome_id: "yes".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Limit,
        size: dec!(100),
        price: Some(dec!(0.55)),
        time_in_force: Some(TimeInForce::GTC),
        client_order_id: Some("demo_order_001".to_string()),
    };

    info!("Sample Order Request:");
    info!("  Market: {}", order_request.market_id);
    info!("  Side: {:?}", order_request.side);
    info!("  Type: {:?}", order_request.order_type);
    info!("  Size: {}", order_request.size);
    info!("  Price: {:?}", order_request.price);

    // Validate order
    match order_request.validate() {
        Ok(_) => info!("  Validation: PASSED"),
        Err(e) => info!("  Validation: FAILED - {}", e),
    }

    // Calculate notional value
    if let Some(price) = order_request.price {
        let notional = price * order_request.size;
        info!("  Notional Value: ${:.2}", notional);
    }

    Ok(())
}

async fn demo_positions(_client: &PolymarketClient) -> Result<()> {
    // Create sample position
    let position = Position {
        market_id: "demo_market".to_string(),
        outcome_id: "yes".to_string(),
        size: dec!(200),
        average_price: dec!(0.55),
        current_price: dec!(0.62),
        unrealized_pnl: dec!(14),
        realized_pnl: dec!(3),
        total_fees: dec!(2.5),
    };

    info!("Sample Position:");
    info!("  Size: {}", position.size);
    info!("  Entry Price: ${:.4}", position.average_price);
    info!("  Current Price: ${:.4}", position.current_price);
    info!("  Current Value: ${:.2}", position.current_value());
    info!("  Cost Basis: ${:.2}", position.cost_basis());
    info!("  Unrealized PnL: ${:.2}", position.unrealized_pnl);
    info!("  Realized PnL: ${:.2}", position.realized_pnl);
    info!("  Total Fees: ${:.2}", position.total_fees);
    info!("  Total PnL: ${:.2}", position.total_pnl());
    info!("  PnL %: {:.2}%", position.pnl_percentage());

    Ok(())
}

async fn demo_market_making(client: PolymarketClient) -> Result<()> {
    let config = MarketMakerConfig {
        spread: dec!(0.04), // 4% spread
        order_size: dec!(100),
        max_position: dec!(1000),
        num_levels: 3,
        min_edge: dec!(0.01),
        inventory_skew: dec!(0.5),
    };

    let mm = PolymarketMM::new(client, config.clone());

    info!("Market Maker Configuration:");
    info!("  Spread: {:.2}%", config.spread * dec!(100));
    info!("  Order Size: {}", config.order_size);
    info!("  Max Position: {}", config.max_position);
    info!("  Price Levels: {}", config.num_levels);
    info!("  Min Edge: {:.2}%", config.min_edge * dec!(100));

    // Generate sample quotes
    let mid_price = dec!(0.50);
    let position = dec!(0); // No position

    let (bid, ask) = mm.calculate_quotes(mid_price, position);
    info!("\nQuotes at mid price ${:.4} with no position:", mid_price);
    info!("  Bid: ${:.4}", bid);
    info!("  Ask: ${:.4}", ask);
    info!("  Spread: ${:.4} ({:.2}%)", ask - bid, (ask - bid) / mid_price * dec!(100));

    // Generate orders
    let orders = mm.generate_orders("demo_market", "yes", mid_price);
    info!("\nGenerated {} orders:", orders.len());
    for (i, order) in orders.iter().enumerate() {
        info!("  Order {}: {:?} {} @ ${:.4}",
            i + 1, order.side, order.size, order.price.unwrap());
    }

    // Show position-adjusted quotes
    let long_position = dec!(500);
    let (bid_long, ask_long) = mm.calculate_quotes(mid_price, long_position);
    info!("\nQuotes with long position ({}):", long_position);
    info!("  Bid: ${:.4} (adjusted down)", bid_long);
    info!("  Ask: ${:.4} (adjusted down)", ask_long);

    Ok(())
}

async fn demo_arbitrage(client: PolymarketClient) -> Result<()> {
    let config = ArbitrageConfig {
        min_profit: dec!(0.02), // 2% minimum
        max_size: dec!(1000),
        fee_rate: dec!(0.02), // 2% fees
        check_interval: 5,
    };

    let arb = PolymarketArbitrage::new(client, config.clone());

    info!("Arbitrage Configuration:");
    info!("  Min Profit: {:.2}%", config.min_profit * dec!(100));
    info!("  Max Size: {}", config.max_size);
    info!("  Fee Rate: {:.2}%", config.fee_rate * dec!(100));
    info!("  Check Interval: {}s", config.check_interval);

    // Demonstrate risk assessment
    info!("\nRisk Assessment Examples:");
    let risks = vec![
        (dec!(15), "15% profit"),
        (dec!(7), "7% profit"),
        (dec!(3), "3% profit"),
        (dec!(1), "1% profit"),
    ];

    for (profit, desc) in risks {
        let risk = arb.assess_risk(profit);
        info!("  {}: {:?}", desc, risk);
    }

    // Show sample opportunity
    info!("\nSample Arbitrage Opportunity:");
    let sample_opp = ArbitrageOpportunity {
        market_id: "demo_market".to_string(),
        outcomes: vec![
            ArbitrageOutcome {
                outcome_id: "yes".to_string(),
                side: OrderSide::Sell,
                price: dec!(0.55),
                size: dec!(1000),
                exchange: "Polymarket".to_string(),
            },
            ArbitrageOutcome {
                outcome_id: "no".to_string(),
                side: OrderSide::Sell,
                price: dec!(0.48),
                size: dec!(1000),
                exchange: "Polymarket".to_string(),
            },
        ],
        total_cost: dec!(1030), // Including fees
        profit: dec!(20),
        profit_percentage: dec!(1.94),
        risk_level: RiskLevel::High,
    };

    info!("  Market: {}", sample_opp.market_id);
    info!("  Total Cost: ${:.2}", sample_opp.total_cost);
    info!("  Profit: ${:.2}", sample_opp.profit);
    info!("  Profit %: {:.2}%", sample_opp.profit_percentage);
    info!("  Risk Level: {:?}", sample_opp.risk_level);
    info!("  Outcomes:");
    for outcome in &sample_opp.outcomes {
        info!("    - {:?} {} @ ${:.4} on {}",
            outcome.side, outcome.outcome_id, outcome.price, outcome.exchange);
    }

    Ok(())
}

async fn demo_websocket_streaming() -> Result<()> {
    info!("WebSocket Streaming Demo:");
    info!("  Setting up stream manager...");

    let stream = StreamBuilder::new().build();

    info!("  Available subscriptions:");
    info!("    - Orderbook updates");
    info!("    - Market updates");
    info!("    - Trade updates");
    info!("    - Order updates");

    info!("\nNote: Actual WebSocket connection requires valid credentials");
    info!("      Use PolymarketStream::connect() to establish connection");
    info!("      Then subscribe to channels with:");
    info!("        - subscribe_orderbook(market_id, outcome_id)");
    info!("        - subscribe_market(market_id)");
    info!("        - subscribe_trades(market_id, outcome_id)");

    // Show subscription management
    info!("\nSubscription Management:");
    let sub_count = stream.get_subscriptions().len();
    info!("  Active subscriptions: {}", sub_count);

    Ok(())
}
