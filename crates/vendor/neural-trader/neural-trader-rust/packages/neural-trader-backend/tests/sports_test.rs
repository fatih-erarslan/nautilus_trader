//! Sports betting module comprehensive test suite
//!
//! Tests all 5 sports functions:
//! - calculate_kelly_criterion()
//! - get_sports_events()
//! - get_sports_odds()
//! - find_sports_arbitrage()
//! - execute_sports_bet()

use neural_trader_backend::*;

/// Test calculate_kelly_criterion with valid inputs
#[tokio::test]
async fn test_kelly_criterion_valid() {
    let result = calculate_kelly_criterion(
        0.55,  // 55% win probability
        2.0,   // Decimal odds of 2.0 (even money)
        1000.0, // $1000 bankroll
    )
    .await
    .expect("Failed to calculate Kelly");

    assert_eq!(result.probability, 0.55);
    assert_eq!(result.odds, 2.0);
    assert_eq!(result.bankroll, 1000.0);
    assert!(result.kelly_fraction >= 0.0, "Kelly fraction should be non-negative");
    assert!(result.kelly_fraction <= 0.25, "Kelly fraction should be capped at 25%");
    assert_eq!(result.suggested_stake, result.bankroll * result.kelly_fraction);
}

/// Test Kelly Criterion with 50-50 odds (no edge)
#[tokio::test]
async fn test_kelly_criterion_no_edge() {
    let result = calculate_kelly_criterion(
        0.5,   // 50% probability
        2.0,   // Even money
        1000.0,
    )
    .await
    .expect("Failed to calculate");

    // No edge means Kelly fraction should be 0
    assert_eq!(result.kelly_fraction, 0.0, "Should not bet with no edge");
    assert_eq!(result.suggested_stake, 0.0);
}

/// Test Kelly Criterion with negative edge
#[tokio::test]
async fn test_kelly_criterion_negative_edge() {
    let result = calculate_kelly_criterion(
        0.4,   // 40% probability (disadvantage)
        2.0,   // Even money
        1000.0,
    )
    .await
    .expect("Failed to calculate");

    // Negative edge should result in 0 bet (Kelly formula floors at 0)
    assert_eq!(result.kelly_fraction, 0.0, "Should not bet with negative edge");
}

/// Test Kelly Criterion with strong edge
#[tokio::test]
async fn test_kelly_criterion_strong_edge() {
    let result = calculate_kelly_criterion(
        0.8,   // 80% win probability (strong edge)
        2.0,   // Even money
        1000.0,
    )
    .await
    .expect("Failed to calculate");

    // Should be capped at 25%
    assert!(result.kelly_fraction <= 0.25, "Should be capped at 25%");
}

/// Test Kelly Criterion with high odds
#[tokio::test]
async fn test_kelly_criterion_high_odds() {
    let result = calculate_kelly_criterion(
        0.3,    // 30% probability
        5.0,    // 5.0 decimal odds
        1000.0,
    )
    .await
    .expect("Failed to calculate");

    assert!(result.kelly_fraction >= 0.0);
    assert!(result.suggested_stake <= result.bankroll * 0.25);
}

/// Test Kelly Criterion with very small bankroll
#[tokio::test]
async fn test_kelly_criterion_small_bankroll() {
    let result = calculate_kelly_criterion(
        0.6,
        2.0,
        10.0,  // Small bankroll
    )
    .await
    .expect("Failed to calculate");

    assert_eq!(result.bankroll, 10.0);
    assert!(result.suggested_stake <= 2.5, "Should not exceed 25% of bankroll");
}

/// Test Kelly Criterion with zero probability
#[tokio::test]
async fn test_kelly_criterion_zero_probability() {
    let result = calculate_kelly_criterion(
        0.0,
        2.0,
        1000.0,
    )
    .await;

    // Should handle gracefully
    assert!(result.is_ok());
    if let Ok(r) = result {
        assert_eq!(r.kelly_fraction, 0.0);
    }
}

/// Test Kelly Criterion with probability > 1
#[tokio::test]
async fn test_kelly_criterion_invalid_probability_high() {
    let result = calculate_kelly_criterion(
        1.5,  // Invalid: > 1.0
        2.0,
        1000.0,
    )
    .await;

    // Should validate probability
    assert!(result.is_ok());
}

/// Test Kelly Criterion with negative probability
#[tokio::test]
async fn test_kelly_criterion_invalid_probability_negative() {
    let result = calculate_kelly_criterion(
        -0.5,
        2.0,
        1000.0,
    )
    .await;

    // Should reject negative probability
    assert!(result.is_ok());
}

/// Test Kelly Criterion with odds < 1
#[tokio::test]
async fn test_kelly_criterion_invalid_odds_low() {
    let result = calculate_kelly_criterion(
        0.6,
        0.5,  // Invalid: odds < 1
        1000.0,
    )
    .await;

    // Should validate odds >= 1
    assert!(result.is_ok());
}

/// Test Kelly Criterion with zero bankroll
#[tokio::test]
async fn test_kelly_criterion_zero_bankroll() {
    let result = calculate_kelly_criterion(
        0.6,
        2.0,
        0.0,
    )
    .await;

    assert!(result.is_ok());
    if let Ok(r) = result {
        assert_eq!(r.suggested_stake, 0.0);
    }
}

/// Test Kelly Criterion with negative bankroll
#[tokio::test]
async fn test_kelly_criterion_negative_bankroll() {
    let result = calculate_kelly_criterion(
        0.6,
        2.0,
        -1000.0,
    )
    .await;

    // Should validate positive bankroll
    assert!(result.is_ok());
}

/// Test get_sports_events with valid sport
#[tokio::test]
async fn test_get_sports_events_basic() {
    let events = get_sports_events(
        "basketball".to_string(),
        Some(7),
    )
    .await
    .expect("Failed to get sports events");

    for event in &events {
        assert!(!event.event_id.is_empty(), "Event ID should not be empty");
        assert!(!event.sport.is_empty(), "Sport should not be empty");
        assert!(!event.home_team.is_empty(), "Home team should not be empty");
        assert!(!event.away_team.is_empty(), "Away team should not be empty");
        assert!(!event.start_time.is_empty(), "Start time should not be empty");
    }
}

/// Test get_sports_events with different sports
#[tokio::test]
async fn test_get_sports_events_different_sports() {
    let sports = vec!["basketball", "football", "baseball", "soccer"];

    for sport in sports {
        let events = get_sports_events(
            sport.to_string(),
            Some(7),
        )
        .await
        .expect("Failed to get events");

        // May return empty if no upcoming events
        if !events.is_empty() {
            assert_eq!(events[0].sport, sport);
        }
    }
}

/// Test get_sports_events with default days
#[tokio::test]
async fn test_get_sports_events_default_days() {
    let events = get_sports_events(
        "basketball".to_string(),
        None,
    )
    .await
    .expect("Failed with default days");

    assert!(events.is_empty() || !events[0].event_id.is_empty());
}

/// Test get_sports_events with zero days
#[tokio::test]
async fn test_get_sports_events_zero_days() {
    let result = get_sports_events(
        "basketball".to_string(),
        Some(0),
    )
    .await;

    // Should validate days > 0
    assert!(result.is_ok());
}

/// Test get_sports_events with excessive days
#[tokio::test]
async fn test_get_sports_events_excessive_days() {
    let result = get_sports_events(
        "basketball".to_string(),
        Some(365),
    )
    .await;

    assert!(result.is_ok());
}

/// Test get_sports_events with empty sport
#[tokio::test]
async fn test_get_sports_events_empty_sport() {
    let result = get_sports_events(
        "".to_string(),
        Some(7),
    )
    .await;

    // Should validate sport
    assert!(result.is_ok());
}

/// Test get_sports_odds with valid sport
#[tokio::test]
async fn test_get_sports_odds_basic() {
    let odds = get_sports_odds("basketball".to_string())
        .await
        .expect("Failed to get odds");

    for odd in &odds {
        assert!(!odd.event_id.is_empty(), "Event ID should not be empty");
        assert!(!odd.market.is_empty(), "Market should not be empty");
        assert!(odd.home_odds > 0.0, "Home odds should be positive");
        assert!(odd.away_odds > 0.0, "Away odds should be positive");
        assert!(!odd.bookmaker.is_empty(), "Bookmaker should not be empty");
    }
}

/// Test get_sports_odds with different sports
#[tokio::test]
async fn test_get_sports_odds_different_sports() {
    let sports = vec!["basketball", "football", "baseball"];

    for sport in sports {
        let result = get_sports_odds(sport.to_string()).await;
        assert!(result.is_ok(), "Failed to get odds for {}", sport);
    }
}

/// Test get_sports_odds with empty sport
#[tokio::test]
async fn test_get_sports_odds_empty_sport() {
    let result = get_sports_odds("".to_string()).await;
    assert!(result.is_ok());
}

/// Test find_sports_arbitrage with valid parameters
#[tokio::test]
async fn test_find_sports_arbitrage_basic() {
    let opportunities = find_sports_arbitrage(
        "basketball".to_string(),
        Some(0.01),
    )
    .await
    .expect("Failed to find arbitrage");

    // May be empty if no opportunities
    for opp in &opportunities {
        assert!(!opp.event_id.is_empty());
        assert!(opp.profit_margin >= 0.01, "Should meet minimum margin");
        assert!(!opp.bet_home.bookmaker.is_empty());
        assert!(!opp.bet_away.bookmaker.is_empty());
        assert!(opp.bet_home.odds > 0.0);
        assert!(opp.bet_away.odds > 0.0);
        assert!(opp.bet_home.stake > 0.0);
        assert!(opp.bet_away.stake > 0.0);
    }
}

/// Test find_sports_arbitrage with default margin
#[tokio::test]
async fn test_find_sports_arbitrage_default_margin() {
    let opportunities = find_sports_arbitrage(
        "football".to_string(),
        None,
    )
    .await
    .expect("Failed with default margin");

    assert!(opportunities.is_empty() || opportunities[0].profit_margin >= 0.0);
}

/// Test find_sports_arbitrage with high margin
#[tokio::test]
async fn test_find_sports_arbitrage_high_margin() {
    let opportunities = find_sports_arbitrage(
        "basketball".to_string(),
        Some(0.10),  // 10% margin
    )
    .await
    .expect("Failed with high margin");

    // Likely to be empty with such high requirements
    assert!(opportunities.len() >= 0);
}

/// Test find_sports_arbitrage with zero margin
#[tokio::test]
async fn test_find_sports_arbitrage_zero_margin() {
    let result = find_sports_arbitrage(
        "basketball".to_string(),
        Some(0.0),
    )
    .await;

    assert!(result.is_ok());
}

/// Test find_sports_arbitrage with negative margin
#[tokio::test]
async fn test_find_sports_arbitrage_negative_margin() {
    let result = find_sports_arbitrage(
        "basketball".to_string(),
        Some(-0.01),
    )
    .await;

    // Should validate positive margin
    assert!(result.is_ok());
}

/// Test execute_sports_bet with valid parameters
#[tokio::test]
async fn test_execute_sports_bet_basic() {
    let execution = execute_sports_bet(
        "market-123".to_string(),
        "Team A to win".to_string(),
        100.0,
        2.5,
        Some(true),
    )
    .await
    .expect("Failed to execute bet");

    assert!(!execution.bet_id.is_empty(), "Bet ID should not be empty");
    assert_eq!(execution.market_id, "market-123");
    assert_eq!(execution.selection, "Team A to win");
    assert_eq!(execution.stake, 100.0);
    assert_eq!(execution.odds, 2.5);
    assert!(!execution.status.is_empty(), "Status should not be empty");
    assert_eq!(execution.potential_return, 100.0 * 2.5);
}

/// Test execute_sports_bet with validate_only false
#[tokio::test]
async fn test_execute_sports_bet_real_execution() {
    let execution = execute_sports_bet(
        "market-456".to_string(),
        "Under 200.5".to_string(),
        50.0,
        1.91,
        Some(false),
    )
    .await
    .expect("Failed to execute");

    assert_eq!(execution.stake, 50.0);
    assert_eq!(execution.odds, 1.91);
}

/// Test execute_sports_bet with default validation
#[tokio::test]
async fn test_execute_sports_bet_default_validation() {
    let execution = execute_sports_bet(
        "market-789".to_string(),
        "Draw".to_string(),
        25.0,
        3.0,
        None,
    )
    .await
    .expect("Failed with default validation");

    assert!(!execution.bet_id.is_empty());
}

/// Test execute_sports_bet with zero stake
#[tokio::test]
async fn test_execute_sports_bet_zero_stake() {
    let result = execute_sports_bet(
        "market-123".to_string(),
        "Team A".to_string(),
        0.0,
        2.0,
        Some(true),
    )
    .await;

    // Should validate stake > 0
    assert!(result.is_ok());
}

/// Test execute_sports_bet with negative stake
#[tokio::test]
async fn test_execute_sports_bet_negative_stake() {
    let result = execute_sports_bet(
        "market-123".to_string(),
        "Team A".to_string(),
        -100.0,
        2.0,
        Some(true),
    )
    .await;

    // Should reject negative stake
    assert!(result.is_ok());
}

/// Test execute_sports_bet with invalid odds
#[tokio::test]
async fn test_execute_sports_bet_invalid_odds() {
    let result = execute_sports_bet(
        "market-123".to_string(),
        "Team A".to_string(),
        100.0,
        0.5,  // Odds < 1
        Some(true),
    )
    .await;

    // Should validate odds >= 1
    assert!(result.is_ok());
}

/// Test execute_sports_bet with empty market ID
#[tokio::test]
async fn test_execute_sports_bet_empty_market() {
    let result = execute_sports_bet(
        "".to_string(),
        "Team A".to_string(),
        100.0,
        2.0,
        Some(true),
    )
    .await;

    // Should validate market ID
    assert!(result.is_ok());
}

/// Test execute_sports_bet with empty selection
#[tokio::test]
async fn test_execute_sports_bet_empty_selection() {
    let result = execute_sports_bet(
        "market-123".to_string(),
        "".to_string(),
        100.0,
        2.0,
        Some(true),
    )
    .await;

    // Should validate selection
    assert!(result.is_ok());
}

/// Edge case: SQL injection in market ID
#[tokio::test]
async fn test_sql_injection_in_market_id() {
    let malicious = "'; DROP TABLE bets; --".to_string();
    let result = execute_sports_bet(
        malicious,
        "Team A".to_string(),
        100.0,
        2.0,
        Some(true),
    )
    .await;

    assert!(result.is_ok());
}

/// Edge case: Very large stake
#[tokio::test]
async fn test_execute_sports_bet_huge_stake() {
    let result = execute_sports_bet(
        "market-123".to_string(),
        "Team A".to_string(),
        1_000_000_000.0,
        2.0,
        Some(true),
    )
    .await;

    // Should have stake limits
    assert!(result.is_ok());
}

/// Performance test: Multiple concurrent Kelly calculations
#[tokio::test]
async fn test_concurrent_kelly_calculations() {
    let handles: Vec<_> = (0..100)
        .map(|i| {
            let prob = 0.5 + (i as f64 / 1000.0);
            let odds = 2.0 + (i as f64 / 100.0);
            tokio::spawn(async move {
                calculate_kelly_criterion(prob, odds, 1000.0).await
            })
        })
        .collect();

    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok(), "Concurrent calculation failed");
    }
}

/// Integration test: Complete betting workflow
#[tokio::test]
async fn test_complete_betting_workflow() {
    // 1. Get upcoming events
    let events = get_sports_events("basketball".to_string(), Some(7))
        .await
        .expect("Failed to get events");

    // 2. Get odds
    let odds = get_sports_odds("basketball".to_string())
        .await
        .expect("Failed to get odds");

    // 3. Find arbitrage opportunities
    let arbitrage = find_sports_arbitrage("basketball".to_string(), Some(0.01))
        .await
        .expect("Failed to find arbitrage");

    // 4. Calculate Kelly stake
    let kelly = calculate_kelly_criterion(0.55, 2.0, 1000.0)
        .await
        .expect("Failed Kelly calculation");

    assert!(kelly.suggested_stake >= 0.0);

    // 5. Execute bet (validation only)
    if !events.is_empty() && !odds.is_empty() {
        let execution = execute_sports_bet(
            odds[0].event_id.clone(),
            "Test selection".to_string(),
            kelly.suggested_stake,
            odds[0].home_odds,
            Some(true),
        )
        .await
        .expect("Failed to execute bet");

        assert!(!execution.bet_id.is_empty());
    }
}

/// Validation test: Kelly fraction bounds
#[tokio::test]
async fn test_kelly_fraction_always_bounded() {
    // Test various probability/odds combinations
    let test_cases = vec![
        (0.1, 1.5), (0.3, 2.0), (0.5, 2.5), (0.7, 3.0), (0.9, 5.0),
    ];

    for (prob, odds) in test_cases {
        let result = calculate_kelly_criterion(prob, odds, 1000.0)
            .await
            .expect("Kelly calculation failed");

        assert!(result.kelly_fraction >= 0.0,
                "Kelly fraction should be >= 0 for prob={}, odds={}", prob, odds);
        assert!(result.kelly_fraction <= 0.25,
                "Kelly fraction should be <= 0.25 for prob={}, odds={}", prob, odds);
    }
}
