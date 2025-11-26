//! Sports betting analysis and execution
//!
//! Provides NAPI bindings for:
//! - Sports event retrieval
//! - Odds analysis
//! - Arbitrage detection
//! - Kelly Criterion betting
//! - Strategy simulation

use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::error::NeuralTraderError;
use nt_sports_betting::{
    models::{BetPosition, BetStatus},
    odds_api::{OddsApiClient, Odds as ApiOdds},
};
use std::collections::HashMap;
use chrono::Utc;
use uuid::Uuid;

/// Get upcoming sports events
#[napi]
pub async fn get_sports_events(
    sport: String,
    days_ahead: Option<u32>,
) -> Result<Vec<SportsEvent>> {
    let days = days_ahead.unwrap_or(7);

    // In a real implementation, this would fetch from a sports data API
    // For now, we'll generate realistic mock data
    let mut events = Vec::new();
    let base_time = Utc::now();

    // Generate sample events based on sport type
    let teams = match sport.as_str() {
        "soccer" | "football" => vec![
            ("Manchester United", "Liverpool"),
            ("Real Madrid", "Barcelona"),
            ("Bayern Munich", "Borussia Dortmund"),
            ("PSG", "Marseille"),
            ("Juventus", "AC Milan"),
        ],
        "basketball" => vec![
            ("Lakers", "Warriors"),
            ("Celtics", "Heat"),
            ("Bucks", "Nets"),
            ("Suns", "Mavericks"),
            ("76ers", "Nuggets"),
        ],
        "baseball" => vec![
            ("Yankees", "Red Sox"),
            ("Dodgers", "Giants"),
            ("Astros", "Rangers"),
            ("Braves", "Mets"),
            ("Cardinals", "Cubs"),
        ],
        _ => vec![
            ("Team A", "Team B"),
            ("Team C", "Team D"),
        ],
    };

    for (idx, (home, away)) in teams.iter().enumerate() {
        let event_time = base_time + chrono::Duration::hours((idx as i64 + 1) * 24);

        if event_time.signed_duration_since(base_time).num_days() <= days as i64 {
            events.push(SportsEvent {
                event_id: format!("evt-{}-{}", sport, Uuid::new_v4()),
                sport: sport.clone(),
                home_team: home.to_string(),
                away_team: away.to_string(),
                start_time: event_time.to_rfc3339(),
            });
        }
    }

    Ok(events)
}

/// Sports event
#[napi(object)]
pub struct SportsEvent {
    pub event_id: String,
    pub sport: String,
    pub home_team: String,
    pub away_team: String,
    pub start_time: String,
}

/// Get sports betting odds
#[napi]
pub async fn get_sports_odds(sport: String) -> Result<Vec<BettingOdds>> {
    // First, get events to attach odds to
    let events = get_sports_events(sport.clone(), Some(3)).await?;

    let mut all_odds = Vec::new();

    // Generate realistic odds from multiple bookmakers
    let bookmakers = vec![
        "Bet365",
        "DraftKings",
        "FanDuel",
        "Caesars",
        "BetMGM",
    ];

    for event in events {
        // Generate base odds with slight variations per bookmaker
        let base_home_odds = 1.75 + (rand_f64() * 0.4); // 1.75 - 2.15
        let base_away_odds = 1.85 + (rand_f64() * 0.4); // 1.85 - 2.25

        for bookmaker in &bookmakers {
            // Add bookmaker-specific variance
            let variance = (rand_f64() - 0.5) * 0.1; // -0.05 to +0.05

            all_odds.push(BettingOdds {
                event_id: event.event_id.clone(),
                market: "moneyline".to_string(),
                home_odds: (base_home_odds + variance).max(1.01),
                away_odds: (base_away_odds - variance).max(1.01),
                bookmaker: bookmaker.to_string(),
            });

            // Add spread market
            all_odds.push(BettingOdds {
                event_id: event.event_id.clone(),
                market: "spread".to_string(),
                home_odds: 1.90 + variance,
                away_odds: 1.90 - variance,
                bookmaker: bookmaker.to_string(),
            });

            // Add totals market
            all_odds.push(BettingOdds {
                event_id: event.event_id.clone(),
                market: "totals".to_string(),
                home_odds: 1.88 + variance,
                away_odds: 1.92 - variance,
                bookmaker: bookmaker.to_string(),
            });
        }
    }

    Ok(all_odds)
}

/// Generate a pseudo-random f64 between 0.0 and 1.0
/// Note: This is for demonstration. In production, use a proper RNG.
fn rand_f64() -> f64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};

    let start = std::time::SystemTime::now();
    let since_epoch = start.duration_since(std::time::UNIX_EPOCH).unwrap();
    let nanos = since_epoch.as_nanos();

    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    nanos.hash(&mut hasher);

    (hasher.finish() as f64) / (u64::MAX as f64)
}

/// Betting odds
#[napi(object)]
pub struct BettingOdds {
    pub event_id: String,
    pub market: String,
    pub home_odds: f64,
    pub away_odds: f64,
    pub bookmaker: String,
}

/// Find arbitrage opportunities
#[napi]
pub async fn find_sports_arbitrage(
    sport: String,
    min_profit_margin: Option<f64>,
) -> Result<Vec<ArbitrageOpportunity>> {
    let min_margin = min_profit_margin.unwrap_or(0.01);

    // Get all odds for the sport
    let all_odds = get_sports_odds(sport).await?;

    // Group odds by event and market
    let mut odds_map: HashMap<String, Vec<BettingOdds>> = HashMap::new();
    for odds in all_odds {
        let key = format!("{}:{}", odds.event_id, odds.market);
        odds_map.entry(key).or_insert_with(Vec::new).push(odds);
    }

    let mut opportunities = Vec::new();

    // Check each event/market combination for arbitrage
    for (key, odds_list) in odds_map {
        if odds_list.len() < 2 {
            continue; // Need at least 2 bookmakers
        }

        // Find best home and away odds
        let mut best_home: Option<&BettingOdds> = None;
        let mut best_away: Option<&BettingOdds> = None;

        for odds in &odds_list {
            if best_home.is_none() || odds.home_odds > best_home.unwrap().home_odds {
                best_home = Some(odds);
            }
            if best_away.is_none() || odds.away_odds > best_away.unwrap().away_odds {
                best_away = Some(odds);
            }
        }

        if let (Some(home), Some(away)) = (best_home, best_away) {
            // Calculate arbitrage
            // For arbitrage: 1/odds_home + 1/odds_away < 1
            let inverse_sum = (1.0 / home.home_odds) + (1.0 / away.away_odds);

            if inverse_sum < 1.0 {
                let profit_margin = (1.0 - inverse_sum) * 100.0;

                if profit_margin >= min_margin * 100.0 {
                    // Calculate optimal stake distribution
                    // Total bankroll = 100 (normalized)
                    let total_stake = 100.0;
                    let home_stake = total_stake / (1.0 + (home.home_odds / away.away_odds));
                    let away_stake = total_stake - home_stake;

                    opportunities.push(ArbitrageOpportunity {
                        event_id: home.event_id.clone(),
                        profit_margin: profit_margin / 100.0, // Convert back to decimal
                        bet_home: BetAllocation {
                            bookmaker: home.bookmaker.clone(),
                            odds: home.home_odds,
                            stake: home_stake,
                        },
                        bet_away: BetAllocation {
                            bookmaker: away.bookmaker.clone(),
                            odds: away.away_odds,
                            stake: away_stake,
                        },
                    });
                }
            }
        }
    }

    Ok(opportunities)
}

/// Arbitrage opportunity
#[napi(object)]
pub struct ArbitrageOpportunity {
    pub event_id: String,
    pub profit_margin: f64,
    pub bet_home: BetAllocation,
    pub bet_away: BetAllocation,
}

/// Bet allocation
#[napi(object)]
pub struct BetAllocation {
    pub bookmaker: String,
    pub odds: f64,
    pub stake: f64,
}

/// Calculate Kelly Criterion bet size
#[napi]
pub async fn calculate_kelly_criterion(
    probability: f64,
    odds: f64,
    bankroll: f64,
) -> Result<KellyCriterion> {
    // Kelly formula: f = (bp - q) / b
    // where f = fraction of bankroll, b = odds - 1, p = probability, q = 1 - p
    let b = odds - 1.0;
    let p = probability;
    let q = 1.0 - p;
    let kelly_fraction = ((b * p) - q) / b;
    let kelly_fraction = kelly_fraction.max(0.0).min(0.25); // Cap at 25%

    Ok(KellyCriterion {
        probability,
        odds,
        bankroll,
        kelly_fraction,
        suggested_stake: bankroll * kelly_fraction,
    })
}

/// Kelly Criterion result
#[napi(object)]
pub struct KellyCriterion {
    pub probability: f64,
    pub odds: f64,
    pub bankroll: f64,
    pub kelly_fraction: f64,
    pub suggested_stake: f64,
}

/// Execute a sports bet
#[napi]
pub async fn execute_sports_bet(
    market_id: String,
    selection: String,
    stake: f64,
    odds: f64,
    validate_only: Option<bool>,
) -> Result<BetExecution> {
    let validate = validate_only.unwrap_or(true);

    // Validation checks
    if stake <= 0.0 {
        return Err(NeuralTraderError::Sports(
            "Stake must be greater than 0".to_string()
        ).into());
    }

    if odds < 1.0 {
        return Err(NeuralTraderError::Sports(
            "Odds must be at least 1.0".to_string()
        ).into());
    }

    if stake > 100000.0 {
        return Err(NeuralTraderError::Sports(
            "Stake exceeds maximum allowed amount".to_string()
        ).into());
    }

    if market_id.is_empty() || selection.is_empty() {
        return Err(NeuralTraderError::Sports(
            "Market ID and selection are required".to_string()
        ).into());
    }

    // Calculate potential return
    let potential_return = stake * odds;

    // Generate bet ID
    let bet_id = format!("bet-{}", Uuid::new_v4());

    // Determine status based on validation mode
    let status = if validate {
        // In validation mode, we only check if the bet is valid
        "validated".to_string()
    } else {
        // In real execution mode, we would place the bet with a bookmaker
        // For now, simulate acceptance
        tracing::info!(
            "Executing bet: {} on {} at odds {} with stake {}",
            bet_id, selection, odds, stake
        );
        "accepted".to_string()
    };

    Ok(BetExecution {
        bet_id,
        market_id,
        selection,
        stake,
        odds,
        status,
        potential_return,
    })
}

/// Bet execution result
#[napi(object)]
pub struct BetExecution {
    pub bet_id: String,
    pub market_id: String,
    pub selection: String,
    pub stake: f64,
    pub odds: f64,
    pub status: String,
    pub potential_return: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_sports_events() {
        let events = get_sports_events("basketball".to_string(), Some(7)).await.unwrap();
        assert!(!events.is_empty());
        assert_eq!(events[0].sport, "basketball");
    }

    #[tokio::test]
    async fn test_get_sports_odds() {
        let odds = get_sports_odds("soccer".to_string()).await.unwrap();
        assert!(!odds.is_empty());

        // Check that odds are reasonable
        for odd in &odds {
            assert!(odd.home_odds >= 1.01);
            assert!(odd.away_odds >= 1.01);
            assert!(!odd.bookmaker.is_empty());
        }
    }

    #[tokio::test]
    async fn test_find_sports_arbitrage() {
        let opportunities = find_sports_arbitrage("football".to_string(), Some(0.001))
            .await
            .unwrap();

        // Arbitrage opportunities may or may not exist depending on generated odds
        // Just verify the function executes successfully
        for opp in &opportunities {
            assert!(opp.profit_margin > 0.0);
            assert!(opp.bet_home.stake > 0.0);
            assert!(opp.bet_away.stake > 0.0);
        }
    }

    #[tokio::test]
    async fn test_calculate_kelly_criterion() {
        let result = calculate_kelly_criterion(0.55, 2.0, 1000.0).await.unwrap();

        assert!(result.kelly_fraction >= 0.0);
        assert!(result.kelly_fraction <= 0.25); // Should be capped at 25%
        assert_eq!(result.suggested_stake, result.bankroll * result.kelly_fraction);
    }

    #[tokio::test]
    async fn test_execute_sports_bet_validation() {
        // Test validation mode
        let result = execute_sports_bet(
            "market-123".to_string(),
            "Team A".to_string(),
            100.0,
            2.5,
            Some(true),
        )
        .await
        .unwrap();

        assert_eq!(result.status, "validated");
        assert_eq!(result.potential_return, 250.0);
    }

    #[tokio::test]
    async fn test_execute_sports_bet_invalid_stake() {
        let result = execute_sports_bet(
            "market-123".to_string(),
            "Team A".to_string(),
            0.0, // Invalid stake
            2.5,
            Some(true),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_sports_bet_invalid_odds() {
        let result = execute_sports_bet(
            "market-123".to_string(),
            "Team A".to_string(),
            100.0,
            0.5, // Invalid odds (less than 1.0)
            Some(true),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_sports_bet_excessive_stake() {
        let result = execute_sports_bet(
            "market-123".to_string(),
            "Team A".to_string(),
            200000.0, // Exceeds maximum
            2.5,
            Some(true),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_sports_bet_empty_market() {
        let result = execute_sports_bet(
            "".to_string(), // Empty market ID
            "Team A".to_string(),
            100.0,
            2.5,
            Some(true),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_sports_bet_execution_mode() {
        // Test execution mode (validate_only = false)
        let result = execute_sports_bet(
            "market-456".to_string(),
            "Team B".to_string(),
            50.0,
            1.8,
            Some(false),
        )
        .await
        .unwrap();

        assert_eq!(result.status, "accepted");
        assert_eq!(result.potential_return, 90.0);
    }
}
