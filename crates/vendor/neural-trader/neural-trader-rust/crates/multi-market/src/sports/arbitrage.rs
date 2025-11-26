//! Arbitrage Detection for Sports Betting
//!
//! Detects risk-free arbitrage opportunities across multiple bookmakers

use crate::error::{MultiMarketError, Result};
use crate::sports::odds_api::{BookmakerOdds, Event};
use crate::types::ArbitrageOpportunity;
use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Arbitrage bet for a specific outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageBet {
    /// Bookmaker name
    pub bookmaker: String,
    /// Outcome name (e.g., "Home Win", "Away Win")
    pub outcome: String,
    /// Decimal odds
    pub odds: Decimal,
    /// Stake amount
    pub stake: Decimal,
    /// Expected payout
    pub payout: Decimal,
}

/// Sports arbitrage opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SportsArbitrageOpportunity {
    /// Event identifier
    pub event_id: String,
    /// Event description
    pub event_description: String,
    /// All bets required for arbitrage
    pub bets: Vec<ArbitrageBet>,
    /// Total stake required
    pub total_stake: Decimal,
    /// Guaranteed profit
    pub profit: Decimal,
    /// Profit margin percentage
    pub profit_margin: Decimal,
    /// Arbitrage confidence score (0-1)
    pub confidence: Decimal,
    /// Time detected
    pub detected_at: chrono::DateTime<Utc>,
}

/// Arbitrage detector configuration
#[derive(Debug, Clone)]
pub struct ArbitrageDetectorConfig {
    /// Minimum profit margin to report (default: 1%)
    pub min_profit_margin: Decimal,
    /// Default stake amount for calculations
    pub default_stake: Decimal,
    /// Bookmaker commission/vig to account for
    pub commission: Decimal,
    /// Minimum confidence score (default: 0.7)
    pub min_confidence: Decimal,
}

impl Default for ArbitrageDetectorConfig {
    fn default() -> Self {
        Self {
            min_profit_margin: dec!(0.01),  // 1%
            default_stake: dec!(1000),
            commission: dec!(0.02),          // 2% commission
            min_confidence: dec!(0.7),
        }
    }
}

/// Arbitrage opportunity detector
pub struct ArbitrageDetector {
    config: ArbitrageDetectorConfig,
}

impl ArbitrageDetector {
    /// Create new arbitrage detector with default config
    pub fn new() -> Self {
        Self {
            config: ArbitrageDetectorConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ArbitrageDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect arbitrage opportunities in a list of events
    pub fn detect_opportunities(&self, events: &[Event]) -> Result<Vec<SportsArbitrageOpportunity>> {
        let mut opportunities = Vec::new();

        for event in events {
            // Check 2-way arbitrage (e.g., home/away)
            if let Some(opp) = self.check_two_way_arbitrage(event)? {
                opportunities.push(opp);
            }

            // Check 3-way arbitrage (e.g., home/draw/away)
            if let Some(opp) = self.check_three_way_arbitrage(event)? {
                opportunities.push(opp);
            }
        }

        // Sort by profit margin descending
        opportunities.sort_by(|a, b| b.profit_margin.cmp(&a.profit_margin));

        Ok(opportunities)
    }

    /// Check for 2-way arbitrage (binary outcome)
    fn check_two_way_arbitrage(&self, event: &Event) -> Result<Option<SportsArbitrageOpportunity>> {
        let home_odds = event.bookmaker_odds.get("home");
        let away_odds = event.bookmaker_odds.get("away");

        if home_odds.is_none() || away_odds.is_none() {
            return Ok(None);
        }

        let home_odds = home_odds.unwrap();
        let away_odds = away_odds.unwrap();

        // Find best odds for each outcome
        let best_home = home_odds.iter().max_by_key(|o| o.odds);
        let best_away = away_odds.iter().max_by_key(|o| o.odds);

        if best_home.is_none() || best_away.is_none() {
            return Ok(None);
        }

        let best_home = best_home.unwrap();
        let best_away = best_away.unwrap();

        // Calculate total implied probability
        let implied_prob_home = Decimal::ONE / best_home.odds;
        let implied_prob_away = Decimal::ONE / best_away.odds;
        let total_implied_prob = implied_prob_home + implied_prob_away;

        // Account for commission
        let total_implied_prob_with_commission = total_implied_prob * (Decimal::ONE + self.config.commission);

        // Check if arbitrage exists
        if total_implied_prob_with_commission >= Decimal::ONE {
            return Ok(None); // No arbitrage
        }

        // Calculate profit margin
        let profit_margin = Decimal::ONE - total_implied_prob_with_commission;

        if profit_margin < self.config.min_profit_margin {
            return Ok(None); // Profit too small
        }

        // Calculate optimal stakes
        let total_stake = self.config.default_stake;
        let stake_home = (total_stake * implied_prob_home) / total_implied_prob;
        let stake_away = (total_stake * implied_prob_away) / total_implied_prob;

        // Calculate payouts and profit
        let payout_home = stake_home * best_home.odds;
        let payout_away = stake_away * best_away.odds;
        let guaranteed_payout = payout_home.min(payout_away);
        let profit = guaranteed_payout - total_stake;

        // Calculate confidence based on odds freshness and margin
        let confidence = self.calculate_confidence(
            &[best_home, best_away],
            profit_margin,
        );

        if confidence < self.config.min_confidence {
            return Ok(None);
        }

        let opportunity = SportsArbitrageOpportunity {
            event_id: event.id.clone(),
            event_description: format!("{} vs {}", event.home_team, event.away_team),
            bets: vec![
                ArbitrageBet {
                    bookmaker: best_home.bookmaker.clone(),
                    outcome: "Home Win".to_string(),
                    odds: best_home.odds,
                    stake: stake_home,
                    payout: payout_home,
                },
                ArbitrageBet {
                    bookmaker: best_away.bookmaker.clone(),
                    outcome: "Away Win".to_string(),
                    odds: best_away.odds,
                    stake: stake_away,
                    payout: payout_away,
                },
            ],
            total_stake,
            profit,
            profit_margin,
            confidence,
            detected_at: Utc::now(),
        };

        Ok(Some(opportunity))
    }

    /// Check for 3-way arbitrage (home/draw/away)
    fn check_three_way_arbitrage(&self, event: &Event) -> Result<Option<SportsArbitrageOpportunity>> {
        let home_odds = event.bookmaker_odds.get("home");
        let draw_odds = event.bookmaker_odds.get("draw");
        let away_odds = event.bookmaker_odds.get("away");

        if home_odds.is_none() || draw_odds.is_none() || away_odds.is_none() {
            return Ok(None);
        }

        let home_odds = home_odds.unwrap();
        let draw_odds = draw_odds.unwrap();
        let away_odds = away_odds.unwrap();

        // Find best odds for each outcome
        let best_home = home_odds.iter().max_by_key(|o| o.odds);
        let best_draw = draw_odds.iter().max_by_key(|o| o.odds);
        let best_away = away_odds.iter().max_by_key(|o| o.odds);

        if best_home.is_none() || best_draw.is_none() || best_away.is_none() {
            return Ok(None);
        }

        let best_home = best_home.unwrap();
        let best_draw = best_draw.unwrap();
        let best_away = best_away.unwrap();

        // Calculate total implied probability
        let implied_prob_home = Decimal::ONE / best_home.odds;
        let implied_prob_draw = Decimal::ONE / best_draw.odds;
        let implied_prob_away = Decimal::ONE / best_away.odds;
        let total_implied_prob = implied_prob_home + implied_prob_draw + implied_prob_away;

        // Account for commission
        let total_implied_prob_with_commission = total_implied_prob * (Decimal::ONE + self.config.commission);

        // Check if arbitrage exists
        if total_implied_prob_with_commission >= Decimal::ONE {
            return Ok(None);
        }

        // Calculate profit margin
        let profit_margin = Decimal::ONE - total_implied_prob_with_commission;

        if profit_margin < self.config.min_profit_margin {
            return Ok(None);
        }

        // Calculate optimal stakes
        let total_stake = self.config.default_stake;
        let stake_home = (total_stake * implied_prob_home) / total_implied_prob;
        let stake_draw = (total_stake * implied_prob_draw) / total_implied_prob;
        let stake_away = (total_stake * implied_prob_away) / total_implied_prob;

        // Calculate payouts and profit
        let payout_home = stake_home * best_home.odds;
        let payout_draw = stake_draw * best_draw.odds;
        let payout_away = stake_away * best_away.odds;
        let guaranteed_payout = payout_home.min(payout_draw).min(payout_away);
        let profit = guaranteed_payout - total_stake;

        // Calculate confidence
        let confidence = self.calculate_confidence(
            &[best_home, best_draw, best_away],
            profit_margin,
        );

        if confidence < self.config.min_confidence {
            return Ok(None);
        }

        let opportunity = SportsArbitrageOpportunity {
            event_id: event.id.clone(),
            event_description: format!("{} vs {}", event.home_team, event.away_team),
            bets: vec![
                ArbitrageBet {
                    bookmaker: best_home.bookmaker.clone(),
                    outcome: "Home Win".to_string(),
                    odds: best_home.odds,
                    stake: stake_home,
                    payout: payout_home,
                },
                ArbitrageBet {
                    bookmaker: best_draw.bookmaker.clone(),
                    outcome: "Draw".to_string(),
                    odds: best_draw.odds,
                    stake: stake_draw,
                    payout: payout_draw,
                },
                ArbitrageBet {
                    bookmaker: best_away.bookmaker.clone(),
                    outcome: "Away Win".to_string(),
                    odds: best_away.odds,
                    stake: stake_away,
                    payout: payout_away,
                },
            ],
            total_stake,
            profit,
            profit_margin,
            confidence,
            detected_at: Utc::now(),
        };

        Ok(Some(opportunity))
    }

    /// Calculate confidence score for an arbitrage opportunity
    fn calculate_confidence(
        &self,
        odds: &[&BookmakerOdds],
        profit_margin: Decimal,
    ) -> Decimal {
        // Base confidence from profit margin (higher margin = higher confidence)
        let margin_factor = (profit_margin / dec!(0.05)).min(Decimal::ONE);

        // Recency factor (fresher odds = higher confidence)
        let now = Utc::now();
        let avg_age_seconds: i64 = odds
            .iter()
            .map(|o| (now - o.last_update).num_seconds())
            .sum::<i64>()
            / odds.len() as i64;

        let recency_factor = if avg_age_seconds < 60 {
            Decimal::ONE
        } else if avg_age_seconds < 300 {
            dec!(0.8)
        } else if avg_age_seconds < 900 {
            dec!(0.6)
        } else {
            dec!(0.4)
        };

        // Bookmaker diversity factor (different bookmakers = higher confidence)
        let unique_bookmakers: std::collections::HashSet<_> =
            odds.iter().map(|o| &o.bookmaker).collect();
        let diversity_factor = if unique_bookmakers.len() == odds.len() {
            Decimal::ONE
        } else {
            dec!(0.8)
        };

        // Combined confidence score
        let confidence = margin_factor * recency_factor * diversity_factor;
        confidence.max(Decimal::ZERO).min(Decimal::ONE)
    }
}

impl Default for ArbitrageDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_event() -> Event {
        use chrono::Utc;
        use std::collections::HashMap;

        Event {
            id: "test_event".to_string(),
            sport: "basketball_nba".to_string(),
            commence_time: Utc::now(),
            home_team: "Lakers".to_string(),
            away_team: "Warriors".to_string(),
            bookmaker_odds: {
                let mut map = HashMap::new();
                map.insert(
                    "home".to_string(),
                    vec![BookmakerOdds {
                        bookmaker: "draftkings".to_string(),
                        odds: dec!(2.1),
                        point: None,
                        last_update: Utc::now(),
                    }],
                );
                map.insert(
                    "away".to_string(),
                    vec![BookmakerOdds {
                        bookmaker: "fanduel".to_string(),
                        odds: dec!(2.1),
                        point: None,
                        last_update: Utc::now(),
                    }],
                );
                map
            },
        }
    }

    #[test]
    fn test_arbitrage_detection() {
        let detector = ArbitrageDetector::new();
        let event = create_test_event();

        let opportunities = detector.detect_opportunities(&[event]).unwrap();

        // With odds of 2.1 on both sides: 1/2.1 + 1/2.1 = 0.952 < 1.0
        // So arbitrage should exist
        assert!(opportunities.len() > 0);
    }

    #[test]
    fn test_no_arbitrage() {
        let detector = ArbitrageDetector::new();

        let mut event = create_test_event();
        event.bookmaker_odds.get_mut("home").unwrap()[0].odds = dec!(1.5);
        event.bookmaker_odds.get_mut("away").unwrap()[0].odds = dec!(1.5);

        let opportunities = detector.detect_opportunities(&[event]).unwrap();

        // With odds of 1.5: 1/1.5 + 1/1.5 = 1.33 > 1.0
        // No arbitrage
        assert_eq!(opportunities.len(), 0);
    }

    #[test]
    fn test_profit_calculation() {
        let detector = ArbitrageDetector::with_config(ArbitrageDetectorConfig {
            min_profit_margin: dec!(0.01),
            default_stake: dec!(1000),
            commission: dec!(0.0), // No commission for testing
            min_confidence: dec!(0.5),
        });

        let event = create_test_event();
        let opportunities = detector.detect_opportunities(&[event]).unwrap();

        assert!(opportunities.len() > 0);
        let opp = &opportunities[0];

        // Verify profit is positive
        assert!(opp.profit > Decimal::ZERO);
        assert!(opp.profit_margin > Decimal::ZERO);

        // Verify stakes sum to default stake
        let total_stake: Decimal = opp.bets.iter().map(|b| b.stake).sum();
        assert_eq!(total_stake, dec!(1000));
    }
}
