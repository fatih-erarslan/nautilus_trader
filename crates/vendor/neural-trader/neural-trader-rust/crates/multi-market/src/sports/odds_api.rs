//! The Odds API Client
//!
//! Integration with The Odds API for real-time sports betting odds

use crate::error::{MultiMarketError, Result};
use chrono::{DateTime, Utc};
use reqwest::{Client, StatusCode};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

const API_BASE_URL: &str = "https://api.the-odds-api.com/v4";

/// Supported sports
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sport {
    #[serde(rename = "americanfootball_nfl")]
    AmericanFootballNfl,
    #[serde(rename = "basketball_nba")]
    BasketballNba,
    #[serde(rename = "basketball_ncaab")]
    BasketballNcaab,
    #[serde(rename = "baseball_mlb")]
    BaseballMlb,
    #[serde(rename = "icehockey_nhl")]
    IcehockeyNhl,
    #[serde(rename = "soccer_epl")]
    SoccerEpl,
    #[serde(rename = "soccer_uefa_champs_league")]
    SoccerUefaChampsLeague,
    #[serde(rename = "tennis_atp")]
    TennisAtp,
    #[serde(rename = "boxing_boxing")]
    Boxing,
    #[serde(rename = "mixed_martial_arts_ufc")]
    Mma,
}

impl Sport {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AmericanFootballNfl => "americanfootball_nfl",
            Self::BasketballNba => "basketball_nba",
            Self::BasketballNcaab => "basketball_ncaab",
            Self::BaseballMlb => "baseball_mlb",
            Self::IcehockeyNhl => "icehockey_nhl",
            Self::SoccerEpl => "soccer_epl",
            Self::SoccerUefaChampsLeague => "soccer_uefa_champs_league",
            Self::TennisAtp => "tennis_atp",
            Self::Boxing => "boxing_boxing",
            Self::Mma => "mixed_martial_arts_ufc",
        }
    }
}

/// Betting markets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Market {
    /// Head to head (moneyline)
    #[serde(rename = "h2h")]
    H2h,
    /// Point spreads
    #[serde(rename = "spreads")]
    Spreads,
    /// Over/under totals
    #[serde(rename = "totals")]
    Totals,
}

/// Bookmaker odds for an outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookmakerOdds {
    pub bookmaker: String,
    pub odds: Decimal,
    pub point: Option<Decimal>,
    pub last_update: DateTime<Utc>,
}

/// Sports event with odds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: String,
    pub sport: String,
    pub commence_time: DateTime<Utc>,
    pub home_team: String,
    pub away_team: String,
    pub bookmaker_odds: HashMap<String, Vec<BookmakerOdds>>,
}

/// Arbitrage opportunity in sports betting
#[derive(Debug, Clone)]
pub struct SportsArbitrage {
    pub event_id: String,
    pub home_team: String,
    pub away_team: String,
    pub bookmaker_1: String,
    pub bookmaker_2: String,
    pub odds_1: Decimal,
    pub odds_2: Decimal,
    pub profit_margin: Decimal,
    pub stake_1: Decimal,
    pub stake_2: Decimal,
}

/// Rate limiter for API requests
#[derive(Debug)]
struct RateLimiter {
    tokens: Arc<RwLock<f64>>,
    capacity: f64,
    refill_rate: f64,
    last_refill: Arc<RwLock<DateTime<Utc>>>,
}

impl RateLimiter {
    fn new(requests_per_second: f64, burst_capacity: f64) -> Self {
        Self {
            tokens: Arc::new(RwLock::new(burst_capacity)),
            capacity: burst_capacity,
            refill_rate: requests_per_second,
            last_refill: Arc::new(RwLock::new(Utc::now())),
        }
    }

    async fn acquire(&self) -> Result<()> {
        loop {
            let now = Utc::now();
            let mut tokens = self.tokens.write().await;
            let mut last_refill = self.last_refill.write().await;

            // Refill tokens
            let elapsed = (now - *last_refill).num_milliseconds() as f64 / 1000.0;
            *tokens = (*tokens + elapsed * self.refill_rate).min(self.capacity);
            *last_refill = now;

            if *tokens >= 1.0 {
                *tokens -= 1.0;
                return Ok(());
            }

            drop(tokens);
            drop(last_refill);

            // Wait and retry
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }
}

/// The Odds API client
pub struct OddsApiClient {
    api_key: String,
    client: Client,
    rate_limiter: RateLimiter,
    event_cache: Arc<RwLock<HashMap<String, Event>>>,
}

impl OddsApiClient {
    /// Create new Odds API client
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: Client::new(),
            rate_limiter: RateLimiter::new(5.0, 50.0), // 5 requests/sec, burst of 50
            event_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get available sports
    pub async fn get_sports(&self) -> Result<Vec<String>> {
        self.rate_limiter.acquire().await?;

        let url = format!("{}/sports", API_BASE_URL);
        let response = self
            .client
            .get(&url)
            .query(&[("apiKey", &self.api_key)])
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(MultiMarketError::ApiError(format!(
                "Failed to get sports: {}",
                response.status()
            )));
        }

        let sports: Vec<serde_json::Value> = response.json().await?;
        Ok(sports.iter().filter_map(|s| s.get("key").and_then(|k| k.as_str()).map(String::from)).collect())
    }

    /// Get odds for a specific sport
    pub async fn get_odds(
        &self,
        sport: Sport,
        markets: &[Market],
        regions: &[&str],
    ) -> Result<Vec<Event>> {
        self.rate_limiter.acquire().await?;

        let markets_str = markets
            .iter()
            .map(|m| match m {
                Market::H2h => "h2h",
                Market::Spreads => "spreads",
                Market::Totals => "totals",
            })
            .collect::<Vec<_>>()
            .join(",");

        let regions_str = regions.join(",");

        let url = format!("{}/sports/{}/odds", API_BASE_URL, sport.as_str());
        let response = self
            .client
            .get(&url)
            .query(&[
                ("apiKey", self.api_key.as_str()),
                ("markets", &markets_str),
                ("regions", &regions_str),
                ("oddsFormat", "decimal"),
                ("dateFormat", "iso"),
            ])
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(MultiMarketError::ApiError(format!(
                "Failed to get odds: {}",
                response.status()
            )));
        }

        let data: Vec<serde_json::Value> = response.json().await?;
        let mut events = Vec::new();

        for event_data in data {
            if let Some(event) = self.parse_event(event_data) {
                // Cache event
                let mut cache = self.event_cache.write().await;
                cache.insert(event.id.clone(), event.clone());
                events.push(event);
            }
        }

        info!("Retrieved {} events for {:?}", events.len(), sport);
        Ok(events)
    }

    /// Find arbitrage opportunities
    pub fn find_arbitrage(
        &self,
        events: &[Event],
        min_profit_margin: Decimal,
    ) -> Vec<SportsArbitrage> {
        let mut opportunities = Vec::new();

        for event in events {
            // Check for h2h arbitrage
            if let Some(home_odds) = event.bookmaker_odds.get("home") {
                if let Some(away_odds) = event.bookmaker_odds.get("away") {
                    // Find best odds for each outcome
                    if let Some(best_home) = home_odds.iter().max_by_key(|o| o.odds) {
                        if let Some(best_away) = away_odds.iter().max_by_key(|o| o.odds) {
                            // Calculate if arbitrage exists
                            let total_implied_prob =
                                Decimal::ONE / best_home.odds + Decimal::ONE / best_away.odds;

                            if total_implied_prob < Decimal::ONE {
                                let profit_margin = Decimal::ONE - total_implied_prob;

                                if profit_margin >= min_profit_margin {
                                    // Calculate optimal stakes for $1000
                                    let total_stake = Decimal::from(1000);
                                    let stake_1 = total_stake / best_home.odds / total_implied_prob;
                                    let stake_2 = total_stake / best_away.odds / total_implied_prob;

                                    opportunities.push(SportsArbitrage {
                                        event_id: event.id.clone(),
                                        home_team: event.home_team.clone(),
                                        away_team: event.away_team.clone(),
                                        bookmaker_1: best_home.bookmaker.clone(),
                                        bookmaker_2: best_away.bookmaker.clone(),
                                        odds_1: best_home.odds,
                                        odds_2: best_away.odds,
                                        profit_margin,
                                        stake_1,
                                        stake_2,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        info!("Found {} arbitrage opportunities", opportunities.len());
        opportunities
    }

    fn parse_event(&self, data: serde_json::Value) -> Option<Event> {
        let id = data.get("id")?.as_str()?.to_string();
        let sport = data.get("sport_key")?.as_str()?.to_string();
        let commence_time = data
            .get("commence_time")?
            .as_str()?
            .parse::<DateTime<Utc>>()
            .ok()?;
        let home_team = data.get("home_team")?.as_str()?.to_string();
        let away_team = data.get("away_team")?.as_str()?.to_string();

        let mut bookmaker_odds: HashMap<String, Vec<BookmakerOdds>> = HashMap::new();

        if let Some(bookmakers) = data.get("bookmakers").and_then(|b| b.as_array()) {
            for bookmaker in bookmakers {
                let bookmaker_name = bookmaker.get("key")?.as_str()?.to_string();
                let last_update = bookmaker
                    .get("last_update")?
                    .as_str()?
                    .parse::<DateTime<Utc>>()
                    .ok()?;

                if let Some(markets) = bookmaker.get("markets").and_then(|m| m.as_array()) {
                    for market in markets {
                        if let Some(outcomes) = market.get("outcomes").and_then(|o| o.as_array()) {
                            for outcome in outcomes {
                                let name = outcome.get("name")?.as_str()?.to_lowercase();
                                let price = outcome
                                    .get("price")?
                                    .as_f64()?;
                                let point = outcome
                                    .get("point")
                                    .and_then(|p| p.as_f64())
                                    .map(Decimal::try_from);

                                let odds = BookmakerOdds {
                                    bookmaker: bookmaker_name.clone(),
                                    odds: Decimal::try_from(price).ok()?,
                                    point: point.and_then(|p| p.ok()),
                                    last_update,
                                };

                                bookmaker_odds
                                    .entry(if name.contains("home") || name == home_team.to_lowercase() {
                                        "home"
                                    } else {
                                        "away"
                                    }.to_string())
                                    .or_default()
                                    .push(odds);
                            }
                        }
                    }
                }
            }
        }

        Some(Event {
            id,
            sport,
            commence_time,
            home_team,
            away_team,
            bookmaker_odds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sport_as_str() {
        assert_eq!(Sport::BasketballNba.as_str(), "basketball_nba");
        assert_eq!(Sport::AmericanFootballNfl.as_str(), "americanfootball_nfl");
    }

    #[test]
    fn test_arbitrage_detection() {
        let event = Event {
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
                        odds: Decimal::new(21, 1), // 2.1
                        point: None,
                        last_update: Utc::now(),
                    }],
                );
                map.insert(
                    "away".to_string(),
                    vec![BookmakerOdds {
                        bookmaker: "fanduel".to_string(),
                        odds: Decimal::new(21, 1), // 2.1
                        point: None,
                        last_update: Utc::now(),
                    }],
                );
                map
            },
        };

        let client = OddsApiClient::new("test_key");
        let opportunities = client.find_arbitrage(&[event], Decimal::new(1, 2)); // 0.01 = 1%

        // With 2.1 odds on both sides: 1/2.1 + 1/2.1 = 0.952 < 1.0, so arbitrage exists
        assert!(opportunities.len() > 0);
    }
}
