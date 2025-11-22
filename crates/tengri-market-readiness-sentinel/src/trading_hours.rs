//! Trading hours validation system for global markets
//!
//! This module provides comprehensive trading hours validation including:
//! - Multi-timezone trading session management
//! - Holiday calendar integration
//! - Market opening/closing validation
//! - Pre-market and after-hours trading support

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc, TimeZone, Datelike, Weekday};
use chrono_tz::Tz;
use uuid::Uuid;

use crate::config::MarketReadinessConfig;
use crate::types::ValidationResult;
use crate::{TradingHoursStatus, ValidationStatus};
use crate::error::MarketReadinessError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSession {
    pub market: String,
    pub timezone: String,
    pub regular_hours: SessionHours,
    pub pre_market: Option<SessionHours>,
    pub after_hours: Option<SessionHours>,
    pub holidays: Vec<HolidayRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHours {
    pub open_time: String,    // HH:MM format
    pub close_time: String,   // HH:MM format
    pub days: Vec<Weekday>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolidayRule {
    pub name: String,
    pub date: String,         // YYYY-MM-DD format
    pub market_closure: MarketClosure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketClosure {
    FullDay,
    EarlyClose(String),      // HH:MM format
    LateOpen(String),        // HH:MM format
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStatus {
    pub market: String,
    pub status: TradingHoursStatus,
    pub next_open: Option<DateTime<Utc>>,
    pub next_close: Option<DateTime<Utc>>,
    pub time_to_event: Option<chrono::Duration>,
    pub session_type: SessionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionType {
    Regular,
    PreMarket,
    AfterHours,
    Closed,
}

#[derive(Debug)]
pub struct TradingHoursValidator {
    config: Arc<MarketReadinessConfig>,
    trading_sessions: Arc<RwLock<HashMap<String, TradingSession>>>,
    market_statuses: Arc<RwLock<HashMap<String, MarketStatus>>>,
    holiday_calendar: Arc<RwLock<HashMap<String, Vec<HolidayRule>>>>,
}

impl TradingHoursValidator {
    pub async fn new(config: Arc<MarketReadinessConfig>) -> Result<Self> {
        Ok(Self {
            config,
            trading_sessions: Arc::new(RwLock::new(HashMap::new())),
            market_statuses: Arc::new(RwLock::new(HashMap::new())),
            holiday_calendar: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing trading hours validator...");
        
        // Load trading sessions configuration
        self.load_trading_sessions().await?;
        
        // Load holiday calendars
        self.load_holiday_calendars().await?;
        
        // Start market status monitoring
        self.start_market_monitoring().await?;
        
        info!("Trading hours validator initialized successfully");
        Ok(())
    }

    pub async fn validate_trading_hours(&self) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        
        // Update all market statuses
        self.update_market_statuses().await?;
        
        let market_statuses = self.market_statuses.read().await;
        let mut validation_issues = Vec::new();
        let mut warnings = Vec::new();
        
        let now = Utc::now();
        
        // Check each market
        for (market, status) in market_statuses.iter() {
            // Validate market status consistency
            if let Some(issue) = self.validate_market_status(market, status, now).await? {
                validation_issues.push(issue);
            }
            
            // Check for upcoming market events
            if let Some(warning) = self.check_upcoming_events(market, status, now).await? {
                warnings.push(warning);
            }
        }
        
        let duration = start_time.elapsed().as_millis() as u64;
        
        // Determine validation result
        let result = if !validation_issues.is_empty() {
            ValidationResult {
                status: ValidationStatus::Failed,
                message: format!("Trading hours validation failed: {}", validation_issues.join(", ")),
                details: Some(serde_json::json!({
                    "issues": validation_issues,
                    "market_statuses": market_statuses.clone(),
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 0.95,
            }
        } else if !warnings.is_empty() {
            ValidationResult {
                status: ValidationStatus::Warning,
                message: format!("Trading hours warnings: {}", warnings.join(", ")),
                details: Some(serde_json::json!({
                    "warnings": warnings,
                    "market_statuses": market_statuses.clone(),
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 0.9,
            }
        } else {
            ValidationResult {
                status: ValidationStatus::Passed,
                message: "All trading hours validations passed".to_string(),
                details: Some(serde_json::json!({
                    "active_markets": market_statuses.values().filter(|s| s.status == TradingHoursStatus::Open).count(),
                    "market_statuses": market_statuses.clone(),
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 1.0,
            }
        };
        
        Ok(result)
    }

    pub async fn get_trading_status(&self) -> Result<TradingHoursStatus> {
        let market_statuses = self.market_statuses.read().await;
        
        // Determine overall trading status
        let mut open_markets = 0;
        let mut total_markets = 0;
        
        for status in market_statuses.values() {
            total_markets += 1;
            if status.status == TradingHoursStatus::Open {
                open_markets += 1;
            }
        }
        
        let status = if open_markets > 0 {
            TradingHoursStatus::Open
        } else if total_markets > 0 {
            // Check if any market is in pre-market or after-hours
            let extended_hours = market_statuses.values().any(|s| {
                matches!(s.status, TradingHoursStatus::PreMarket | TradingHoursStatus::PostMarket)
            });
            
            if extended_hours {
                TradingHoursStatus::PreMarket // Or PostMarket - simplified
            } else {
                TradingHoursStatus::Closed
            }
        } else {
            TradingHoursStatus::Closed
        };
        
        Ok(status)
    }

    async fn load_trading_sessions(&self) -> Result<()> {
        let mut sessions = self.trading_sessions.write().await;
        
        // US Markets
        sessions.insert("NYSE".to_string(), TradingSession {
            market: "NYSE".to_string(),
            timezone: "America/New_York".to_string(),
            regular_hours: SessionHours {
                open_time: "09:30".to_string(),
                close_time: "16:00".to_string(),
                days: vec![Weekday::Mon, Weekday::Tue, Weekday::Wed, Weekday::Thu, Weekday::Fri],
            },
            pre_market: Some(SessionHours {
                open_time: "04:00".to_string(),
                close_time: "09:30".to_string(),
                days: vec![Weekday::Mon, Weekday::Tue, Weekday::Wed, Weekday::Thu, Weekday::Fri],
            }),
            after_hours: Some(SessionHours {
                open_time: "16:00".to_string(),
                close_time: "20:00".to_string(),
                days: vec![Weekday::Mon, Weekday::Tue, Weekday::Wed, Weekday::Thu, Weekday::Fri],
            }),
            holidays: vec![],
        });
        
        // London Stock Exchange
        sessions.insert("LSE".to_string(), TradingSession {
            market: "LSE".to_string(),
            timezone: "Europe/London".to_string(),
            regular_hours: SessionHours {
                open_time: "08:00".to_string(),
                close_time: "16:30".to_string(),
                days: vec![Weekday::Mon, Weekday::Tue, Weekday::Wed, Weekday::Thu, Weekday::Fri],
            },
            pre_market: None,
            after_hours: None,
            holidays: vec![],
        });
        
        // Tokyo Stock Exchange
        sessions.insert("TSE".to_string(), TradingSession {
            market: "TSE".to_string(),
            timezone: "Asia/Tokyo".to_string(),
            regular_hours: SessionHours {
                open_time: "09:00".to_string(),
                close_time: "15:00".to_string(),
                days: vec![Weekday::Mon, Weekday::Tue, Weekday::Wed, Weekday::Thu, Weekday::Fri],
            },
            pre_market: None,
            after_hours: None,
            holidays: vec![],
        });
        
        // Forex (24/5)
        sessions.insert("FX".to_string(), TradingSession {
            market: "FX".to_string(),
            timezone: "UTC".to_string(),
            regular_hours: SessionHours {
                open_time: "22:00".to_string(), // Sunday 22:00 UTC
                close_time: "22:00".to_string(), // Friday 22:00 UTC
                days: vec![Weekday::Mon, Weekday::Tue, Weekday::Wed, Weekday::Thu, Weekday::Fri],
            },
            pre_market: None,
            after_hours: None,
            holidays: vec![],
        });
        
        // Crypto (24/7)
        sessions.insert("CRYPTO".to_string(), TradingSession {
            market: "CRYPTO".to_string(),
            timezone: "UTC".to_string(),
            regular_hours: SessionHours {
                open_time: "00:00".to_string(),
                close_time: "23:59".to_string(),
                days: vec![Weekday::Mon, Weekday::Tue, Weekday::Wed, Weekday::Thu, Weekday::Fri, Weekday::Sat, Weekday::Sun],
            },
            pre_market: None,
            after_hours: None,
            holidays: vec![],
        });
        
        Ok(())
    }

    async fn load_holiday_calendars(&self) -> Result<()> {
        let mut calendars = self.holiday_calendar.write().await;
        
        // Load US holidays
        let us_holidays = vec![
            HolidayRule {
                name: "New Year's Day".to_string(),
                date: "2024-01-01".to_string(),
                market_closure: MarketClosure::FullDay,
            },
            HolidayRule {
                name: "Martin Luther King Jr. Day".to_string(),
                date: "2024-01-15".to_string(),
                market_closure: MarketClosure::FullDay,
            },
            HolidayRule {
                name: "Presidents' Day".to_string(),
                date: "2024-02-19".to_string(),
                market_closure: MarketClosure::FullDay,
            },
            HolidayRule {
                name: "Good Friday".to_string(),
                date: "2024-03-29".to_string(),
                market_closure: MarketClosure::FullDay,
            },
            HolidayRule {
                name: "Memorial Day".to_string(),
                date: "2024-05-27".to_string(),
                market_closure: MarketClosure::FullDay,
            },
            HolidayRule {
                name: "Independence Day".to_string(),
                date: "2024-07-04".to_string(),
                market_closure: MarketClosure::FullDay,
            },
            HolidayRule {
                name: "Labor Day".to_string(),
                date: "2024-09-02".to_string(),
                market_closure: MarketClosure::FullDay,
            },
            HolidayRule {
                name: "Thanksgiving".to_string(),
                date: "2024-11-28".to_string(),
                market_closure: MarketClosure::FullDay,
            },
            HolidayRule {
                name: "Black Friday".to_string(),
                date: "2024-11-29".to_string(),
                market_closure: MarketClosure::EarlyClose("13:00".to_string()),
            },
            HolidayRule {
                name: "Christmas Eve".to_string(),
                date: "2024-12-24".to_string(),
                market_closure: MarketClosure::EarlyClose("13:00".to_string()),
            },
            HolidayRule {
                name: "Christmas Day".to_string(),
                date: "2024-12-25".to_string(),
                market_closure: MarketClosure::FullDay,
            },
            HolidayRule {
                name: "New Year's Eve".to_string(),
                date: "2024-12-31".to_string(),
                market_closure: MarketClosure::EarlyClose("13:00".to_string()),
            },
        ];
        
        calendars.insert("NYSE".to_string(), us_holidays);
        
        Ok(())
    }

    async fn start_market_monitoring(&self) -> Result<()> {
        let market_statuses = self.market_statuses.clone();
        let trading_sessions = self.trading_sessions.clone();
        let holiday_calendar = self.holiday_calendar.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                let sessions = trading_sessions.read().await;
                let calendars = holiday_calendar.read().await;
                let mut statuses = market_statuses.write().await;
                
                for (market, session) in sessions.iter() {
                    let status = Self::calculate_market_status(
                        session,
                        calendars.get(market).unwrap_or(&vec![]),
                        Utc::now(),
                    ).await;
                    
                    if let Ok(status) = status {
                        statuses.insert(market.clone(), status);
                    }
                }
            }
        });
        
        Ok(())
    }

    async fn update_market_statuses(&self) -> Result<()> {
        let sessions = self.trading_sessions.read().await;
        let calendars = self.holiday_calendar.read().await;
        let mut statuses = self.market_statuses.write().await;
        
        for (market, session) in sessions.iter() {
            let status = Self::calculate_market_status(
                session,
                calendars.get(market).unwrap_or(&vec![]),
                Utc::now(),
            ).await?;
            
            statuses.insert(market.clone(), status);
        }
        
        Ok(())
    }

    async fn calculate_market_status(
        session: &TradingSession,
        holidays: &[HolidayRule],
        now: DateTime<Utc>,
    ) -> Result<MarketStatus> {
        let tz: Tz = session.timezone.parse().map_err(|_| {
            MarketReadinessError::ValidationError("Invalid timezone".to_string())
        })?;
        
        let local_time = now.with_timezone(&tz);
        let today = local_time.date_naive();
        let current_weekday = local_time.weekday();
        
        // Check if today is a holiday
        if let Some(holiday) = holidays.iter().find(|h| {
            chrono::NaiveDate::parse_from_str(&h.date, "%Y-%m-%d").map_or(false, |d| d == today)
        }) {
            match &holiday.market_closure {
                MarketClosure::FullDay => {
                    return Ok(MarketStatus {
                        market: session.market.clone(),
                        status: TradingHoursStatus::Holiday,
                        next_open: Self::calculate_next_open(session, holidays, now).await?,
                        next_close: None,
                        time_to_event: None,
                        session_type: SessionType::Closed,
                    });
                },
                MarketClosure::EarlyClose(close_time) => {
                    // Handle early close
                    let early_close = Self::parse_time_in_timezone(close_time, &tz, today)?;
                    let regular_open = Self::parse_time_in_timezone(&session.regular_hours.open_time, &tz, today)?;
                    
                    if now < regular_open {
                        return Ok(MarketStatus {
                            market: session.market.clone(),
                            status: TradingHoursStatus::PreMarket,
                            next_open: Some(regular_open),
                            next_close: Some(early_close),
                            time_to_event: Some(regular_open.signed_duration_since(now)),
                            session_type: SessionType::PreMarket,
                        });
                    } else if now < early_close {
                        return Ok(MarketStatus {
                            market: session.market.clone(),
                            status: TradingHoursStatus::Open,
                            next_open: None,
                            next_close: Some(early_close),
                            time_to_event: Some(early_close.signed_duration_since(now)),
                            session_type: SessionType::Regular,
                        });
                    } else {
                        return Ok(MarketStatus {
                            market: session.market.clone(),
                            status: TradingHoursStatus::Closed,
                            next_open: Self::calculate_next_open(session, holidays, now).await?,
                            next_close: None,
                            time_to_event: None,
                            session_type: SessionType::Closed,
                        });
                    }
                },
                MarketClosure::LateOpen(open_time) => {
                    // Handle late open
                    let late_open = Self::parse_time_in_timezone(open_time, &tz, today)?;
                    let regular_close = Self::parse_time_in_timezone(&session.regular_hours.close_time, &tz, today)?;
                    
                    if now < late_open {
                        return Ok(MarketStatus {
                            market: session.market.clone(),
                            status: TradingHoursStatus::Closed,
                            next_open: Some(late_open),
                            next_close: Some(regular_close),
                            time_to_event: Some(late_open.signed_duration_since(now)),
                            session_type: SessionType::Closed,
                        });
                    } else if now < regular_close {
                        return Ok(MarketStatus {
                            market: session.market.clone(),
                            status: TradingHoursStatus::Open,
                            next_open: None,
                            next_close: Some(regular_close),
                            time_to_event: Some(regular_close.signed_duration_since(now)),
                            session_type: SessionType::Regular,
                        });
                    } else {
                        return Ok(MarketStatus {
                            market: session.market.clone(),
                            status: TradingHoursStatus::Closed,
                            next_open: Self::calculate_next_open(session, holidays, now).await?,
                            next_close: None,
                            time_to_event: None,
                            session_type: SessionType::Closed,
                        });
                    }
                },
            }
        }
        
        // Check if today is a trading day
        if !session.regular_hours.days.contains(&current_weekday) {
            return Ok(MarketStatus {
                market: session.market.clone(),
                status: TradingHoursStatus::Closed,
                next_open: Self::calculate_next_open(session, holidays, now).await?,
                next_close: None,
                time_to_event: None,
                session_type: SessionType::Closed,
            });
        }
        
        // Calculate trading hours for today
        let regular_open = Self::parse_time_in_timezone(&session.regular_hours.open_time, &tz, today)?;
        let regular_close = Self::parse_time_in_timezone(&session.regular_hours.close_time, &tz, today)?;
        
        // Check pre-market hours
        if let Some(pre_market) = &session.pre_market {
            let pre_open = Self::parse_time_in_timezone(&pre_market.open_time, &tz, today)?;
            
            if now >= pre_open && now < regular_open {
                return Ok(MarketStatus {
                    market: session.market.clone(),
                    status: TradingHoursStatus::PreMarket,
                    next_open: Some(regular_open),
                    next_close: Some(regular_close),
                    time_to_event: Some(regular_open.signed_duration_since(now)),
                    session_type: SessionType::PreMarket,
                });
            }
        }
        
        // Check regular hours
        if now >= regular_open && now < regular_close {
            return Ok(MarketStatus {
                market: session.market.clone(),
                status: TradingHoursStatus::Open,
                next_open: None,
                next_close: Some(regular_close),
                time_to_event: Some(regular_close.signed_duration_since(now)),
                session_type: SessionType::Regular,
            });
        }
        
        // Check after-hours
        if let Some(after_hours) = &session.after_hours {
            let after_close = Self::parse_time_in_timezone(&after_hours.close_time, &tz, today)?;
            
            if now >= regular_close && now < after_close {
                return Ok(MarketStatus {
                    market: session.market.clone(),
                    status: TradingHoursStatus::PostMarket,
                    next_open: Self::calculate_next_open(session, holidays, now).await?,
                    next_close: Some(after_close),
                    time_to_event: Some(after_close.signed_duration_since(now)),
                    session_type: SessionType::AfterHours,
                });
            }
        }
        
        // Market is closed
        Ok(MarketStatus {
            market: session.market.clone(),
            status: TradingHoursStatus::Closed,
            next_open: Self::calculate_next_open(session, holidays, now).await?,
            next_close: None,
            time_to_event: None,
            session_type: SessionType::Closed,
        })
    }

    async fn calculate_next_open(
        session: &TradingSession,
        holidays: &[HolidayRule],
        now: DateTime<Utc>,
    ) -> Result<Option<DateTime<Utc>>> {
        let tz: Tz = session.timezone.parse().map_err(|_| {
            MarketReadinessError::ValidationError("Invalid timezone".to_string())
        })?;
        
        let local_time = now.with_timezone(&tz);
        let mut check_date = local_time.date_naive();
        
        // Look for next trading day within reasonable time (30 days)
        for _ in 0..30 {
            check_date = check_date.succ_opt().unwrap_or(check_date);
            let check_weekday = check_date.weekday();
            
            // Check if it's a trading day
            if !session.regular_hours.days.contains(&check_weekday) {
                continue;
            }
            
            // Check if it's a holiday
            if holidays.iter().any(|h| {
                chrono::NaiveDate::parse_from_str(&h.date, "%Y-%m-%d").map_or(false, |d| d == check_date)
            }) {
                continue;
            }
            
            // Found next trading day
            let next_open = Self::parse_time_in_timezone(&session.regular_hours.open_time, &tz, check_date)?;
            return Ok(Some(next_open));
        }
        
        Ok(None)
    }

    fn parse_time_in_timezone(
        time_str: &str,
        tz: &Tz,
        date: chrono::NaiveDate,
    ) -> Result<DateTime<Utc>> {
        let time_parts: Vec<&str> = time_str.split(':').collect();
        if time_parts.len() != 2 {
            return Err(MarketReadinessError::ValidationError(
                format!("Invalid time format: {}", time_str)
            ).into());
        }
        
        let hour: u32 = time_parts[0].parse().map_err(|_| {
            MarketReadinessError::ValidationError(format!("Invalid hour: {}", time_parts[0]))
        })?;
        
        let minute: u32 = time_parts[1].parse().map_err(|_| {
            MarketReadinessError::ValidationError(format!("Invalid minute: {}", time_parts[1]))
        })?;
        
        let naive_datetime = date.and_hms_opt(hour, minute, 0).ok_or_else(|| {
            MarketReadinessError::ValidationError("Invalid time".to_string())
        })?;
        
        let local_datetime = tz.from_local_datetime(&naive_datetime).single().ok_or_else(|| {
            MarketReadinessError::ValidationError("Timezone conversion failed".to_string())
        })?;
        
        Ok(local_datetime.with_timezone(&Utc))
    }

    async fn validate_market_status(
        &self,
        market: &str,
        status: &MarketStatus,
        now: DateTime<Utc>,
    ) -> Result<Option<String>> {
        // Validate status consistency
        if let Some(next_close) = status.next_close {
            if next_close < now {
                return Ok(Some(format!("Market {} next close time is in the past", market)));
            }
        }
        
        if let Some(next_open) = status.next_open {
            if next_open < now {
                return Ok(Some(format!("Market {} next open time is in the past", market)));
            }
        }
        
        // Validate status and session type consistency
        let valid_combination = match (&status.status, &status.session_type) {
            (TradingHoursStatus::Open, SessionType::Regular) => true,
            (TradingHoursStatus::PreMarket, SessionType::PreMarket) => true,
            (TradingHoursStatus::PostMarket, SessionType::AfterHours) => true,
            (TradingHoursStatus::Closed, SessionType::Closed) => true,
            (TradingHoursStatus::Holiday, SessionType::Closed) => true,
            _ => false,
        };
        
        if !valid_combination {
            return Ok(Some(format!(
                "Market {} has inconsistent status {:?} and session type {:?}",
                market, status.status, status.session_type
            )));
        }
        
        Ok(None)
    }

    async fn check_upcoming_events(
        &self,
        market: &str,
        status: &MarketStatus,
        now: DateTime<Utc>,
    ) -> Result<Option<String>> {
        // Check for upcoming market events within 30 minutes
        let warning_threshold = chrono::Duration::minutes(30);
        
        if let Some(next_close) = status.next_close {
            let time_to_close = next_close.signed_duration_since(now);
            if time_to_close > chrono::Duration::zero() && time_to_close <= warning_threshold {
                return Ok(Some(format!(
                    "Market {} closing in {} minutes",
                    market,
                    time_to_close.num_minutes()
                )));
            }
        }
        
        if let Some(next_open) = status.next_open {
            let time_to_open = next_open.signed_duration_since(now);
            if time_to_open > chrono::Duration::zero() && time_to_open <= warning_threshold {
                return Ok(Some(format!(
                    "Market {} opening in {} minutes",
                    market,
                    time_to_open.num_minutes()
                )));
            }
        }
        
        Ok(None)
    }
}