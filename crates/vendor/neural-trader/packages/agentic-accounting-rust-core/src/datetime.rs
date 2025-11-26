/*!
 * Date/time utilities for tax calculations
 *
 * Handles ISO 8601 date parsing, formatting, and wash sale period calculations
 */

use chrono::{DateTime, Utc, Duration, NaiveDateTime, Datelike, TimeZone};
use crate::error::{RustCoreError, Result};

/// Parse an ISO 8601 date string to a DateTime<Utc>
pub fn parse_datetime_internal(date_str: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(date_str)
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|_| {
            // Try parsing as naive datetime and assume UTC
            NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S")
                .map(|ndt| Utc.from_utc_datetime(&ndt))
        })
        .map_err(|e| RustCoreError::DateTimeError(format!("Failed to parse datetime: {}", e)))
}

/// Format a DateTime<Utc> as ISO 8601 string
pub fn format_datetime_internal(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339()
}

/// Parse an ISO 8601 date string (NAPI export)
#[napi]
pub fn parse_datetime(date_str: String) -> napi::Result<String> {
    let dt = parse_datetime_internal(&date_str)
        .map_err(|e| napi::Error::from_reason(format!("{}", e)))?;
    Ok(format_datetime_internal(&dt))
}

/// Format a date string to ISO 8601 (NAPI export)
#[napi]
pub fn format_datetime(date_str: String) -> napi::Result<String> {
    let dt = parse_datetime_internal(&date_str)
        .map_err(|e| napi::Error::from_reason(format!("{}", e)))?;
    Ok(format_datetime_internal(&dt))
}

/// Calculate the number of days between two dates
#[napi]
pub fn days_between(date1: String, date2: String) -> napi::Result<i64> {
    let dt1 = parse_datetime_internal(&date1)
        .map_err(|e| napi::Error::from_reason(format!("Invalid date1: {}", e)))?;
    let dt2 = parse_datetime_internal(&date2)
        .map_err(|e| napi::Error::from_reason(format!("Invalid date2: {}", e)))?;

    let duration = dt2.signed_duration_since(dt1);
    Ok(duration.num_days())
}

/// Check if a date is within the wash sale period (30 days before or after)
/// US tax law: A wash sale occurs when you sell a security at a loss and
/// purchase the same or substantially identical security within 30 days
/// before or after the sale
#[napi]
pub fn is_within_wash_sale_period(sale_date: String, purchase_date: String) -> napi::Result<bool> {
    let sale_dt = parse_datetime_internal(&sale_date)
        .map_err(|e| napi::Error::from_reason(format!("Invalid sale date: {}", e)))?;
    let purchase_dt = parse_datetime_internal(&purchase_date)
        .map_err(|e| napi::Error::from_reason(format!("Invalid purchase date: {}", e)))?;

    let days_diff = (purchase_dt.signed_duration_since(sale_dt)).num_days();

    // Within 30 days before or 30 days after
    Ok(days_diff >= -30 && days_diff <= 30)
}

/// Add days to a date
#[napi]
pub fn add_days(date_str: String, days: i64) -> napi::Result<String> {
    let dt = parse_datetime_internal(&date_str)
        .map_err(|e| napi::Error::from_reason(format!("Invalid date: {}", e)))?;

    let new_dt = dt + Duration::days(days);
    Ok(format_datetime_internal(&new_dt))
}

/// Subtract days from a date
#[napi]
pub fn subtract_days(date_str: String, days: i64) -> napi::Result<String> {
    let dt = parse_datetime_internal(&date_str)
        .map_err(|e| napi::Error::from_reason(format!("Invalid date: {}", e)))?;

    let new_dt = dt - Duration::days(days);
    Ok(format_datetime_internal(&new_dt))
}

/// Get the start of the tax year for a given date (January 1st)
#[napi]
pub fn get_tax_year_start(date_str: String) -> napi::Result<String> {
    let dt = parse_datetime_internal(&date_str)
        .map_err(|e| napi::Error::from_reason(format!("Invalid date: {}", e)))?;

    let year = dt.year();
    let tax_year_start = DateTime::parse_from_rfc3339(&format!("{}-01-01T00:00:00Z", year))
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| napi::Error::from_reason(format!("Failed to create tax year start: {}", e)))?;

    Ok(format_datetime_internal(&tax_year_start))
}

/// Get the end of the tax year for a given date (December 31st)
#[napi]
pub fn get_tax_year_end(date_str: String) -> napi::Result<String> {
    let dt = parse_datetime_internal(&date_str)
        .map_err(|e| napi::Error::from_reason(format!("Invalid date: {}", e)))?;

    let year = dt.year();
    let tax_year_end = DateTime::parse_from_rfc3339(&format!("{}-12-31T23:59:59Z", year))
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| napi::Error::from_reason(format!("Failed to create tax year end: {}", e)))?;

    Ok(format_datetime_internal(&tax_year_end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_datetime() {
        let result = parse_datetime("2024-01-15T10:30:00Z".to_string()).unwrap();
        assert!(result.contains("2024-01-15"));
    }

    #[test]
    fn test_days_between() {
        let days = days_between(
            "2024-01-01T00:00:00Z".to_string(),
            "2024-01-31T00:00:00Z".to_string()
        ).unwrap();
        assert_eq!(days, 30);
    }

    #[test]
    fn test_wash_sale_within_period() {
        // Sale on Jan 15, purchase on Jan 20 (5 days after) = wash sale
        let is_wash = is_within_wash_sale_period(
            "2024-01-15T00:00:00Z".to_string(),
            "2024-01-20T00:00:00Z".to_string()
        ).unwrap();
        assert!(is_wash);
    }

    #[test]
    fn test_wash_sale_outside_period() {
        // Sale on Jan 15, purchase on Mar 1 (45+ days after) = not a wash sale
        let is_wash = is_within_wash_sale_period(
            "2024-01-15T00:00:00Z".to_string(),
            "2024-03-01T00:00:00Z".to_string()
        ).unwrap();
        assert!(!is_wash);
    }

    #[test]
    fn test_wash_sale_before_sale() {
        // Purchase on Jan 1, sale on Jan 15 (14 days after purchase) = wash sale
        let is_wash = is_within_wash_sale_period(
            "2024-01-15T00:00:00Z".to_string(),
            "2024-01-01T00:00:00Z".to_string()
        ).unwrap();
        assert!(is_wash);
    }

    #[test]
    fn test_add_days() {
        let result = add_days("2024-01-01T00:00:00Z".to_string(), 10).unwrap();
        assert!(result.contains("2024-01-11"));
    }

    #[test]
    fn test_tax_year_bounds() {
        let start = get_tax_year_start("2024-06-15T00:00:00Z".to_string()).unwrap();
        let end = get_tax_year_end("2024-06-15T00:00:00Z".to_string()).unwrap();

        assert!(start.contains("2024-01-01"));
        assert!(end.contains("2024-12-31"));
    }
}
