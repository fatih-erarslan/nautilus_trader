use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::AppState;

// ============ Request/Response Types ============

#[derive(Debug, Deserialize)]
pub struct DashboardQuery {
    #[serde(default = "default_timeframe")]
    pub timeframe: String, // "24h", "7d", "30d"
}

#[derive(Debug, Deserialize)]
pub struct UsageQuery {
    pub user_id: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ActivityQuery {
    pub user_id: Option<String>,
    #[serde(default = "default_page")]
    pub page: i64,
    #[serde(default = "default_activity_limit")]
    pub limit: i64,
    pub action: Option<String>,
    pub entity_type: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ActivityLogEntry {
    pub user_id: String,
    pub action: String,
    pub entity_type: String,
    pub entity_id: Option<String>,
    pub entity_name: Option<String>,
    pub description: String,
    pub severity: Option<String>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct DashboardStats {
    pub total_api_calls: i64,
    pub total_workflows: i64,
    pub total_scans: i64,
    pub active_users: i64,
    pub avg_response_time_ms: f64,
    pub error_rate: f64,
    pub recent_activity: Vec<ActivitySummary>,
    pub performance_metrics: PerformanceSnapshot,
    pub timeframe: String,
}

#[derive(Debug, Serialize)]
pub struct ActivitySummary {
    pub id: String,
    pub user_id: String,
    pub action: String,
    pub entity_type: String,
    pub entity_name: Option<String>,
    pub description: String,
    pub severity: String,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct PerformanceSnapshot {
    pub api_latency_p50: f64,
    pub api_latency_p95: f64,
    pub api_latency_p99: f64,
    pub total_events: i64,
    pub success_rate: f64,
}

#[derive(Debug, Serialize)]
pub struct UsageReport {
    pub user_id: String,
    pub period: String,
    pub total_api_calls: i64,
    pub total_workflows: i64,
    pub total_scans: i64,
    pub total_execution_time_ms: i64,
    pub success_rate: f64,
    pub daily_breakdown: Vec<DailyUsage>,
}

#[derive(Debug, Serialize)]
pub struct DailyUsage {
    pub date: String,
    pub api_calls: i64,
    pub workflows: i64,
    pub scans: i64,
    pub success_count: i64,
    pub error_count: i64,
}

fn default_timeframe() -> String {
    "24h".to_string()
}

fn default_page() -> i64 {
    1
}

fn default_activity_limit() -> i64 {
    50
}

// ============ API Handlers ============

/// GET /api/analytics/dashboard
/// Returns comprehensive dashboard statistics
pub async fn get_dashboard(
    State(state): State<AppState>,
    Query(query): Query<DashboardQuery>,
) -> impl IntoResponse {
    tracing::info!("Fetching dashboard analytics for timeframe: {}", query.timeframe);

    match state.db.get_dashboard_stats(&query.timeframe).await {
        Ok(stats) => Json(json!(stats)),
        Err(e) => {
            tracing::error!("Failed to get dashboard stats: {}", e);
            Json(json!({
                "error": "Failed to retrieve dashboard statistics",
                "message": e.to_string()
            }))
        }
    }
}

/// GET /api/analytics/usage
/// Returns detailed usage analytics
pub async fn get_usage_analytics(
    State(state): State<AppState>,
    Query(query): Query<UsageQuery>,
) -> impl IntoResponse {
    let user_id = query.user_id.as_deref().unwrap_or("user-1");
    tracing::info!("Fetching usage analytics for user: {}", user_id);

    match state.db.get_usage_analytics(user_id, query.start_date.as_deref(), query.end_date.as_deref()).await {
        Ok(report) => Json(json!(report)),
        Err(e) => {
            tracing::error!("Failed to get usage analytics: {}", e);
            Json(json!({
                "error": "Failed to retrieve usage analytics",
                "message": e.to_string()
            }))
        }
    }
}

/// GET /api/activity/feed
/// Returns paginated activity feed
pub async fn get_activity_feed(
    State(state): State<AppState>,
    Query(query): Query<ActivityQuery>,
) -> impl IntoResponse {
    tracing::info!("Fetching activity feed: page={}, limit={}", query.page, query.limit);

    match state.db.get_activity_feed(
        query.user_id.as_deref(),
        query.page,
        query.limit,
        query.action.as_deref(),
        query.entity_type.as_deref(),
    ).await {
        Ok(activities) => Json(json!({
            "activities": activities,
            "page": query.page,
            "limit": query.limit,
            "total": activities.len()
        })),
        Err(e) => {
            tracing::error!("Failed to get activity feed: {}", e);
            Json(json!({
                "error": "Failed to retrieve activity feed",
                "message": e.to_string(),
                "activities": []
            }))
        }
    }
}

/// POST /api/activity/log
/// Log a new activity entry
pub async fn log_activity(
    State(state): State<AppState>,
    Json(payload): Json<ActivityLogEntry>,
) -> impl IntoResponse {
    tracing::info!("Logging activity: {} {} {}", payload.action, payload.entity_type,
                   payload.entity_name.as_deref().unwrap_or("unknown"));

    match state.db.log_activity(
        &payload.user_id,
        &payload.action,
        &payload.entity_type,
        payload.entity_id.as_deref(),
        payload.entity_name.as_deref(),
        &payload.description,
        payload.severity.as_deref(),
        payload.metadata,
    ).await {
        Ok(activity_id) => Json(json!({
            "status": "success",
            "activity_id": activity_id,
            "message": "Activity logged successfully"
        })),
        Err(e) => {
            tracing::error!("Failed to log activity: {}", e);
            Json(json!({
                "error": "Failed to log activity",
                "message": e.to_string()
            }))
        }
    }
}

/// GET /api/analytics/performance
/// Returns performance metrics and monitoring data
pub async fn get_performance_metrics(
    State(state): State<AppState>,
    Query(query): Query<DashboardQuery>,
) -> impl IntoResponse {
    tracing::info!("Fetching performance metrics for timeframe: {}", query.timeframe);

    match state.db.get_performance_metrics(&query.timeframe).await {
        Ok(metrics) => Json(json!(metrics)),
        Err(e) => {
            tracing::error!("Failed to get performance metrics: {}", e);
            Json(json!({
                "error": "Failed to retrieve performance metrics",
                "message": e.to_string()
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        assert_eq!(default_timeframe(), "24h");
        assert_eq!(default_page(), 1);
        assert_eq!(default_activity_limit(), 50);
    }

    #[test]
    fn test_activity_log_serialization() {
        let entry = ActivityLogEntry {
            user_id: "user-1".to_string(),
            action: "create".to_string(),
            entity_type: "workflow".to_string(),
            entity_id: Some("wf-1".to_string()),
            entity_name: Some("Test Workflow".to_string()),
            description: "Created test workflow".to_string(),
            severity: Some("info".to_string()),
            metadata: Some(json!({"key": "value"})),
        };

        let json_str = serde_json::to_string(&entry).unwrap();
        assert!(json_str.contains("user-1"));
        assert!(json_str.contains("create"));
    }
}
