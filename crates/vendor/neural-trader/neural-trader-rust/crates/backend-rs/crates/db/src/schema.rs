// @generated automatically by Diesel CLI.
// Updated for SQLite compatibility

diesel::table! {
    profiles (id) {
        id -> Text,  // UUID as TEXT in SQLite
        username -> Nullable<Text>,
        full_name -> Nullable<Text>,
        avatar_url -> Nullable<Text>,
        metadata -> Text,  // JSON as TEXT in SQLite
        created_at -> Timestamp,
        updated_at -> Timestamp,
    }
}

diesel::table! {
    workflows (id) {
        id -> Text,  // UUID as TEXT
        user_id -> Text,  // UUID as TEXT
        name -> Text,
        description -> Nullable<Text>,
        config -> Text,  // JSON as TEXT
        status -> Text,
        created_at -> Timestamp,
        updated_at -> Timestamp,
    }
}

diesel::table! {
    workflow_executions (id) {
        id -> Text,  // UUID as TEXT
        workflow_id -> Text,  // UUID as TEXT
        user_id -> Text,  // UUID as TEXT
        status -> Text,
        input -> Nullable<Text>,  // JSON as TEXT
        output -> Nullable<Text>,  // JSON as TEXT
        metrics -> Nullable<Text>,  // JSON as TEXT
        error -> Nullable<Text>,
        started_at -> Timestamp,
        completed_at -> Nullable<Timestamp>,
        execution_time_ms -> Nullable<Integer>,
    }
}

diesel::table! {
    vectors (id) {
        id -> Text,  // UUID as TEXT
        user_id -> Text,  // UUID as TEXT
        #[sql_name = "embedding"]
        embedding_data -> Text,  // Store as JSON string
        metadata -> Text,  // JSON as TEXT
        created_at -> Timestamp,
    }
}

diesel::table! {
    security_events (id) {
        id -> Text,  // UUID as TEXT
        user_id -> Text,  // UUID as TEXT
        event_type -> Text,
        threat_types -> Text,  // JSON array as TEXT
        confidence -> Double,
        input_hash -> Nullable<Text>,
        metadata -> Nullable<Text>,  // JSON as TEXT
        created_at -> Timestamp,
    }
}

diesel::table! {
    background_jobs (id) {
        id -> Text,  // UUID as TEXT
        user_id -> Text,  // UUID as TEXT
        job_type -> Text,
        status -> Text,
        payload -> Nullable<Text>,  // JSON as TEXT
        result -> Nullable<Text>,  // JSON as TEXT
        error -> Nullable<Text>,
        retries -> Integer,
        created_at -> Timestamp,
        started_at -> Nullable<Timestamp>,
        completed_at -> Nullable<Timestamp>,
    }
}

diesel::table! {
    foxruv_metrics (id) {
        id -> Text,  // UUID as TEXT
        user_id -> Text,  // UUID as TEXT
        package_name -> Text,
        operation -> Text,
        execution_time_ms -> Integer,
        speedup_factor -> Nullable<Double>,
        metadata -> Nullable<Text>,  // JSON as TEXT
        created_at -> Timestamp,
    }
}

diesel::joinable!(workflow_executions -> workflows (workflow_id));
diesel::joinable!(workflows -> profiles (user_id));

diesel::allow_tables_to_appear_in_same_query!(
    profiles,
    workflows,
    workflow_executions,
    vectors,
    security_events,
    background_jobs,
    foxruv_metrics,
);
