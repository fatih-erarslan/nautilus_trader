use beclever_common::Config;
use beclever_db::DbPool;

pub struct AppState {
    pub config: Config,
    pub db_pool: DbPool,
}

impl AppState {
    pub fn new(config: Config, db_pool: DbPool) -> Self {
        Self { config, db_pool }
    }
}
