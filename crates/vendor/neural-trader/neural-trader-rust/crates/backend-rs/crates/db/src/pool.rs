use beclever_common::{Error, Result};
use diesel::sqlite::SqliteConnection;
use diesel::r2d2::{ConnectionManager, Pool, PooledConnection};
use std::time::Duration;

pub type DbPool = Pool<ConnectionManager<SqliteConnection>>;
pub type DbConnection = PooledConnection<ConnectionManager<SqliteConnection>>;

/// Create a new database connection pool
pub fn create_pool(database_url: &str, max_size: u32) -> Result<DbPool> {
    let manager = ConnectionManager::<SqliteConnection>::new(database_url);

    Pool::builder()
        .max_size(max_size)
        .connection_timeout(Duration::from_secs(30))
        .build(manager)
        .map_err(|e| Error::Database(format!("Failed to create connection pool: {}", e)))
}

/// Trait for database operations that can be mocked
#[cfg_attr(test, mockall::automock)]
pub trait DatabaseOperations: Send + Sync {
    fn get_connection(&self) -> Result<DbConnection>;
    fn health_check(&self) -> Result<()>;
}

/// Production implementation
pub struct DatabasePool {
    pool: DbPool,
}

impl DatabasePool {
    pub fn new(pool: DbPool) -> Self {
        Self { pool }
    }
}

impl DatabaseOperations for DatabasePool {
    fn get_connection(&self) -> Result<DbConnection> {
        self.pool
            .get()
            .map_err(|e| Error::Database(format!("Failed to get connection: {}", e)))
    }

    fn health_check(&self) -> Result<()> {
        let _conn = self.get_connection()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_database_operations() {
        let mut mock_db = MockDatabaseOperations::new();

        mock_db
            .expect_health_check()
            .times(1)
            .returning(|| Ok(()));

        assert!(mock_db.health_check().is_ok());
    }

    #[test]
    fn test_create_pool_invalid_url() {
        let result = create_pool("invalid://url", 5);
        assert!(result.is_err());
    }
}
