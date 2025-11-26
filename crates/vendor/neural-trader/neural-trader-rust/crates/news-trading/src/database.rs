use crate::error::{NewsError, Result};
use crate::models::{NewsArticle, NewsQuery};
use serde_json;
use std::path::Path;

pub struct NewsDB {
    db: sled::Db,
}

impl NewsDB {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = sled::open(path).map_err(|e| NewsError::Database(e.to_string()))?;
        Ok(Self { db })
    }

    pub fn in_memory() -> Result<Self> {
        let db = sled::Config::new()
            .temporary(true)
            .open()
            .map_err(|e| NewsError::Database(e.to_string()))?;
        Ok(Self { db })
    }

    /// Store a news article
    pub fn store(&self, article: &NewsArticle) -> Result<()> {
        let key = article.id.as_bytes();
        let value = serde_json::to_vec(article).map_err(|e| NewsError::Database(e.to_string()))?;

        self.db
            .insert(key, value)
            .map_err(|e| NewsError::Database(e.to_string()))?;

        // Index by symbol
        for symbol in &article.symbols {
            self.index_by_symbol(symbol, &article.id)?;
        }

        // Index by date
        self.index_by_date(&article.published_at.to_rfc3339(), &article.id)?;

        Ok(())
    }

    /// Store multiple articles in a batch
    pub fn store_batch(&self, articles: &[NewsArticle]) -> Result<usize> {
        let mut count = 0;
        for article in articles {
            if self.store(article).is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Get an article by ID
    pub fn get(&self, id: &str) -> Result<Option<NewsArticle>> {
        match self.db.get(id.as_bytes()) {
            Ok(Some(bytes)) => {
                let article: NewsArticle = serde_json::from_slice(&bytes)
                    .map_err(|e| NewsError::Database(e.to_string()))?;
                Ok(Some(article))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NewsError::Database(e.to_string())),
        }
    }

    /// Query articles using filters
    pub fn query(&self, query: &NewsQuery) -> Result<Vec<NewsArticle>> {
        let mut results = Vec::new();

        // If symbols are specified, use the symbol index
        if let Some(ref symbols) = query.symbols {
            for symbol in symbols {
                if let Some(ids) = self.get_symbol_index(symbol)? {
                    for id in ids {
                        if let Some(article) = self.get(&id)? {
                            if query.matches(&article) {
                                results.push(article);
                            }
                        }
                    }
                }
            }
        } else {
            // Full scan
            for item in self.db.iter() {
                if let Ok((_, value)) = item {
                    if let Ok(article) = serde_json::from_slice::<NewsArticle>(&value) {
                        if query.matches(&article) {
                            results.push(article);
                        }
                    }
                }
            }
        }

        // Sort by date
        results.sort_by(|a, b| b.published_at.cmp(&a.published_at));

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Get article history for a symbol
    pub fn get_history(&self, symbol: &str, days: u32) -> Result<Vec<NewsArticle>> {
        let start_date = chrono::Utc::now() - chrono::Duration::days(days as i64);

        let query = NewsQuery::new()
            .with_symbols(vec![symbol.to_string()])
            .with_date_range(start_date, chrono::Utc::now());

        self.query(&query)
    }

    /// Delete an article
    pub fn delete(&self, id: &str) -> Result<()> {
        self.db
            .remove(id.as_bytes())
            .map_err(|e| NewsError::Database(e.to_string()))?;
        Ok(())
    }

    /// Get total article count
    pub fn count(&self) -> usize {
        self.db.len()
    }

    /// Clear all articles
    pub fn clear(&self) -> Result<()> {
        self.db
            .clear()
            .map_err(|e| NewsError::Database(e.to_string()))?;
        Ok(())
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| NewsError::Database(e.to_string()))?;
        Ok(())
    }

    fn index_by_symbol(&self, symbol: &str, article_id: &str) -> Result<()> {
        let index_key = format!("idx:symbol:{}", symbol);
        let mut ids = self.get_symbol_index(symbol)?.unwrap_or_default();

        if !ids.contains(&article_id.to_string()) {
            ids.push(article_id.to_string());
            let value = serde_json::to_vec(&ids).map_err(|e| NewsError::Database(e.to_string()))?;
            self.db
                .insert(index_key.as_bytes(), value)
                .map_err(|e| NewsError::Database(e.to_string()))?;
        }

        Ok(())
    }

    fn get_symbol_index(&self, symbol: &str) -> Result<Option<Vec<String>>> {
        let index_key = format!("idx:symbol:{}", symbol);
        match self.db.get(index_key.as_bytes()) {
            Ok(Some(bytes)) => {
                let ids: Vec<String> = serde_json::from_slice(&bytes)
                    .map_err(|e| NewsError::Database(e.to_string()))?;
                Ok(Some(ids))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NewsError::Database(e.to_string())),
        }
    }

    fn index_by_date(&self, date: &str, article_id: &str) -> Result<()> {
        let index_key = format!("idx:date:{}", date);
        self.db
            .insert(index_key.as_bytes(), article_id.as_bytes())
            .map_err(|e| NewsError::Database(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_creation() {
        let db = NewsDB::in_memory().unwrap();
        assert_eq!(db.count(), 0);
    }

    #[test]
    fn test_store_and_retrieve() {
        let db = NewsDB::in_memory().unwrap();

        let article = NewsArticle::new(
            "test1".to_string(),
            "Test Title".to_string(),
            "Test content".to_string(),
            "test".to_string(),
        );

        db.store(&article).unwrap();
        // Note: count includes indices, so it may be > 1
        assert!(db.count() >= 1, "Expected at least 1 item in DB");

        let retrieved = db.get("test1").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().title, "Test Title");
    }

    #[test]
    fn test_query_by_symbol() {
        let db = NewsDB::in_memory().unwrap();

        let article = NewsArticle::new(
            "test1".to_string(),
            "AAPL news".to_string(),
            "Apple earnings".to_string(),
            "test".to_string(),
        )
        .with_symbols(vec!["AAPL".to_string()]);

        db.store(&article).unwrap();

        let query = NewsQuery::new().with_symbols(vec!["AAPL".to_string()]);
        let results = db.query(&query).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "test1");
    }

    #[test]
    fn test_batch_store() {
        let db = NewsDB::in_memory().unwrap();

        let articles = vec![
            NewsArticle::new(
                "1".to_string(),
                "News 1".to_string(),
                "Content 1".to_string(),
                "test".to_string(),
            ),
            NewsArticle::new(
                "2".to_string(),
                "News 2".to_string(),
                "Content 2".to_string(),
                "test".to_string(),
            ),
        ];

        let count = db.store_batch(&articles).unwrap();
        assert_eq!(count, 2, "Should store 2 articles");

        // Verify articles were stored
        assert!(db.get("1").unwrap().is_some());
        assert!(db.get("2").unwrap().is_some());
    }
}
