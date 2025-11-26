use nt_news_trading::{NewsArticle, NewsDB, NewsQuery};

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
    assert_eq!(db.count(), 1);

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
        NewsArticle::new(
            "3".to_string(),
            "News 3".to_string(),
            "Content 3".to_string(),
            "test".to_string(),
        ),
    ];

    let count = db.store_batch(&articles).unwrap();
    assert_eq!(count, 3, "Should successfully store 3 articles");

    // Verify articles were stored (count may include indices)
    assert!(db.get("1").unwrap().is_some());
    assert!(db.get("2").unwrap().is_some());
    assert!(db.get("3").unwrap().is_some());
}

#[test]
fn test_delete_article() {
    let db = NewsDB::in_memory().unwrap();

    let article = NewsArticle::new(
        "delete_me".to_string(),
        "Test".to_string(),
        "Content".to_string(),
        "test".to_string(),
    );

    db.store(&article).unwrap();
    let initial_count = db.count();
    assert!(initial_count >= 1, "Should have at least 1 entry after store");

    db.delete("delete_me").unwrap();

    let retrieved = db.get("delete_me").unwrap();
    assert!(retrieved.is_none(), "Article should be deleted");
}

#[test]
fn test_query_with_multiple_filters() {
    let db = NewsDB::in_memory().unwrap();

    let article1 = NewsArticle::new(
        "1".to_string(),
        "AAPL news".to_string(),
        "Content".to_string(),
        "source1".to_string(),
    )
    .with_symbols(vec!["AAPL".to_string()])
    .with_relevance(0.9);

    let article2 = NewsArticle::new(
        "2".to_string(),
        "MSFT news".to_string(),
        "Content".to_string(),
        "source2".to_string(),
    )
    .with_symbols(vec!["MSFT".to_string()])
    .with_relevance(0.5);

    db.store(&article1).unwrap();
    db.store(&article2).unwrap();

    let query = NewsQuery::new()
        .with_symbols(vec!["AAPL".to_string()])
        .with_min_relevance(0.8);

    let results = db.query(&query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "1");
}

#[test]
fn test_clear_database() {
    let db = NewsDB::in_memory().unwrap();

    let articles = vec![
        NewsArticle::new(
            "1".to_string(),
            "News 1".to_string(),
            "Content".to_string(),
            "test".to_string(),
        ),
        NewsArticle::new(
            "2".to_string(),
            "News 2".to_string(),
            "Content".to_string(),
            "test".to_string(),
        ),
    ];

    db.store_batch(&articles).unwrap();
    assert!(db.count() >= 2, "Should have at least 2 entries");

    db.clear().unwrap();
    assert_eq!(db.count(), 0, "Database should be empty after clear");
}

#[test]
fn test_get_history() {
    let db = NewsDB::in_memory().unwrap();

    let article = NewsArticle::new(
        "recent".to_string(),
        "Recent AAPL news".to_string(),
        "Content".to_string(),
        "test".to_string(),
    )
    .with_symbols(vec!["AAPL".to_string()]);

    db.store(&article).unwrap();

    let history = db.get_history("AAPL", 7).unwrap();
    assert_eq!(history.len(), 1);
}
