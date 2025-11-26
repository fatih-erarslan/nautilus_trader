use nt_news_trading::{Sentiment, SentimentAnalyzer, SentimentLabel};

#[test]
fn test_sentiment_analyzer_creation() {
    let analyzer = SentimentAnalyzer::default();
    let sentiment = analyzer.analyze("test");
    assert!(sentiment.score >= -1.0 && sentiment.score <= 1.0);
}

#[test]
fn test_positive_sentiment() {
    let analyzer = SentimentAnalyzer::default();
    let sentiment = analyzer.analyze("The stock is showing strong growth and bullish momentum with record profits");

    assert!(sentiment.is_positive(), "Expected positive sentiment");
    assert!(sentiment.score > 0.3, "Expected score > 0.3, got {}", sentiment.score);
}

#[test]
fn test_negative_sentiment() {
    let analyzer = SentimentAnalyzer::default();
    let sentiment = analyzer.analyze("The company faces a major crisis with declining profits and market crash concerns");

    assert!(sentiment.is_negative(), "Expected negative sentiment");
    assert!(sentiment.score < -0.3, "Expected score < -0.3, got {}", sentiment.score);
}

#[test]
fn test_neutral_sentiment() {
    let analyzer = SentimentAnalyzer::default();
    let sentiment = analyzer.analyze("The company held a meeting today to discuss general matters");

    assert!(sentiment.is_neutral(), "Expected neutral sentiment");
    assert!(sentiment.score.abs() < 0.2, "Expected abs(score) < 0.2, got {}", sentiment.score);
}

#[test]
fn test_batch_analysis() {
    let analyzer = SentimentAnalyzer::default();
    let texts = vec![
        "Bullish rally continues with strong gains",
        "Market crash expected with severe losses",
        "Normal trading day without major events",
    ];

    let results = analyzer.analyze_batch(&texts);
    assert_eq!(results.len(), 3);

    assert!(results[0].is_positive(), "First should be positive");
    assert!(results[1].is_negative(), "Second should be negative");
    assert!(results[2].is_neutral(), "Third should be neutral");
}

#[test]
fn test_sentiment_labels() {
    let very_positive = Sentiment::new(0.8, 0.9);
    assert_eq!(very_positive.label, SentimentLabel::VeryPositive);

    let positive = Sentiment::new(0.4, 0.7);
    assert_eq!(positive.label, SentimentLabel::Positive);

    let neutral = Sentiment::new(0.0, 0.5);
    assert_eq!(neutral.label, SentimentLabel::Neutral);

    let negative = Sentiment::new(-0.4, 0.7);
    assert_eq!(negative.label, SentimentLabel::Negative);

    let very_negative = Sentiment::new(-0.8, 0.9);
    assert_eq!(very_negative.label, SentimentLabel::VeryNegative);
}

#[test]
fn test_sentiment_magnitude() {
    let analyzer = SentimentAnalyzer::default();

    let strong_sentiment = analyzer.analyze("Massive surge in profits with record breaking growth and exceptional performance");
    assert!(strong_sentiment.magnitude > 0.3, "Expected magnitude > 0.3, got {}", strong_sentiment.magnitude);

    let weak_sentiment = analyzer.analyze("The company exists and operates");
    assert!(weak_sentiment.magnitude < 0.4, "Expected magnitude < 0.4, got {}", weak_sentiment.magnitude);
}

#[test]
fn test_detailed_analysis() {
    let analyzer = SentimentAnalyzer::default();
    let detailed = analyzer.analyze_detailed("The stock shows strong growth but faces some risks");

    assert!(detailed.positive_words.contains(&"strong".to_string()));
    assert!(detailed.positive_words.contains(&"growth".to_string()));
    assert!(detailed.word_count > 0);
}

#[test]
fn test_financial_terms() {
    let analyzer = SentimentAnalyzer::default();

    // Test earnings-related terms
    let earnings = analyzer.analyze("Company beats earnings expectations with strong profits");
    assert!(earnings.score > 0.0, "Expected positive for earnings beat, got {}", earnings.score);

    // Test loss-related terms
    let loss = analyzer.analyze("Company reports significant losses and declining profits");
    assert!(loss.score < 0.0, "Expected negative for losses, got {}", loss.score);

    // Test merger-related terms - neutral without additional context
    let merger = analyzer.analyze("Company announces major acquisition with strong growth prospects");
    assert!(merger.score >= -0.2, "Expected neutral or positive for acquisition, got {}", merger.score);
}

#[test]
fn test_sentiment_score_bounds() {
    let analyzer = SentimentAnalyzer::default();

    let long_text = "test ".repeat(1000);
    let sentiment = analyzer.analyze(&long_text);
    assert!(sentiment.score >= -1.0 && sentiment.score <= 1.0);
    assert!(sentiment.magnitude >= 0.0 && sentiment.magnitude <= 1.0);
}
