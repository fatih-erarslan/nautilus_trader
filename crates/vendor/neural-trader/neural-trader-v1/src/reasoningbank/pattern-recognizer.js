/**
 * Pattern Recognizer - Vector-Based Pattern Similarity Search
 *
 * Uses AgentDB's 150x faster vector search for pattern recognition:
 * - Stores learned patterns as embeddings
 * - Performs semantic similarity search
 * - Retrieves relevant past decisions
 * - Supports HNSW indexing for ultra-fast retrieval
 * - Quantization for 4-32x memory reduction
 *
 * @module reasoningbank/pattern-recognizer
 */

const EventEmitter = require('events');

/**
 * Embedding dimensions for different pattern types
 */
const EmbeddingConfig = {
  MARKET_CONDITION: 32,    // Market state features
  DECISION: 64,            // Trading decision features
  OUTCOME: 16,             // Outcome features
  COMPOSITE: 128           // Combined features
};

/**
 * Pattern Recognizer Class
 */
class PatternRecognizer extends EventEmitter {
  constructor(options = {}) {
    super();

    this.options = {
      topK: options.topK || 10,
      minSimilarity: options.minSimilarity || 0.7,
      enableHNSW: options.enableHNSW !== false,
      enableQuantization: options.enableQuantization !== false,
      ...options
    };

    this.agentDB = null;
    this.patterns = new Map();

    this.stats = {
      totalPatterns: 0,
      totalQueries: 0,
      avgQueryTime: 0,
      cacheHits: 0,
      cacheMisses: 0
    };

    // Simple cache for recent queries
    this.queryCache = new Map();
    this.maxCacheSize = 100;
  }

  /**
   * Initialize pattern recognizer
   *
   * @param {AgentDBClient} agentDB - AgentDB client instance
   * @returns {Promise<void>}
   */
  async initialize(agentDB) {
    this.agentDB = agentDB;

    if (this.agentDB) {
      // Initialize vector collection in AgentDB
      await this.agentDB.updateState({
        collection: 'patterns',
        enableHNSW: this.options.enableHNSW,
        enableQuantization: this.options.enableQuantization,
        initialized: true
      });
    }

    console.log('✅ Pattern Recognizer initialized with AgentDB vector search');
  }

  /**
   * Store a pattern in AgentDB
   *
   * @param {Object} pattern - Pattern to store
   * @returns {Promise<Object>} Stored pattern
   */
  async storePattern(pattern) {
    // Generate embedding for pattern
    const embedding = this.createEmbedding(pattern);

    // Add embedding to pattern
    const patternWithEmbedding = {
      ...pattern,
      embedding,
      storedAt: Date.now()
    };

    // Store locally
    this.patterns.set(pattern.id, patternWithEmbedding);

    // Store in AgentDB for distributed access
    if (this.agentDB) {
      await this.agentDB.updateState({
        collection: 'patterns',
        patternId: pattern.id,
        data: patternWithEmbedding
      });
    }

    this.stats.totalPatterns++;

    this.emit('pattern:stored', { patternId: pattern.id });

    return patternWithEmbedding;
  }

  /**
   * Find similar patterns using vector similarity
   *
   * @param {Object} query - Query state/pattern
   * @param {Object} options - Search options
   * @returns {Promise<Array>} Similar patterns
   */
  async findSimilar(query, options = {}) {
    const startTime = Date.now();

    const topK = options.topK || this.options.topK;
    const minSimilarity = options.minSimilarity || this.options.minSimilarity;
    const includeContext = options.includeContext !== false;

    // Generate query embedding
    const queryEmbedding = this.createEmbedding(query);

    // Check cache first
    const cacheKey = this.getCacheKey(queryEmbedding, topK, minSimilarity);
    if (this.queryCache.has(cacheKey)) {
      this.stats.cacheHits++;
      return this.queryCache.get(cacheKey);
    }

    this.stats.cacheMisses++;

    // Search in AgentDB (150x faster with HNSW)
    let results = [];

    if (this.agentDB) {
      try {
        // Use AgentDB's vector similarity search
        const agentDBResults = await this.agentDB.vectorSearch({
          collection: 'patterns',
          queryEmbedding,
          topK,
          minSimilarity
        });

        results = agentDBResults.map(r => ({
          ...r.pattern,
          similarity: r.similarity,
          distance: r.distance
        }));

      } catch (error) {
        console.warn('  ⚠️  AgentDB vector search failed, using local fallback');
        results = this.localVectorSearch(queryEmbedding, topK, minSimilarity);
      }
    } else {
      // Fallback to local search
      results = this.localVectorSearch(queryEmbedding, topK, minSimilarity);
    }

    // Add context if requested
    if (includeContext) {
      for (const result of results) {
        result.context = this.getPatternContext(result);
      }
    }

    const queryTime = Date.now() - startTime;
    this.updateQueryStats(queryTime);

    // Cache results
    this.cacheResults(cacheKey, results);

    this.emit('pattern:searched', {
      resultsCount: results.length,
      queryTime,
      cacheHit: false
    });

    return results;
  }

  /**
   * Create embedding vector from pattern/state
   * @private
   */
  createEmbedding(data) {
    // Composite embedding combining different features
    const embedding = [];

    // Market condition features (32 dims)
    const marketFeatures = this.extractMarketFeatures(data);
    embedding.push(...marketFeatures);

    // Decision features (64 dims)
    const decisionFeatures = this.extractDecisionFeatures(data);
    embedding.push(...decisionFeatures);

    // Outcome features (16 dims)
    const outcomeFeatures = this.extractOutcomeFeatures(data);
    embedding.push(...outcomeFeatures);

    // Pad to 128 dimensions
    while (embedding.length < EmbeddingConfig.COMPOSITE) {
      embedding.push(0);
    }

    // Normalize
    return this.normalizeEmbedding(embedding);
  }

  /**
   * Extract market condition features
   * @private
   */
  extractMarketFeatures(data) {
    const marketState = data.marketState || data.conditions || data.decision?.marketState || {};
    const features = new Array(EmbeddingConfig.MARKET_CONDITION).fill(0);

    // Volatility (0-1)
    features[0] = this.normalize(marketState.volatility, 0, 100);

    // Volume (log scale)
    features[1] = marketState.volume ? Math.log10(marketState.volume + 1) / 10 : 0;

    // Trend encoding (one-hot)
    if (marketState.trend === 'up') features[2] = 1;
    else if (marketState.trend === 'down') features[3] = 1;
    else if (marketState.trend === 'sideways') features[4] = 1;

    // Expected direction
    if (marketState.expectedDirection === 'up') features[5] = 1;
    else if (marketState.expectedDirection === 'down') features[6] = 1;

    // Expected volatility
    features[7] = this.normalize(marketState.expectedVolatility, 0, 100);

    // Actual volatility
    features[8] = this.normalize(marketState.actualVolatility, 0, 100);

    // Price features (if available)
    if (marketState.price) {
      features[9] = this.normalize(marketState.price, 0, 1000);
    }

    // Moving averages (if available)
    if (marketState.sma20) features[10] = this.normalize(marketState.sma20, 0, 1000);
    if (marketState.sma50) features[11] = this.normalize(marketState.sma50, 0, 1000);

    return features;
  }

  /**
   * Extract decision features
   * @private
   */
  extractDecisionFeatures(data) {
    const decision = data.decision || data;
    const features = new Array(EmbeddingConfig.DECISION).fill(0);

    // Action encoding (one-hot)
    if (decision.action === 'buy') features[0] = 1;
    else if (decision.action === 'sell') features[1] = 1;
    else if (decision.action === 'hold') features[2] = 1;

    // Type encoding
    if (decision.type) {
      const typeHash = this.hashString(decision.type, 10);
      features[3 + typeHash] = 1;
    }

    // Quantity (normalized)
    features[13] = this.normalize(decision.quantity, 0, 1000);

    // Price
    features[14] = this.normalize(decision.price, 0, 1000);

    // Timing (hour of day)
    if (decision.timestamp) {
      const hour = new Date(decision.timestamp).getHours();
      features[15] = hour / 24;
    }

    // Reasoning confidence
    if (decision.reasoning?.confidence) {
      features[16] = decision.reasoning.confidence;
    }

    // Risk level encoding
    if (decision.reasoning?.riskLevel) {
      const riskMap = { low: 0.25, medium: 0.5, high: 0.75, extreme: 1.0 };
      features[17] = riskMap[decision.reasoning.riskLevel] || 0.5;
    }

    // Number of reasoning factors
    if (decision.reasoning?.factors) {
      features[18] = Math.min(1.0, decision.reasoning.factors.length / 10);
    }

    return features;
  }

  /**
   * Extract outcome features
   * @private
   */
  extractOutcomeFeatures(data) {
    const outcome = data.outcome || {};
    const verdict = data.verdict || {};
    const features = new Array(EmbeddingConfig.OUTCOME).fill(0);

    // Execution status
    features[0] = outcome.executed ? 1 : 0;

    // P&L (normalized, clamped to ±50%)
    if (outcome.pnlPercent !== undefined) {
      features[1] = this.normalize(outcome.pnlPercent, -50, 50);
    }

    // Risk-adjusted return
    if (outcome.riskAdjustedReturn !== undefined) {
      features[2] = this.normalize(outcome.riskAdjustedReturn, -50, 50);
    }

    // Slippage
    if (outcome.slippage !== undefined) {
      features[3] = this.normalize(Math.abs(outcome.slippage), 0, 5);
    }

    // Execution time (normalized to 0-10 seconds)
    if (outcome.executionTime !== undefined) {
      features[4] = this.normalize(outcome.executionTime, 0, 10000);
    }

    // Verdict score
    features[5] = verdict.score || 0.5;

    // Quality encoding
    if (verdict.quality) {
      const qualityMap = {
        excellent: 1.0,
        good: 0.75,
        neutral: 0.5,
        poor: 0.25,
        terrible: 0.0
      };
      features[6] = qualityMap[verdict.quality] || 0.5;
    }

    return features;
  }

  /**
   * Local vector search fallback (cosine similarity)
   * @private
   */
  localVectorSearch(queryEmbedding, topK, minSimilarity) {
    const results = [];

    for (const [patternId, pattern] of this.patterns.entries()) {
      if (!pattern.embedding) continue;

      const similarity = this.cosineSimilarity(queryEmbedding, pattern.embedding);

      if (similarity >= minSimilarity) {
        results.push({
          ...pattern,
          similarity,
          distance: 1 - similarity
        });
      }
    }

    // Sort by similarity (descending)
    results.sort((a, b) => b.similarity - a.similarity);

    return results.slice(0, topK);
  }

  /**
   * Calculate cosine similarity
   * @private
   */
  cosineSimilarity(vec1, vec2) {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    const magnitude = Math.sqrt(norm1) * Math.sqrt(norm2);

    return magnitude > 0 ? dotProduct / magnitude : 0;
  }

  /**
   * Normalize embedding to unit vector
   * @private
   */
  normalizeEmbedding(embedding) {
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));

    if (norm === 0) return embedding;

    return embedding.map(val => val / norm);
  }

  /**
   * Normalize value to [0, 1]
   * @private
   */
  normalize(value, min, max) {
    if (value === undefined || value === null) return 0;
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
  }

  /**
   * Hash string to integer
   * @private
   */
  hashString(str, range) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash) % range;
  }

  /**
   * Get pattern context
   * @private
   */
  getPatternContext(pattern) {
    return {
      storedAt: pattern.storedAt,
      type: pattern.type,
      occurrences: pattern.occurrences,
      avgScore: pattern.avgScore,
      confidence: pattern.confidence
    };
  }

  /**
   * Get cache key
   * @private
   */
  getCacheKey(embedding, topK, minSimilarity) {
    const embeddingHash = embedding.slice(0, 10).join(',');
    return `${embeddingHash}-${topK}-${minSimilarity}`;
  }

  /**
   * Cache results
   * @private
   */
  cacheResults(key, results) {
    this.queryCache.set(key, results);

    // Limit cache size (LRU-like)
    if (this.queryCache.size > this.maxCacheSize) {
      const firstKey = this.queryCache.keys().next().value;
      this.queryCache.delete(firstKey);
    }
  }

  /**
   * Update query statistics
   * @private
   */
  updateQueryStats(queryTime) {
    const currentAvg = this.stats.avgQueryTime;
    const currentTotal = this.stats.totalQueries;

    this.stats.totalQueries++;
    this.stats.avgQueryTime = (currentAvg * currentTotal + queryTime) / this.stats.totalQueries;
  }

  /**
   * Get statistics
   *
   * @returns {Object} Recognizer statistics
   */
  getStats() {
    return {
      ...this.stats,
      cacheSize: this.queryCache.size,
      cacheHitRate: this.stats.totalQueries > 0
        ? this.stats.cacheHits / this.stats.totalQueries
        : 0
    };
  }
}

module.exports = PatternRecognizer;
module.exports.EmbeddingConfig = EmbeddingConfig;
