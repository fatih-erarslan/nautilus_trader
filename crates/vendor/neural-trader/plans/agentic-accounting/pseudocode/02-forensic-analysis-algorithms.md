# Forensic Analysis Algorithms - Pseudocode

## 1. Vector-Based Fraud Detection

```
FUNCTION detect_fraud_patterns(transaction, fraud_signature_library, threshold = 0.85):
  INPUT:
    transaction: Transaction to analyze
    fraud_signature_library: Collection of known fraud pattern embeddings
    threshold: Similarity score threshold (0-1)

  OUTPUT:
    is_suspicious: Boolean
    matches: Array of similar fraud patterns
    similarity_scores: Array of scores

  ALGORITHM:
    # Generate embedding for transaction
    transaction_text = serialize_transaction_for_embedding(transaction)
    transaction_embedding = generate_embedding(transaction_text)

    # Search AgentDB for similar fraud patterns
    similar_patterns = agentdb.search(
      collection: "fraud_signatures",
      query_vector: transaction_embedding,
      top_k: 10,
      distance_metric: COSINE
    )

    matches = []
    scores = []

    FOR EACH pattern IN similar_patterns:
      similarity_score = pattern.score

      IF similarity_score >= threshold:
        matches.append(pattern)
        scores.append(similarity_score)

    is_suspicious = matches.length > 0

    IF is_suspicious:
      # Log to audit trail
      log_audit_event(
        action: "FRAUD_PATTERN_DETECTED",
        transaction: transaction,
        matches: matches,
        scores: scores
      )

    RETURN is_suspicious, matches, scores

FUNCTION serialize_transaction_for_embedding(transaction):
  INPUT:
    transaction: Transaction record

  OUTPUT:
    text: String representation for embedding model

  ALGORITHM:
    # Convert transaction to natural language description
    text = CONCAT(
      "Transaction: ", transaction.type,
      " Amount: ", transaction.quantity, " ", transaction.asset,
      " Price: ", transaction.price, " ", transaction.currency,
      " Source: ", transaction.source,
      " Timestamp: ", transaction.timestamp,
      " Fees: ", transaction.fees
    )

    # Add metadata if available
    IF transaction.metadata:
      FOR EACH key, value IN transaction.metadata:
        text += " " + key + ": " + value

    RETURN text
```

---

## 2. Outlier Detection Using Clustering

```
FUNCTION detect_outliers(transactions, sensitivity = 2.0):
  INPUT:
    transactions: Array of Transaction records
    sensitivity: Standard deviations from mean (higher = fewer outliers)

  OUTPUT:
    outliers: Array of suspicious transactions
    cluster_info: Clustering analysis results

  ALGORITHM:
    # Generate embeddings for all transactions
    embeddings = []
    FOR EACH txn IN transactions:
      text = serialize_transaction_for_embedding(txn)
      embedding = generate_embedding(text)
      embeddings.append({
        transaction: txn,
        embedding: embedding
      })

    # Perform clustering (e.g., DBSCAN or K-means)
    clusters = perform_clustering(
      embeddings: MAP(embeddings, e -> e.embedding),
      algorithm: DBSCAN,
      eps: 0.3,  # Distance threshold
      min_samples: 5
    )

    # Identify noise points (outliers)
    outliers = []
    FOR EACH idx, cluster_id IN clusters:
      IF cluster_id == -1:  # DBSCAN noise label
        outliers.append(embeddings[idx].transaction)

    # Calculate distance from cluster centroids
    cluster_analysis = analyze_clusters(embeddings, clusters)

    # Apply statistical outlier detection
    statistical_outliers = detect_statistical_outliers(
      transactions,
      features: ["amount", "frequency", "timing"],
      sensitivity: sensitivity
    )

    # Combine results
    combined_outliers = UNION(outliers, statistical_outliers)

    RETURN combined_outliers, cluster_analysis

FUNCTION detect_statistical_outliers(transactions, features, sensitivity):
  INPUT:
    transactions: Array of transactions
    features: Array of feature names to analyze
    sensitivity: Z-score threshold

  OUTPUT:
    outliers: Array of transactions

  ALGORITHM:
    outliers = []

    FOR EACH feature IN features:
      # Extract feature values
      values = MAP(transactions, t -> get_feature(t, feature))

      # Calculate mean and standard deviation
      mean = MEAN(values)
      std_dev = STD_DEV(values)

      # Find outliers using Z-score
      FOR EACH idx, value IN values:
        z_score = ABS((value - mean) / std_dev)

        IF z_score > sensitivity:
          outliers.append(transactions[idx])

    # Remove duplicates
    outliers = UNIQUE(outliers)

    RETURN outliers
```

---

## 3. Transaction-Communication Pattern Linking

```
FUNCTION link_transactions_to_communications(transactions, communications, threshold = 0.75):
  INPUT:
    transactions: Array of Transaction records with embeddings
    communications: Array of Communication records (emails, messages) with embeddings
    threshold: Similarity threshold

  OUTPUT:
    links: Array of TransactionCommunicationLink records

  ALGORITHM:
    links = []

    FOR EACH transaction IN transactions:
      # Get transaction embedding
      txn_embedding = get_or_create_embedding(transaction)

      # Search for related communications within time window
      time_window_start = transaction.timestamp - 7_DAYS
      time_window_end = transaction.timestamp + 7_DAYS

      relevant_comms = FILTER(communications, WHERE
        timestamp >= time_window_start AND
        timestamp <= time_window_end
      )

      # Find similar communications via vector search
      FOR EACH comm IN relevant_comms:
        comm_embedding = get_or_create_embedding(comm)

        similarity = cosine_similarity(txn_embedding, comm_embedding)

        IF similarity >= threshold:
          link = TransactionCommunicationLink {
            transactionId: transaction.id,
            communicationId: comm.id,
            similarityScore: similarity,
            timeDelta: comm.timestamp - transaction.timestamp,
            context: extract_relevant_context(comm, transaction)
          }

          links.append(link)

    # Rank links by suspiciousness
    FOR EACH link IN links:
      link.suspicionScore = calculate_suspicion_score(link)

    SORT(links, BY suspicionScore DESC)

    RETURN links

FUNCTION calculate_suspicion_score(link):
  INPUT:
    link: TransactionCommunicationLink

  OUTPUT:
    score: Number (0-100)

  ALGORITHM:
    score = 0

    # High similarity between transaction and communication
    score += link.similarityScore * 50

    # Communication shortly before transaction (potential coordination)
    IF link.timeDelta > 0 AND link.timeDelta < 24_HOURS:
      score += 30

    # Keywords indicating potential fraud
    fraud_keywords = ["hide", "secret", "off the books", "cash", "undeclared"]
    keyword_matches = count_keywords(link.context, fraud_keywords)
    score += keyword_matches * 5

    # Large transaction amount
    IF link.transaction.quantity * link.transaction.price > 100000:
      score += 10

    RETURN CLAMP(score, 0, 100)
```

---

## 4. Merkle Proof Generation for Audit Trail

```
FUNCTION generate_merkle_proof(audit_entry_id, audit_trail):
  INPUT:
    audit_entry_id: ID of entry to prove
    audit_trail: Array of AuditEntry records

  OUTPUT:
    merkle_proof: MerkleProof object

  ALGORITHM:
    # Find target entry
    target_entry = FIND(audit_trail, WHERE id == audit_entry_id)
    target_index = INDEX_OF(audit_trail, target_entry)

    # Build Merkle tree
    leaves = MAP(audit_trail, entry -> SHA256(entry.hash))
    tree = build_merkle_tree(leaves)

    # Generate proof (path from leaf to root)
    proof_path = []
    current_index = target_index
    current_level = 0

    WHILE current_level < tree.height:
      # Find sibling node
      is_left_child = (current_index % 2 == 0)
      sibling_index = is_left_child ? current_index + 1 : current_index - 1

      IF sibling_index < tree.levels[current_level].length:
        sibling_hash = tree.levels[current_level][sibling_index]

        proof_path.append({
          hash: sibling_hash,
          position: is_left_child ? "RIGHT" : "LEFT"
        })

      # Move to parent level
      current_index = FLOOR(current_index / 2)
      current_level += 1

    merkle_proof = MerkleProof {
      entryId: audit_entry_id,
      entryHash: target_entry.hash,
      path: proof_path,
      root: tree.root,
      timestamp: NOW()
    }

    RETURN merkle_proof

FUNCTION verify_merkle_proof(proof, claimed_entry):
  INPUT:
    proof: MerkleProof object
    claimed_entry: AuditEntry to verify

  OUTPUT:
    is_valid: Boolean

  ALGORITHM:
    # Hash the claimed entry
    current_hash = SHA256(claimed_entry.hash)

    # Reconstruct root by following proof path
    FOR EACH step IN proof.path:
      IF step.position == "LEFT":
        current_hash = SHA256(step.hash + current_hash)
      ELSE:
        current_hash = SHA256(current_hash + step.hash)

    # Verify reconstructed root matches proof root
    is_valid = (current_hash == proof.root)

    RETURN is_valid

FUNCTION build_merkle_tree(leaves):
  INPUT:
    leaves: Array of leaf hashes

  OUTPUT:
    tree: MerkleTree object with levels and root

  ALGORITHM:
    tree = MerkleTree {
      levels: [leaves],
      height: 0
    }

    current_level = leaves

    WHILE current_level.length > 1:
      next_level = []

      FOR i FROM 0 TO current_level.length STEP 2:
        left = current_level[i]
        right = i + 1 < current_level.length ? current_level[i + 1] : left

        parent = SHA256(left + right)
        next_level.append(parent)

      tree.levels.append(next_level)
      tree.height += 1
      current_level = next_level

    tree.root = current_level[0]

    RETURN tree
```

---

## 5. Anomaly Scoring System

```
FUNCTION calculate_anomaly_score(transaction, historical_data, fraud_signatures):
  INPUT:
    transaction: Transaction to score
    historical_data: User's transaction history
    fraud_signatures: Known fraud patterns

  OUTPUT:
    anomaly_score: Number (0-100)
    risk_factors: Array of identified risk factors

  ALGORITHM:
    risk_factors = []
    anomaly_score = 0

    # 1. Statistical anomaly (25 points)
    stat_score = check_statistical_anomaly(transaction, historical_data)
    anomaly_score += stat_score * 0.25
    IF stat_score > 50:
      risk_factors.append("STATISTICAL_OUTLIER")

    # 2. Fraud pattern similarity (35 points)
    fraud_score = check_fraud_similarity(transaction, fraud_signatures)
    anomaly_score += fraud_score * 0.35
    IF fraud_score > 70:
      risk_factors.append("FRAUD_PATTERN_MATCH")

    # 3. Behavioral anomaly (20 points)
    behavior_score = check_behavioral_anomaly(transaction, historical_data)
    anomaly_score += behavior_score * 0.20
    IF behavior_score > 60:
      risk_factors.append("ABNORMAL_BEHAVIOR")

    # 4. Temporal anomaly (10 points)
    temporal_score = check_temporal_anomaly(transaction, historical_data)
    anomaly_score += temporal_score * 0.10
    IF temporal_score > 70:
      risk_factors.append("UNUSUAL_TIMING")

    # 5. Network anomaly (10 points)
    network_score = check_network_anomaly(transaction)
    anomaly_score += network_score * 0.10
    IF network_score > 80:
      risk_factors.append("SUSPICIOUS_SOURCE")

    # Normalize to 0-100
    anomaly_score = CLAMP(anomaly_score, 0, 100)

    # Determine risk level
    risk_level = SWITCH anomaly_score:
      CASE < 30: "LOW"
      CASE < 60: "MEDIUM"
      CASE < 80: "HIGH"
      DEFAULT: "CRITICAL"

    RETURN {
      score: anomaly_score,
      riskLevel: risk_level,
      factors: risk_factors,
      timestamp: NOW()
    }

FUNCTION check_behavioral_anomaly(transaction, historical_data):
  INPUT:
    transaction: Current transaction
    historical_data: Historical transactions

  OUTPUT:
    score: Number (0-100)

  ALGORITHM:
    score = 0

    # Unusual amount for this user
    typical_amount = MEDIAN(historical_data, t -> t.quantity * t.price)
    amount_ratio = transaction.quantity * transaction.price / typical_amount
    IF amount_ratio > 10:
      score += 40
    ELSE IF amount_ratio > 5:
      score += 20

    # Unusual asset for this user
    user_assets = UNIQUE(historical_data, t -> t.asset)
    IF transaction.asset NOT IN user_assets:
      score += 30

    # Unusual source/exchange
    user_sources = UNIQUE(historical_data, t -> t.source)
    IF transaction.source NOT IN user_sources:
      score += 20

    # High frequency (velocity)
    recent_count = COUNT(historical_data, WHERE
      timestamp > NOW() - 1_HOUR
    )
    IF recent_count > 10:
      score += 10

    RETURN CLAMP(score, 0, 100)
```

---

## 6. Real-Time Fraud Alert System

```
FUNCTION process_transaction_for_fraud(transaction):
  INPUT:
    transaction: Incoming transaction

  OUTPUT:
    alert: FraudAlert object or null

  ALGORITHM:
    # Load user context
    user_history = load_user_transactions(transaction.userId)
    fraud_signatures = load_fraud_signatures()

    # Calculate anomaly score
    anomaly_result = calculate_anomaly_score(
      transaction,
      user_history,
      fraud_signatures
    )

    # Check if alert threshold exceeded
    ALERT_THRESHOLD = 70

    IF anomaly_result.score >= ALERT_THRESHOLD:
      # Generate alert
      alert = FraudAlert {
        id: generate_uuid(),
        transactionId: transaction.id,
        userId: transaction.userId,
        anomalyScore: anomaly_result.score,
        riskLevel: anomaly_result.riskLevel,
        riskFactors: anomaly_result.factors,
        timestamp: NOW(),
        status: "PENDING_REVIEW",
        escalated: anomaly_result.riskLevel == "CRITICAL"
      }

      # Store alert
      save_fraud_alert(alert)

      # Trigger notifications
      IF alert.escalated:
        notify_compliance_team(alert)

      # Block transaction if critical
      IF anomaly_result.score >= 90:
        block_transaction(transaction, alert)

      # Store in AgentDB for learning
      store_fraud_case(transaction, alert, anomaly_result)

      RETURN alert

    RETURN null
```

---

## Performance Considerations

1. **Vector Index**: Use HNSW for O(log n) similarity search
2. **Caching**: Cache user profiles and fraud signatures
3. **Batch Processing**: Process multiple transactions in parallel
4. **Incremental Updates**: Update Merkle trees incrementally
5. **GPU Acceleration**: Use GPU for large-scale embedding generation
6. **Rust Implementation**: Critical path in Rust for sub-millisecond performance
