# Agentic Accounting System - Testing Strategy

## Test-Driven Development (TDD) Approach

Following SPARC methodology, we implement comprehensive testing at all levels before and during development.

## Testing Pyramid

```
           ┌────────────────┐
           │  E2E Tests     │  10%
           │  (Slow)        │
           ├────────────────┤
           │ Integration    │  20%
           │   Tests        │
           ├────────────────┤
           │  Unit Tests    │  70%
           │  (Fast)        │
           └────────────────┘
```

---

## 1. Unit Tests (70% of test suite)

**Target**: Individual functions and modules in isolation

**Coverage Goal**: >90% code coverage

### Tax Calculation Tests

```typescript
describe('Tax Calculations', () => {
  describe('FIFO Method', () => {
    it('should calculate simple FIFO disposal correctly', () => {
      const lots = [
        createLot({ date: '2023-01-01', quantity: 10, costBasis: 100 }),
        createLot({ date: '2023-02-01', quantity: 5, costBasis: 60 }),
      ];
      const sale = createSale({ date: '2023-03-01', quantity: 12, proceeds: 180 });

      const result = calculateFifo(sale, lots);

      expect(result.disposals).toHaveLength(2);
      expect(result.disposals[0].quantity).toBe(10);
      expect(result.disposals[0].costBasis).toBe(100);
      expect(result.disposals[1].quantity).toBe(2);
      expect(result.disposals[1].costBasis).toBe(24);
      expect(result.totalGain).toBe(56); // 180 - (100 + 24)
    });

    it('should handle insufficient lots gracefully', () => {
      const lots = [createLot({ quantity: 5, costBasis: 50 })];
      const sale = createSale({ quantity: 10, proceeds: 100 });

      expect(() => calculateFifo(sale, lots)).toThrow(InsufficientQuantityError);
    });

    it('should correctly determine short vs long term', () => {
      const lots = [
        createLot({ date: '2022-01-01', quantity: 10, costBasis: 100 }),
      ];
      const sale = createSale({ date: '2023-02-01', quantity: 10, proceeds: 150 });

      const result = calculateFifo(sale, lots);

      expect(result.disposals[0].term).toBe('LONG');
      expect(result.disposals[0].gain).toBe(50);
    });
  });

  describe('Wash Sale Detection', () => {
    it('should detect wash sale within 30-day window', () => {
      const disposal = createDisposal({
        date: '2023-06-15',
        asset: 'BTC',
        quantity: 1,
        gain: -1000,
      });
      const transactions = [
        createTransaction({ type: 'BUY', asset: 'BTC', date: '2023-06-20', quantity: 1 }),
      ];

      const result = detectWashSale(disposal, transactions);

      expect(result.isWashSale).toBe(true);
      expect(result.disallowedLoss).toBe(1000);
    });

    it('should not flag wash sale for gains', () => {
      const disposal = createDisposal({ gain: 1000 });
      const transactions = [createTransaction({ type: 'BUY', date: '2023-06-20' })];

      const result = detectWashSale(disposal, transactions);

      expect(result.isWashSale).toBe(false);
    });

    it('should not flag if replacement is >30 days away', () => {
      const disposal = createDisposal({ date: '2023-06-01', gain: -1000 });
      const transactions = [
        createTransaction({ type: 'BUY', date: '2023-07-15', quantity: 1 }),
      ];

      const result = detectWashSale(disposal, transactions);

      expect(result.isWashSale).toBe(false);
    });
  });

  describe('Decimal Precision', () => {
    it('should handle precise decimal calculations without rounding errors', () => {
      const lot = createLot({ quantity: 0.00123456, costBasis: 10.987654321 });
      const sale = createSale({ quantity: 0.00123456, proceeds: 15.123456789 });

      const result = calculateFifo(sale, [lot]);

      // Should maintain full precision
      expect(result.totalGain.toString()).toBe('4.135802468');
    });
  });
});
```

### Forensic Analysis Tests

```typescript
describe('Fraud Detection', () => {
  describe('Vector Similarity', () => {
    it('should find similar fraud patterns above threshold', async () => {
      const transaction = createTransaction({
        type: 'TRANSFER',
        amount: 50000,
        source: 'offshore-account',
      });

      const fraudSignatures = await loadFraudSignatures();
      const result = await detectFraudPatterns(transaction, fraudSignatures, 0.85);

      expect(result.isSuspicious).toBe(true);
      expect(result.matches.length).toBeGreaterThan(0);
      expect(result.matches[0].score).toBeGreaterThanOrEqual(0.85);
    });

    it('should not flag normal transactions', async () => {
      const transaction = createTransaction({
        type: 'BUY',
        amount: 100,
        source: 'coinbase',
      });

      const result = await detectFraudPatterns(transaction, await loadFraudSignatures());

      expect(result.isSuspicious).toBe(false);
    });
  });

  describe('Outlier Detection', () => {
    it('should identify statistical outliers', () => {
      const transactions = [
        ...generateNormalTransactions(100, { mean: 1000, stdDev: 200 }),
        createTransaction({ amount: 50000 }), // Outlier
      ];

      const outliers = detectStatisticalOutliers(transactions, 2.0);

      expect(outliers).toHaveLength(1);
      expect(outliers[0].amount).toBe(50000);
    });
  });

  describe('Merkle Proofs', () => {
    it('should generate valid Merkle proof for audit entry', () => {
      const auditTrail = generateAuditEntries(100);
      const targetEntry = auditTrail[42];

      const proof = generateMerkleProof(targetEntry.id, auditTrail);

      expect(proof.path).toBeDefined();
      expect(proof.root).toBeDefined();
    });

    it('should verify valid Merkle proofs', () => {
      const auditTrail = generateAuditEntries(100);
      const targetEntry = auditTrail[42];
      const proof = generateMerkleProof(targetEntry.id, auditTrail);

      const isValid = verifyMerkleProof(proof, targetEntry);

      expect(isValid).toBe(true);
    });

    it('should reject invalid proofs', () => {
      const auditTrail = generateAuditEntries(100);
      const targetEntry = auditTrail[42];
      const proof = generateMerkleProof(targetEntry.id, auditTrail);

      // Tamper with entry
      const tamperedEntry = { ...targetEntry, amount: 99999 };

      const isValid = verifyMerkleProof(proof, tamperedEntry);

      expect(isValid).toBe(false);
    });
  });
});
```

### Agent Behavior Tests

```typescript
describe('Agent Coordination', () => {
  describe('Task Allocation', () => {
    it('should route tax calculation tasks to tax agent', async () => {
      const coordinator = new CoordinatorAgent();
      const task = createTask({ type: 'CALCULATE_TAX' });

      const assignment = await coordinator.assignTask(task);

      expect(assignment.agentType).toBe('TAX_COMPUTE');
    });

    it('should load balance across multiple agent instances', async () => {
      const coordinator = new CoordinatorAgent();
      const agents = [
        new TaxComputeAgent({ id: 'agent-1', load: 0.2 }),
        new TaxComputeAgent({ id: 'agent-2', load: 0.8 }),
      ];

      const task = createTask({ type: 'CALCULATE_TAX' });
      const assignment = await coordinator.assignTask(task, agents);

      expect(assignment.agentId).toBe('agent-1'); // Lower load
    });
  });

  describe('ReasoningBank Integration', () => {
    it('should store successful decisions', async () => {
      const agent = new TaxComputeAgent();
      const scenario = 'complex_multi_lot_disposal';
      const decision = 'use_hifo_method';

      await agent.logDecision(scenario, decision, 'Minimizes tax liability', 'SUCCESS');

      const entries = await agent.retrieveSimilarDecisions(scenario, 5);

      expect(entries).toHaveLength(1);
      expect(entries[0].decision).toBe(decision);
    });

    it('should retrieve similar decisions for guidance', async () => {
      const agent = new TaxComputeAgent();

      // Pre-populate with successful decisions
      await seedReasoningBank([
        { scenario: 'high_value_disposal', decision: 'use_hifo', outcome: 'SUCCESS' },
        { scenario: 'high_value_disposal', decision: 'use_fifo', outcome: 'FAILURE' },
      ]);

      const similar = await agent.retrieveSimilarDecisions('high_value_disposal', 5);

      expect(similar[0].decision).toBe('use_hifo'); // Higher ranked
      expect(similar[0].outcome).toBe('SUCCESS');
    });
  });
});
```

---

## 2. Integration Tests (20% of test suite)

**Target**: Interaction between multiple components

**Environment**: Uses test database and AgentDB instance

### Database Integration

```typescript
describe('Database Integration', () => {
  let db: DatabaseClient;

  beforeAll(async () => {
    db = await createTestDatabase();
    await runMigrations(db);
  });

  afterAll(async () => {
    await db.close();
  });

  describe('Transaction Persistence', () => {
    it('should persist transactions with embeddings', async () => {
      const transaction = createTransaction({ amount: 1000, asset: 'BTC' });

      await db.transactions.insert(transaction);

      const retrieved = await db.transactions.findById(transaction.id);
      expect(retrieved).toMatchObject(transaction);

      // Check embedding created
      const embedding = await db.embeddings.findByEntityId(transaction.id);
      expect(embedding).toBeDefined();
      expect(embedding.vector).toHaveLength(768);
    });

    it('should support vector similarity queries', async () => {
      await db.transactions.insertMany(generateTestTransactions(100));

      const queryTransaction = createTransaction({
        type: 'SELL',
        amount: 50000,
        asset: 'BTC',
      });

      const similar = await db.transactions.findSimilar(queryTransaction, 10);

      expect(similar).toHaveLength(10);
      expect(similar[0].similarityScore).toBeGreaterThan(0.7);
    });
  });

  describe('Audit Trail Integrity', () => {
    it('should maintain immutable audit log chain', async () => {
      const entries = generateAuditEntries(10);

      for (const entry of entries) {
        await db.auditTrail.append(entry);
      }

      const retrieved = await db.auditTrail.getAll();

      // Verify chain integrity
      for (let i = 1; i < retrieved.length; i++) {
        expect(retrieved[i].previousHash).toBe(retrieved[i - 1].hash);
      }
    });

    it('should detect tampered entries', async () => {
      await db.auditTrail.append(generateAuditEntries(10));

      // Attempt to modify entry directly (simulate attack)
      await db.query('UPDATE audit_trail SET amount = 9999 WHERE id = $1', [
        entries[5].id,
      ]);

      const isValid = await db.auditTrail.verifyIntegrity();

      expect(isValid).toBe(false);
    });
  });
});
```

### Agent Communication

```typescript
describe('Agent Communication', () => {
  let coordinator: CoordinatorAgent;
  let taxAgent: TaxComputeAgent;
  let complianceAgent: ComplianceAgent;

  beforeAll(async () => {
    coordinator = new CoordinatorAgent();
    taxAgent = new TaxComputeAgent();
    complianceAgent = new ComplianceAgent();

    await coordinator.registerAgent(taxAgent);
    await coordinator.registerAgent(complianceAgent);
  });

  it('should coordinate multi-agent workflow', async () => {
    const transaction = createTransaction({ type: 'SELL', amount: 10000 });

    const result = await coordinator.processTransaction(transaction);

    expect(result.taxCalculation).toBeDefined();
    expect(result.complianceCheck).toBeDefined();
    expect(result.complianceCheck.approved).toBe(true);
  });

  it('should handle agent failures gracefully', async () => {
    // Simulate agent failure
    jest.spyOn(taxAgent, 'execute').mockRejectedValue(new Error('Agent crashed'));

    const transaction = createTransaction({ type: 'SELL' });

    await expect(coordinator.processTransaction(transaction)).rejects.toThrow();

    // Verify retry logic
    expect(taxAgent.execute).toHaveBeenCalledTimes(3); // Initial + 2 retries
  });
});
```

---

## 3. End-to-End Tests (10% of test suite)

**Target**: Complete user workflows from start to finish

**Environment**: Fully deployed system (staging)

### Full Tax Year Workflow

```typescript
describe('Complete Tax Year Workflow', () => {
  it('should process full year of transactions and generate tax forms', async () => {
    // 1. Import transactions
    const transactions = await importTransactions('./test-data/2023-transactions.csv');
    expect(transactions).toHaveLength(500);

    // 2. Calculate taxes
    const taxSummary = await calculateAnnualTax(2023, 'HIFO');
    expect(taxSummary.totalGains).toBeDefined();

    // 3. Identify harvest opportunities
    const opportunities = await identifyHarvestOpportunities(2023);
    expect(opportunities.length).toBeGreaterThan(0);

    // 4. Execute harvesting
    const harvested = await executeHarvesting(opportunities.slice(0, 5));
    expect(harvested.totalLossBanked).toBeGreaterThan(0);

    // 5. Generate reports
    const scheduleD = await generateReport('SCHEDULE_D', 2023);
    const form8949 = await generateReport('FORM_8949', 2023);

    expect(scheduleD).toContain('Schedule D');
    expect(form8949).toContain('Form 8949');

    // 6. Verify audit trail
    const auditLog = await getAuditTrail(2023);
    expect(auditLog.length).toBeGreaterThan(500);
    expect(await verifyAuditIntegrity(auditLog)).toBe(true);
  });
});
```

### Compliance & Forensics

```typescript
describe('Compliance & Forensics E2E', () => {
  it('should detect and flag fraudulent activity', async () => {
    // Inject suspicious transactions
    await injectSuspiciousTransactions([
      { type: 'TRANSFER', amount: 100000, source: 'unknown' },
      { type: 'SELL', amount: 50000, timing: 'unusual' },
    ]);

    // Trigger forensic analysis
    const alerts = await runForensicScan();

    expect(alerts.length).toBeGreaterThanOrEqual(2);
    expect(alerts[0].severity).toBe('HIGH');
    expect(alerts[0].riskFactors).toContain('UNUSUAL_AMOUNT');

    // Verify compliance agent blocked transactions
    const blockedTxns = await getBlockedTransactions();
    expect(blockedTxns.length).toBeGreaterThan(0);
  });
});
```

---

## 4. Performance Tests

**Target**: Verify performance SLAs

### Benchmarks

```typescript
describe('Performance Benchmarks', () => {
  it('should handle 10,000 transactions in <5 seconds', async () => {
    const transactions = generateTestTransactions(10000);
    const start = Date.now();

    await bulkImportTransactions(transactions);

    const duration = Date.now() - start;
    expect(duration).toBeLessThan(5000);
  });

  it('should execute vector search in <100µs', async () => {
    await seedAgentDB(100000); // 100K vectors

    const query = generateRandomVector(768);
    const iterations = 1000;

    const start = process.hrtime.bigint();
    for (let i = 0; i < iterations; i++) {
      await agentdb.search(query, 10);
    }
    const end = process.hrtime.bigint();

    const avgMicroseconds = Number(end - start) / 1000 / iterations;
    expect(avgMicroseconds).toBeLessThan(100);
  });

  it('should complete Rust tax calculation in <10ms', async () => {
    const lots = generateTestLots(1000);
    const sale = createTestSale({ quantity: 500 });

    const start = Date.now();
    const result = await rustCore.calculateFifo(sale, lots);
    const duration = Date.now() - start;

    expect(duration).toBeLessThan(10);
    expect(result.disposals.length).toBeGreaterThan(0);
  });
});
```

---

## 5. Security Tests

**Target**: Identify vulnerabilities and attack vectors

### Penetration Testing

```typescript
describe('Security Tests', () => {
  it('should prevent SQL injection attacks', async () => {
    const maliciousInput = "'; DROP TABLE transactions; --";

    await expect(
      api.post('/api/transactions', { asset: maliciousInput })
    ).rejects.toThrow();

    // Verify table still exists
    const count = await db.transactions.count();
    expect(count).toBeGreaterThanOrEqual(0);
  });

  it('should reject tampered audit entries', async () => {
    const entry = await db.auditTrail.findById('some-id');
    const proof = await generateMerkleProof(entry.id);

    // Tamper with entry
    entry.amount = 99999;

    const isValid = await verifyMerkleProof(proof, entry);
    expect(isValid).toBe(false);
  });

  it('should enforce RBAC permissions', async () => {
    const userToken = await authenticate({ role: 'VIEWER' });

    await expect(
      api.post('/api/transactions', {}, { headers: { Authorization: userToken } })
    ).rejects.toThrow('Forbidden');
  });
});
```

---

## Test Data Management

### Fixtures & Factories

```typescript
// factories.ts
export const createTransaction = (overrides: Partial<Transaction> = {}) => ({
  id: uuid(),
  type: 'BUY',
  asset: 'BTC',
  quantity: 1.0,
  price: 50000,
  timestamp: new Date(),
  ...overrides,
});

export const createLot = (overrides: Partial<TaxLot> = {}) => ({
  id: uuid(),
  asset: 'BTC',
  quantity: 1.0,
  costBasis: 50000,
  acquiredDate: new Date(),
  ...overrides,
});

// fixtures.ts
export const fixtures = {
  transactions: loadJSON('./fixtures/transactions.json'),
  lots: loadJSON('./fixtures/lots.json'),
  disposals: loadJSON('./fixtures/disposals.json'),
};
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm run test:unit

  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:latest
      redis:
        image: redis:7-alpine
    steps:
      - run: npm run test:integration

  e2e-tests:
    needs: integration-tests
    runs-on: ubuntu-latest
    steps:
      - run: npm run test:e2e

  performance-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - run: npm run test:perf
```

---

## Test Coverage Reporting

- **Tool**: Istanbul/nyc
- **Threshold**: 90% coverage
- **Reports**: HTML, LCOV, JSON
- **CI**: Block merge if coverage drops below threshold

---

## Testing Best Practices

1. **Test isolation**: Each test should be independent
2. **Fast feedback**: Unit tests run in <5 seconds
3. **Deterministic**: Tests produce same result every time
4. **Descriptive names**: Clear test intent
5. **Arrange-Act-Assert**: Clear test structure
6. **Mock external dependencies**: Isolate system under test
7. **Test edge cases**: Boundary conditions, errors
8. **Continuous testing**: Run on every commit
