/**
 * Comprehensive Type Validation Tests
 * Coverage Target: 95%+
 */

import {
  Transaction,
  Position,
  Lot,
  TaxResult,
  TaxTransaction,
  TransactionSource,
  IngestionResult,
  ComplianceRule,
  ComplianceViolation,
  AgentConfig,
} from '../src/index';

describe('Transaction Interface', () => {
  it('should validate complete transaction object', () => {
    const transaction: Transaction = {
      id: 'txn-001',
      timestamp: new Date('2024-01-15'),
      type: 'BUY',
      asset: 'BTC',
      quantity: 1.5,
      price: 45000,
      fees: 50,
      exchange: 'coinbase',
      walletAddress: '0x123',
      metadata: { note: 'test' },
      source: 'api',
    };

    expect(transaction.id).toBe('txn-001');
    expect(transaction.type).toBe('BUY');
    expect(transaction.quantity).toBe(1.5);
  });

  it('should support all transaction types', () => {
    const types: Transaction['type'][] = [
      'BUY',
      'SELL',
      'TRADE',
      'CONVERT',
      'INCOME',
      'DIVIDEND',
      'FEE',
      'TRANSFER',
    ];

    types.forEach((type) => {
      const txn: Transaction = {
        id: 'test',
        timestamp: new Date(),
        type,
        asset: 'BTC',
        quantity: 1,
        price: 50000,
      };
      expect(txn.type).toBe(type);
    });
  });

  it('should handle optional fields', () => {
    const minimal: Transaction = {
      id: 'txn-002',
      timestamp: new Date(),
      type: 'SELL',
      asset: 'ETH',
      quantity: 10,
      price: 2500,
    };

    expect(minimal.fees).toBeUndefined();
    expect(minimal.exchange).toBeUndefined();
    expect(minimal.metadata).toBeUndefined();
  });

  it('should support TransactionSource as object', () => {
    const source: TransactionSource = {
      type: 'exchange',
      name: 'binance',
      credentials: { apiKey: 'test' },
    };

    const transaction: Transaction = {
      id: 'txn-003',
      timestamp: new Date(),
      type: 'BUY',
      asset: 'SOL',
      quantity: 100,
      price: 75,
      source,
    };

    expect(transaction.source).toEqual(source);
  });
});

describe('Position Interface', () => {
  it('should validate complete position object', () => {
    const position: Position = {
      id: 'pos-001',
      asset: 'BTC',
      quantity: 2.5,
      averageCost: 40000,
      currentValue: 112500,
      unrealizedGainLoss: 12500,
      lots: [],
      lastUpdated: new Date(),
      totalCost: 100000,
      averageCostBasis: 40000,
    };

    expect(position.asset).toBe('BTC');
    expect(position.quantity).toBe(2.5);
    expect(position.unrealizedGainLoss).toBe(12500);
  });

  it('should contain array of lots', () => {
    const lot: Lot = {
      id: 'lot-001',
      asset: 'ETH',
      quantity: 10,
      purchasePrice: 2000,
      purchaseDate: new Date('2024-01-01'),
      acquisitionDate: new Date('2024-01-01'),
      transactionId: 'txn-001',
      disposed: false,
      isOpen: true,
      remainingQuantity: 10,
      costBasis: 20000,
    };

    const position: Position = {
      id: 'pos-002',
      asset: 'ETH',
      quantity: 10,
      averageCost: 2000,
      currentValue: 25000,
      unrealizedGainLoss: 5000,
      lots: [lot],
      lastUpdated: new Date(),
      totalCost: 20000,
      averageCostBasis: 2000,
    };

    expect(position.lots).toHaveLength(1);
    expect(position.lots[0].asset).toBe('ETH');
  });
});

describe('Lot Interface', () => {
  it('should validate lot with all fields', () => {
    const lot: Lot = {
      id: 'lot-001',
      asset: 'BTC',
      quantity: 0.5,
      purchasePrice: 45000,
      purchaseDate: new Date('2024-01-01'),
      acquisitionDate: new Date('2024-01-01'),
      transactionId: 'txn-001',
      disposed: true,
      disposedDate: new Date('2024-06-01'),
      disposedPrice: 60000,
      isOpen: false,
      remainingQuantity: 0,
      costBasis: 22500,
    };

    expect(lot.disposed).toBe(true);
    expect(lot.disposedPrice).toBe(60000);
    expect(lot.remainingQuantity).toBe(0);
  });

  it('should handle open lot without disposal info', () => {
    const lot: Lot = {
      id: 'lot-002',
      asset: 'ETH',
      quantity: 5,
      purchasePrice: 2500,
      purchaseDate: new Date('2024-03-01'),
      acquisitionDate: new Date('2024-03-01'),
      transactionId: 'txn-002',
      disposed: false,
      isOpen: true,
      remainingQuantity: 5,
      costBasis: 12500,
    };

    expect(lot.disposed).toBe(false);
    expect(lot.isOpen).toBe(true);
    expect(lot.disposedDate).toBeUndefined();
  });

  it('should support both purchaseDate and acquisitionDate aliases', () => {
    const lot: Lot = {
      id: 'lot-003',
      asset: 'SOL',
      quantity: 100,
      purchasePrice: 50,
      purchaseDate: new Date('2024-02-01'),
      acquisitionDate: new Date('2024-02-01'),
      transactionId: 'txn-003',
      isOpen: true,
      remainingQuantity: 100,
      costBasis: 5000,
    };

    expect(lot.purchaseDate).toEqual(lot.acquisitionDate);
  });
});

describe('TaxResult Interface', () => {
  it('should validate complete tax result', () => {
    const taxResult: TaxResult = {
      totalGain: 15000,
      totalLoss: 3000,
      shortTermGain: 8000,
      shortTermLoss: 2000,
      longTermGain: 7000,
      longTermLoss: 1000,
      transactions: [],
      year: 2024,
    };

    expect(taxResult.totalGain).toBe(15000);
    expect(taxResult.year).toBe(2024);
    expect(taxResult.transactions).toEqual([]);
  });

  it('should contain array of tax transactions', () => {
    const taxTxn: TaxTransaction = {
      id: 'tax-001',
      asset: 'BTC',
      buyDate: new Date('2023-01-01'),
      sellDate: new Date('2024-06-01'),
      acquisitionDate: new Date('2023-01-01'),
      disposalDate: new Date('2024-06-01'),
      quantity: 0.5,
      costBasis: 20000,
      proceeds: 30000,
      gainLoss: 10000,
      holdingPeriod: 518,
      type: 'long-term',
      isLongTerm: true,
    };

    const result: TaxResult = {
      totalGain: 10000,
      totalLoss: 0,
      shortTermGain: 0,
      shortTermLoss: 0,
      longTermGain: 10000,
      longTermLoss: 0,
      transactions: [taxTxn],
      year: 2024,
    };

    expect(result.transactions).toHaveLength(1);
    expect(result.transactions[0].type).toBe('long-term');
  });
});

describe('TaxTransaction Interface', () => {
  it('should validate short-term capital gain', () => {
    const txn: TaxTransaction = {
      id: 'tax-001',
      asset: 'ETH',
      buyDate: new Date('2024-01-01'),
      sellDate: new Date('2024-06-01'),
      acquisitionDate: new Date('2024-01-01'),
      disposalDate: new Date('2024-06-01'),
      quantity: 10,
      costBasis: 25000,
      proceeds: 30000,
      gainLoss: 5000,
      holdingPeriod: 152,
      type: 'short-term',
      isLongTerm: false,
    };

    expect(txn.type).toBe('short-term');
    expect(txn.isLongTerm).toBe(false);
    expect(txn.gainLoss).toBe(5000);
  });

  it('should validate long-term capital loss', () => {
    const txn: TaxTransaction = {
      id: 'tax-002',
      asset: 'SOL',
      buyDate: new Date('2022-01-01'),
      sellDate: new Date('2024-01-01'),
      acquisitionDate: new Date('2022-01-01'),
      disposalDate: new Date('2024-01-01'),
      quantity: 100,
      costBasis: 10000,
      proceeds: 7500,
      gainLoss: -2500,
      holdingPeriod: 730,
      type: 'long-term',
      isLongTerm: true,
    };

    expect(txn.type).toBe('long-term');
    expect(txn.gainLoss).toBe(-2500);
  });

  it('should support wash sale adjustment', () => {
    const txn: TaxTransaction = {
      id: 'tax-003',
      asset: 'BTC',
      buyDate: new Date('2024-01-01'),
      sellDate: new Date('2024-02-01'),
      acquisitionDate: new Date('2024-01-01'),
      disposalDate: new Date('2024-02-01'),
      quantity: 0.5,
      costBasis: 25000,
      proceeds: 22000,
      gainLoss: -3000,
      washSaleAdjustment: 3000,
      holdingPeriod: 31,
      type: 'short-term',
      isLongTerm: false,
      method: 'FIFO',
      metadata: { washSaleDetected: true },
    };

    expect(txn.washSaleAdjustment).toBe(3000);
    expect(txn.method).toBe('FIFO');
  });
});

describe('TransactionSource Interface', () => {
  it('should validate exchange source', () => {
    const source: TransactionSource = {
      type: 'exchange',
      name: 'coinbase',
      credentials: { apiKey: 'test', apiSecret: 'secret' },
    };

    expect(source.type).toBe('exchange');
    expect(source.credentials).toBeDefined();
  });

  it('should validate wallet source', () => {
    const source: TransactionSource = {
      type: 'wallet',
      name: 'metamask',
    };

    expect(source.type).toBe('wallet');
    expect(source.credentials).toBeUndefined();
  });

  it('should support all source types', () => {
    const types: TransactionSource['type'][] = ['exchange', 'wallet', 'csv', 'api'];

    types.forEach((type) => {
      const source: TransactionSource = { type, name: 'test' };
      expect(source.type).toBe(type);
    });
  });
});

describe('IngestionResult Interface', () => {
  it('should validate successful ingestion', () => {
    const result: IngestionResult = {
      success: true,
      transactionsImported: 50,
      errors: [],
      warnings: ['Price missing for 3 transactions'],
      source: { type: 'exchange', name: 'binance' },
      timestamp: new Date(),
      total: 50,
      duration: 2500,
      successful: 50,
      failed: 0,
    };

    expect(result.success).toBe(true);
    expect(result.transactionsImported).toBe(50);
    expect(result.errors).toHaveLength(0);
  });

  it('should validate failed ingestion', () => {
    const result: IngestionResult = {
      success: false,
      transactionsImported: 0,
      errors: ['API authentication failed', 'Network timeout'],
      warnings: [],
      source: { type: 'api', name: 'kraken' },
      timestamp: new Date(),
    };

    expect(result.success).toBe(false);
    expect(result.errors).toHaveLength(2);
  });

  it('should include transaction array when available', () => {
    const transactions: Transaction[] = [
      {
        id: 'txn-001',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
      },
    ];

    const result: IngestionResult = {
      success: true,
      transactionsImported: 1,
      errors: [],
      warnings: [],
      source: { type: 'csv', name: 'upload' },
      timestamp: new Date(),
      transactions,
    };

    expect(result.transactions).toHaveLength(1);
  });
});

describe('ComplianceRule Interface', () => {
  it('should validate tax compliance rule', () => {
    const rule: ComplianceRule = {
      id: 'rule-001',
      name: 'Wash Sale Rule',
      description: '30-day wash sale period for securities',
      category: 'tax',
      jurisdiction: 'US',
      severity: 'warning',
    };

    expect(rule.category).toBe('tax');
    expect(rule.severity).toBe('warning');
  });

  it('should support all rule categories', () => {
    const categories: ComplianceRule['category'][] = ['tax', 'regulatory', 'reporting'];

    categories.forEach((category) => {
      const rule: ComplianceRule = {
        id: `rule-${category}`,
        name: `Test ${category}`,
        description: 'Test rule',
        category,
        jurisdiction: 'US',
        severity: 'info',
      };
      expect(rule.category).toBe(category);
    });
  });

  it('should support all severity levels', () => {
    const severities: ComplianceRule['severity'][] = ['info', 'warning', 'error', 'critical'];

    severities.forEach((severity) => {
      const rule: ComplianceRule = {
        id: `rule-${severity}`,
        name: 'Test rule',
        description: 'Test rule',
        category: 'tax',
        jurisdiction: 'US',
        severity,
      };
      expect(rule.severity).toBe(severity);
    });
  });
});

describe('ComplianceViolation Interface', () => {
  it('should validate complete violation', () => {
    const violation: ComplianceViolation = {
      ruleId: 'rule-001',
      severity: 'error',
      message: 'Wash sale detected within 30-day window',
      transactionId: 'txn-001',
      details: {
        daysApart: 15,
        asset: 'BTC',
        loss: -5000,
      },
      timestamp: new Date(),
    };

    expect(violation.severity).toBe('error');
    expect(violation.transactionId).toBe('txn-001');
    expect(violation.details).toBeDefined();
  });

  it('should handle violation without transaction ID', () => {
    const violation: ComplianceViolation = {
      ruleId: 'rule-002',
      severity: 'warning',
      message: 'Missing cost basis information',
      timestamp: new Date(),
    };

    expect(violation.transactionId).toBeUndefined();
    expect(violation.details).toBeUndefined();
  });
});

describe('AgentConfig Interface', () => {
  it('should validate complete agent config', () => {
    const config: AgentConfig = {
      agentId: 'tax-compute-001',
      agentType: 'TAX_COMPUTE',
      enableLearning: true,
      enableMetrics: true,
      logLevel: 'info',
    };

    expect(config.agentId).toBe('tax-compute-001');
    expect(config.enableLearning).toBe(true);
  });

  it('should support all log levels', () => {
    const logLevels: AgentConfig['logLevel'][] = ['debug', 'info', 'warn', 'error'];

    logLevels.forEach((logLevel) => {
      const config: AgentConfig = {
        agentId: 'test-agent',
        agentType: 'TEST',
        logLevel,
      };
      expect(config.logLevel).toBe(logLevel);
    });
  });

  it('should handle minimal config', () => {
    const config: AgentConfig = {
      agentId: 'minimal-agent',
      agentType: 'MINIMAL',
    };

    expect(config.enableLearning).toBeUndefined();
    expect(config.logLevel).toBeUndefined();
  });
});

describe('Type Compatibility', () => {
  it('should allow date aliases in tax transactions', () => {
    const txn: TaxTransaction = {
      id: 'tax-001',
      asset: 'BTC',
      buyDate: new Date('2024-01-01'),
      sellDate: new Date('2024-06-01'),
      acquisitionDate: new Date('2024-01-01'),
      disposalDate: new Date('2024-06-01'),
      quantity: 1,
      costBasis: 50000,
      proceeds: 60000,
      gainLoss: 10000,
      holdingPeriod: 152,
      type: 'short-term',
      isLongTerm: false,
    };

    // Both aliases should work
    expect(txn.buyDate).toEqual(txn.acquisitionDate);
    expect(txn.sellDate).toEqual(txn.disposalDate);
  });

  it('should allow string or object for transaction source', () => {
    const txn1: Transaction = {
      id: 'txn-001',
      timestamp: new Date(),
      type: 'BUY',
      asset: 'BTC',
      quantity: 1,
      price: 50000,
      source: 'coinbase',
    };

    const txn2: Transaction = {
      id: 'txn-002',
      timestamp: new Date(),
      type: 'BUY',
      asset: 'ETH',
      quantity: 10,
      price: 2500,
      source: { type: 'exchange', name: 'binance' },
    };

    expect(typeof txn1.source).toBe('string');
    expect(typeof txn2.source).toBe('object');
  });
});

describe('Edge Cases', () => {
  it('should handle zero quantities', () => {
    const lot: Lot = {
      id: 'lot-empty',
      asset: 'BTC',
      quantity: 0,
      purchasePrice: 50000,
      purchaseDate: new Date(),
      acquisitionDate: new Date(),
      transactionId: 'txn-001',
      isOpen: false,
      remainingQuantity: 0,
      costBasis: 0,
    };

    expect(lot.quantity).toBe(0);
    expect(lot.costBasis).toBe(0);
  });

  it('should handle negative gain/loss', () => {
    const taxTxn: TaxTransaction = {
      id: 'tax-loss',
      asset: 'SOL',
      buyDate: new Date('2024-01-01'),
      sellDate: new Date('2024-02-01'),
      acquisitionDate: new Date('2024-01-01'),
      disposalDate: new Date('2024-02-01'),
      quantity: 100,
      costBasis: 10000,
      proceeds: 7500,
      gainLoss: -2500,
      holdingPeriod: 31,
      type: 'short-term',
      isLongTerm: false,
    };

    expect(taxTxn.gainLoss).toBeLessThan(0);
  });

  it('should handle very large numbers', () => {
    const position: Position = {
      id: 'pos-whale',
      asset: 'BTC',
      quantity: 1000000,
      averageCost: 50000,
      currentValue: 60000000000,
      unrealizedGainLoss: 10000000000,
      lots: [],
      lastUpdated: new Date(),
      totalCost: 50000000000,
      averageCostBasis: 50000,
    };

    expect(position.currentValue).toBe(60000000000);
  });
});
