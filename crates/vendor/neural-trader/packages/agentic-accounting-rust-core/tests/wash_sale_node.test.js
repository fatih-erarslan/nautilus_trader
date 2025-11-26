/**
 * Node.js integration test for wash sale detection
 *
 * Tests the NAPI bindings to ensure Rust wash sale functions
 * are properly exported and working from JavaScript/TypeScript
 */

const {
  detectWashSale,
  applyWashSaleAdjustment,
  detectWashSalesBatch,
  isWashSaleReplacement,
  calculateWashSaleHoldingPeriod,
} = require('../index.js');

// Helper to create test data
function createDisposal(asset, gainLoss, disposalDate, acquisitionDate) {
  return {
    id: 'disposal1',
    saleTransactionId: 'sale1',
    lotId: 'lot1',
    asset,
    quantity: '100',
    proceeds: '9000',
    costBasis: '10000',
    gainLoss: gainLoss,
    acquisitionDate: acquisitionDate,
    disposalDate: disposalDate,
    isLongTerm: false,
  };
}

function createTransaction(id, type, asset, timestamp, quantity = '100', price = '100') {
  return {
    id,
    transactionType: type,
    asset,
    quantity,
    price,
    timestamp,
    source: 'test',
    fees: '0',
  };
}

function createTaxLot(id, asset, costBasis, acquisitionDate) {
  return {
    id,
    transactionId: `tx_${id}`,
    asset,
    quantity: '100',
    remainingQuantity: '100',
    costBasis: costBasis,
    acquisitionDate: acquisitionDate,
  };
}

console.log('🧪 Running Node.js wash sale tests...\n');

try {
  // Test 1: Basic wash sale detection (loss with replacement within 30 days)
  console.log('Test 1: Basic wash sale detection');
  const disposal1 = createDisposal('BTC', '-1000', '2024-01-15T00:00:00Z', '2023-12-01T00:00:00Z');
  const transactions1 = [
    createTransaction('tx1', 'BUY', 'BTC', '2024-01-25T00:00:00Z'),
  ];

  const result1 = detectWashSale(disposal1, transactions1);
  console.log('  Result:', {
    isWashSale: result1.isWashSale,
    disallowedLoss: result1.disallowedLoss,
    hasReplacement: !!result1.replacementTransactionId,
  });

  if (!result1.isWashSale) throw new Error('Should detect wash sale');
  if (result1.disallowedLoss !== '1000') throw new Error('Wrong disallowed loss');
  console.log('  ✅ PASSED\n');

  // Test 2: Gains are exempt from wash sale
  console.log('Test 2: Gains exempt from wash sale');
  const disposal2 = createDisposal('ETH', '1000', '2024-01-15T00:00:00Z', '2023-12-01T00:00:00Z');
  const transactions2 = [
    createTransaction('tx2', 'BUY', 'ETH', '2024-01-20T00:00:00Z'),
  ];

  const result2 = detectWashSale(disposal2, transactions2);
  console.log('  Result:', {
    isWashSale: result2.isWashSale,
    disallowedLoss: result2.disallowedLoss,
  });

  if (result2.isWashSale) throw new Error('Gains should not trigger wash sale');
  console.log('  ✅ PASSED\n');

  // Test 3: Outside wash window (no wash sale)
  console.log('Test 3: Outside 30-day window');
  const disposal3 = createDisposal('AAPL', '-500', '2024-01-15T00:00:00Z', '2023-12-01T00:00:00Z');
  const transactions3 = [
    createTransaction('tx3', 'BUY', 'AAPL', '2024-03-01T00:00:00Z'), // 45+ days later
  ];

  const result3 = detectWashSale(disposal3, transactions3);
  console.log('  Result:', {
    isWashSale: result3.isWashSale,
  });

  if (result3.isWashSale) throw new Error('Should not detect wash sale outside window');
  console.log('  ✅ PASSED\n');

  // Test 4: Cost basis adjustment
  console.log('Test 4: Cost basis adjustment');
  const disposal4 = createDisposal('MSFT', '-1500', '2024-01-15T00:00:00Z', '2023-11-01T00:00:00Z');
  const lot4 = createTaxLot('lot2', 'MSFT', '10000', '2024-01-20T00:00:00Z');

  const result4 = applyWashSaleAdjustment(disposal4, lot4, '1500');
  console.log('  Result:', {
    adjustedGainLoss: result4.adjustedDisposal.gainLoss,
    newCostBasis: result4.adjustedLot.costBasis,
    adjustmentAmount: result4.adjustmentAmount,
  });

  if (result4.adjustedDisposal.gainLoss !== '0') throw new Error('Loss should be disallowed');
  if (result4.adjustedLot.costBasis !== '11500') throw new Error('Cost basis should increase');
  console.log('  ✅ PASSED\n');

  // Test 5: Batch detection
  console.log('Test 5: Batch wash sale detection');
  const disposals5 = [
    createDisposal('BTC', '-1000', '2024-01-15T00:00:00Z', '2023-12-01T00:00:00Z'),
    createDisposal('ETH', '500', '2024-01-16T00:00:00Z', '2023-11-01T00:00:00Z'),
    createDisposal('AAPL', '-750', '2024-01-17T00:00:00Z', '2023-10-01T00:00:00Z'),
  ];

  const transactions5 = [
    createTransaction('tx1', 'BUY', 'BTC', '2024-01-25T00:00:00Z'),
    createTransaction('tx2', 'BUY', 'ETH', '2024-01-26T00:00:00Z'),
    createTransaction('tx3', 'BUY', 'AAPL', '2024-01-27T00:00:00Z'),
  ];

  const result5 = detectWashSalesBatch(disposals5, transactions5);
  console.log('  Results:', result5.map((r, i) => ({
    disposal: i + 1,
    isWashSale: r.isWashSale,
  })));

  if (!result5[0].isWashSale) throw new Error('BTC loss should trigger wash sale');
  if (result5[1].isWashSale) throw new Error('ETH gain should not trigger wash sale');
  if (!result5[2].isWashSale) throw new Error('AAPL loss should trigger wash sale');
  console.log('  ✅ PASSED\n');

  // Test 6: Helper functions
  console.log('Test 6: Helper functions');
  const isReplacement = isWashSaleReplacement(
    '2024-01-15T00:00:00Z',
    '2024-01-25T00:00:00Z'
  );
  console.log('  is_wash_sale_replacement (10 days):', isReplacement);
  if (!isReplacement) throw new Error('Should be replacement');

  const holdingPeriod = calculateWashSaleHoldingPeriod(
    '2024-01-01T00:00:00Z',
    '2024-02-01T00:00:00Z',
    '2024-02-05T00:00:00Z',
    '2024-03-01T00:00:00Z'
  );
  console.log('  Adjusted holding period (days):', holdingPeriod);
  if (holdingPeriod !== 56) throw new Error('Wrong holding period calculation');
  console.log('  ✅ PASSED\n');

  console.log('🎉 All tests passed!');
  console.log('\n✅ Wash sale detection successfully implemented and working!');
  process.exit(0);

} catch (error) {
  console.error('\n❌ Test failed:', error.message);
  console.error(error.stack);
  process.exit(1);
}
