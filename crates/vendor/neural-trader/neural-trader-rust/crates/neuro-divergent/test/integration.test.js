const assert = require('assert');

try {
  const binding = require('../index.js');

  console.log('Testing Neuro-Divergent NAPI bindings...\n');

  // Test basic functions
  console.log('1. Testing add function...');
  const sum = binding.add(2, 3);
  assert.strictEqual(sum, 5, 'add(2, 3) should equal 5');
  console.log('   ✓ add(2, 3) =', sum);

  console.log('\n2. Testing version function...');
  const ver = binding.version();
  assert.ok(ver, 'version should return a string');
  console.log('   ✓ version =', ver);

  console.log('\n3. Testing platformInfo function...');
  const info = binding.platformInfo();
  const parsed = JSON.parse(info);
  assert.ok(parsed.platform, 'platform info should include platform');
  assert.ok(parsed.arch, 'platform info should include arch');
  console.log('   ✓ platform =', parsed.platform);
  console.log('   ✓ arch =', parsed.arch);
  console.log('   ✓ version =', parsed.version);

  console.log('\n4. Testing NeuroDivergent class...');
  const strategy = new binding.NeuroDivergent('momentum');
  assert.ok(strategy, 'NeuroDivergent should be instantiable');
  console.log('   ✓ Created NeuroDivergent instance');

  const name = strategy.getStrategyName();
  assert.strictEqual(name, 'momentum', 'strategy name should match');
  console.log('   ✓ Strategy name:', name);

  const params = JSON.stringify({ period: 14, threshold: 0.02 });
  strategy.setParameters(params);
  console.log('   ✓ Set parameters');

  const retrieved = strategy.getParameters();
  assert.strictEqual(retrieved, params, 'retrieved parameters should match');
  console.log('   ✓ Retrieved parameters:', retrieved);

  const marketData = JSON.stringify({
    price: 100,
    volume: 1000,
    timestamp: Date.now()
  });
  const analysis = strategy.analyze(marketData);
  const result = JSON.parse(analysis);
  assert.ok(result.strategy, 'analysis should include strategy');
  assert.ok(result.signal, 'analysis should include signal');
  console.log('   ✓ Analysis result:', result);

  console.log('\n✅ All tests passed!');
  process.exit(0);

} catch (error) {
  console.error('\n❌ Test failed:', error.message);
  console.error(error.stack);
  process.exit(1);
}
