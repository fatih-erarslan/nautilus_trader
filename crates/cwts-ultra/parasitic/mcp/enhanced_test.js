const WebSocket = require('ws');

console.log('🚀 CWTS Real-Time Market Data Monitor');
console.log('=====================================\n');

const ws = new WebSocket('ws://localhost:8501');

let messageCount = 0;
let symbolStats = {};
let startTime = Date.now();

ws.on('open', () => {
  console.log('✅ Connected to CWTS Quantum Trading System');
  console.log('📊 Processing live Binance market data...\n');
});

ws.on('message', (data) => {
  messageCount++;
  const msg = JSON.parse(data);
  
  // Track stats per symbol
  const symbol = msg.s || msg.symbol || 'UNKNOWN';
  if (!symbolStats[symbol]) {
    symbolStats[symbol] = { count: 0, lastPrice: 0, lastTime: 0 };
  }
  symbolStats[symbol].count++;
  symbolStats[symbol].lastPrice = msg.p || msg.price || msg.c || 0;
  symbolStats[symbol].lastTime = msg.E || msg.timestamp || Date.now();
  
  // Log detailed info for first few messages and then periodically
  if (messageCount <= 5 || messageCount % 50 === 0) {
    console.log(`📈 Message #${messageCount}:`);
    console.log(`   Type: ${msg.e || msg.type || 'market'}`);
    console.log(`   Symbol: ${symbol}`);
    console.log(`   Price: ${symbolStats[symbol].lastPrice}`);
    console.log(`   Time: ${new Date(symbolStats[symbol].lastTime).toISOString()}`);
    console.log(`   Raw: ${JSON.stringify(msg).substring(0, 100)}...`);
    console.log('');
  }
  
  // Show aggregated stats every 5 seconds
  if (messageCount % 100 === 0) {
    const elapsed = (Date.now() - startTime) / 1000;
    console.log('📊 === CWTS PERFORMANCE METRICS ===');
    console.log(`⚡ Total Messages: ${messageCount}`);
    console.log(`🚀 Rate: ${Math.round(messageCount / elapsed)} msg/sec`);
    console.log(`⏱️  Elapsed: ${elapsed.toFixed(1)}s`);
    console.log('\n📈 Symbol Activity:');
    Object.entries(symbolStats).forEach(([sym, stats]) => {
      console.log(`   ${sym}: ${stats.count} messages, Last: $${stats.lastPrice}`);
    });
    console.log('=====================================\n');
  }
});

ws.on('error', (error) => {
  console.error('❌ Error:', error.message);
});

ws.on('close', () => {
  console.log('\n🔌 Connection closed');
  console.log(`📊 Final: ${messageCount} messages processed`);
});

// Run for 20 seconds
setTimeout(() => {
  console.log('\n✅ Demo complete - CWTS Quantum Trading System operational');
  ws.close();
  process.exit(0);
}, 20000);
