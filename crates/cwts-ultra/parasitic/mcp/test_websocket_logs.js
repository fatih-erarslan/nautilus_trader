const WebSocket = require('ws');

console.log('🔌 Connecting to CWTS WebSocket server...');
const ws = new WebSocket('ws://localhost:8501');

let messageCount = 0;
let lastStats = Date.now();

ws.on('open', () => {
  console.log('✅ Connected to CWTS WebSocket');
  console.log('📊 Listening for real-time market data...\n');
});

ws.on('message', (data) => {
  messageCount++;
  const message = JSON.parse(data);
  
  // Log every 10th message to avoid flooding
  if (messageCount % 10 === 0) {
    console.log(`📈 Message #${messageCount}:`, {
      type: message.type,
      symbol: message.symbol,
      price: message.price,
      timestamp: new Date(message.timestamp).toISOString()
    });
  }
  
  // Show stats every 5 seconds
  const now = Date.now();
  if (now - lastStats > 5000) {
    console.log(`\n📊 Stats: ${messageCount} messages received (${Math.round(messageCount / ((now - lastStats) / 1000))} msg/sec)\n`);
    lastStats = now;
    messageCount = 0;
  }
});

ws.on('error', (error) => {
  console.error('❌ WebSocket error:', error.message);
});

ws.on('close', () => {
  console.log('\n🔌 WebSocket connection closed');
});

// Keep running for 30 seconds then exit
setTimeout(() => {
  console.log('\n✅ Test complete - closing connection');
  ws.close();
  process.exit(0);
}, 30000);
