const m = require('midstreamer');
console.log('Available exports:');
for (const k in m) {
  if (!k.startsWith('__')) {
    console.log(`  ${k}: ${typeof m[k]}`);
  }
}
