#!/usr/bin/env node
/**
 * Validate all generated JSON Schema files
 */

const fs = require('fs');
const path = require('path');

const toolsDir = path.join(__dirname, '../tools');
const files = fs.readdirSync(toolsDir).filter(f => f.endsWith('.json'));

console.log('\n📋 Schema Validation Report\n');
console.log('='.repeat(60));

let validCount = 0;
let invalidCount = 0;
const errors = [];

files.forEach(file => {
  try {
    const content = fs.readFileSync(path.join(toolsDir, file), 'utf8');
    const schema = JSON.parse(content);

    // Validate required fields
    const required = ['$schema', '$id', 'title', 'description', 'category', 'properties', 'metadata'];
    const missing = required.filter(field => !schema[field]);

    if (missing.length > 0) {
      errors.push({ file, error: `Missing fields: ${missing.join(', ')}` });
      invalidCount++;
    } else {
      // Validate input/output schemas exist
      if (!schema.properties.input_schema || !schema.properties.output_schema) {
        errors.push({ file, error: 'Missing input_schema or output_schema' });
        invalidCount++;
      } else {
        validCount++;
      }
    }
  } catch (error) {
    errors.push({ file, error: error.message });
    invalidCount++;
  }
});

console.log(`Total schemas: ${files.length}`);
console.log(`✅ Valid: ${validCount}`);
console.log(`❌ Invalid: ${invalidCount}`);

if (errors.length > 0) {
  console.log('\n⚠️  Errors:');
  errors.forEach(({ file, error }) => {
    console.log(`   - ${file}: ${error}`);
  });
}

// Group by category
console.log('\n📂 Schemas by Category:\n');
const byCategory = {};
files.forEach(file => {
  try {
    const content = fs.readFileSync(path.join(toolsDir, file), 'utf8');
    const schema = JSON.parse(content);
    const cat = schema.category || 'uncategorized';
    byCategory[cat] = (byCategory[cat] || 0) + 1;
  } catch (e) {
    // Skip invalid files
  }
});

Object.entries(byCategory)
  .sort((a, b) => b[1] - a[1])
  .forEach(([cat, count]) => {
    console.log(`   ${cat.padEnd(20)}: ${count} tools`);
  });

// Show metadata distribution
console.log('\n💰 Cost Distribution:\n');
const byCost = {};
files.forEach(file => {
  try {
    const content = fs.readFileSync(path.join(toolsDir, file), 'utf8');
    const schema = JSON.parse(content);
    const cost = schema.metadata?.cost || 'unknown';
    byCost[cost] = (byCost[cost] || 0) + 1;
  } catch (e) {}
});

Object.entries(byCost)
  .sort((a, b) => b[1] - a[1])
  .forEach(([cost, count]) => {
    console.log(`   ${cost.padEnd(15)}: ${count} tools`);
  });

// GPU capable tools
console.log('\n🚀 GPU Capability:\n');
let gpuCapable = 0;
let nonGpu = 0;
files.forEach(file => {
  try {
    const content = fs.readFileSync(path.join(toolsDir, file), 'utf8');
    const schema = JSON.parse(content);
    if (schema.metadata?.gpu_capable) {
      gpuCapable++;
    } else {
      nonGpu++;
    }
  } catch (e) {}
});

console.log(`   GPU-capable    : ${gpuCapable} tools`);
console.log(`   Non-GPU        : ${nonGpu} tools`);

console.log('\n' + '='.repeat(60));
console.log();

process.exit(invalidCount > 0 ? 1 : 0);
