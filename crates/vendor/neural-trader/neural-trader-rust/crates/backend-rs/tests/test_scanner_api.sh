#!/bin/bash

# Test Scanner API Endpoints
# This script tests all scanner endpoints to ensure they work correctly

set -e

API_BASE="http://localhost:8080"
SCAN_ID=""

echo "============================================"
echo "Testing BeClever Scanner API Endpoints"
echo "============================================"
echo ""

# Test 1: Health Check
echo "1. Testing Health Check..."
curl -s "$API_BASE/health" | jq '.'
echo ""

# Test 2: Start New Scan
echo "2. Starting New API Scan..."
SCAN_RESPONSE=$(curl -s -X POST "$API_BASE/api/scanner/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://api.example.com/openapi.json",
    "scan_type": "openapi",
    "options": {
      "deep_scan": true,
      "check_auth": true
    }
  }')
echo "$SCAN_RESPONSE" | jq '.'
SCAN_ID=$(echo "$SCAN_RESPONSE" | jq -r '.scan_id')
echo "Scan ID: $SCAN_ID"
echo ""

# Test 3: List All Scans
echo "3. Listing All Scans (page 1, limit 10)..."
curl -s "$API_BASE/api/scanner/scans?page=1&limit=10" | jq '.'
echo ""

# Test 4: List Scans with Status Filter
echo "4. Listing Queued Scans..."
curl -s "$API_BASE/api/scanner/scans?status=queued" | jq '.'
echo ""

# Test 5: Get Scan Details
echo "5. Getting Scan Details for ID: $SCAN_ID..."
curl -s "$API_BASE/api/scanner/scans/$SCAN_ID" | jq '.'
echo ""

# Test 6: Get AI-Generated Report
echo "6. Getting AI-Generated Report..."
curl -s "$API_BASE/api/scanner/scans/$SCAN_ID/report" | jq '.'
echo ""

# Test 7: Get Scanner Statistics
echo "7. Getting Scanner Statistics..."
curl -s "$API_BASE/api/scanner/stats" | jq '.'
echo ""

# Test 8: Create Multiple Scans
echo "8. Creating Multiple Scans for Testing..."
for i in {1..3}; do
  curl -s -X POST "$API_BASE/api/scanner/scan" \
    -H "Content-Type: application/json" \
    -d "{
      \"url\": \"https://api.test$i.com/spec.json\",
      \"scan_type\": \"auto\",
      \"options\": {}
    }" | jq -r '.scan_id'
  echo "Created scan $i"
done
echo ""

# Test 9: List Updated Scans
echo "9. Listing All Scans After Creating Multiple..."
curl -s "$API_BASE/api/scanner/scans?limit=20" | jq '.scans | length'
echo " scans found"
echo ""

# Test 10: Delete Scan
echo "10. Deleting Scan ID: $SCAN_ID..."
curl -s -X DELETE "$API_BASE/api/scanner/scans/$SCAN_ID" | jq '.'
echo ""

# Test 11: Verify Deletion
echo "11. Verifying Scan Was Deleted..."
curl -s "$API_BASE/api/scanner/scans/$SCAN_ID" | jq '.'
echo ""

# Test 12: Final Statistics
echo "12. Final Scanner Statistics..."
curl -s "$API_BASE/api/scanner/stats" | jq '.'
echo ""

echo "============================================"
echo "All Scanner API Tests Completed!"
echo "============================================"
