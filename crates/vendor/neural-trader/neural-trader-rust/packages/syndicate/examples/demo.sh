#!/bin/bash
# Demo script for Syndicate CLI

set -e

SYNDICATE_BIN="../bin/syndicate.js"

echo "========================================="
echo "  Syndicate CLI Demo"
echo "========================================="
echo ""

# 1. Create Syndicate
echo "1. Creating syndicate..."
node $SYNDICATE_BIN create demo-syndicate --bankroll 100000 --rules examples/rules.json
echo ""

# 2. Add Members
echo "2. Adding members..."
node $SYNDICATE_BIN member add "Alice Johnson" alice@example.com senior-trader --capital 30000
node $SYNDICATE_BIN member add "Bob Smith" bob@example.com analyst --capital 25000
node $SYNDICATE_BIN member add "Carol Davis" carol@example.com risk-manager --capital 20000
echo ""

# 3. List Members
echo "3. Listing members..."
node $SYNDICATE_BIN member list
echo ""

# 4. Allocate Funds
echo "4. Allocating funds using Kelly Criterion..."
node $SYNDICATE_BIN allocate examples/opportunity.json --strategy kelly
echo ""

# 5. Preview Distribution
echo "5. Previewing profit distribution..."
node $SYNDICATE_BIN distribute preview 5000 --model hybrid
echo ""

# 6. Distribute Profits
echo "6. Distributing profits..."
node $SYNDICATE_BIN distribute 5000 --model hybrid
echo ""

# 7. Create Vote
echo "7. Creating governance vote..."
node $SYNDICATE_BIN vote create "Should we increase the risk allocation to 30%?" --options "Yes,No,Abstain"
echo ""

# 8. Show Statistics
echo "8. Showing syndicate statistics..."
node $SYNDICATE_BIN stats --syndicate demo-syndicate
echo ""

echo "========================================="
echo "  Demo Complete!"
echo "========================================="
