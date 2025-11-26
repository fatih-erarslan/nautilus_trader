/**
 * MCP Tools Usage Example
 *
 * Demonstrates:
 * - Using MCP tools for syndicate management
 * - All 10 syndicate MCP tools
 * - Integration with Neural Trader MCP server
 * - Automated workflows
 */

// Note: This example assumes the Neural Trader MCP server is running
// Start with: npx neural-trader mcp start

async function mcpToolsUsageExample() {
  console.log('=== MCP Tools Usage Example ===\n');
  console.log('This example demonstrates using MCP tools for syndicate management.\n');

  // Example 1: Create Syndicate (create_syndicate_tool)
  console.log('Example 1: Create Syndicate via MCP');
  console.log('===================================\n');

  console.log('MCP Tool: create_syndicate_tool');
  console.log('Parameters:');
  const createParams = {
    syndicate_id: 'mcp-demo-syndicate',
    name: 'MCP Demo Syndicate',
    initial_capital: 100000,
    config: {
      max_single_bet: 0.05,
      max_daily_exposure: 0.20,
      min_reserve: 0.10
    }
  };
  console.log(JSON.stringify(createParams, null, 2));
  console.log('\nExpected Response:');
  console.log('{');
  console.log('  "success": true,');
  console.log('  "syndicate_id": "mcp-demo-syndicate",');
  console.log('  "total_capital": 100000');
  console.log('}\n');

  // Example 2: Add Members (add_syndicate_member)
  console.log('\nExample 2: Add Syndicate Member via MCP');
  console.log('=======================================\n');

  console.log('MCP Tool: add_syndicate_member');
  console.log('Parameters:');
  const addMemberParams = {
    syndicate_id: 'mcp-demo-syndicate',
    name: 'Alice Johnson',
    email: 'alice@example.com',
    role: 'lead_investor',
    initial_contribution: 40000
  };
  console.log(JSON.stringify(addMemberParams, null, 2));
  console.log('\nExpected Response:');
  console.log('{');
  console.log('  "success": true,');
  console.log('  "member_id": "alice-001",');
  console.log('  "tier": "platinum",');
  console.log('  "voting_weight": 0.40');
  console.log('}\n');

  // Example 3: Get Syndicate Status (get_syndicate_status_tool)
  console.log('\nExample 3: Get Syndicate Status via MCP');
  console.log('=======================================\n');

  console.log('MCP Tool: get_syndicate_status_tool');
  console.log('Parameters:');
  const statusParams = {
    syndicate_id: 'mcp-demo-syndicate'
  };
  console.log(JSON.stringify(statusParams, null, 2));
  console.log('\nExpected Response:');
  console.log('{');
  console.log('  "syndicate_id": "mcp-demo-syndicate",');
  console.log('  "total_capital": 100000,');
  console.log('  "available_capital": 95000,');
  console.log('  "total_invested": 5000,');
  console.log('  "member_count": 4,');
  console.log('  "roi": 12.5,');
  console.log('  "win_rate": 58.3,');
  console.log('  "sharpe_ratio": 1.85');
  console.log('}\n');

  // Example 4: Allocate Funds (allocate_syndicate_funds)
  console.log('\nExample 4: Allocate Funds via MCP');
  console.log('=================================\n');

  console.log('MCP Tool: allocate_syndicate_funds');
  console.log('Parameters:');
  const allocateParams = {
    syndicate_id: 'mcp-demo-syndicate',
    opportunities: [
      {
        sport: 'NFL',
        event: 'Chiefs vs Bills',
        odds: 2.10,
        probability: 0.55,
        confidence: 0.85
      }
    ],
    strategy: 'kelly_criterion',
    fractional_kelly: 0.25
  };
  console.log(JSON.stringify(allocateParams, null, 2));
  console.log('\nExpected Response:');
  console.log('{');
  console.log('  "success": true,');
  console.log('  "allocations": [');
  console.log('    {');
  console.log('      "opportunity": "Chiefs vs Bills",');
  console.log('      "amount": 3525,');
  console.log('      "kelly_percentage": 14.1,');
  console.log('      "expected_value": 546.38');
  console.log('    }');
  console.log('  ],');
  console.log('  "total_allocated": 3525');
  console.log('}\n');

  // Example 5: Distribute Profits (distribute_syndicate_profits)
  console.log('\nExample 5: Distribute Profits via MCP');
  console.log('=====================================\n');

  console.log('MCP Tool: distribute_syndicate_profits');
  console.log('Parameters:');
  const distributeParams = {
    syndicate_id: 'mcp-demo-syndicate',
    total_profit: 10000,
    model: 'proportional'
  };
  console.log(JSON.stringify(distributeParams, null, 2));
  console.log('\nExpected Response:');
  console.log('{');
  console.log('  "success": true,');
  console.log('  "distributions": [');
  console.log('    { "member": "Alice", "amount": 4000, "percentage": 40 },');
  console.log('    { "member": "Bob", "amount": 3000, "percentage": 30 },');
  console.log('    { "member": "Carol", "amount": 2000, "percentage": 20 },');
  console.log('    { "member": "David", "amount": 1000, "percentage": 10 }');
  console.log('  ],');
  console.log('  "total_distributed": 10000');
  console.log('}\n');

  // Example 6: Create Vote (create_syndicate_vote)
  console.log('\nExample 6: Create Syndicate Vote via MCP');
  console.log('========================================\n');

  console.log('MCP Tool: create_syndicate_vote');
  console.log('Parameters:');
  const voteParams = {
    syndicate_id: 'mcp-demo-syndicate',
    vote_type: 'strategy',
    proposal: 'Increase fractional Kelly to 0.30',
    options: ['approve', 'reject', 'defer'],
    duration_hours: 48
  };
  console.log(JSON.stringify(voteParams, null, 2));
  console.log('\nExpected Response:');
  console.log('{');
  console.log('  "success": true,');
  console.log('  "vote_id": "vote-001",');
  console.log('  "expires_at": "2024-01-15T12:00:00Z",');
  console.log('  "quorum_required": 0.60');
  console.log('}\n');

  // Example 7: Cast Vote (cast_syndicate_vote)
  console.log('\nExample 7: Cast Vote via MCP');
  console.log('============================\n');

  console.log('MCP Tool: cast_syndicate_vote');
  console.log('Parameters:');
  const castVoteParams = {
    syndicate_id: 'mcp-demo-syndicate',
    vote_id: 'vote-001',
    member_id: 'alice-001',
    option: 'approve'
  };
  console.log(JSON.stringify(castVoteParams, null, 2));
  console.log('\nExpected Response:');
  console.log('{');
  console.log('  "success": true,');
  console.log('  "vote_recorded": true,');
  console.log('  "current_participation": 0.75,');
  console.log('  "quorum_met": true,');
  console.log('  "current_results": {');
  console.log('    "approve": 0.70,');
  console.log('    "reject": 0.20,');
  console.log('    "defer": 0.10');
  console.log('  }');
  console.log('}\n');

  // Example 8: Process Withdrawal (process_syndicate_withdrawal)
  console.log('\nExample 8: Process Withdrawal via MCP');
  console.log('=====================================\n');

  console.log('MCP Tool: process_syndicate_withdrawal');
  console.log('Parameters:');
  const withdrawParams = {
    syndicate_id: 'mcp-demo-syndicate',
    member_id: 'david-001',
    amount: 5000,
    reason: 'Personal expense',
    is_emergency: false
  };
  console.log(JSON.stringify(withdrawParams, null, 2));
  console.log('\nExpected Response:');
  console.log('{');
  console.log('  "success": true,');
  console.log('  "withdrawal_id": "withdrawal-001",');
  console.log('  "status": "pending",');
  console.log('  "estimated_completion": "2024-01-16T12:00:00Z",');
  console.log('  "requires_approval": false');
  console.log('}\n');

  // Example 9: Get Member Performance (get_syndicate_member_performance)
  console.log('\nExample 9: Get Member Performance via MCP');
  console.log('=========================================\n');

  console.log('MCP Tool: get_syndicate_member_performance');
  console.log('Parameters:');
  const perfParams = {
    syndicate_id: 'mcp-demo-syndicate',
    member_id: 'alice-001'
  };
  console.log(JSON.stringify(perfParams, null, 2));
  console.log('\nExpected Response:');
  console.log('{');
  console.log('  "member_id": "alice-001",');
  console.log('  "member_name": "Alice Johnson",');
  console.log('  "total_contribution": 40000,');
  console.log('  "current_equity": 45200,');
  console.log('  "total_profit": 5200,');
  console.log('  "roi": 13.0,');
  console.log('  "win_rate": 62.5,');
  console.log('  "sharpe_ratio": 2.1,');
  console.log('  "tier": "platinum"');
  console.log('}\n');

  // Example 10: Update Allocation Strategy (update_syndicate_allocation_strategy)
  console.log('\nExample 10: Update Allocation Strategy via MCP');
  console.log('==============================================\n');

  console.log('MCP Tool: update_syndicate_allocation_strategy');
  console.log('Parameters:');
  const updateStrategyParams = {
    syndicate_id: 'mcp-demo-syndicate',
    strategy_config: {
      default_strategy: 'kelly_criterion',
      fractional_kelly: 0.30,
      max_kelly_override: 0.15,
      risk_adjustment: true,
      correlation_adjustment: true
    }
  };
  console.log(JSON.stringify(updateStrategyParams, null, 2));
  console.log('\nExpected Response:');
  console.log('{');
  console.log('  "success": true,');
  console.log('  "strategy_updated": true,');
  console.log('  "effective_date": "2024-01-13T12:00:00Z",');
  console.log('  "requires_vote": true,');
  console.log('  "vote_id": "vote-002"');
  console.log('}\n');

  // Example 11: Integration Example - Full Workflow
  console.log('\nExample 11: Full Workflow Integration');
  console.log('====================================\n');

  console.log('Step 1: Create syndicate');
  console.log('  → create_syndicate_tool');
  console.log('\nStep 2: Add 4 members');
  console.log('  → add_syndicate_member (x4)');
  console.log('\nStep 3: Check initial status');
  console.log('  → get_syndicate_status_tool');
  console.log('\nStep 4: Make first allocation');
  console.log('  → allocate_syndicate_funds');
  console.log('\nStep 5: Simulate win and distribute');
  console.log('  → distribute_syndicate_profits');
  console.log('\nStep 6: Check member performance');
  console.log('  → get_syndicate_member_performance (x4)');
  console.log('\nStep 7: Member requests withdrawal');
  console.log('  → process_syndicate_withdrawal');
  console.log('\nStep 8: Create strategy vote');
  console.log('  → create_syndicate_vote');
  console.log('\nStep 9: All members vote');
  console.log('  → cast_syndicate_vote (x4)');
  console.log('\nStep 10: Update strategy if approved');
  console.log('  → update_syndicate_allocation_strategy\n');

  // Example 12: Error Handling
  console.log('\nExample 12: Error Handling');
  console.log('=========================\n');

  console.log('Common Error Responses:\n');

  console.log('1. Syndicate not found:');
  console.log('{');
  console.log('  "success": false,');
  console.log('  "error": "Syndicate not found",');
  console.log('  "code": "SYNDICATE_NOT_FOUND"');
  console.log('}\n');

  console.log('2. Insufficient capital:');
  console.log('{');
  console.log('  "success": false,');
  console.log('  "error": "Insufficient capital for allocation",');
  console.log('  "code": "INSUFFICIENT_CAPITAL",');
  console.log('  "available": 5000,');
  console.log('  "requested": 10000');
  console.log('}\n');

  console.log('3. Permission denied:');
  console.log('{');
  console.log('  "success": false,');
  console.log('  "error": "Member lacks permission for this action",');
  console.log('  "code": "PERMISSION_DENIED",');
  console.log('  "required_permission": "allocate_funds"');
  console.log('}\n');

  // Summary
  console.log('\n=== MCP Tools Summary ===');
  console.log('Total MCP Tools: 10');
  console.log('\nCategories:');
  console.log('  - Management: create, status, update');
  console.log('  - Members: add, performance');
  console.log('  - Finance: allocate, distribute, withdraw');
  console.log('  - Governance: vote create, vote cast');
  console.log('\nIntegration:');
  console.log('  - Works with Neural Trader MCP server');
  console.log('  - Can be called from Claude Desktop');
  console.log('  - Supports automation workflows');
  console.log('  - Returns structured JSON responses');

  console.log('\n=== Example Complete ===');
  console.log('\nTo use MCP tools:');
  console.log('1. Start Neural Trader MCP server:');
  console.log('   npx neural-trader mcp start');
  console.log('\n2. Configure in Claude Desktop MCP settings');
  console.log('\n3. Call tools via MCP protocol');
}

// Run example
if (require.main === module) {
  mcpToolsUsageExample()
    .then(() => {
      console.log('\n✓ MCP tools example completed');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n✗ Example failed:', error);
      process.exit(1);
    });
}

module.exports = { mcpToolsUsageExample };
