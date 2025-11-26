# TDD Framework Quick Reference

## Test Categories & Commands

### Unit Tests
```bash
# Run all unit tests
pytest test_framework/unit -v

# Run specific component tests
pytest test_framework/unit/agents -v
pytest test_framework/unit/commands -v
pytest test_framework/unit/mcp_tools -v

# Run with coverage
pytest test_framework/unit --cov=swarm --cov-report=html
```

### Integration Tests
```bash
# Run integration tests
pytest test_framework/integration -v

# Run specific integration scenarios
pytest test_framework/integration/agent_coordination -v
pytest test_framework/integration/sdk_integration -v
pytest test_framework/integration/mcp_integration -v

# Run with timeout
pytest test_framework/integration --timeout=300
```

### Performance Tests
```bash
# Run performance benchmarks
pytest test_framework/performance --benchmark-only

# Run with profiling
pytest test_framework/performance --profile

# Generate report
pytest test_framework/performance --benchmark-json=report.json
```

### Resilience Tests
```bash
# Run chaos tests
pytest test_framework/resilience -v

# Run specific failure scenarios
pytest test_framework/resilience -k "agent_failure"
pytest test_framework/resilience -k "network_partition"
```

## Key Test Patterns

### Agent Testing Pattern
```python
# 1. Create fixture
@pytest.fixture
async def agent(config):
    agent = SwarmAgent(**config)
    await agent.initialize()
    yield agent
    await agent.shutdown()

# 2. Test state transitions
async def test_state_transition(agent):
    await agent.start_task()
    assert agent.state == AgentState.BUSY

# 3. Test task processing
async def test_task_processing(agent):
    result = await agent.process_task(task)
    assert result["status"] == "completed"
```

### Coordination Testing Pattern
```python
# 1. Create swarm
swarm = await create_test_swarm(agent_count=5)

# 2. Test consensus
consensus = await swarm.build_consensus(proposal)
assert consensus["agreement_ratio"] > 0.7

# 3. Test pipeline
result = await swarm.execute_pipeline(stages)
assert all(s["status"] == "success" for s in result)
```

### MCP Tool Testing Pattern
```python
# 1. Mock MCP client
@patch('mcp_client.invoke_tool')
async def test_mcp_tool(mock_invoke):
    mock_invoke.return_value = {"status": "success"}
    
# 2. Test tool chain
analysis = await invoke_tool("quick_analysis", params)
if analysis["recommendation"] == "buy":
    trade = await invoke_tool("execute_trade", params)
```

## Test Fixtures

### Agent Fixtures
- `create_mock_agent(type)` - Creates mock agent
- `create_test_swarm(configs)` - Creates test swarm
- `create_agent_pool(count)` - Creates agent pool

### Data Fixtures
- `generate_market_data(symbol, trend)` - Market data
- `generate_market_event(type)` - Market events
- `create_command_sequence(scenario)` - Commands

### Environment Fixtures
- `create_test_environment()` - Isolated env
- `inject_failure(type, target)` - Failure injection
- `create_network_partition()` - Network issues

## Assertion Helpers

### State Assertions
```python
# Eventually consistent
assert_eventually_consistent(agents, key, timeout=5)

# Message ordering
assert_message_ordering(message_log)

# Health status
assert_swarm_healthy(swarm, threshold=0.8)
```

### Performance Assertions
```python
# Throughput
assert throughput > 1000  # msgs/sec

# Latency
assert p95_latency < 100  # ms

# Scaling efficiency
assert efficiency > 0.7  # 70%
```

## Common Test Scenarios

### 1. Agent Lifecycle
- Initialize → Idle → Busy → Complete → Shutdown

### 2. Coordination Flows
- Task Distribution → Parallel Execution → Result Aggregation
- Signal → Consensus Building → Decision → Action

### 3. Failure Scenarios
- Agent Crash → Detection → Recovery → Rebalance
- Network Partition → Degraded Mode → Heal → Reconcile

### 4. Performance Scenarios
- Load Ramp → Measure → Scale → Stabilize
- Spike → Adapt → Recover → Normal

## Debugging Tips

### Capture Failure State
```python
# On test failure
debug_dir = await capture_failure_state(swarm, test_name)
# Check: debug_dir/agent_*.json, messages.json, metrics.json
```

### Enable Verbose Logging
```bash
pytest -v -s --log-cli-level=DEBUG
```

### Use Time Control
```python
# Control time in tests
with TimeControl() as tc:
    tc.advance(seconds=10)
    # Test timeout behavior
```

### Profile Performance
```bash
# CPU profiling
pytest --profile --profile-svg

# Memory profiling
pytest --memprof
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run TDD Suite
  run: |
    pytest test_framework/unit -v
    pytest test_framework/integration -v -m "not slow"
    pytest test_framework/performance --benchmark-compare
```

### Pre-commit Hooks
```yaml
- repo: local
  hooks:
    - id: swarm-tests
      name: Swarm Unit Tests
      entry: pytest test_framework/unit -x
      language: system
      pass_filenames: false
```

## Best Practices Checklist

- [ ] Tests are isolated and independent
- [ ] Each test has a single clear purpose
- [ ] Test names describe what they test
- [ ] Fixtures are reusable and composable
- [ ] Async tests use proper await/async
- [ ] Performance tests have baselines
- [ ] Failure tests verify recovery
- [ ] Integration tests mock external deps
- [ ] E2E tests cover critical paths
- [ ] All tests have timeout protection