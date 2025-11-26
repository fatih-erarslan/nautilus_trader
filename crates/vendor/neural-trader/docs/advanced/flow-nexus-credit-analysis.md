# Flow Nexus Credit Consumption Analysis

## Test Results Summary

### Starting Balance: 2,887 credits
### Current Balance: 2,886 credits
### Total Consumed: 1 credit

## Operations Tested

### Sandbox Operations
| Operation | Template | Resources | Credits Charged |
|-----------|----------|-----------|-----------------|
| Create Python Sandbox | python | 1024MB RAM, 2hr timeout | 1 credit |
| Create React Sandbox | react | 2048MB RAM, 2 CPUs | 0 credits |
| Create Node.js Sandbox | node | Default, with env vars | 0 credits |
| Execute Code (Failed) | python | - | 0 credits |
| Execute Code (Success) | node | - | 0 credits |

### Key Findings

1. **Initial Sandbox Creation**: Only the first sandbox creation was charged (1 credit)
2. **Subsequent Sandboxes**: No charges for additional sandbox creations
3. **Code Execution**: Free regardless of success/failure
4. **Environment Variables**: No additional charge for env var configuration

## Credit Pricing Model Observations

### Confirmed Free Operations
- Code execution in sandboxes
- Sandbox status checks
- Listing sandboxes
- Environment variable configuration

### Potential Pricing Structure
- First sandbox per session: 1 credit
- Subsequent sandboxes: Free (possibly rate-limited)
- Extended runtime sandboxes: May incur additional charges
- High-resource sandboxes (CPU/RAM): May have different pricing

## Comparison with Previous Test User

### testuser1757267578@flow-test.com
- Started with: 256 credits
- Ended with: 165 credits
- Operations: Swarms, sandboxes, workflows
- Consumption: ~91 credits

### ruv@ruv.net (current)
- Started with: 2,887 credits
- Current: 2,886 credits
- Operations: 3 sandboxes, 2 code executions
- Consumption: 1 credit

## Recommendations

1. **Sandbox Strategy**: Create one sandbox and reuse it for multiple executions
2. **Resource Optimization**: Default resources appear sufficient for most tasks
3. **Batch Operations**: Group related work in single sandbox sessions
4. **Credit Monitoring**: Check balance periodically during intensive operations

## Notes

- Sandboxes appear to auto-terminate after timeout
- No active sandboxes shown in list (despite recent creation)
- Credit deduction may be delayed or batched
- Different account tiers may have different pricing

---
*Analysis Date: 2025-09-07*
*Account: ruv@ruv.net*