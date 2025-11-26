# Supabase Integration Setup Guide

Complete guide for setting up Supabase persistence and real-time capabilities for the Neural Trading Platform.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Supabase Project Setup](#supabase-project-setup)
3. [Database Schema Setup](#database-schema-setup)
4. [Row Level Security (RLS) Configuration](#row-level-security-rls-configuration)
5. [Edge Functions Deployment](#edge-functions-deployment)
6. [Real-time Configuration](#real-time-configuration)
7. [Environment Variables](#environment-variables)
8. [Client Integration](#client-integration)
9. [E2B Sandbox Integration](#e2b-sandbox-integration)
10. [Testing](#testing)
11. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

## Prerequisites

- Node.js 18+ installed
- Supabase CLI installed (`npm install -g supabase`)
- Git repository access
- E2B account (for sandbox integration)

## Supabase Project Setup

### 1. Create Supabase Project

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Click "New Project"
3. Choose organization and set project details:
   - **Name**: `neural-trader`
   - **Database Password**: Generate strong password
   - **Region**: Choose closest to your users
4. Wait for project creation (2-3 minutes)

### 2. Get Project Credentials

From your Supabase project dashboard:

- **Project URL**: `https://your-project-id.supabase.co`
- **Anon Key**: Used for client-side operations
- **Service Role Key**: Used for admin operations (keep secure!)

## Database Schema Setup

### 1. Run Database Migrations

Execute the SQL files in order:

```bash
# Connect to your Supabase project
supabase link --project-ref your-project-id

# Run schema setup
psql -h db.your-project-id.supabase.co -U postgres -d postgres -f src/supabase/database/schema.sql

# Set up RLS policies
psql -h db.your-project-id.supabase.co -U postgres -d postgres -f src/supabase/database/rls_policies.sql

# Add custom functions
psql -h db.your-project-id.supabase.co -U postgres -d postgres -f src/supabase/database/functions.sql

# Set up triggers
psql -h db.your-project-id.supabase.co -U postgres -d postgres -f src/supabase/real-time/triggers.sql
```

### 2. Verify Schema

Check that all tables were created:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY table_name;
```

Expected tables:
- `profiles`
- `symbols`
- `market_data`
- `news_data`
- `trading_accounts`
- `positions`
- `orders`
- `neural_models`
- `training_runs`
- `model_predictions`
- `trading_bots`
- `bot_executions`
- `sandbox_deployments`
- `performance_metrics`
- `alerts`
- `audit_logs`

## Row Level Security (RLS) Configuration

### 1. Enable RLS

RLS is automatically enabled in the setup scripts. Verify it's active:

```sql
SELECT schemaname, tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public' AND rowsecurity = true;
```

### 2. Test RLS Policies

Create test users and verify data isolation:

```sql
-- Create test user
INSERT INTO auth.users (id, email, email_confirmed_at, created_at, updated_at)
VALUES (
  gen_random_uuid(),
  'test@neuraltrade.ai',
  now(),
  now(),
  now()
);

-- Test profile access (should only see own data)
```

## Edge Functions Deployment

### 1. Deploy Edge Functions

```bash
# Deploy market data processor
supabase functions deploy market-data-processor --project-ref your-project-id

# Deploy signal generator
supabase functions deploy signal-generator --project-ref your-project-id

# Deploy risk calculator
supabase functions deploy risk-calculator --project-ref your-project-id
```

### 2. Test Edge Functions

```bash
# Test market data processor
curl -X POST \
  'https://your-project-id.supabase.co/functions/v1/market-data-processor' \
  -H 'Authorization: Bearer YOUR_ANON_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "symbol": "AAPL",
    "timestamp": "2024-01-01T10:00:00Z",
    "open": 150.00,
    "high": 152.00,
    "low": 149.00,
    "close": 151.00,
    "volume": 1000000,
    "timeframe": "1h"
  }'
```

## Real-time Configuration

### 1. Enable Real-time

In Supabase Dashboard:
1. Go to Settings → API
2. Enable Realtime
3. Configure realtime settings:
   - **Max events per second**: 100
   - **Max concurrent users**: 200

### 2. Test Real-time Subscriptions

```typescript
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY)

// Test market data subscription
const channel = supabase
  .channel('market-data-test')
  .on(
    'postgres_changes',
    {
      event: 'INSERT',
      schema: 'public',
      table: 'market_data'
    },
    (payload) => console.log('New market data:', payload)
  )
  .subscribe()
```

## Environment Variables

### 1. Local Development (.env.local)

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# E2B Configuration (for sandbox integration)
E2B_API_KEY=your-e2b-api-key

# Database Configuration (for direct connections)
DATABASE_URL=postgresql://postgres:password@db.your-project-id.supabase.co:5432/postgres

# Security
JWT_SECRET=your-jwt-secret
```

### 2. Production Environment

Set the same variables in your production environment (Vercel, Railway, etc.).

**⚠️ Security Note**: Never commit the service role key to version control!

## Client Integration

### 1. Install Dependencies

```bash
npm install @supabase/supabase-js @supabase/realtime-js
```

### 2. Initialize Supabase Client

```typescript
import { createClient } from '@supabase/supabase-js'
import { Database } from './src/supabase/types/database.types'

const supabaseUrl = process.env.SUPABASE_URL!
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY!

export const supabase = createClient<Database>(supabaseUrl, supabaseAnonKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: true
  },
  realtime: {
    params: {
      eventsPerSecond: 100
    }
  }
})
```

### 3. Use Client Libraries

```typescript
import { neuralModelsClient } from './src/supabase/client/neural-models'
import { tradingBotsClient } from './src/supabase/client/trading-bots'
import { sandboxIntegrationClient } from './src/supabase/client/sandbox-integration'

// Create neural model
const { model, error } = await neuralModelsClient.createModel(userId, {
  name: 'My LSTM Model',
  model_type: 'lstm',
  architecture: { /* model config */ }
})

// Deploy bot to sandbox
const { deployment } = await sandboxIntegrationClient.deployBotToSandbox(userId, {
  bot_id: 'bot-id',
  sandbox_config: {
    name: 'Trading Bot Sandbox',
    cpu_count: 2,
    memory_mb: 1024
  }
})
```

## E2B Sandbox Integration

### 1. Setup E2B Account

1. Sign up at [E2B Platform](https://e2b.dev)
2. Create API key
3. Install E2B CLI: `npm install -g @e2b/cli`

### 2. Create Trading Bot Template

```bash
# Create custom template
e2b template build --name neural-trader-base --path ./e2b-template

# Deploy template
e2b template deploy neural-trader-base
```

### 3. Configure Sandbox Integration

Add E2B configuration to your environment:

```bash
E2B_API_KEY=your-e2b-api-key
E2B_TEMPLATE_ID=neural-trader-base
```

### 4. Test Sandbox Deployment

```typescript
import { sandboxIntegrationClient } from './src/supabase/client/sandbox-integration'

const deployment = await sandboxIntegrationClient.deployBotToSandbox(userId, {
  bot_id: 'your-bot-id',
  sandbox_config: {
    name: 'Test Deployment',
    template: 'neural-trader-base',
    cpu_count: 1,
    memory_mb: 512,
    timeout_seconds: 3600
  },
  trading_config: {
    symbols: ['AAPL', 'GOOGL'],
    risk_limits: { max_position_size: 1000 }
  }
})
```

## Testing

### 1. Run Database Tests

```bash
# Set test environment variables
export SUPABASE_URL=https://your-project-id.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Run tests
npm test tests/supabase/
```

### 2. Test Real-time Features

```bash
# Test real-time subscriptions
npm run test:realtime
```

### 3. Test Edge Functions

```bash
# Test all edge functions
npm run test:edge-functions
```

## Monitoring & Troubleshooting

### 1. Enable Performance Monitoring

```typescript
import { performanceMonitor } from './src/supabase/monitoring/performance-monitor'

// Start monitoring
await performanceMonitor.startMonitoring(60000) // 1 minute intervals

// Check system health
const health = await performanceMonitor.checkSystemHealth()
console.log('System Health:', health)
```

### 2. Common Issues

#### Database Connection Issues

```bash
# Check database status
supabase status

# View logs
supabase logs db
```

#### RLS Policy Issues

```sql
-- Check if RLS is enabled
SELECT schemaname, tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public';

-- View active policies
SELECT * FROM pg_policies WHERE schemaname = 'public';
```

#### Real-time Issues

```typescript
// Check connection status
const status = supabase.realtime.connection.status
console.log('Realtime status:', status)

// Monitor connection events
supabase.realtime.connection.on('error', (error) => {
  console.error('Realtime error:', error)
})
```

### 3. Performance Optimization

#### Database Indexes

Ensure proper indexes exist:

```sql
-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan 
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;

-- Add custom indexes if needed
CREATE INDEX CONCURRENTLY idx_market_data_symbol_time 
ON market_data (symbol_id, timestamp DESC);
```

#### Connection Pooling

Configure connection pooling in production:

```typescript
const supabase = createClient(url, key, {
  global: {
    headers: {
      'x-connection-pool-size': '10'
    }
  }
})
```

## Security Best Practices

### 1. API Key Management

- Use environment variables for all keys
- Rotate service role keys regularly
- Never expose service role key in client-side code
- Use anon key for client operations only

### 2. RLS Policies

- Test all RLS policies thoroughly
- Use principle of least privilege
- Regularly audit policy effectiveness
- Monitor for policy bypass attempts

### 3. Edge Function Security

- Validate all inputs
- Implement rate limiting
- Use CORS headers properly
- Sanitize outputs

## Backup and Recovery

### 1. Database Backups

Supabase provides automatic backups, but for critical data:

```bash
# Manual backup
pg_dump "postgresql://postgres:password@db.your-project-id.supabase.co:5432/postgres" \
  --no-owner --no-privileges --clean > backup.sql

# Restore from backup
psql "postgresql://postgres:password@db.your-project-id.supabase.co:5432/postgres" \
  < backup.sql
```

### 2. Data Export

```typescript
// Export trading data
const { data } = await supabase
  .from('trading_accounts')
  .select(`
    *,
    positions (*),
    orders (*)
  `)
  .eq('user_id', userId)

// Save to file or cloud storage
```

## Support and Resources

- [Supabase Documentation](https://supabase.com/docs)
- [E2B Documentation](https://e2b.dev/docs)
- [Project GitHub Issues](https://github.com/ruvnet/claude-neural-trader/issues)
- [Neural Trading Discord](https://discord.gg/neural-trading)

---

**Next Steps**: After completing this setup, proceed to the [API Integration Guide](./API_INTEGRATION_GUIDE.md) for connecting your trading systems.