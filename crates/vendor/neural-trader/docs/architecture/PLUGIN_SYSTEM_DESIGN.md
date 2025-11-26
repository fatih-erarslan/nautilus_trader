# Plugin System Design Specification
## Neural Trader CLI v3.0

**Version:** 1.0.0
**Date:** 2025-11-17

---

## Overview

The Neural Trader CLI plugin system enables developers to extend functionality without modifying core code. Plugins can add commands, hooks, services, and UI components.

---

## Plugin Architecture

### Plugin Lifecycle

```
Discovery → Validation → Loading → Initialization → Activation → Execution → Deactivation → Unloading
```

### Plugin Types

1. **Command Plugins** - Add new commands
2. **Hook Plugins** - Intercept command execution
3. **Service Plugins** - Provide backend services
4. **UI Plugins** - Add dashboard widgets
5. **Integration Plugins** - Connect external systems

---

## Plugin Structure

### Directory Layout

```
plugins/my-plugin/
├── package.json                 # Plugin metadata
├── index.ts                     # Plugin entry point
├── commands/                    # Command implementations
│   ├── analyze.ts
│   └── report.ts
├── hooks/                       # Hook implementations
│   ├── before-trade.ts
│   └── after-trade.ts
├── services/                    # Service implementations
│   └── custom-analyzer.ts
├── ui/                         # UI components
│   └── widgets/
│       └── custom-chart.ts
├── config/                      # Plugin configuration
│   ├── schema.json
│   └── defaults.yaml
├── tests/                       # Plugin tests
│   ├── commands.test.ts
│   └── hooks.test.ts
└── README.md                    # Plugin documentation
```

### package.json

```json
{
  "name": "neural-trader-plugin-advanced-analysis",
  "version": "1.0.0",
  "description": "Advanced market analysis plugin",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "keywords": ["neural-trader", "plugin", "analysis"],
  "author": "Your Name",
  "license": "MIT",
  "neuralTrader": {
    "plugin": true,
    "minVersion": "3.0.0",
    "maxVersion": "3.x"
  },
  "dependencies": {
    "ta-lib": "^1.0.0",
    "chart.js": "^4.0.0"
  },
  "peerDependencies": {
    "neural-trader": "^3.0.0"
  }
}
```

### index.ts

```typescript
import { Plugin, PluginContext } from '@neural-trader/plugin-api';

export default class AdvancedAnalysisPlugin implements Plugin {
  name = 'advanced-analysis';
  version = '1.0.0';
  description = 'Advanced market analysis tools';

  // Plugin lifecycle
  async initialize(context: PluginContext): Promise<void> {
    this.context = context;
    this.logger = context.logger.child({ plugin: this.name });
    this.logger.info('Initializing advanced analysis plugin');
  }

  async activate(): Promise<void> {
    // Register commands
    this.context.commands.register({
      name: 'analyze:technical',
      description: 'Perform technical analysis',
      options: [
        { name: 'symbol', type: 'string', required: true },
        { name: 'indicators', type: 'string[]', default: ['RSI', 'MACD'] }
      ],
      execute: this.analyzeTechnical.bind(this)
    });

    // Register hooks
    this.context.hooks.register('before:trade', this.beforeTrade.bind(this));
    this.context.hooks.register('after:trade', this.afterTrade.bind(this));

    // Register services
    this.context.services.register('analyzer', new TechnicalAnalyzer());

    // Register UI widgets
    this.context.ui.registerWidget({
      name: 'technical-chart',
      component: TechnicalChartWidget,
      defaultConfig: { indicators: ['RSI', 'MACD'] }
    });

    this.logger.info('Plugin activated');
  }

  async deactivate(): Promise<void> {
    this.logger.info('Plugin deactivated');
  }

  async dispose(): Promise<void> {
    // Cleanup resources
    this.logger.info('Plugin disposed');
  }

  // Command implementation
  private async analyzeTechnical(args: any): Promise<void> {
    const { symbol, indicators } = args;
    const analyzer = this.context.services.get('analyzer');
    const results = await analyzer.analyze(symbol, indicators);
    this.context.ui.display(results);
  }

  // Hook implementations
  private async beforeTrade(trade: Trade): Promise<void> {
    // Validate trade based on technical analysis
    const signal = await this.checkTradeSignal(trade.symbol);
    if (!signal.isValid) {
      throw new Error(`Trade rejected: ${signal.reason}`);
    }
  }

  private async afterTrade(trade: Trade, result: TradeResult): Promise<void> {
    // Log trade for analysis
    this.logger.info('Trade executed', { trade, result });
  }
}
```

---

## Plugin API

### PluginContext

```typescript
export interface PluginContext {
  // Version info
  version: string;
  apiVersion: string;

  // Core services
  logger: Logger;
  config: ConfigService;
  state: StateService;
  events: EventEmitter;

  // Registry services
  commands: CommandRegistry;
  hooks: HookRegistry;
  services: ServiceRegistry;
  ui: UIRegistry;

  // Data access
  data: {
    marketData: MarketDataService;
    positions: PositionService;
    orders: OrderService;
  };

  // Utilities
  utils: {
    validation: ValidationUtils;
    formatting: FormattingUtils;
    http: HttpClient;
  };
}
```

### Command Registration

```typescript
export interface CommandDefinition {
  name: string;
  description: string;
  category?: string;
  options?: CommandOption[];
  aliases?: string[];
  examples?: string[];
  execute: (args: CommandArgs) => Promise<void>;
  validate?: (args: CommandArgs) => ValidationResult;
}

export interface CommandOption {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'string[]';
  description?: string;
  required?: boolean;
  default?: any;
  choices?: any[];
}

// Registration
context.commands.register({
  name: 'analyze:sentiment',
  description: 'Analyze market sentiment',
  category: 'analysis',
  options: [
    { name: 'symbol', type: 'string', required: true },
    { name: 'source', type: 'string', choices: ['twitter', 'reddit', 'news'] }
  ],
  examples: [
    'neural-trader analyze:sentiment AAPL --source twitter',
    'neural-trader analyze:sentiment TSLA --source reddit'
  ],
  execute: async (args) => {
    // Implementation
  }
});
```

### Hook Registration

```typescript
export type HookType =
  | 'before:command'
  | 'after:command'
  | 'before:trade'
  | 'after:trade'
  | 'before:backtest'
  | 'after:backtest'
  | 'on:error';

export interface HookHandler {
  (data: any, context: HookContext): Promise<any>;
}

export interface HookContext {
  plugin: string;
  command?: string;
  timestamp: number;
  metadata: Record<string, any>;
}

// Registration
context.hooks.register('before:trade', async (trade, hookContext) => {
  // Validate or modify trade before execution
  const validated = await validateTrade(trade);
  if (!validated.isValid) {
    throw new Error(`Trade validation failed: ${validated.reason}`);
  }
  return { ...trade, validated: true };
});
```

### Service Registration

```typescript
export interface Service {
  name: string;
  version: string;
  methods: Record<string, Function>;
}

// Registration
context.services.register('sentiment-analyzer', {
  name: 'sentiment-analyzer',
  version: '1.0.0',
  methods: {
    async analyze(symbol: string): Promise<SentimentScore> {
      // Implementation
    },
    async getHistory(symbol: string, days: number): Promise<SentimentHistory> {
      // Implementation
    }
  }
});

// Usage in other plugins
const analyzer = context.services.get('sentiment-analyzer');
const sentiment = await analyzer.analyze('AAPL');
```

### UI Widget Registration

```typescript
export interface Widget {
  name: string;
  component: React.ComponentType<WidgetProps>;
  defaultConfig: WidgetConfig;
  updateInterval?: number;
}

export interface WidgetProps {
  config: WidgetConfig;
  data: any;
  onUpdate: (data: any) => void;
}

// Registration
context.ui.registerWidget({
  name: 'sentiment-gauge',
  component: SentimentGaugeWidget,
  defaultConfig: {
    symbol: 'AAPL',
    refreshInterval: 5000,
    showHistory: true
  },
  updateInterval: 5000
});
```

---

## Plugin Security

### Sandboxing

```typescript
class PluginSandbox {
  private vm: VM;
  private context: SandboxContext;

  constructor(plugin: Plugin) {
    this.vm = new VM({
      timeout: 5000,
      sandbox: this.createSandbox(plugin)
    });
  }

  private createSandbox(plugin: Plugin): SandboxContext {
    return {
      // Allowed modules
      require: this.createRequire([
        'fs', 'path', 'crypto', 'util',
        'neural-trader/api', 'neural-trader/utils'
      ]),

      // Restricted globals
      console: this.createRestrictedConsole(),
      process: this.createRestrictedProcess(),

      // Plugin API
      neuralTrader: this.createPluginAPI(plugin),

      // Utilities
      setTimeout: this.createSafeTimeout(),
      setInterval: this.createSafeInterval()
    };
  }

  execute(code: string): any {
    return this.vm.run(code);
  }
}
```

### Permission System

```typescript
export interface PluginPermissions {
  // File system access
  filesystem: {
    read: string[];    // Allowed read paths
    write: string[];   // Allowed write paths
  };

  // Network access
  network: {
    allowHttp: boolean;
    allowedDomains: string[];
  };

  // System access
  system: {
    allowShell: boolean;
    allowEnv: string[];  // Allowed env vars
  };

  // Data access
  data: {
    marketData: boolean;
    positions: boolean;
    orders: boolean;
    history: boolean;
  };
}

// Permission check
class PermissionChecker {
  check(plugin: Plugin, permission: string, resource: string): boolean {
    const permissions = plugin.permissions;

    if (permission === 'fs:read') {
      return permissions.filesystem.read.some(
        pattern => this.matchPattern(resource, pattern)
      );
    }

    if (permission === 'network:http') {
      if (!permissions.network.allowHttp) return false;
      const domain = this.extractDomain(resource);
      return permissions.network.allowedDomains.includes(domain);
    }

    return false;
  }
}
```

### Code Signing

```typescript
class PluginValidator {
  async validateSignature(plugin: Plugin): Promise<boolean> {
    const signature = await fs.readFile(
      path.join(plugin.path, 'signature.txt'),
      'utf8'
    );

    const publicKey = await this.getPublicKey(plugin.author);
    const verified = crypto.verify(
      'sha256',
      Buffer.from(plugin.code),
      publicKey,
      Buffer.from(signature, 'base64')
    );

    return verified;
  }

  async validateChecksum(plugin: Plugin): Promise<boolean> {
    const expectedChecksum = plugin.manifest.checksum;
    const actualChecksum = this.calculateChecksum(plugin.code);
    return expectedChecksum === actualChecksum;
  }
}
```

---

## Plugin Discovery

### Discovery Sources

1. **Local Directory** - `./plugins/`
2. **User Directory** - `~/.neural-trader/plugins/`
3. **NPM Registry** - `@neural-trader/plugin-*`
4. **Plugin Registry** - `https://plugins.neural-trader.io`

### Discovery Process

```typescript
class PluginDiscovery {
  async discover(): Promise<PluginManifest[]> {
    const manifests: PluginManifest[] = [];

    // 1. Scan local directory
    manifests.push(...await this.scanDirectory('./plugins'));

    // 2. Scan user directory
    manifests.push(...await this.scanDirectory('~/.neural-trader/plugins'));

    // 3. Check installed NPM packages
    manifests.push(...await this.scanNpmPackages());

    // 4. Query plugin registry
    if (config.plugins.registryEnabled) {
      manifests.push(...await this.queryRegistry());
    }

    return this.deduplicateManifests(manifests);
  }

  private async scanDirectory(dir: string): Promise<PluginManifest[]> {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const manifests: PluginManifest[] = [];

    for (const entry of entries) {
      if (entry.isDirectory()) {
        const manifestPath = path.join(dir, entry.name, 'package.json');
        if (await fs.exists(manifestPath)) {
          const pkg = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
          if (pkg.neuralTrader?.plugin) {
            manifests.push(this.parseManifest(pkg, path.join(dir, entry.name)));
          }
        }
      }
    }

    return manifests;
  }
}
```

---

## Plugin Configuration

### Schema Definition

```typescript
// plugins/my-plugin/config/schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "enabled": {
      "type": "boolean",
      "default": true,
      "description": "Enable/disable plugin"
    },
    "analysisWindow": {
      "type": "number",
      "default": 30,
      "minimum": 1,
      "maximum": 365,
      "description": "Analysis window in days"
    },
    "indicators": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["RSI", "MACD", "SMA", "EMA", "BB"]
      },
      "default": ["RSI", "MACD"],
      "description": "Technical indicators to use"
    }
  },
  "required": ["enabled"]
}
```

### Configuration Loading

```typescript
class PluginConfigManager {
  async loadConfig(plugin: Plugin): Promise<PluginConfig> {
    // 1. Load defaults from plugin
    const defaults = await this.loadDefaults(plugin);

    // 2. Load user config
    const userConfig = await this.loadUserConfig(plugin.name);

    // 3. Load project config
    const projectConfig = await this.loadProjectConfig(plugin.name);

    // 4. Merge configurations
    const merged = this.mergeConfigs(defaults, userConfig, projectConfig);

    // 5. Validate against schema
    await this.validateConfig(plugin, merged);

    return merged;
  }

  private async validateConfig(
    plugin: Plugin,
    config: PluginConfig
  ): Promise<void> {
    const schema = await this.loadSchema(plugin);
    const validator = new Ajv();
    const valid = validator.validate(schema, config);

    if (!valid) {
      throw new Error(
        `Invalid config for plugin ${plugin.name}: ${validator.errorsText()}`
      );
    }
  }
}
```

---

## Plugin Examples

### Example 1: Technical Analysis Plugin

```typescript
// plugins/technical-analysis/index.ts
import { Plugin, PluginContext } from '@neural-trader/plugin-api';
import * as talib from 'ta-lib';

export default class TechnicalAnalysisPlugin implements Plugin {
  name = 'technical-analysis';
  version = '1.0.0';

  async activate(context: PluginContext): Promise<void> {
    // Register analyze command
    context.commands.register({
      name: 'ta:analyze',
      description: 'Perform technical analysis',
      options: [
        { name: 'symbol', type: 'string', required: true },
        { name: 'indicator', type: 'string', required: true },
        { name: 'period', type: 'number', default: 14 }
      ],
      execute: async (args) => {
        const { symbol, indicator, period } = args;

        // Fetch market data
        const data = await context.data.marketData.fetch(symbol, {
          period: '1d',
          count: period * 2
        });

        // Calculate indicator
        let result;
        switch (indicator) {
          case 'RSI':
            result = talib.RSI(data.close, period);
            break;
          case 'MACD':
            result = talib.MACD(data.close, 12, 26, 9);
            break;
          // ... more indicators
        }

        // Display results
        context.ui.displayTable({
          title: `${indicator} for ${symbol}`,
          headers: ['Date', 'Value'],
          rows: result.map((val, i) => [data.dates[i], val.toFixed(2)])
        });
      }
    });

    // Register hook to add TA to trades
    context.hooks.register('before:trade', async (trade) => {
      const rsi = await this.calculateRSI(trade.symbol);
      trade.metadata = { ...trade.metadata, rsi };
      return trade;
    });
  }

  private async calculateRSI(symbol: string): Promise<number> {
    // Implementation
  }
}
```

### Example 2: Custom Dashboard Widget

```typescript
// plugins/custom-widget/index.ts
import React from 'react';
import { Widget, WidgetProps } from '@neural-trader/plugin-api';

const SentimentWidget: React.FC<WidgetProps> = ({ config, data, onUpdate }) => {
  const [sentiment, setSentiment] = React.useState(data.sentiment || 0);

  React.useEffect(() => {
    const interval = setInterval(async () => {
      const newData = await fetchSentiment(config.symbol);
      setSentiment(newData.score);
      onUpdate(newData);
    }, config.refreshInterval);

    return () => clearInterval(interval);
  }, [config]);

  return (
    <Box flexDirection="column">
      <Text bold>Market Sentiment - {config.symbol}</Text>
      <Text>
        Score: {sentiment > 0 ? '🟢' : '🔴'} {sentiment.toFixed(2)}
      </Text>
      <ProgressBar value={Math.abs(sentiment)} max={1} />
    </Box>
  );
};

export default class SentimentWidgetPlugin implements Plugin {
  name = 'sentiment-widget';
  version = '1.0.0';

  async activate(context: PluginContext): Promise<void> {
    context.ui.registerWidget({
      name: 'sentiment',
      component: SentimentWidget,
      defaultConfig: {
        symbol: 'AAPL',
        refreshInterval: 5000
      }
    });
  }
}
```

---

## Plugin Distribution

### Publishing to NPM

```bash
# Build plugin
npm run build

# Test plugin
npm test

# Publish
npm publish --access public
```

### Plugin Registry Submission

```bash
# Submit to neural-trader plugin registry
neural-trader plugin submit ./plugins/my-plugin

# Registry validates and publishes
# Plugin becomes discoverable via:
neural-trader plugin search "sentiment"
neural-trader plugin install sentiment-analyzer
```

---

## Best Practices

1. **Version Compatibility**
   - Specify min/max neural-trader versions
   - Use semantic versioning
   - Test with target versions

2. **Performance**
   - Minimize startup time
   - Use lazy loading
   - Cache expensive operations
   - Limit memory usage

3. **Error Handling**
   - Handle errors gracefully
   - Provide helpful error messages
   - Log errors appropriately
   - Don't crash the CLI

4. **Testing**
   - Write unit tests
   - Write integration tests
   - Test with real CLI
   - Test error cases

5. **Documentation**
   - Comprehensive README
   - Command examples
   - Configuration options
   - API documentation

6. **Security**
   - Validate all inputs
   - Use least privilege
   - Don't expose secrets
   - Sign your plugin

---

## Conclusion

The plugin system enables extensive customization while maintaining security and stability. Developers can add commands, hooks, services, and UI widgets to create a personalized trading experience.

**Next Steps:**
1. Implement plugin loader
2. Create plugin API package
3. Build example plugins
4. Set up plugin registry
5. Write plugin development guide
