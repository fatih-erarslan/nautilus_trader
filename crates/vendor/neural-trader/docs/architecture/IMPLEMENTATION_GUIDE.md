# Implementation Quick-Start Guide
## Neural Trader CLI v3.0 Enhancement

**Version:** 1.0.0
**Date:** 2025-11-17

---

## Overview

This guide provides step-by-step instructions to begin implementing the neural-trader CLI enhancements. It covers the initial setup, foundation work, and first features to implement.

---

## Prerequisites

### Required Tools
- Node.js >= 18.0.0
- npm >= 9.0.0
- TypeScript >= 5.0
- Git

### Recommended Tools
- VS Code with TypeScript extension
- Postman or similar (for MCP testing)
- Terminal multiplexer (tmux/screen)

### Knowledge Requirements
- TypeScript/JavaScript
- Node.js CLI development
- Async/await patterns
- Event-driven architecture
- Basic Rust (for NAPI integration)

---

## Phase 1: Foundation Setup (Week 1)

### Step 1.1: Project Structure Setup

```bash
# Create new directory structure
cd /home/user/neural-trader

# Create TypeScript source directories
mkdir -p src/cli
mkdir -p src/commands/{base,core,package,mcp,agent,monitor,deploy,profile,template,configure}
mkdir -p src/ui/{components,dashboard/widgets}
mkdir -p src/core/{config,state,events,logger,plugin}
mkdir -p src/services/{mcp,agent,monitor,deploy,profile}
mkdir -p src/integration
mkdir -p src/utils

# Create config directories
mkdir -p config
mkdir -p plugins/examples
mkdir -p templates/{trading,backtesting,examples}

# Create test directories
mkdir -p tests/{unit,integration,e2e}
```

### Step 1.2: TypeScript Configuration

Create `/home/user/neural-trader/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "moduleResolution": "node",
    "types": ["node"],
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@commands/*": ["src/commands/*"],
      "@core/*": ["src/core/*"],
      "@services/*": ["src/services/*"],
      "@utils/*": ["src/utils/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

Create `/home/user/neural-trader/tsconfig.build.json`:

```json
{
  "extends": "./tsconfig.json",
  "exclude": ["node_modules", "dist", "tests", "**/*.test.ts", "**/*.spec.ts"]
}
```

### Step 1.3: Package.json Updates

Update `/home/user/neural-trader/package.json`:

```json
{
  "name": "neural-trader",
  "version": "3.0.0",
  "description": "High-performance neural trading system with interactive CLI",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "bin": {
    "neural-trader": "./dist/cli/index.js"
  },
  "scripts": {
    "dev": "tsx watch src/cli/index.ts",
    "build": "tsc -p tsconfig.build.json",
    "build:fast": "tsup src/cli/index.ts --format cjs --dts --clean",
    "start": "node dist/cli/index.js",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "lint": "eslint src --ext .ts",
    "lint:fix": "eslint src --ext .ts --fix",
    "format": "prettier --write \"src/**/*.ts\"",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "commander": "^11.1.0",
    "inquirer": "^9.2.12",
    "chalk": "^5.3.0",
    "ora": "^8.0.1",
    "cli-table3": "^0.6.3",
    "boxen": "^7.1.1",
    "ink": "^4.4.1",
    "react": "^18.2.0",
    "ws": "^8.14.2",
    "axios": "^1.6.2",
    "zod": "^3.22.4",
    "yaml": "^2.3.4",
    "cosmiconfig": "^9.0.0",
    "winston": "^3.11.0",
    "eventemitter3": "^5.0.1"
  },
  "devDependencies": {
    "@types/node": "^20.10.4",
    "@types/inquirer": "^9.0.7",
    "@types/ws": "^8.5.9",
    "@typescript-eslint/eslint-plugin": "^6.13.2",
    "@typescript-eslint/parser": "^6.13.2",
    "eslint": "^8.55.0",
    "prettier": "^3.1.0",
    "tsx": "^4.6.2",
    "tsup": "^8.0.1",
    "typescript": "^5.3.3",
    "jest": "^29.7.0",
    "@types/jest": "^29.5.10",
    "ts-jest": "^29.1.1"
  }
}
```

### Step 1.4: Install Dependencies

```bash
cd /home/user/neural-trader
npm install
```

---

## Step 2: Core Systems (Week 1)

### Step 2.1: Config Manager

Create `/home/user/neural-trader/src/core/config/manager.ts`:

```typescript
import { cosmiconfigSync } from 'cosmiconfig';
import { z } from 'zod';
import path from 'path';
import fs from 'fs';
import yaml from 'yaml';

// Configuration schema
const ConfigSchema = z.object({
  version: z.string(),
  cli: z.object({
    theme: z.string().default('default'),
    interactive: z.object({
      enabled: z.boolean().default(true),
      historySize: z.number().default(1000),
      autoComplete: z.boolean().default(true),
    }),
    output: z.object({
      format: z.enum(['pretty', 'json', 'yaml']).default('pretty'),
      colors: z.boolean().default(true),
      unicode: z.boolean().default(true),
    }),
  }),
  logging: z.object({
    level: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
    file: z.object({
      enabled: z.boolean().default(true),
      path: z.string().default('~/.neural-trader/logs'),
      maxSize: z.string().default('10mb'),
      maxFiles: z.number().default(5),
    }),
    audit: z.object({
      enabled: z.boolean().default(true),
      path: z.string().default('~/.neural-trader/audit'),
    }),
  }),
  mcp: z.object({
    enabled: z.boolean().default(true),
    port: z.number().default(3000),
    host: z.string().default('localhost'),
  }),
});

export type Config = z.infer<typeof ConfigSchema>;

export class ConfigManager {
  private config: Config | null = null;
  private configPath: string | null = null;

  constructor(private searchFrom?: string) {}

  async load(): Promise<Config> {
    // Load configuration using cosmiconfig
    const explorer = cosmiconfigSync('neural-trader', {
      searchPlaces: [
        'package.json',
        '.neural-traderrc',
        '.neural-traderrc.json',
        '.neural-traderrc.yaml',
        '.neural-traderrc.yml',
        'neural-trader.config.js',
        'neural-trader.config.ts',
      ],
    });

    const result = explorer.search(this.searchFrom);

    if (result) {
      this.configPath = result.filepath;
      this.config = this.validateAndMerge(result.config);
    } else {
      // Load default config
      this.config = this.loadDefaults();
    }

    return this.config;
  }

  private loadDefaults(): Config {
    const defaultConfigPath = path.join(__dirname, '../../../config/default.yaml');
    if (fs.existsSync(defaultConfigPath)) {
      const content = fs.readFileSync(defaultConfigPath, 'utf8');
      return yaml.parse(content);
    }

    // Fallback to minimal config
    return {
      version: '3.0.0',
      cli: {
        theme: 'default',
        interactive: {
          enabled: true,
          historySize: 1000,
          autoComplete: true,
        },
        output: {
          format: 'pretty',
          colors: true,
          unicode: true,
        },
      },
      logging: {
        level: 'info',
        file: {
          enabled: true,
          path: '~/.neural-trader/logs',
          maxSize: '10mb',
          maxFiles: 5,
        },
        audit: {
          enabled: true,
          path: '~/.neural-trader/audit',
        },
      },
      mcp: {
        enabled: true,
        port: 3000,
        host: 'localhost',
      },
    };
  }

  private validateAndMerge(config: any): Config {
    // Validate config against schema
    const validated = ConfigSchema.parse(config);

    // Expand environment variables
    return this.expandEnvVars(validated);
  }

  private expandEnvVars(config: any): any {
    if (typeof config === 'string') {
      return config.replace(/\$\{([^}]+)\}/g, (_, key) => {
        return process.env[key] || '';
      });
    }

    if (Array.isArray(config)) {
      return config.map(item => this.expandEnvVars(item));
    }

    if (typeof config === 'object' && config !== null) {
      const result: any = {};
      for (const [key, value] of Object.entries(config)) {
        result[key] = this.expandEnvVars(value);
      }
      return result;
    }

    return config;
  }

  get(key: string): any {
    if (!this.config) {
      throw new Error('Config not loaded');
    }

    const keys = key.split('.');
    let value: any = this.config;

    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        return undefined;
      }
    }

    return value;
  }

  set(key: string, value: any): void {
    if (!this.config) {
      throw new Error('Config not loaded');
    }

    const keys = key.split('.');
    const lastKey = keys.pop()!;
    let target: any = this.config;

    for (const k of keys) {
      if (!(k in target)) {
        target[k] = {};
      }
      target = target[k];
    }

    target[lastKey] = value;
  }

  async save(): Promise<void> {
    if (!this.config || !this.configPath) {
      throw new Error('No config to save');
    }

    const content = yaml.stringify(this.config);
    fs.writeFileSync(this.configPath, content, 'utf8');
  }

  getConfig(): Config {
    if (!this.config) {
      throw new Error('Config not loaded');
    }
    return this.config;
  }
}
```

### Step 2.2: Logger

Create `/home/user/neural-trader/src/core/logger/logger.ts`:

```typescript
import winston from 'winston';
import path from 'path';
import fs from 'fs';

export class Logger {
  private logger: winston.Logger;

  constructor(options?: {
    level?: string;
    logDir?: string;
    enableFile?: boolean;
    enableConsole?: boolean;
  }) {
    const {
      level = 'info',
      logDir = path.join(process.env.HOME || '', '.neural-trader', 'logs'),
      enableFile = true,
      enableConsole = true,
    } = options || {};

    // Ensure log directory exists
    if (enableFile && !fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }

    const transports: winston.transport[] = [];

    // Console transport
    if (enableConsole) {
      transports.push(
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
            winston.format.printf(({ timestamp, level, message, ...meta }) => {
              let msg = `[${timestamp}] ${level}: ${message}`;
              if (Object.keys(meta).length > 0) {
                msg += ` ${JSON.stringify(meta)}`;
              }
              return msg;
            })
          ),
        })
      );
    }

    // File transport
    if (enableFile) {
      transports.push(
        new winston.transports.File({
          filename: path.join(logDir, 'neural-trader.log'),
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.json()
          ),
          maxsize: 10 * 1024 * 1024, // 10MB
          maxFiles: 5,
        })
      );

      // Error log file
      transports.push(
        new winston.transports.File({
          filename: path.join(logDir, 'error.log'),
          level: 'error',
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.json()
          ),
          maxsize: 10 * 1024 * 1024,
          maxFiles: 5,
        })
      );
    }

    this.logger = winston.createLogger({
      level,
      transports,
    });
  }

  debug(message: string, meta?: object): void {
    this.logger.debug(message, meta);
  }

  info(message: string, meta?: object): void {
    this.logger.info(message, meta);
  }

  warn(message: string, meta?: object): void {
    this.logger.warn(message, meta);
  }

  error(message: string, error?: Error, meta?: object): void {
    this.logger.error(message, { error: error?.stack, ...meta });
  }

  child(options: object): Logger {
    const childLogger = new Logger();
    childLogger.logger = this.logger.child(options);
    return childLogger;
  }
}

// Singleton instance
let defaultLogger: Logger | null = null;

export function getLogger(): Logger {
  if (!defaultLogger) {
    defaultLogger = new Logger();
  }
  return defaultLogger;
}

export function initLogger(options?: any): Logger {
  defaultLogger = new Logger(options);
  return defaultLogger;
}
```

### Step 2.3: Event System

Create `/home/user/neural-trader/src/core/events/emitter.ts`:

```typescript
import EventEmitter3 from 'eventemitter3';

export enum EventType {
  // CLI Events
  CLI_START = 'cli:start',
  CLI_STOP = 'cli:stop',
  COMMAND_START = 'command:start',
  COMMAND_END = 'command:end',
  COMMAND_ERROR = 'command:error',

  // MCP Events
  MCP_START = 'mcp:start',
  MCP_STOP = 'mcp:stop',
  MCP_ERROR = 'mcp:error',

  // Agent Events
  AGENT_SPAWN = 'agent:spawn',
  AGENT_STOP = 'agent:stop',
  AGENT_ERROR = 'agent:error',
}

export interface Event {
  type: EventType;
  timestamp: number;
  data: any;
  metadata?: {
    userId?: string;
    sessionId?: string;
    commandId?: string;
  };
}

export class EventSystem extends EventEmitter3 {
  private static instance: EventSystem;

  private constructor() {
    super();
  }

  static getInstance(): EventSystem {
    if (!EventSystem.instance) {
      EventSystem.instance = new EventSystem();
    }
    return EventSystem.instance;
  }

  emit(eventType: EventType, data: any, metadata?: Event['metadata']): boolean {
    const event: Event = {
      type: eventType,
      timestamp: Date.now(),
      data,
      metadata,
    };

    return super.emit(eventType, event);
  }

  onEvent(eventType: EventType, handler: (event: Event) => void): void {
    this.on(eventType, handler);
  }

  offEvent(eventType: EventType, handler: (event: Event) => void): void {
    this.off(eventType, handler);
  }

  onceEvent(eventType: EventType, handler: (event: Event) => void): void {
    this.once(eventType, handler);
  }
}

// Singleton instance
export const events = EventSystem.getInstance();
```

### Step 2.4: State Store

Create `/home/user/neural-trader/src/core/state/store.ts`:

```typescript
import fs from 'fs';
import path from 'path';

export interface StateStore {
  agents: {
    [agentId: string]: any;
  };
  deployments: {
    [deploymentId: string]: any;
  };
  sessions: {
    current?: string;
    history: string[];
  };
  settings: {
    [key: string]: any;
  };
}

export class StateManager {
  private state: StateStore;
  private statePath: string;
  private autoSave: boolean;
  private saveInterval?: NodeJS.Timeout;

  constructor(options?: { statePath?: string; autoSave?: boolean; autoSaveInterval?: number }) {
    this.statePath =
      options?.statePath ||
      path.join(process.env.HOME || '', '.neural-trader', 'state', 'state.json');

    this.autoSave = options?.autoSave !== false;

    this.state = this.load();

    if (this.autoSave) {
      this.saveInterval = setInterval(() => {
        this.save();
      }, options?.autoSaveInterval || 60000);
    }
  }

  private load(): StateStore {
    if (fs.existsSync(this.statePath)) {
      try {
        const content = fs.readFileSync(this.statePath, 'utf8');
        return JSON.parse(content);
      } catch (error) {
        console.error('Failed to load state:', error);
      }
    }

    return {
      agents: {},
      deployments: {},
      sessions: {
        history: [],
      },
      settings: {},
    };
  }

  save(): void {
    const dir = path.dirname(this.statePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    fs.writeFileSync(this.statePath, JSON.stringify(this.state, null, 2), 'utf8');
  }

  get(key: string): any {
    const keys = key.split('.');
    let value: any = this.state;

    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        return undefined;
      }
    }

    return value;
  }

  set(key: string, value: any): void {
    const keys = key.split('.');
    const lastKey = keys.pop()!;
    let target: any = this.state;

    for (const k of keys) {
      if (!(k in target)) {
        target[k] = {};
      }
      target = target[k];
    }

    target[lastKey] = value;

    if (this.autoSave) {
      this.save();
    }
  }

  delete(key: string): void {
    const keys = key.split('.');
    const lastKey = keys.pop()!;
    let target: any = this.state;

    for (const k of keys) {
      if (!(k in target)) {
        return;
      }
      target = target[k];
    }

    delete target[lastKey];

    if (this.autoSave) {
      this.save();
    }
  }

  getState(): StateStore {
    return this.state;
  }

  dispose(): void {
    if (this.saveInterval) {
      clearInterval(this.saveInterval);
    }
    this.save();
  }
}
```

---

## Step 3: Base Command Class (Week 1)

Create `/home/user/neural-trader/src/commands/base/command.ts`:

```typescript
import { Logger, getLogger } from '@/core/logger/logger';
import { EventSystem, EventType, events } from '@/core/events/emitter';
import { ConfigManager } from '@/core/config/manager';
import { StateManager } from '@/core/state/store';

export interface CommandArgs {
  [key: string]: any;
}

export interface CommandOptions {
  name: string;
  description: string;
  aliases?: string[];
  category?: string;
}

export interface ValidationResult {
  valid: boolean;
  errors?: string[];
}

export abstract class BaseCommand {
  protected logger: Logger;
  protected events: EventSystem;
  protected config?: ConfigManager;
  protected state?: StateManager;

  constructor(
    protected options: CommandOptions,
    deps?: {
      logger?: Logger;
      config?: ConfigManager;
      state?: StateManager;
    }
  ) {
    this.logger = deps?.logger || getLogger();
    this.events = events;
    this.config = deps?.config;
    this.state = deps?.state;
  }

  /**
   * Main execution method - must be implemented by subclasses
   */
  abstract execute(args: CommandArgs): Promise<void>;

  /**
   * Validate command arguments
   */
  validate(args: CommandArgs): ValidationResult {
    // Default implementation - can be overridden
    return { valid: true };
  }

  /**
   * Hook: runs before command execution
   */
  async beforeExecute(args: CommandArgs): Promise<void> {
    this.logger.debug(`Executing command: ${this.options.name}`, { args });
    this.events.emit(EventType.COMMAND_START, {
      command: this.options.name,
      args,
    });
  }

  /**
   * Hook: runs after successful command execution
   */
  async afterExecute(args: CommandArgs, result?: any): Promise<void> {
    this.logger.debug(`Command completed: ${this.options.name}`);
    this.events.emit(EventType.COMMAND_END, {
      command: this.options.name,
      args,
      result,
    });
  }

  /**
   * Hook: runs when command execution fails
   */
  async onError(error: Error, args: CommandArgs): Promise<void> {
    this.logger.error(`Command failed: ${this.options.name}`, error, { args });
    this.events.emit(EventType.COMMAND_ERROR, {
      command: this.options.name,
      args,
      error: error.message,
      stack: error.stack,
    });
  }

  /**
   * Run command with lifecycle hooks
   */
  async run(args: CommandArgs): Promise<void> {
    try {
      // Validate
      const validation = this.validate(args);
      if (!validation.valid) {
        throw new Error(`Validation failed: ${validation.errors?.join(', ')}`);
      }

      // Before hook
      await this.beforeExecute(args);

      // Execute
      const result = await this.execute(args);

      // After hook
      await this.afterExecute(args, result);
    } catch (error) {
      await this.onError(error as Error, args);
      throw error;
    }
  }

  getName(): string {
    return this.options.name;
  }

  getDescription(): string {
    return this.options.description;
  }

  getAliases(): string[] {
    return this.options.aliases || [];
  }

  getCategory(): string {
    return this.options.category || 'general';
  }
}
```

---

## Step 4: First Command Implementation (Week 1)

Create `/home/user/neural-trader/src/commands/core/version.ts`:

```typescript
import { BaseCommand, CommandArgs } from '../base/command';
import chalk from 'chalk';
import boxen from 'boxen';

export class VersionCommand extends BaseCommand {
  constructor() {
    super({
      name: 'version',
      description: 'Show version and system information',
      aliases: ['v', '--version'],
      category: 'core',
    });
  }

  async execute(args: CommandArgs): Promise<void> {
    const pkg = require('../../../../package.json');

    // Check NAPI bindings
    let napiAvailable = false;
    let napiFunctions = 0;
    try {
      const napi = require('../../../../neural-trader-rust/index.js');
      napiAvailable = true;
      napiFunctions = Object.keys(napi).length;
    } catch (error) {
      // NAPI not available
    }

    // Output format
    if (args.json) {
      console.log(
        JSON.stringify(
          {
            version: pkg.version,
            node: process.version,
            platform: `${process.platform}-${process.arch}`,
            napi: {
              available: napiAvailable,
              functions: napiFunctions,
            },
          },
          null,
          2
        )
      );
    } else {
      const lines = [
        chalk.bold.cyan('Neural Trader CLI'),
        '',
        `Version: ${chalk.bold(pkg.version)}`,
        `Node: ${chalk.dim(process.version)}`,
        `Platform: ${chalk.dim(process.platform)}-${chalk.dim(process.arch)}`,
        '',
        napiAvailable
          ? `${chalk.green('✓')} NAPI Bindings: ${chalk.bold('Available')} (${napiFunctions} functions)`
          : `${chalk.yellow('⚠')} NAPI Bindings: ${chalk.dim('Not loaded')}`,
        '',
        chalk.blue('https://github.com/ruvnet/neural-trader'),
      ];

      console.log(
        boxen(lines.join('\n'), {
          padding: 1,
          margin: 1,
          borderStyle: 'round',
          borderColor: 'cyan',
        })
      );
    }
  }
}
```

---

## Step 5: CLI Entry Point (Week 1)

Create `/home/user/neural-trader/src/cli/index.ts`:

```typescript
#!/usr/bin/env node

import { Command } from 'commander';
import { ConfigManager } from '@/core/config/manager';
import { initLogger } from '@/core/logger/logger';
import { StateManager } from '@/core/state/store';
import { events, EventType } from '@/core/events/emitter';

// Commands
import { VersionCommand } from '@/commands/core/version';

async function main() {
  const program = new Command();

  // Load config
  const config = new ConfigManager();
  await config.load();

  // Initialize logger
  const logger = initLogger({
    level: config.get('logging.level'),
    logDir: config.get('logging.file.path'),
    enableFile: config.get('logging.file.enabled'),
  });

  // Initialize state
  const state = new StateManager({
    autoSave: config.get('state.autoSave'),
    autoSaveInterval: config.get('state.autoSaveInterval'),
  });

  // Emit CLI start event
  events.emit(EventType.CLI_START, {});

  // Setup CLI
  program
    .name('neural-trader')
    .description('High-performance neural trading system')
    .version('3.0.0');

  // Global options
  program
    .option('--debug', 'Enable debug logging')
    .option('--config <path>', 'Config file path')
    .option('--json', 'JSON output format')
    .option('--no-color', 'Disable colors')
    .option('--quiet, -q', 'Quiet mode');

  // Register version command
  const versionCmd = new VersionCommand();
  program
    .command('version')
    .description(versionCmd.getDescription())
    .action(async (options) => {
      try {
        await versionCmd.run({ ...options, ...program.opts() });
      } catch (error) {
        logger.error('Command failed', error as Error);
        process.exit(1);
      }
    });

  // Parse arguments
  await program.parseAsync(process.argv);

  // Emit CLI stop event
  events.emit(EventType.CLI_STOP, {});

  // Cleanup
  state.dispose();
}

// Handle errors
process.on('unhandledRejection', (error) => {
  console.error('Unhandled rejection:', error);
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  process.exit(1);
});

// Run
main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
```

---

## Step 6: Build and Test (Week 1)

### Build the project

```bash
cd /home/user/neural-trader
npm run build
```

### Test the version command

```bash
node dist/cli/index.js version
```

Expected output:
```
╭────────────────────────────────────────╮
│                                        │
│   Neural Trader CLI                    │
│                                        │
│   Version: 3.0.0                       │
│   Node: v18.17.0                       │
│   Platform: linux-x64                  │
│                                        │
│   ✓ NAPI Bindings: Available (42       │
│   functions)                           │
│                                        │
│   https://github.com/ruvnet/neural-    │
│   trader                               │
│                                        │
╰────────────────────────────────────────╯
```

---

## Step 7: Development Workflow

### Development mode with hot reload

```bash
npm run dev
```

### Run tests

```bash
npm test
```

### Lint code

```bash
npm run lint
npm run lint:fix
```

### Format code

```bash
npm run format
```

### Type check

```bash
npm run typecheck
```

---

## Next Steps

After completing the foundation:

1. **Week 2: Enhanced Commands**
   - Implement help command
   - Implement package commands
   - Implement MCP commands

2. **Week 3-4: Agent Coordination**
   - Implement agent spawning
   - Implement agent monitoring
   - Integrate with agentic-flow

3. **Week 5-6: Monitoring Dashboard**
   - Build dashboard UI
   - Implement WebSocket streaming
   - Add visualization components

4. **Week 7-8: Cloud Deployment**
   - E2B integration
   - Flow Nexus integration
   - Deployment management

5. **Week 9-10: Interactive Mode**
   - Build REPL
   - Add auto-completion
   - Add command history

6. **Week 11-12: Polish & Documentation**
   - Write comprehensive docs
   - Create tutorials
   - Migration guide

---

## Development Best Practices

1. **Write tests first** (TDD approach)
2. **Use TypeScript strict mode**
3. **Document public APIs**
4. **Follow consistent code style**
5. **Commit frequently with clear messages**
6. **Review before merging**
7. **Keep PRs small and focused**

---

## Useful Commands

```bash
# Development
npm run dev                    # Hot reload development
npm run build                  # Build for production
npm run build:fast             # Fast build with tsup

# Testing
npm test                       # Run tests
npm test -- --watch            # Watch mode
npm run test:coverage          # Coverage report

# Code Quality
npm run lint                   # Check linting
npm run lint:fix               # Fix linting issues
npm run format                 # Format code
npm run typecheck              # Type checking

# Debugging
node --inspect dist/cli/index.js version
```

---

## Troubleshooting

### Issue: Module resolution errors

**Solution:** Check tsconfig.json paths and ensure they match directory structure.

### Issue: NAPI bindings not found

**Solution:** Ensure neural-trader-rust is built:
```bash
cd neural-trader-rust/crates/napi-bindings
napi build --release
```

### Issue: Permission denied on CLI

**Solution:** Make CLI executable:
```bash
chmod +x dist/cli/index.js
```

---

## Resources

- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Commander.js Documentation](https://github.com/tj/commander.js)
- [Ink Documentation](https://github.com/vadimdemedes/ink)
- [Winston Logger](https://github.com/winstonjs/winston)
- [Zod Validation](https://github.com/colinhacks/zod)

---

## Conclusion

This guide provides everything needed to begin implementing the neural-trader CLI v3.0 enhancements. Follow the steps sequentially, test frequently, and refer to the full architecture documentation for detailed specifications.

**Happy coding! 🚀**
