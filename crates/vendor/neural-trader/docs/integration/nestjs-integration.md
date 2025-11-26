# NestJS Integration Guide

## Overview

This guide demonstrates how to integrate `@neural-trader/backend` with NestJS applications, following NestJS architectural patterns and best practices.

## Platform Requirements

- **Node.js**: >= 16.0.0 (Recommended: v18 LTS or v20 LTS)
- **NestJS**: >= 9.0.0
- **TypeScript**: >= 4.7.0
- **Operating Systems**: Linux, macOS, Windows

## Installation

```bash
npm install @neural-trader/backend @nestjs/core @nestjs/common rxjs reflect-metadata
npm install --save-dev @types/node
```

## Module Structure

### Neural Trader Module

```typescript
// src/neural-trader/neural-trader.module.ts
import { Module, Global } from '@nestjs/common';
import { NeuralTraderService } from './neural-trader.service';
import { TradingController } from './controllers/trading.controller';
import { PortfolioController } from './controllers/portfolio.controller';
import { BacktestController } from './controllers/backtest.controller';
import { SwarmController } from './controllers/swarm.controller';
import { SyndicateController } from './controllers/syndicate.controller';

@Global()
@Module({
  providers: [NeuralTraderService],
  controllers: [
    TradingController,
    PortfolioController,
    BacktestController,
    SwarmController,
    SyndicateController,
  ],
  exports: [NeuralTraderService],
})
export class NeuralTraderModule {}
```

### Core Service

```typescript
// src/neural-trader/neural-trader.service.ts
import { Injectable, OnModuleInit, OnModuleDestroy, Logger } from '@nestjs/common';
import * as backend from '@neural-trader/backend';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class NeuralTraderService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(NeuralTraderService.name);
  private initialized = false;

  constructor(private configService: ConfigService) {}

  async onModuleInit() {
    await this.initialize();
  }

  async onModuleDestroy() {
    await this.shutdown();
  }

  private async initialize() {
    if (this.initialized) return;

    try {
      const jwtSecret = this.configService.get<string>('JWT_SECRET');
      if (!jwtSecret) {
        throw new Error('JWT_SECRET is required');
      }

      // Initialize authentication
      backend.initAuth(jwtSecret);
      this.logger.log('Authentication initialized');

      // Initialize rate limiter
      backend.initRateLimiter({
        maxRequestsPerMinute: this.configService.get('RATE_LIMIT_MAX_REQUESTS', 100),
        burstSize: this.configService.get('RATE_LIMIT_BURST_SIZE', 20),
        windowDurationSecs: this.configService.get('RATE_LIMIT_WINDOW_SECS', 60),
      });
      this.logger.log('Rate limiter initialized');

      // Initialize audit logger
      backend.initAuditLogger(
        this.configService.get('AUDIT_MAX_EVENTS', 10000),
        this.configService.get('AUDIT_LOG_CONSOLE', true),
        this.configService.get('AUDIT_LOG_FILE', true)
      );
      this.logger.log('Audit logger initialized');

      // Initialize security config
      const corsConfig = {
        allowedOrigins: this.configService.get('CORS_ALLOWED_ORIGINS', '*').split(','),
        allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key'],
        exposedHeaders: ['X-Total-Count', 'X-Page-Number'],
        allowCredentials: true,
        maxAge: 86400,
      };

      backend.initSecurityConfig(
        corsConfig,
        this.configService.get('REQUIRE_HTTPS', true)
      );
      this.logger.log('Security config initialized');

      // Initialize neural trader
      await backend.initNeuralTrader(
        JSON.stringify({
          logLevel: this.configService.get('LOG_LEVEL', 'info'),
          enableGpu: this.configService.get('ENABLE_GPU', false),
        })
      );
      this.logger.log('Neural Trader initialized');

      this.initialized = true;

      const systemInfo = backend.getSystemInfo();
      this.logger.log(`Neural Trader v${systemInfo.version} ready`);
    } catch (error) {
      this.logger.error('Failed to initialize Neural Trader', error);
      throw error;
    }
  }

  private async shutdown() {
    if (!this.initialized) return;

    try {
      await backend.shutdown();
      this.logger.log('Neural Trader shut down gracefully');
    } catch (error) {
      this.logger.error('Error during shutdown', error);
    }
  }

  // Trading operations
  async quickAnalysis(symbol: string, useGpu?: boolean) {
    return backend.quickAnalysis(symbol, useGpu);
  }

  async executeTrade(
    strategy: string,
    symbol: string,
    action: string,
    quantity: number,
    orderType?: string,
    limitPrice?: number
  ) {
    return backend.executeTrade(strategy, symbol, action, quantity, orderType, limitPrice);
  }

  async simulateTrade(strategy: string, symbol: string, action: string, useGpu?: boolean) {
    return backend.simulateTrade(strategy, symbol, action, useGpu);
  }

  // Portfolio operations
  async getPortfolioStatus(includeAnalytics?: boolean) {
    return backend.getPortfolioStatus(includeAnalytics);
  }

  async portfolioRebalance(targetAllocations: string, currentPortfolio?: string) {
    return backend.portfolioRebalance(targetAllocations, currentPortfolio);
  }

  async riskAnalysis(portfolio: string, useGpu?: boolean) {
    return backend.riskAnalysis(portfolio, useGpu);
  }

  // Backtesting
  async runBacktest(
    strategy: string,
    symbol: string,
    startDate: string,
    endDate: string,
    useGpu?: boolean
  ) {
    return backend.runBacktest(strategy, symbol, startDate, endDate, useGpu);
  }

  async optimizeStrategy(
    strategy: string,
    symbol: string,
    parameterRanges: string,
    useGpu?: boolean
  ) {
    return backend.optimizeStrategy(strategy, symbol, parameterRanges, useGpu);
  }

  // Strategies
  async listStrategies() {
    return backend.listStrategies();
  }

  async getStrategyInfo(strategy: string) {
    return backend.getStrategyInfo(strategy);
  }

  // System operations
  getSystemInfo() {
    return backend.getSystemInfo();
  }

  async healthCheck() {
    return backend.healthCheck();
  }

  // Security operations
  createApiKey(username: string, role: string, rateLimit?: number, expiresInDays?: number) {
    return backend.createApiKey(username, role, rateLimit, expiresInDays);
  }

  validateApiKey(apiKey: string) {
    return backend.validateApiKey(apiKey);
  }

  checkAuthorization(apiKey: string, operation: string, requiredRole: string) {
    return backend.checkAuthorization(apiKey, operation, requiredRole);
  }

  checkRateLimit(identifier: string, tokens?: number) {
    return backend.checkRateLimit(identifier, tokens);
  }

  getRateLimitStats(identifier: string) {
    return backend.getRateLimitStats(identifier);
  }

  logAuditEvent(
    level: string,
    category: string,
    action: string,
    outcome: string,
    userId?: string,
    username?: string,
    ipAddress?: string,
    resource?: string,
    details?: string
  ) {
    return backend.logAuditEvent(
      level,
      category,
      action,
      outcome,
      userId,
      username,
      ipAddress,
      resource,
      details
    );
  }

  sanitizeInput(input: string) {
    return backend.sanitizeInput(input);
  }

  checkSecurityThreats(input: string) {
    return backend.checkSecurityThreats(input);
  }

  validateTradingParams(symbol: string, quantity: number, price?: number) {
    return backend.validateTradingParams(symbol, quantity, price);
  }
}
```

## Guards and Interceptors

### API Key Guard

```typescript
// src/guards/api-key.guard.ts
import { Injectable, CanActivate, ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { NeuralTraderService } from '../neural-trader/neural-trader.service';

@Injectable()
export class ApiKeyGuard implements CanActivate {
  constructor(private readonly neuralTrader: NeuralTraderService) {}

  canActivate(context: ExecutionContext): boolean {
    const request = context.switchToHttp().getRequest();
    const apiKey = request.headers['x-api-key'];

    if (!apiKey) {
      this.neuralTrader.logAuditEvent(
        'Warning',
        'Authentication',
        'missing_api_key',
        'failure',
        null,
        null,
        request.ip
      );
      throw new UnauthorizedException('API key required');
    }

    try {
      const user = this.neuralTrader.validateApiKey(apiKey);
      request.user = user;

      this.neuralTrader.logAuditEvent(
        'Info',
        'Authentication',
        'api_key_validated',
        'success',
        user.userId,
        user.username,
        request.ip
      );

      return true;
    } catch (error) {
      this.neuralTrader.logAuditEvent(
        'Security',
        'Authentication',
        'invalid_api_key',
        'failure',
        null,
        null,
        request.ip
      );
      throw new UnauthorizedException('Invalid API key');
    }
  }
}
```

### Rate Limit Guard

```typescript
// src/guards/rate-limit.guard.ts
import { Injectable, CanActivate, ExecutionContext, HttpException, HttpStatus } from '@nestjs/common';
import { NeuralTraderService } from '../neural-trader/neural-trader.service';

@Injectable()
export class RateLimitGuard implements CanActivate {
  constructor(private readonly neuralTrader: NeuralTraderService) {}

  canActivate(context: ExecutionContext): boolean {
    const request = context.switchToHttp().getRequest();
    const identifier = request.user?.userId || request.ip;

    if (!this.neuralTrader.checkRateLimit(identifier, 1)) {
      this.neuralTrader.logAuditEvent(
        'Warning',
        'Security',
        'rate_limit_exceeded',
        'failure',
        request.user?.userId,
        request.user?.username,
        request.ip
      );

      const stats = this.neuralTrader.getRateLimitStats(identifier);
      throw new HttpException(
        {
          statusCode: HttpStatus.TOO_MANY_REQUESTS,
          message: 'Rate limit exceeded',
          retryAfter: Math.ceil((60 - (stats.totalRequests % 60)) / stats.refillRate),
          stats,
        },
        HttpStatus.TOO_MANY_REQUESTS
      );
    }

    return true;
  }
}
```

### Audit Logging Interceptor

```typescript
// src/interceptors/audit-logging.interceptor.ts
import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
  Logger,
} from '@nestjs/common';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import { NeuralTraderService } from '../neural-trader/neural-trader.service';

@Injectable()
export class AuditLoggingInterceptor implements NestInterceptor {
  private readonly logger = new Logger(AuditLoggingInterceptor.name);

  constructor(private readonly neuralTrader: NeuralTraderService) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const request = context.switchToHttp().getRequest();
    const { method, url, user, ip } = request;
    const startTime = Date.now();

    return next.handle().pipe(
      tap({
        next: (data) => {
          const responseTime = Date.now() - startTime;

          this.neuralTrader.logAuditEvent(
            'Info',
            'DataAccess',
            `${method} ${url}`,
            'success',
            user?.userId,
            user?.username,
            ip,
            url,
            JSON.stringify({ responseTime, statusCode: 200 })
          );
        },
        error: (error) => {
          const responseTime = Date.now() - startTime;

          this.neuralTrader.logAuditEvent(
            'Error',
            'DataAccess',
            `${method} ${url}`,
            'failure',
            user?.userId,
            user?.username,
            ip,
            url,
            JSON.stringify({
              responseTime,
              error: error.message,
              statusCode: error.status || 500,
            })
          );
        },
      })
    );
  }
}
```

### Input Sanitization Pipe

```typescript
// src/pipes/sanitization.pipe.ts
import { PipeTransform, Injectable, ArgumentMetadata, BadRequestException } from '@nestjs/common';
import { NeuralTraderService } from '../neural-trader/neural-trader.service';

@Injectable()
export class SanitizationPipe implements PipeTransform {
  constructor(private readonly neuralTrader: NeuralTraderService) {}

  transform(value: any, metadata: ArgumentMetadata) {
    if (typeof value === 'object' && value !== null) {
      return this.sanitizeObject(value);
    }

    if (typeof value === 'string') {
      return this.sanitizeString(value);
    }

    return value;
  }

  private sanitizeObject(obj: any): any {
    const sanitized: any = {};

    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === 'string') {
        sanitized[key] = this.sanitizeString(value);
      } else if (typeof value === 'object' && value !== null) {
        sanitized[key] = this.sanitizeObject(value);
      } else {
        sanitized[key] = value;
      }
    }

    return sanitized;
  }

  private sanitizeString(input: string): string {
    const sanitized = this.neuralTrader.sanitizeInput(input);
    const threats = this.neuralTrader.checkSecurityThreats(sanitized);

    if (threats.length > 0) {
      throw new BadRequestException({
        message: 'Security threat detected in input',
        threats,
      });
    }

    return sanitized;
  }
}
```

## Controllers

### Trading Controller

```typescript
// src/neural-trader/controllers/trading.controller.ts
import {
  Controller,
  Get,
  Post,
  Body,
  Param,
  Query,
  UseGuards,
  UseInterceptors,
  UsePipes,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { NeuralTraderService } from '../neural-trader.service';
import { ApiKeyGuard } from '../../guards/api-key.guard';
import { RateLimitGuard } from '../../guards/rate-limit.guard';
import { AuditLoggingInterceptor } from '../../interceptors/audit-logging.interceptor';
import { SanitizationPipe } from '../../pipes/sanitization.pipe';

@ApiTags('trading')
@Controller('api/trading')
@UseGuards(ApiKeyGuard, RateLimitGuard)
@UseInterceptors(AuditLoggingInterceptor)
export class TradingController {
  constructor(private readonly neuralTrader: NeuralTraderService) {}

  @Get('strategies')
  @ApiOperation({ summary: 'List all available trading strategies' })
  async listStrategies() {
    return this.neuralTrader.listStrategies();
  }

  @Get('strategies/:name')
  @ApiOperation({ summary: 'Get detailed strategy information' })
  async getStrategyInfo(@Param('name') name: string) {
    return this.neuralTrader.getStrategyInfo(name);
  }

  @Get('analysis/:symbol')
  @ApiOperation({ summary: 'Quick market analysis for a symbol' })
  async quickAnalysis(
    @Param('symbol') symbol: string,
    @Query('useGpu') useGpu?: boolean
  ) {
    return this.neuralTrader.quickAnalysis(symbol, useGpu);
  }

  @Post('simulate')
  @UsePipes(SanitizationPipe)
  @ApiOperation({ summary: 'Simulate a trade operation' })
  async simulateTrade(
    @Body() body: {
      strategy: string;
      symbol: string;
      action: string;
      useGpu?: boolean;
    }
  ) {
    return this.neuralTrader.simulateTrade(
      body.strategy,
      body.symbol,
      body.action,
      body.useGpu
    );
  }

  @Post('execute')
  @UsePipes(SanitizationPipe)
  @ApiOperation({ summary: 'Execute a live trade' })
  async executeTrade(
    @Body() body: {
      strategy: string;
      symbol: string;
      action: string;
      quantity: number;
      orderType?: string;
      limitPrice?: number;
    }
  ) {
    // Validate trading parameters
    if (!this.neuralTrader.validateTradingParams(body.symbol, body.quantity, body.limitPrice)) {
      throw new BadRequestException('Invalid trading parameters');
    }

    return this.neuralTrader.executeTrade(
      body.strategy,
      body.symbol,
      body.action,
      body.quantity,
      body.orderType,
      body.limitPrice
    );
  }
}
```

### Portfolio Controller

```typescript
// src/neural-trader/controllers/portfolio.controller.ts
import { Controller, Get, Post, Body, Query, UseGuards } from '@nestjs/common';
import { ApiTags, ApiOperation } from '@nestjs/swagger';
import { NeuralTraderService } from '../neural-trader.service';
import { ApiKeyGuard } from '../../guards/api-key.guard';
import { RateLimitGuard } from '../../guards/rate-limit.guard';

@ApiTags('portfolio')
@Controller('api/portfolio')
@UseGuards(ApiKeyGuard, RateLimitGuard)
export class PortfolioController {
  constructor(private readonly neuralTrader: NeuralTraderService) {}

  @Get('status')
  @ApiOperation({ summary: 'Get current portfolio status' })
  async getStatus(@Query('includeAnalytics') includeAnalytics?: boolean) {
    return this.neuralTrader.getPortfolioStatus(includeAnalytics);
  }

  @Post('rebalance')
  @ApiOperation({ summary: 'Calculate portfolio rebalancing' })
  async rebalance(
    @Body() body: {
      targetAllocations: string;
      currentPortfolio?: string;
    }
  ) {
    return this.neuralTrader.portfolioRebalance(
      body.targetAllocations,
      body.currentPortfolio
    );
  }

  @Post('risk-analysis')
  @ApiOperation({ summary: 'Comprehensive portfolio risk analysis' })
  async riskAnalysis(
    @Body() body: {
      portfolio: string;
      useGpu?: boolean;
    }
  ) {
    return this.neuralTrader.riskAnalysis(body.portfolio, body.useGpu);
  }
}
```

## DTOs and Validation

```typescript
// src/neural-trader/dto/execute-trade.dto.ts
import { IsString, IsNumber, IsEnum, IsOptional, Min } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export enum TradeAction {
  BUY = 'buy',
  SELL = 'sell',
}

export enum OrderType {
  MARKET = 'market',
  LIMIT = 'limit',
  STOP = 'stop',
  STOP_LIMIT = 'stop_limit',
}

export class ExecuteTradeDto {
  @ApiProperty({ description: 'Trading strategy name' })
  @IsString()
  strategy: string;

  @ApiProperty({ description: 'Trading symbol' })
  @IsString()
  symbol: string;

  @ApiProperty({ description: 'Trade action', enum: TradeAction })
  @IsEnum(TradeAction)
  action: TradeAction;

  @ApiProperty({ description: 'Quantity to trade', minimum: 1 })
  @IsNumber()
  @Min(1)
  quantity: number;

  @ApiProperty({ description: 'Order type', enum: OrderType, required: false })
  @IsOptional()
  @IsEnum(OrderType)
  orderType?: OrderType;

  @ApiProperty({ description: 'Limit price for limit orders', required: false })
  @IsOptional()
  @IsNumber()
  @Min(0)
  limitPrice?: number;
}
```

## Main Application

```typescript
// src/main.ts
import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  // Global validation pipe
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
    })
  );

  // Swagger documentation
  const config = new DocumentBuilder()
    .setTitle('Neural Trader API')
    .setDescription('High-performance algorithmic trading API')
    .setVersion('2.1.1')
    .addApiKey({ type: 'apiKey', name: 'x-api-key', in: 'header' }, 'api-key')
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('api/docs', app, document);

  // Enable CORS
  app.enableCors();

  const port = process.env.PORT || 3000;
  await app.listen(port);

  console.log(`Application is running on: http://localhost:${port}`);
  console.log(`Swagger docs available at: http://localhost:${port}/api/docs`);
}

bootstrap();
```

## Testing

```typescript
// src/neural-trader/neural-trader.service.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { NeuralTraderService } from './neural-trader.service';

describe('NeuralTraderService', () => {
  let service: NeuralTraderService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        NeuralTraderService,
        {
          provide: ConfigService,
          useValue: {
            get: jest.fn((key: string, defaultValue?: any) => {
              const config = {
                JWT_SECRET: 'test-secret-key',
                RATE_LIMIT_MAX_REQUESTS: 100,
                LOG_LEVEL: 'info',
              };
              return config[key] || defaultValue;
            }),
          },
        },
      ],
    }).compile();

    service = module.get<NeuralTraderService>(NeuralTraderService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('should return system info', () => {
    const info = service.getSystemInfo();
    expect(info).toHaveProperty('version');
    expect(info).toHaveProperty('features');
  });
});
```

## Environment Configuration

```typescript
// src/config/configuration.ts
export default () => ({
  port: parseInt(process.env.PORT, 10) || 3000,
  jwtSecret: process.env.JWT_SECRET,
  rateLimit: {
    maxRequests: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS, 10) || 100,
    burstSize: parseInt(process.env.RATE_LIMIT_BURST_SIZE, 10) || 20,
    windowSecs: parseInt(process.env.RATE_LIMIT_WINDOW_SECS, 10) || 60,
  },
  cors: {
    allowedOrigins: (process.env.CORS_ALLOWED_ORIGINS || '*').split(','),
  },
  security: {
    requireHttps: process.env.REQUIRE_HTTPS === 'true',
  },
  neuralTrader: {
    logLevel: process.env.LOG_LEVEL || 'info',
    enableGpu: process.env.ENABLE_GPU === 'true',
  },
});
```

## Deployment

### Docker

```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:20-alpine

WORKDIR /app

COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./

EXPOSE 3000

CMD ["node", "dist/main"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-trader-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neural-trader-api
  template:
    metadata:
      labels:
        app: neural-trader-api
    spec:
      containers:
      - name: api
        image: neural-trader-api:latest
        ports:
        - containerPort: 3000
        env:
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: neural-trader-secrets
              key: jwt-secret
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "1Gi"
            cpu: "500m"
```

## Best Practices

1. Use dependency injection throughout the application
2. Implement proper error handling with exception filters
3. Use DTOs for request/response validation
4. Enable API documentation with Swagger
5. Implement comprehensive testing (unit, integration, e2e)
6. Use configuration modules for environment management
7. Implement health checks for Kubernetes/cloud deployments
8. Use interceptors for cross-cutting concerns
9. Implement proper logging and monitoring
10. Follow NestJS architectural patterns consistently
