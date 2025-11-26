/**
 * NestJS Integration Example
 *
 * Complete NestJS module with neural-trader backend
 * Features:
 * - Dependency injection
 * - Guards and interceptors
 * - DTOs and validation
 * - Swagger documentation
 * - Exception filters
 * - Configuration management
 */

import {
  Module,
  Global,
  Injectable,
  OnModuleInit,
  OnModuleDestroy,
  Logger,
  Controller,
  Get,
  Post,
  Body,
  Param,
  Query,
  UseGuards,
  UseInterceptors,
  UsePipes,
  CanActivate,
  ExecutionContext,
  NestInterceptor,
  CallHandler,
  PipeTransform,
  ArgumentMetadata,
  Injectable as InjectableDecorator,
  UnauthorizedException,
  BadRequestException,
  HttpException,
  HttpStatus,
} from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ApiTags, ApiOperation, ApiResponse, ApiHeader } from '@nestjs/swagger';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import {
  IsString,
  IsNumber,
  IsEnum,
  IsOptional,
  IsBoolean,
  Min,
  Max,
  IsDateString,
} from 'class-validator';
import * as backend from '@neural-trader/backend';

// ========================================
// DTOs
// ========================================

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

export class QuickAnalysisDto {
  @IsString()
  symbol: string;

  @IsOptional()
  @IsBoolean()
  useGpu?: boolean;
}

export class SimulateTradeDto {
  @IsString()
  strategy: string;

  @IsString()
  symbol: string;

  @IsEnum(TradeAction)
  action: TradeAction;

  @IsOptional()
  @IsBoolean()
  useGpu?: boolean;
}

export class ExecuteTradeDto {
  @IsString()
  strategy: string;

  @IsString()
  symbol: string;

  @IsEnum(TradeAction)
  action: TradeAction;

  @IsNumber()
  @Min(1)
  quantity: number;

  @IsOptional()
  @IsEnum(OrderType)
  orderType?: OrderType;

  @IsOptional()
  @IsNumber()
  @Min(0)
  limitPrice?: number;
}

export class BacktestDto {
  @IsString()
  strategy: string;

  @IsString()
  symbol: string;

  @IsDateString()
  startDate: string;

  @IsDateString()
  endDate: string;

  @IsOptional()
  @IsBoolean()
  useGpu?: boolean;
}

export class RiskAnalysisDto {
  @IsString()
  portfolio: string;

  @IsOptional()
  @IsBoolean()
  useGpu?: boolean;
}

export class SwarmInitDto {
  @IsEnum(['mesh', 'hierarchical', 'ring', 'star'])
  topology: string;

  config: {
    maxAgents?: number;
    distributionStrategy?: string;
    enableGpu?: boolean;
    autoScaling?: boolean;
    minAgents?: number;
    maxMemoryMb?: number;
    timeoutSecs?: number;
  };
}

// ========================================
// Core Service
// ========================================

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
      this.logger.log('✓ Authentication initialized');

      // Initialize rate limiter
      backend.initRateLimiter({
        maxRequestsPerMinute: this.configService.get('RATE_LIMIT_MAX_REQUESTS', 100),
        burstSize: this.configService.get('RATE_LIMIT_BURST_SIZE', 20),
        windowDurationSecs: this.configService.get('RATE_LIMIT_WINDOW_SECS', 60),
      });
      this.logger.log('✓ Rate limiter initialized');

      // Initialize audit logger
      backend.initAuditLogger(
        this.configService.get('AUDIT_MAX_EVENTS', 10000),
        this.configService.get('AUDIT_LOG_CONSOLE', true),
        this.configService.get('AUDIT_LOG_FILE', true)
      );
      this.logger.log('✓ Audit logger initialized');

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
      this.logger.log('✓ Security config initialized');

      // Initialize neural trader
      await backend.initNeuralTrader(
        JSON.stringify({
          logLevel: this.configService.get('LOG_LEVEL', 'info'),
          enableGpu: this.configService.get('ENABLE_GPU', false),
        })
      );
      this.logger.log('✓ Neural Trader initialized');

      this.initialized = true;

      const systemInfo = backend.getSystemInfo();
      this.logger.log(`🚀 Neural Trader v${systemInfo.version} ready`);
    } catch (error) {
      this.logger.error('Failed to initialize Neural Trader', error);
      throw error;
    }
  }

  private async shutdown() {
    if (!this.initialized) return;

    try {
      await backend.shutdown();
      this.logger.log('✓ Neural Trader shut down gracefully');
    } catch (error) {
      this.logger.error('Error during shutdown', error);
    }
  }

  // Trading operations
  async quickAnalysis(symbol: string, useGpu?: boolean) {
    return backend.quickAnalysis(symbol, useGpu);
  }

  async simulateTrade(strategy: string, symbol: string, action: string, useGpu?: boolean) {
    return backend.simulateTrade(strategy, symbol, action, useGpu);
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

  // Portfolio operations
  async getPortfolioStatus(includeAnalytics?: boolean) {
    return backend.getPortfolioStatus(includeAnalytics);
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

  // Strategies
  async listStrategies() {
    return backend.listStrategies();
  }

  async getStrategyInfo(strategy: string) {
    return backend.getStrategyInfo(strategy);
  }

  // Swarm operations
  async initE2bSwarm(topology: string, config: string) {
    return backend.initE2bSwarm(topology, config);
  }

  async getSwarmStatus(swarmId?: string) {
    return backend.getSwarmStatus(swarmId);
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

// ========================================
// Guards
// ========================================

@Injectable()
export class ApiKeyGuard implements CanActivate {
  private readonly logger = new Logger(ApiKeyGuard.name);

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

@Injectable()
export class RateLimitGuard implements CanActivate {
  private readonly logger = new Logger(RateLimitGuard.name);

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

// ========================================
// Interceptors
// ========================================

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
        next: () => {
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

// ========================================
// Pipes
// ========================================

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

// ========================================
// Controllers
// ========================================

@ApiTags('system')
@Controller('api/system')
export class SystemController {
  constructor(private readonly neuralTrader: NeuralTraderService) {}

  @Get('health')
  @ApiOperation({ summary: 'Health check endpoint' })
  @ApiResponse({ status: 200, description: 'System is healthy' })
  async healthCheck() {
    return this.neuralTrader.healthCheck();
  }

  @Get('info')
  @ApiOperation({ summary: 'Get system information' })
  @ApiResponse({ status: 200, description: 'System information retrieved' })
  getSystemInfo() {
    return this.neuralTrader.getSystemInfo();
  }
}

@ApiTags('trading')
@Controller('api/trading')
@UseGuards(ApiKeyGuard, RateLimitGuard)
@UseInterceptors(AuditLoggingInterceptor)
@ApiHeader({ name: 'x-api-key', description: 'API Key', required: true })
export class TradingController {
  constructor(private readonly neuralTrader: NeuralTraderService) {}

  @Get('strategies')
  @ApiOperation({ summary: 'List all available strategies' })
  async listStrategies() {
    const strategies = await this.neuralTrader.listStrategies();
    return { strategies };
  }

  @Get('strategies/:name')
  @ApiOperation({ summary: 'Get strategy information' })
  async getStrategyInfo(@Param('name') name: string) {
    const info = await this.neuralTrader.getStrategyInfo(name);
    return { strategy: name, info: JSON.parse(info) };
  }

  @Get('analysis/:symbol')
  @ApiOperation({ summary: 'Quick market analysis' })
  async quickAnalysis(
    @Param('symbol') symbol: string,
    @Query('useGpu') useGpu?: string
  ) {
    const analysis = await this.neuralTrader.quickAnalysis(symbol, useGpu === 'true');
    return {
      symbol,
      analysis,
      timestamp: new Date().toISOString(),
    };
  }

  @Post('simulate')
  @UsePipes(SanitizationPipe)
  @ApiOperation({ summary: 'Simulate trade operation' })
  async simulateTrade(@Body() dto: SimulateTradeDto) {
    const result = await this.neuralTrader.simulateTrade(
      dto.strategy,
      dto.symbol,
      dto.action,
      dto.useGpu
    );

    return {
      simulation: result,
      timestamp: new Date().toISOString(),
    };
  }

  @Post('execute')
  @UsePipes(SanitizationPipe)
  @ApiOperation({ summary: 'Execute live trade' })
  async executeTrade(@Body() dto: ExecuteTradeDto) {
    if (!this.neuralTrader.validateTradingParams(dto.symbol, dto.quantity, dto.limitPrice)) {
      throw new BadRequestException('Invalid trading parameters');
    }

    const result = await this.neuralTrader.executeTrade(
      dto.strategy,
      dto.symbol,
      dto.action,
      dto.quantity,
      dto.orderType,
      dto.limitPrice
    );

    return {
      trade: result,
      timestamp: new Date().toISOString(),
    };
  }
}

@ApiTags('portfolio')
@Controller('api/portfolio')
@UseGuards(ApiKeyGuard, RateLimitGuard)
@UseInterceptors(AuditLoggingInterceptor)
@ApiHeader({ name: 'x-api-key', description: 'API Key', required: true })
export class PortfolioController {
  constructor(private readonly neuralTrader: NeuralTraderService) {}

  @Get()
  @ApiOperation({ summary: 'Get portfolio status' })
  async getStatus(@Query('includeAnalytics') includeAnalytics?: string) {
    const status = await this.neuralTrader.getPortfolioStatus(
      includeAnalytics === 'true'
    );

    return {
      portfolio: status,
      timestamp: new Date().toISOString(),
    };
  }

  @Post('risk')
  @UsePipes(SanitizationPipe)
  @ApiOperation({ summary: 'Portfolio risk analysis' })
  async riskAnalysis(@Body() dto: RiskAnalysisDto) {
    const analysis = await this.neuralTrader.riskAnalysis(dto.portfolio, dto.useGpu);

    return {
      riskAnalysis: analysis,
      timestamp: new Date().toISOString(),
    };
  }
}

@ApiTags('backtest')
@Controller('api/backtest')
@UseGuards(ApiKeyGuard, RateLimitGuard)
@UseInterceptors(AuditLoggingInterceptor)
@ApiHeader({ name: 'x-api-key', description: 'API Key', required: true })
export class BacktestController {
  constructor(private readonly neuralTrader: NeuralTraderService) {}

  @Post()
  @UsePipes(SanitizationPipe)
  @ApiOperation({ summary: 'Run strategy backtest' })
  async runBacktest(@Body() dto: BacktestDto) {
    const result = await this.neuralTrader.runBacktest(
      dto.strategy,
      dto.symbol,
      dto.startDate,
      dto.endDate,
      dto.useGpu
    );

    return {
      backtest: result,
      timestamp: new Date().toISOString(),
    };
  }
}

@ApiTags('swarm')
@Controller('api/swarm')
@UseGuards(ApiKeyGuard, RateLimitGuard)
@UseInterceptors(AuditLoggingInterceptor)
@ApiHeader({ name: 'x-api-key', description: 'API Key', required: true })
export class SwarmController {
  constructor(private readonly neuralTrader: NeuralTraderService) {}

  @Post('init')
  @ApiOperation({ summary: 'Initialize E2B trading swarm' })
  async initSwarm(@Body() dto: SwarmInitDto) {
    const swarm = await this.neuralTrader.initE2bSwarm(
      dto.topology,
      JSON.stringify(dto.config)
    );

    return {
      swarm,
      timestamp: new Date().toISOString(),
    };
  }

  @Get(':swarmId/status')
  @ApiOperation({ summary: 'Get swarm status' })
  async getStatus(@Param('swarmId') swarmId: string) {
    const status = await this.neuralTrader.getSwarmStatus(swarmId);

    return {
      swarm: status,
      timestamp: new Date().toISOString(),
    };
  }
}

// ========================================
// Module Definition
// ========================================

@Global()
@Module({
  providers: [
    NeuralTraderService,
    ApiKeyGuard,
    RateLimitGuard,
    AuditLoggingInterceptor,
    SanitizationPipe,
  ],
  controllers: [
    SystemController,
    TradingController,
    PortfolioController,
    BacktestController,
    SwarmController,
  ],
  exports: [NeuralTraderService],
})
export class NeuralTraderModule {}

// Export all components
export {
  // Service
  NeuralTraderService,
  // Guards
  ApiKeyGuard,
  RateLimitGuard,
  // Interceptors
  AuditLoggingInterceptor,
  // Pipes
  SanitizationPipe,
  // Controllers
  SystemController,
  TradingController,
  PortfolioController,
  BacktestController,
  SwarmController,
  // DTOs
  QuickAnalysisDto,
  SimulateTradeDto,
  ExecuteTradeDto,
  BacktestDto,
  RiskAnalysisDto,
  SwarmInitDto,
};
