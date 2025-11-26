# MCP Best Practices and Security Guide
## For AI News Trading Platform

### Table of Contents

1. [Security Architecture](#security-architecture)
2. [Authentication and Authorization](#authentication-and-authorization)
3. [Input Validation and Sanitization](#input-validation-and-sanitization)
4. [Rate Limiting and Resource Management](#rate-limiting-and-resource-management)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Production Deployment Best Practices](#production-deployment-best-practices)
7. [Security Incident Response](#security-incident-response)
8. [Compliance and Regulatory Considerations](#compliance-and-regulatory-considerations)

## Security Architecture

### Defense in Depth Strategy

The AI News Trading platform implements multiple security layers:

```
┌─────────────────────────────────────────────────────────────┐
│                   External Firewall                           │
├─────────────────────────────────────────────────────────────┤
│                    WAF (Web Application Firewall)             │
├─────────────────────────────────────────────────────────────┤
│              API Gateway (Rate Limiting, Auth)                │
├─────────────────────────────────────────────────────────────┤
│                    MCP Security Layer                         │
│  ┌─────────────┬──────────────┬──────────────┬────────────┐ │
│  │   OAuth 2.1 │ Input Valid. │ Audit Logs   │ Encryption │ │
│  └─────────────┴──────────────┴──────────────┴────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    MCP Server Layer                           │
│  ┌─────────────┬──────────────┬──────────────┬────────────┐ │
│  │News Analysis│ Market Data  │Trade Executor│Risk Manager│ │
│  └─────────────┴──────────────┴──────────────┴────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                 Internal Security Controls                    │
│  ┌─────────────┬──────────────┬──────────────┬────────────┐ │
│  │   Network   │   Database   │     Key      │   Service  │ │
│  │ Segmentation│  Encryption  │  Management  │    Mesh    │ │
│  └─────────────┴──────────────┴──────────────┴────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Security Principles

1. **Zero Trust Architecture**: Never trust, always verify
2. **Least Privilege Access**: Minimal permissions for each component
3. **Defense in Depth**: Multiple security layers
4. **Fail Secure**: Default to secure state on failure
5. **Audit Everything**: Comprehensive logging and monitoring

## Authentication and Authorization

### OAuth 2.1 Implementation

```python
# src/security/oauth_manager.py
from authlib.integrations.flask_oauth2 import ResourceProtector
from authlib.oauth2.rfc7662 import IntrospectTokenValidator
import time

class MCPOAuthManager:
    def __init__(self, introspection_endpoint: str, client_credentials: dict):
        self.introspection_endpoint = introspection_endpoint
        self.client_credentials = client_credentials
        self.token_cache = TTLCache(maxsize=10000, ttl=300)  # 5-minute cache
        
    async def validate_token(self, token: str) -> dict:
        """Validate OAuth token with caching"""
        # Check cache first
        cached = self.token_cache.get(token)
        if cached:
            if cached['exp'] > time.time():
                return cached
            else:
                del self.token_cache[token]
        
        # Introspect token
        response = await self._introspect_token(token)
        
        if response['active']:
            # Cache valid token
            self.token_cache[token] = response
            return response
        else:
            raise UnauthorizedError("Invalid or expired token")
    
    async def _introspect_token(self, token: str) -> dict:
        """Call OAuth introspection endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.introspection_endpoint,
                data={'token': token},
                auth=aiohttp.BasicAuth(
                    self.client_credentials['client_id'],
                    self.client_credentials['client_secret']
                )
            ) as response:
                return await response.json()

# MCP Server integration
class SecureMCPServer(Server):
    def __init__(self, oauth_manager: MCPOAuthManager):
        super().__init__("secure-trading-server")
        self.oauth = oauth_manager
        
    async def _validate_request(self, context):
        """Validate every incoming request"""
        auth_header = context.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            raise UnauthorizedError("Missing Bearer token")
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        token_info = await self.oauth.validate_token(token)
        
        # Store user info in context
        context.user = {
            'id': token_info['sub'],
            'permissions': token_info.get('scope', '').split(),
            'email': token_info.get('email'),
            'roles': token_info.get('roles', [])
        }
        
        return token_info
```

### Permission-Based Access Control

```python
# src/security/permissions.py
from enum import Enum
from functools import wraps

class TradingPermissions(Enum):
    # Read permissions
    READ_MARKET_DATA = "trading:market:read"
    READ_NEWS = "trading:news:read"
    READ_POSITIONS = "trading:positions:read"
    
    # Write permissions
    EXECUTE_TRADES = "trading:orders:execute"
    MODIFY_ORDERS = "trading:orders:modify"
    CANCEL_ORDERS = "trading:orders:cancel"
    
    # Admin permissions
    MANAGE_RISK_LIMITS = "trading:risk:manage"
    VIEW_ALL_ACCOUNTS = "trading:accounts:view_all"
    SYSTEM_ADMIN = "trading:system:admin"

def require_permission(permission: TradingPermissions):
    """Decorator to enforce permission requirements"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, context, *args, **kwargs):
            user_permissions = context.user.get('permissions', [])
            
            if permission.value not in user_permissions:
                # Log unauthorized attempt
                await self.audit_logger.log_unauthorized_access(
                    user_id=context.user.get('id'),
                    requested_permission=permission.value,
                    method=func.__name__,
                    timestamp=datetime.utcnow()
                )
                
                raise ForbiddenError(
                    f"Insufficient permissions. Required: {permission.value}"
                )
            
            return await func(self, context, *args, **kwargs)
        return wrapper
    return decorator

# Usage in MCP tools
class TradingMCPServer(SecureMCPServer):
    @Tool("execute_market_order")
    @require_permission(TradingPermissions.EXECUTE_TRADES)
    async def execute_market_order(self, context, params):
        # User has been authenticated and authorized
        return await self._execute_order(params)
```

### API Key Management

```python
# src/security/api_key_manager.py
import secrets
import hashlib
from datetime import datetime, timedelta

class APIKeyManager:
    def __init__(self, db_connection):
        self.db = db_connection
        
    async def create_api_key(self, user_id: str, name: str, permissions: List[str], 
                            expires_in_days: int = 90) -> dict:
        """Create a new API key with specific permissions"""
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Store in database
        api_key_record = {
            'key_hash': key_hash,
            'user_id': user_id,
            'name': name,
            'permissions': permissions,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(days=expires_in_days),
            'last_used': None,
            'is_active': True
        }
        
        await self.db.api_keys.insert_one(api_key_record)
        
        # Return key only once
        return {
            'api_key': raw_key,
            'key_id': str(api_key_record['_id']),
            'expires_at': api_key_record['expires_at']
        }
    
    async def validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate API key and return permissions"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        record = await self.db.api_keys.find_one({
            'key_hash': key_hash,
            'is_active': True,
            'expires_at': {'$gt': datetime.utcnow()}
        })
        
        if record:
            # Update last used
            await self.db.api_keys.update_one(
                {'_id': record['_id']},
                {'$set': {'last_used': datetime.utcnow()}}
            )
            
            return {
                'user_id': record['user_id'],
                'permissions': record['permissions'],
                'key_id': str(record['_id'])
            }
        
        return None
```

## Input Validation and Sanitization

### Comprehensive Input Validation

```python
# src/security/input_validation.py
from pydantic import BaseModel, validator, Field
from typing import List, Optional, Union
import re

class OrderParameters(BaseModel):
    """Strict validation for trading orders"""
    symbol: str = Field(..., regex=r'^[A-Z]{1,5}$', description="Stock symbol")
    side: str = Field(..., regex=r'^(BUY|SELL)$')
    quantity: int = Field(..., gt=0, le=100000)
    order_type: str = Field(..., regex=r'^(MARKET|LIMIT|STOP|STOP_LIMIT)$')
    limit_price: Optional[float] = Field(None, gt=0, le=1000000)
    stop_price: Optional[float] = Field(None, gt=0, le=1000000)
    
    @validator('symbol')
    def validate_symbol(cls, v):
        # Additional validation against allowed symbols
        if v not in ALLOWED_SYMBOLS:
            raise ValueError(f"Symbol {v} not in allowed list")
        return v
    
    @validator('limit_price')
    def validate_limit_price(cls, v, values):
        if values.get('order_type') in ['LIMIT', 'STOP_LIMIT'] and v is None:
            raise ValueError("Limit price required for LIMIT orders")
        return v

class InputSanitizer:
    """Sanitize inputs to prevent injection attacks"""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000) -> str:
        """Remove potentially harmful content from text"""
        # Truncate to max length
        text = text[:max_length]
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Remove potential prompt injection patterns
        injection_patterns = [
            r'ignore previous instructions',
            r'disregard all prior',
            r'</system>',
            r'```system',
            r'SYSTEM:',
            r'Assistant:',
        ]
        
        for pattern in injection_patterns:
            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def sanitize_sql_identifier(identifier: str) -> str:
        """Sanitize SQL identifiers to prevent SQL injection"""
        # Only allow alphanumeric and underscore
        if not re.match(r'^[a-zA-Z0-9_]+$', identifier):
            raise ValueError(f"Invalid identifier: {identifier}")
        return identifier

# MCP Tool with validation
class ValidatedTradingServer(MCPServer):
    @Tool("place_order")
    async def place_order(self, params: dict) -> dict:
        # Validate with Pydantic
        try:
            validated_params = OrderParameters(**params)
        except ValidationError as e:
            return {
                "status": "REJECTED",
                "errors": e.errors()
            }
        
        # Additional business logic validation
        if validated_params.order_type == 'MARKET' and self._is_market_closed():
            return {
                "status": "REJECTED",
                "reason": "Market orders not allowed outside market hours"
            }
        
        # Proceed with validated parameters
        return await self._execute_validated_order(validated_params)
```

### Output Sanitization

```python
# src/security/output_sanitization.py
class OutputSanitizer:
    """Sanitize outputs to prevent information leakage"""
    
    @staticmethod
    def sanitize_error_message(error: Exception, debug_mode: bool = False) -> str:
        """Sanitize error messages for external consumption"""
        if debug_mode:
            # Development environment - return full error
            return str(error)
        
        # Production - return generic messages
        error_mapping = {
            ConnectionError: "Service temporarily unavailable",
            ValueError: "Invalid input provided",
            PermissionError: "Access denied",
            TimeoutError: "Request timed out"
        }
        
        for error_type, message in error_mapping.items():
            if isinstance(error, error_type):
                return message
        
        return "An error occurred processing your request"
    
    @staticmethod
    def redact_sensitive_data(data: dict) -> dict:
        """Redact sensitive information from responses"""
        sensitive_fields = [
            'password', 'token', 'api_key', 'secret',
            'ssn', 'account_number', 'routing_number'
        ]
        
        def _redact_dict(d: dict) -> dict:
            result = {}
            for key, value in d.items():
                if any(field in key.lower() for field in sensitive_fields):
                    result[key] = '[REDACTED]'
                elif isinstance(value, dict):
                    result[key] = _redact_dict(value)
                elif isinstance(value, list):
                    result[key] = [_redact_dict(v) if isinstance(v, dict) else v for v in value]
                else:
                    result[key] = value
            return result
        
        return _redact_dict(data)
```

## Rate Limiting and Resource Management

### Comprehensive Rate Limiting

```python
# src/security/rate_limiter.py
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

class HierarchicalRateLimiter:
    """Multi-level rate limiting for MCP servers"""
    
    def __init__(self):
        self.limits = {
            'global': RateLimit(1000, timedelta(minutes=1)),  # 1000 req/min globally
            'user': RateLimit(100, timedelta(minutes=1)),     # 100 req/min per user
            'tool': RateLimit(10, timedelta(seconds=1)),      # 10 req/sec per tool
            'expensive': RateLimit(5, timedelta(minutes=1))   # 5 expensive ops/min
        }
        self.buckets = defaultdict(lambda: defaultdict(list))
    
    async def check_rate_limit(self, level: str, identifier: str) -> bool:
        """Check if request is within rate limits"""
        limit = self.limits[level]
        now = datetime.utcnow()
        
        # Clean old entries
        self.buckets[level][identifier] = [
            ts for ts in self.buckets[level][identifier]
            if now - ts < limit.window
        ]
        
        # Check limit
        if len(self.buckets[level][identifier]) >= limit.max_requests:
            return False
        
        # Add current request
        self.buckets[level][identifier].append(now)
        return True
    
    async def rate_limit_decorator(self, level: str = 'user', 
                                  identifier_func=None,
                                  expensive: bool = False):
        """Decorator for rate limiting MCP tools"""
        def decorator(func):
            @wraps(func)
            async def wrapper(self, context, *args, **kwargs):
                # Get identifier
                if identifier_func:
                    identifier = identifier_func(context)
                else:
                    identifier = context.user.get('id', 'anonymous')
                
                # Check multiple levels
                checks = [
                    ('global', 'global'),
                    (level, identifier),
                ]
                
                if expensive:
                    checks.append(('expensive', identifier))
                
                for check_level, check_id in checks:
                    if not await self.rate_limiter.check_rate_limit(check_level, check_id):
                        raise RateLimitError(
                            f"Rate limit exceeded for {check_level}: {check_id}"
                        )
                
                return await func(self, context, *args, **kwargs)
            return wrapper
        return decorator

# Usage in MCP Server
class RateLimitedTradingServer(MCPServer):
    def __init__(self):
        super().__init__("trading-server")
        self.rate_limiter = HierarchicalRateLimiter()
    
    @Tool("get_quote")
    @rate_limiter.rate_limit_decorator(level='tool', identifier_func=lambda c: 'get_quote')
    async def get_quote(self, context, params):
        return await self._fetch_quote(params['symbol'])
    
    @Tool("analyze_portfolio")
    @rate_limiter.rate_limit_decorator(expensive=True)
    async def analyze_portfolio(self, context, params):
        # Expensive GPU operation
        return await self._run_portfolio_analysis()
```

### Resource Management

```python
# src/security/resource_manager.py
import resource
import psutil
from contextlib import asynccontextmanager

class ResourceManager:
    """Manage computational resources for MCP operations"""
    
    def __init__(self):
        self.active_operations = {}
        self.resource_limits = {
            'max_memory_mb': 4096,
            'max_cpu_seconds': 60,
            'max_file_handles': 100,
            'max_gpu_memory_mb': 8192
        }
    
    @asynccontextmanager
    async def limit_resources(self, operation_id: str, limits: dict = None):
        """Context manager to limit resources for an operation"""
        limits = limits or self.resource_limits
        
        # Set process limits
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, 
                          (limits['max_memory_mb'] * 1024 * 1024, hard))
        
        # Track operation
        self.active_operations[operation_id] = {
            'start_time': datetime.utcnow(),
            'pid': os.getpid(),
            'limits': limits
        }
        
        try:
            # Monitor in background
            monitor_task = asyncio.create_task(
                self._monitor_resources(operation_id)
            )
            
            yield
            
        finally:
            # Clean up
            monitor_task.cancel()
            del self.active_operations[operation_id]
            
            # Reset limits
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    
    async def _monitor_resources(self, operation_id: str):
        """Monitor resource usage and kill if exceeded"""
        while operation_id in self.active_operations:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            limits = self.active_operations[operation_id]['limits']
            
            if memory_mb > limits['max_memory_mb']:
                # Log and terminate
                await self.audit_logger.log_resource_violation(
                    operation_id=operation_id,
                    resource='memory',
                    used=memory_mb,
                    limit=limits['max_memory_mb']
                )
                process.terminate()
                raise ResourceError(f"Memory limit exceeded: {memory_mb}MB")
            
            await asyncio.sleep(1)
```

## Monitoring and Observability

### Comprehensive Logging

```python
# src/monitoring/structured_logging.py
import structlog
from pythonjsonlogger import jsonlogger
import logging.handlers

class MCPAuditLogger:
    """Structured logging for security audit trail"""
    
    def __init__(self, log_path: str):
        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
        
        # Set up secure file handler with rotation
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=100_000_000,  # 100MB
            backupCount=10,
            encoding='utf-8'
        )
        handler.setFormatter(jsonlogger.JsonFormatter())
        
    async def log_mcp_request(self, event_type: str, **kwargs):
        """Log MCP request with full context"""
        self.logger.info(
            "mcp_request",
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            **self._sanitize_log_data(kwargs)
        )
    
    async def log_security_event(self, event_type: str, severity: str, **kwargs):
        """Log security-related events"""
        log_method = getattr(self.logger, severity.lower())
        log_method(
            "security_event",
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def _sanitize_log_data(self, data: dict) -> dict:
        """Remove sensitive data from logs"""
        sensitive_keys = {'password', 'token', 'api_key', 'secret'}
        
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {
                    k: '[REDACTED]' if k.lower() in sensitive_keys else _sanitize(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [_sanitize(item) for item in obj]
            return obj
        
        return _sanitize(data)

# Integration with MCP Server
class MonitoredMCPServer(MCPServer):
    def __init__(self):
        super().__init__("monitored-server")
        self.audit_logger = MCPAuditLogger("/logs/mcp_audit.log")
        self.metrics = MetricsCollector()
    
    async def handle_tool_call(self, context, tool_name, params):
        """Wrap tool calls with monitoring"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Log request
        await self.audit_logger.log_mcp_request(
            "tool_call_start",
            request_id=request_id,
            user_id=context.user.get('id'),
            tool_name=tool_name,
            params=params
        )
        
        try:
            # Execute tool
            result = await super().handle_tool_call(context, tool_name, params)
            
            # Log success
            await self.audit_logger.log_mcp_request(
                "tool_call_success",
                request_id=request_id,
                duration_ms=(time.time() - start_time) * 1000
            )
            
            # Update metrics
            self.metrics.record_request(tool_name, "success", time.time() - start_time)
            
            return result
            
        except Exception as e:
            # Log failure
            await self.audit_logger.log_mcp_request(
                "tool_call_failure",
                request_id=request_id,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            
            # Update metrics
            self.metrics.record_request(tool_name, "failure", time.time() - start_time)
            
            raise
```

### Metrics and Monitoring

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
import prometheus_client

class MCPMetricsCollector:
    """Prometheus metrics for MCP servers"""
    
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'mcp_requests_total',
            'Total MCP requests',
            ['server', 'tool', 'status']
        )
        
        self.request_duration = Histogram(
            'mcp_request_duration_seconds',
            'MCP request duration',
            ['server', 'tool'],
            buckets=[.001, .005, .01, .05, .1, .5, 1, 5, 10]
        )
        
        # Security metrics
        self.auth_failures = Counter(
            'mcp_auth_failures_total',
            'Authentication failures',
            ['reason']
        )
        
        self.rate_limit_hits = Counter(
            'mcp_rate_limit_hits_total',
            'Rate limit violations',
            ['level', 'identifier']
        )
        
        # Resource metrics
        self.active_connections = Gauge(
            'mcp_active_connections',
            'Active MCP connections',
            ['server']
        )
        
        self.gpu_utilization = Gauge(
            'mcp_gpu_utilization_percent',
            'GPU utilization percentage',
            ['device_id', 'server']
        )
        
        # Business metrics
        self.trades_executed = Counter(
            'trades_executed_total',
            'Total trades executed',
            ['symbol', 'side', 'strategy']
        )
        
        self.trade_value = Histogram(
            'trade_value_dollars',
            'Trade value in dollars',
            ['symbol', 'side'],
            buckets=[100, 1000, 10000, 100000, 1000000]
        )
    
    def record_request(self, tool: str, status: str, duration: float):
        """Record request metrics"""
        self.request_count.labels(
            server=self.server_name,
            tool=tool,
            status=status
        ).inc()
        
        self.request_duration.labels(
            server=self.server_name,
            tool=tool
        ).observe(duration)
    
    def export_metrics(self):
        """Export metrics in Prometheus format"""
        return prometheus_client.generate_latest()
```

### Real-time Alerting

```python
# src/monitoring/alerting.py
from typing import List, Dict
import aiohttp

class SecurityAlertManager:
    """Real-time security alerting for MCP servers"""
    
    def __init__(self, webhook_urls: List[str]):
        self.webhook_urls = webhook_urls
        self.alert_rules = self._define_alert_rules()
    
    def _define_alert_rules(self) -> List[AlertRule]:
        """Define security alert rules"""
        return [
            AlertRule(
                name="multiple_auth_failures",
                condition=lambda metrics: metrics['auth_failures_1m'] > 10,
                severity="high",
                message="Multiple authentication failures detected"
            ),
            AlertRule(
                name="unusual_trade_volume",
                condition=lambda metrics: metrics['trade_volume_5m'] > 10000000,
                severity="medium",
                message="Unusual trading volume detected"
            ),
            AlertRule(
                name="gpu_memory_exhaustion",
                condition=lambda metrics: any(
                    gpu['memory_used_percent'] > 95 
                    for gpu in metrics['gpu_status'].values()
                ),
                severity="critical",
                message="GPU memory exhaustion detected"
            ),
            AlertRule(
                name="suspicious_api_pattern",
                condition=lambda metrics: self._detect_suspicious_pattern(metrics),
                severity="high",
                message="Suspicious API usage pattern detected"
            )
        ]
    
    async def check_alerts(self, metrics: Dict):
        """Check metrics against alert rules"""
        for rule in self.alert_rules:
            if rule.condition(metrics):
                await self._send_alert(rule, metrics)
    
    async def _send_alert(self, rule: AlertRule, metrics: Dict):
        """Send alert to configured webhooks"""
        alert_data = {
            "alert_name": rule.name,
            "severity": rule.severity,
            "message": rule.message,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "recommended_action": self._get_recommended_action(rule.name)
        }
        
        async with aiohttp.ClientSession() as session:
            for webhook_url in self.webhook_urls:
                try:
                    await session.post(webhook_url, json=alert_data)
                except Exception as e:
                    # Log webhook failure but don't crash
                    logger.error(f"Failed to send alert to {webhook_url}: {e}")
    
    def _detect_suspicious_pattern(self, metrics: Dict) -> bool:
        """Detect suspicious API usage patterns"""
        # Example: Rapid succession of different tool calls
        tool_calls = metrics.get('recent_tool_calls', [])
        
        if len(tool_calls) < 10:
            return False
        
        # Check for pattern indicative of automated scanning
        unique_tools = set(call['tool'] for call in tool_calls[-10:])
        time_span = tool_calls[-1]['timestamp'] - tool_calls[-10]['timestamp']
        
        # Many different tools in short time = suspicious
        return len(unique_tools) > 8 and time_span < 5  # seconds
```

## Production Deployment Best Practices

### Deployment Configuration

```yaml
# deployment/production-mcp-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mcp-security-config
data:
  security_config.yaml: |
    security:
      # TLS Configuration
      tls:
        min_version: "1.3"
        cipher_suites:
          - TLS_AES_256_GCM_SHA384
          - TLS_CHACHA20_POLY1305_SHA256
        require_client_cert: true
        client_ca_file: /certs/client-ca.crt
      
      # Authentication
      auth:
        providers:
          - type: oauth2
            issuer: https://auth.trading-platform.com
            jwks_uri: https://auth.trading-platform.com/.well-known/jwks.json
            required_scopes:
              - trading:read
              - trading:write
          - type: api_key
            header_name: X-API-Key
            validation_endpoint: https://api.trading-platform.com/v1/validate
      
      # Rate Limiting
      rate_limiting:
        global:
          requests_per_minute: 10000
          burst_size: 1000
        per_user:
          requests_per_minute: 100
          burst_size: 20
        per_ip:
          requests_per_minute: 50
          burst_size: 10
      
      # IP Restrictions
      ip_filtering:
        mode: whitelist
        allowed_ranges:
          - 10.0.0.0/8      # Internal network
          - 172.16.0.0/12   # VPN range
        geolocation:
          allowed_countries: ["US", "GB", "JP", "SG"]
          block_tor: true
          block_proxies: true
      
      # Content Security
      content_security:
        max_request_size: 1048576  # 1MB
        max_response_size: 10485760  # 10MB
        allowed_content_types:
          - application/json
          - text/plain
        sanitize_logs: true
        redact_sensitive_fields: true
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-trading-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-trading-server
  template:
    metadata:
      labels:
        app: mcp-trading-server
    spec:
      serviceAccountName: mcp-server
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: mcp-server
        image: trading-platform/mcp-server:v1.0.0
        ports:
        - containerPort: 8443
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENABLE_PROFILING
          value: "false"
        volumeMounts:
        - name: security-config
          mountPath: /config
          readOnly: true
        - name: tls-certs
          mountPath: /certs
          readOnly: true
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        livenessProbe:
          httpGet:
            path: /health
            port: 8443
            scheme: HTTPS
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8443
            scheme: HTTPS
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: security-config
        configMap:
          name: mcp-security-config
      - name: tls-certs
        secret:
          secretName: mcp-tls-certs
```

### Network Security

```yaml
# deployment/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mcp-server-network-policy
spec:
  podSelector:
    matchLabels:
      app: mcp-trading-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: api-gateway
    - podSelector:
        matchLabels:
          app: mcp-client
    ports:
    - protocol: TCP
      port: 8443
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: databases
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
  - to:
    - namespaceSelector:
        matchLabels:
          name: message-queue
    ports:
    - protocol: TCP
      port: 5672  # RabbitMQ
  - to:
    - podSelector:
        matchLabels:
          app: gpu-inference-server
    ports:
    - protocol: TCP
      port: 8001  # Triton
```

## Security Incident Response

### Incident Response Plan

```python
# src/security/incident_response.py
class SecurityIncidentHandler:
    """Automated security incident response"""
    
    def __init__(self):
        self.response_actions = {
            'auth_brute_force': self._handle_brute_force,
            'data_exfiltration': self._handle_data_exfiltration,
            'privilege_escalation': self._handle_privilege_escalation,
            'injection_attempt': self._handle_injection_attempt
        }
    
    async def handle_incident(self, incident_type: str, context: dict):
        """Handle security incident based on type"""
        handler = self.response_actions.get(incident_type)
        if handler:
            await handler(context)
        else:
            await self._handle_unknown_incident(incident_type, context)
    
    async def _handle_brute_force(self, context: dict):
        """Response to authentication brute force attempts"""
        user_id = context.get('user_id')
        ip_address = context.get('ip_address')
        
        # 1. Block IP address
        await self.firewall.block_ip(ip_address, duration=3600)  # 1 hour
        
        # 2. Lock user account
        if user_id:
            await self.user_manager.lock_account(user_id, reason="Brute force detected")
        
        # 3. Alert security team
        await self.alert_manager.send_critical_alert(
            "Brute force attack detected",
            context
        )
        
        # 4. Increase monitoring
        await self.monitoring.enable_enhanced_logging(ip_address)
    
    async def _handle_data_exfiltration(self, context: dict):
        """Response to potential data exfiltration"""
        session_id = context.get('session_id')
        
        # 1. Terminate session immediately
        await self.session_manager.terminate_session(session_id)
        
        # 2. Revoke all tokens
        await self.token_manager.revoke_user_tokens(context.get('user_id'))
        
        # 3. Capture forensic data
        await self.forensics.capture_session_data(session_id)
        
        # 4. Alert incident response team
        await self.alert_manager.page_on_call_team(
            "Potential data exfiltration detected",
            severity="CRITICAL",
            context=context
        )
```

## Compliance and Regulatory Considerations

### Compliance Framework

```python
# src/compliance/regulatory_compliance.py
class RegulatoryComplianceManager:
    """Ensure MCP operations comply with financial regulations"""
    
    def __init__(self):
        self.regulations = {
            'mifid2': MiFID2Compliance(),
            'dodd_frank': DoddFrankCompliance(),
            'gdpr': GDPRCompliance(),
            'sox': SOXCompliance()
        }
    
    async def validate_trade_compliance(self, trade_params: dict) -> ComplianceResult:
        """Validate trade against all applicable regulations"""
        results = []
        
        for reg_name, regulation in self.regulations.items():
            if regulation.applicable(trade_params):
                result = await regulation.validate(trade_params)
                results.append(result)
                
                if not result.compliant:
                    # Log non-compliance
                    await self.audit_logger.log_compliance_violation(
                        regulation=reg_name,
                        violation=result.violation_reason,
                        trade_params=trade_params
                    )
        
        return ComplianceResult(
            compliant=all(r.compliant for r in results),
            violations=[r.violation_reason for r in results if not r.compliant],
            audit_trail=self._generate_audit_trail(results)
        )
    
    def _generate_audit_trail(self, results: List[ComplianceResult]) -> dict:
        """Generate comprehensive audit trail for compliance"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'regulations_checked': list(self.regulations.keys()),
            'results': [r.to_dict() for r in results],
            'overall_compliant': all(r.compliant for r in results)
        }

class MiFID2Compliance:
    """MiFID II compliance checks"""
    
    async def validate(self, trade_params: dict) -> ComplianceResult:
        checks = []
        
        # Best execution requirement
        if trade_params.get('order_type') == 'MARKET':
            best_price = await self.get_best_execution_price(trade_params['symbol'])
            if abs(trade_params.get('expected_price', 0) - best_price) > 0.01:
                checks.append("Best execution requirement not met")
        
        # Transaction reporting
        if trade_params.get('value', 0) > 10000:
            if not trade_params.get('lei_code'):
                checks.append("LEI code required for large transactions")
        
        # Pre-trade transparency
        if trade_params.get('dark_pool', False):
            if trade_params.get('value', 0) < 100000:
                checks.append("Dark pool trading not allowed for small orders")
        
        return ComplianceResult(
            compliant=len(checks) == 0,
            violation_reason="; ".join(checks) if checks else None
        )
```

### Data Privacy and Retention

```python
# src/compliance/data_privacy.py
class DataPrivacyManager:
    """Manage data privacy and retention for MCP servers"""
    
    def __init__(self):
        self.retention_policies = {
            'trade_data': timedelta(days=2555),  # 7 years
            'audit_logs': timedelta(days=2555),  # 7 years
            'user_sessions': timedelta(days=90),
            'api_logs': timedelta(days=365),
            'metrics': timedelta(days=30)
        }
    
    async def anonymize_user_data(self, user_id: str):
        """Anonymize user data for GDPR compliance"""
        # Replace PII with anonymized values
        anonymous_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        
        updates = {
            'user_id': anonymous_id,
            'email': f"{anonymous_id}@anonymized.local",
            'name': 'REDACTED',
            'ip_address': '0.0.0.0'
        }
        
        # Update all collections
        for collection in ['trades', 'audit_logs', 'sessions']:
            await self.db[collection].update_many(
                {'user_id': user_id},
                {'$set': updates}
            )
    
    async def enforce_retention_policies(self):
        """Delete data according to retention policies"""
        for data_type, retention_period in self.retention_policies.items():
            cutoff_date = datetime.utcnow() - retention_period
            
            if data_type == 'trade_data':
                # Archive before deletion
                await self.archive_old_trades(cutoff_date)
            
            # Delete old data
            result = await self.db[data_type].delete_many({
                'created_at': {'$lt': cutoff_date}
            })
            
            await self.audit_logger.log_retention_enforcement(
                data_type=data_type,
                records_deleted=result.deleted_count,
                cutoff_date=cutoff_date
            )
```

## Security Checklist

### Pre-Production Security Checklist

- [ ] **Authentication & Authorization**
  - [ ] OAuth 2.1 implementation tested
  - [ ] API key rotation mechanism in place
  - [ ] Permission model documented and tested
  - [ ] MFA enforced for sensitive operations

- [ ] **Input/Output Security**
  - [ ] All inputs validated with Pydantic models
  - [ ] Injection attack patterns blocked
  - [ ] Output sanitization for all responses
  - [ ] Error messages don't leak sensitive info

- [ ] **Rate Limiting & Resources**
  - [ ] Multi-level rate limiting configured
  - [ ] Resource limits enforced
  - [ ] GPU memory management tested
  - [ ] Circuit breakers implemented

- [ ] **Monitoring & Alerting**
  - [ ] Structured logging deployed
  - [ ] Security metrics exposed
  - [ ] Alert rules configured
  - [ ] Incident response plan tested

- [ ] **Network Security**
  - [ ] TLS 1.3 enforced
  - [ ] Network policies configured
  - [ ] IP whitelisting active
  - [ ] DDoS protection enabled

- [ ] **Compliance**
  - [ ] Regulatory checks automated
  - [ ] Audit trail complete
  - [ ] Data retention policies enforced
  - [ ] Privacy controls implemented

### Production Monitoring Checklist

- [ ] **Real-time Monitoring**
  - [ ] Authentication failures < 0.1%
  - [ ] Rate limit violations < 1%
  - [ ] Response times < 100ms p99
  - [ ] Error rates < 0.01%

- [ ] **Security Metrics**
  - [ ] Failed auth attempts tracked
  - [ ] Suspicious patterns detected
  - [ ] Resource usage within limits
  - [ ] Compliance violations = 0

- [ ] **Incident Response**
  - [ ] On-call rotation active
  - [ ] Runbooks up to date
  - [ ] Forensics tools ready
  - [ ] Communication plan tested

## Conclusion

This comprehensive security guide provides the foundation for building secure, compliant MCP servers for the AI News Trading platform. Key takeaways:

1. **Defense in Depth**: Multiple security layers protect against various attack vectors
2. **Zero Trust**: Every request is authenticated and authorized
3. **Comprehensive Monitoring**: All operations are logged and monitored
4. **Regulatory Compliance**: Automated checks ensure regulatory requirements are met
5. **Incident Response**: Automated and manual response procedures minimize damage

Regular security audits, penetration testing, and updates to this guide ensure the platform maintains the highest security standards while enabling powerful AI-driven trading capabilities.