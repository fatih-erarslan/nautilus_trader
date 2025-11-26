/**
 * Unit Tests for HTTPS Enforcement and Security Headers
 *
 * Tests:
 * - Security headers generation
 * - HTTPS redirection logic
 * - Certificate validation
 * - CSP and Permissions Policy formatting
 */

const {
  SECURITY_CONFIG,
  generateCSP,
  generatePermissionsPolicy,
  securityHeadersMiddleware,
  httpsRedirectMiddleware,
  validateCertificate,
  getSecuritySummary,
} = require('../../src/infrastructure/https-security');

describe('HTTPS Security - Unit Tests', () => {
  describe('Content Security Policy (CSP)', () => {
    test('generates CSP header from directives', () => {
      const directives = {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'unsafe-inline'"],
        styleSrc: ["'self'"],
        imgSrc: ["'self'", 'data:', 'https:'],
        upgradeInsecureRequests: [],
      };

      const csp = generateCSP(directives);

      expect(csp).toContain("default-src 'self'");
      expect(csp).toContain("script-src 'self' 'unsafe-inline'");
      expect(csp).toContain("style-src 'self'");
      expect(csp).toContain("img-src 'self' data: https:");
      expect(csp).toContain('upgrade-insecure-requests');
    });

    test('handles camelCase to kebab-case conversion', () => {
      const directives = {
        defaultSrc: ["'self'"],
        connectSrc: ["'self'", 'https://api.example.com'],
        frameAncestors: ["'none'"],
      };

      const csp = generateCSP(directives);

      expect(csp).toContain('default-src');
      expect(csp).toContain('connect-src');
      expect(csp).toContain('frame-ancestors');
    });

    test('handles empty directive arrays (flags)', () => {
      const directives = {
        upgradeInsecureRequests: [],
        blockAllMixedContent: [],
      };

      const csp = generateCSP(directives);

      expect(csp).toContain('upgrade-insecure-requests');
      expect(csp).toContain('block-all-mixed-content');
      expect(csp).not.toContain('upgrade-insecure-requests ');
    });
  });

  describe('Permissions Policy', () => {
    test('generates Permissions Policy header', () => {
      const policy = {
        camera: [],
        microphone: [],
        geolocation: ["'self'"],
        payment: ["'self'", 'https://payment.example.com'],
      };

      const permissionsPolicy = generatePermissionsPolicy(policy);

      expect(permissionsPolicy).toContain('camera=()');
      expect(permissionsPolicy).toContain('microphone=()');
      expect(permissionsPolicy).toContain("geolocation=('self')");
      expect(permissionsPolicy).toContain("payment=('self' https://payment.example.com)");
    });

    test('handles empty allowlists', () => {
      const policy = {
        camera: [],
        microphone: [],
      };

      const permissionsPolicy = generatePermissionsPolicy(policy);

      expect(permissionsPolicy).toBe('camera=(), microphone=()');
    });
  });

  describe('Security Headers Middleware', () => {
    test('sets all required security headers', () => {
      const req = {};
      const res = {
        headers: {},
        setHeader(name, value) {
          this.headers[name] = value;
        },
        removeHeader(name) {
          delete this.headers[name];
        },
      };
      const next = jest.fn();

      securityHeadersMiddleware(req, res, next);

      expect(res.headers['Strict-Transport-Security']).toBeDefined();
      expect(res.headers['Content-Security-Policy']).toBeDefined();
      expect(res.headers['X-Frame-Options']).toBe('DENY');
      expect(res.headers['X-Content-Type-Options']).toBe('nosniff');
      expect(res.headers['X-XSS-Protection']).toBe('1; mode=block');
      expect(res.headers['Referrer-Policy']).toBe('strict-origin-when-cross-origin');
      expect(res.headers['Permissions-Policy']).toBeDefined();
      expect(next).toHaveBeenCalled();
    });

    test('HSTS header includes all directives', () => {
      const req = {};
      const res = {
        headers: {},
        setHeader(name, value) {
          this.headers[name] = value;
        },
        removeHeader() {},
      };
      const next = jest.fn();

      securityHeadersMiddleware(req, res, next);

      const hsts = res.headers['Strict-Transport-Security'];
      expect(hsts).toContain('max-age=31536000');
      expect(hsts).toContain('includeSubDomains');
      expect(hsts).toContain('preload');
    });

    test('removes X-Powered-By header', () => {
      const req = {};
      const res = {
        headers: { 'X-Powered-By': 'Express' },
        setHeader(name, value) {
          this.headers[name] = value;
        },
        removeHeader(name) {
          delete this.headers[name];
        },
      };
      const next = jest.fn();

      securityHeadersMiddleware(req, res, next);

      expect(res.headers['X-Powered-By']).toBeUndefined();
    });
  });

  describe('HTTPS Redirect Middleware', () => {
    test('redirects HTTP to HTTPS', () => {
      const req = {
        secure: false,
        headers: {},
        hostname: 'example.com',
        url: '/api/test',
        path: '/api/test',
      };
      const res = {
        redirect: jest.fn(),
      };
      const next = jest.fn();

      httpsRedirectMiddleware(req, res, next);

      expect(res.redirect).toHaveBeenCalledWith(301, 'https://example.com/api/test');
      expect(next).not.toHaveBeenCalled();
    });

    test('allows HTTPS requests through', () => {
      const req = {
        secure: true,
        headers: {},
        hostname: 'example.com',
        url: '/api/test',
        path: '/api/test',
      };
      const res = {
        redirect: jest.fn(),
      };
      const next = jest.fn();

      httpsRedirectMiddleware(req, res, next);

      expect(res.redirect).not.toHaveBeenCalled();
      expect(next).toHaveBeenCalled();
    });

    test('detects HTTPS from x-forwarded-proto header', () => {
      const req = {
        secure: false,
        headers: { 'x-forwarded-proto': 'https' },
        hostname: 'example.com',
        url: '/api/test',
        path: '/api/test',
      };
      const res = {
        redirect: jest.fn(),
      };
      const next = jest.fn();

      httpsRedirectMiddleware(req, res, next);

      expect(res.redirect).not.toHaveBeenCalled();
      expect(next).toHaveBeenCalled();
    });

    test('allows health check endpoints on HTTP', () => {
      const healthReq = {
        secure: false,
        headers: {},
        hostname: 'example.com',
        url: '/health',
        path: '/health',
      };
      const res = {
        redirect: jest.fn(),
      };
      const next = jest.fn();

      httpsRedirectMiddleware(healthReq, res, next);

      expect(res.redirect).not.toHaveBeenCalled();
      expect(next).toHaveBeenCalled();

      // Test /ping endpoint
      healthReq.url = '/ping';
      healthReq.path = '/ping';
      httpsRedirectMiddleware(healthReq, res, next);

      expect(res.redirect).not.toHaveBeenCalled();
      expect(next).toHaveBeenCalledTimes(2);
    });
  });

  describe('Security Configuration', () => {
    test('uses TLS 1.3 by default', () => {
      expect(SECURITY_CONFIG.tls.minVersion).toBe('TLSv1.3');
      expect(SECURITY_CONFIG.tls.maxVersion).toBe('TLSv1.3');
    });

    test('includes modern cipher suites', () => {
      expect(SECURITY_CONFIG.tls.ciphers).toContain('TLS_AES_128_GCM_SHA256');
      expect(SECURITY_CONFIG.tls.ciphers).toContain('TLS_AES_256_GCM_SHA384');
      expect(SECURITY_CONFIG.tls.ciphers).toContain('TLS_CHACHA20_POLY1305_SHA256');
    });

    test('HSTS configured for 1 year with preload', () => {
      expect(SECURITY_CONFIG.hsts.maxAge).toBe(31536000);
      expect(SECURITY_CONFIG.hsts.includeSubDomains).toBe(true);
      expect(SECURITY_CONFIG.hsts.preload).toBe(true);
    });

    test('CSP blocks frames and objects', () => {
      expect(SECURITY_CONFIG.csp.directives.frameSrc).toEqual(["'none'"]);
      expect(SECURITY_CONFIG.csp.directives.objectSrc).toEqual(["'none'"]);
    });

    test('CSP upgrades insecure requests', () => {
      expect(SECURITY_CONFIG.csp.directives.upgradeInsecureRequests).toEqual([]);
    });
  });

  describe('Security Summary', () => {
    test('returns comprehensive security summary', () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = 'production';

      const summary = getSecuritySummary();

      expect(summary.https.enabled).toBe(true);
      expect(summary.https.tlsVersion).toBe('TLSv1.3');
      expect(summary.environment).toBe('production');
      expect(summary.securityScore).toBe('9.5/10');

      process.env.NODE_ENV = originalEnv;
    });

    test('shows lower security score in development', () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = 'development';

      const summary = getSecuritySummary();

      expect(summary.https.enabled).toBe(false);
      expect(summary.securityScore).toBe('8.5/10');

      process.env.NODE_ENV = originalEnv;
    });
  });
});
