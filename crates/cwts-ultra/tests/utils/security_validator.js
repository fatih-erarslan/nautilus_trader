/**
 * Security Validator for Trading System Testing
 * Provides comprehensive security validation and cryptographic utilities
 */

const crypto = require('crypto');

class SecurityValidator {
  constructor() {
    this.encryptionKey = crypto.randomBytes(32); // 256-bit key
    this.signingKey = crypto.randomBytes(64); // 512-bit signing key
    this.algorithm = 'aes-256-gcm';
    this.hashAlgorithm = 'sha256';
  }

  /**
   * Calculate cryptographic hash for order data
   */
  calculateOrderHash(order) {
    // Create deterministic string representation
    const orderString = JSON.stringify({
      id: order.id,
      symbol: order.symbol,
      side: order.side,
      quantity: order.quantity,
      price: order.price,
      type: order.type,
      timestamp: order.timestamp,
      userId: order.userId
    }, Object.keys(order).sort());

    return crypto.createHash(this.hashAlgorithm)
      .update(orderString)
      .digest('hex');
  }

  /**
   * Calculate hash for audit log entry
   */
  calculateLogHash(logEntry) {
    const logString = JSON.stringify({
      event: logEntry.event,
      orderId: logEntry.orderId,
      userId: logEntry.userId,
      timestamp: logEntry.timestamp,
      data: logEntry.data
    }, Object.keys(logEntry).filter(k => k !== 'hash' && k !== 'signature').sort());

    return crypto.createHash(this.hashAlgorithm)
      .update(logString)
      .digest('hex');
  }

  /**
   * Encrypt sensitive data
   */
  encryptSensitiveData(data) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher(this.algorithm, this.encryptionKey);
    cipher.setAAD(Buffer.from('trading-system'));

    let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
    encrypted += cipher.final('hex');

    const authTag = cipher.getAuthTag();

    return {
      encrypted,
      iv: iv.toString('hex'),
      authTag: authTag.toString('hex')
    };
  }

  /**
   * Decrypt sensitive data
   */
  async decryptSensitiveData(encryptedData) {
    try {
      const decipher = crypto.createDecipher(this.algorithm, this.encryptionKey);
      decipher.setAAD(Buffer.from('trading-system'));
      decipher.setAuthTag(Buffer.from(encryptedData.authTag, 'hex'));

      let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
      decrypted += decipher.final('utf8');

      return JSON.parse(decrypted);
    } catch (error) {
      throw new Error('Decryption failed: ' + error.message);
    }
  }

  /**
   * Generate secure random identifier
   */
  generateSecureRandomId() {
    return crypto.randomBytes(16).toString('hex');
  }

  /**
   * Generate session token
   */
  generateSessionToken(userId) {
    const payload = {
      userId,
      timestamp: Date.now(),
      nonce: crypto.randomBytes(16).toString('hex')
    };

    const token = Buffer.from(JSON.stringify(payload)).toString('base64');
    const signature = this.signData(token);

    return `${token}.${signature}`;
  }

  /**
   * Generate expired token for testing
   */
  generateExpiredToken(userId) {
    const payload = {
      userId,
      timestamp: Date.now() - (24 * 60 * 60 * 1000), // 24 hours ago
      nonce: crypto.randomBytes(16).toString('hex')
    };

    const token = Buffer.from(JSON.stringify(payload)).toString('base64');
    const signature = this.signData(token);

    return `${token}.${signature}`;
  }

  /**
   * Validate session token
   */
  validateSessionToken(token) {
    try {
      const [payload, signature] = token.split('.');
      
      // Verify signature
      if (!this.verifyDataSignature(payload, signature)) {
        return { valid: false, reason: 'Invalid signature' };
      }

      // Parse payload
      const data = JSON.parse(Buffer.from(payload, 'base64').toString('utf8'));
      
      // Check expiration (1 hour)
      const maxAge = 60 * 60 * 1000;
      if (Date.now() - data.timestamp > maxAge) {
        return { valid: false, reason: 'Token expired' };
      }

      return { valid: true, userId: data.userId };
    } catch (error) {
      return { valid: false, reason: 'Malformed token' };
    }
  }

  /**
   * Sign data with HMAC
   */
  signData(data) {
    return crypto.createHmac('sha256', this.signingKey)
      .update(data)
      .digest('hex');
  }

  /**
   * Verify data signature
   */
  verifyDataSignature(data, signature) {
    const expectedSignature = this.signData(data);
    return crypto.timingSafeEqual(
      Buffer.from(signature, 'hex'),
      Buffer.from(expectedSignature, 'hex')
    );
  }

  /**
   * Create digital signature for audit logs
   */
  async verifySignature(logEntry) {
    try {
      // Recreate the data that was signed
      const dataToSign = this.calculateLogHash(logEntry);
      const expectedSignature = this.signData(dataToSign);
      
      return crypto.timingSafeEqual(
        Buffer.from(logEntry.signature, 'hex'),
        Buffer.from(expectedSignature, 'hex')
      );
    } catch (error) {
      return false;
    }
  }

  /**
   * Calculate entropy of a string
   */
  calculateEntropy(str) {
    const frequencies = {};
    
    // Count character frequencies
    for (const char of str) {
      frequencies[char] = (frequencies[char] || 0) + 1;
    }

    // Calculate entropy
    let entropy = 0;
    const length = str.length;
    
    for (const freq of Object.values(frequencies)) {
      const probability = freq / length;
      entropy -= probability * Math.log2(probability);
    }

    return entropy;
  }

  /**
   * Detect tampering in audit trail
   */
  async detectTampering(auditLogs) {
    for (const log of auditLogs) {
      // Verify hash integrity
      const calculatedHash = this.calculateLogHash(log);
      if (calculatedHash !== log.hash) {
        return true; // Tampering detected
      }

      // Verify signature
      const signatureValid = await this.verifySignature(log);
      if (!signatureValid) {
        return true; // Tampering detected
      }
    }

    return false; // No tampering detected
  }

  /**
   * Validate input for security threats
   */
  validateInput(input, type) {
    const validations = {
      orderId: {
        maxLength: 64,
        pattern: /^[a-zA-Z0-9_-]+$/,
        blacklist: ['script', 'eval', 'function', 'delete', 'drop', 'exec']
      },
      symbol: {
        maxLength: 16,
        pattern: /^[A-Z0-9.]+$/,
        blacklist: ['..', '/', '\\', '<', '>', '&', '"', "'"]
      },
      userId: {
        maxLength: 32,
        pattern: /^[a-zA-Z0-9_]+$/,
        blacklist: ['admin', 'root', 'system', 'null', 'undefined']
      },
      price: {
        type: 'number',
        min: 0.0001,
        max: 999999.9999,
        decimalPlaces: 4
      },
      quantity: {
        type: 'number',
        min: 1,
        max: Number.MAX_SAFE_INTEGER,
        integer: true
      }
    };

    const validation = validations[type];
    if (!validation) {
      throw new Error(`Unknown input type: ${type}`);
    }

    // Type checking
    if (validation.type === 'number') {
      if (typeof input !== 'number' || isNaN(input) || !isFinite(input)) {
        return { valid: false, reason: 'Invalid number' };
      }

      if (input < validation.min || input > validation.max) {
        return { valid: false, reason: 'Number out of range' };
      }

      if (validation.integer && !Number.isInteger(input)) {
        return { valid: false, reason: 'Must be integer' };
      }

      if (validation.decimalPlaces) {
        const decimalCount = (input.toString().split('.')[1] || '').length;
        if (decimalCount > validation.decimalPlaces) {
          return { valid: false, reason: 'Too many decimal places' };
        }
      }

      return { valid: true };
    }

    // String validation
    if (typeof input !== 'string') {
      return { valid: false, reason: 'Must be string' };
    }

    // Length check
    if (input.length > validation.maxLength) {
      return { valid: false, reason: 'String too long' };
    }

    // Pattern check
    if (validation.pattern && !validation.pattern.test(input)) {
      return { valid: false, reason: 'Invalid format' };
    }

    // Blacklist check
    const lowerInput = input.toLowerCase();
    for (const blacklisted of validation.blacklist) {
      if (lowerInput.includes(blacklisted.toLowerCase())) {
        return { valid: false, reason: 'Contains blacklisted content' };
      }
    }

    // XSS check
    if (this.containsXSS(input)) {
      return { valid: false, reason: 'XSS detected' };
    }

    // SQL injection check
    if (this.containsSQLInjection(input)) {
      return { valid: false, reason: 'SQL injection detected' };
    }

    // Path traversal check
    if (this.containsPathTraversal(input)) {
      return { valid: false, reason: 'Path traversal detected' };
    }

    return { valid: true };
  }

  /**
   * Check for XSS patterns
   */
  containsXSS(input) {
    const xssPatterns = [
      /<script[^>]*>.*?<\/script>/gi,
      /<iframe[^>]*>.*?<\/iframe>/gi,
      /<object[^>]*>.*?<\/object>/gi,
      /<embed[^>]*>/gi,
      /javascript:/gi,
      /vbscript:/gi,
      /onload=/gi,
      /onerror=/gi,
      /onclick=/gi,
      /onmouseover=/gi
    ];

    return xssPatterns.some(pattern => pattern.test(input));
  }

  /**
   * Check for SQL injection patterns
   */
  containsSQLInjection(input) {
    const sqlPatterns = [
      /('|(\\')|(;)|(\\;)|(--|#|\/\*))/gi,
      /(union|select|insert|update|delete|drop|create|alter|exec|execute)/gi,
      /(\s|^)(or|and)(\s|$)/gi,
      /(sleep|benchmark|waitfor|delay)/gi
    ];

    return sqlPatterns.some(pattern => pattern.test(input));
  }

  /**
   * Check for path traversal patterns
   */
  containsPathTraversal(input) {
    const pathPatterns = [
      /\.\.\//g,
      /\.\.\\g,
      /%2e%2e%2f/gi,
      /%2e%2e%5c/gi,
      /\.\.%2f/gi,
      /\.\.%5c/gi
    ];

    return pathPatterns.some(pattern => pattern.test(input));
  }

  /**
   * Sanitize input string
   */
  sanitizeInput(input) {
    if (typeof input !== 'string') {
      return input;
    }

    return input
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#x27;')
      .replace(/\//g, '&#x2F;')
      .replace(/\\/g, '&#x5C;')
      .replace(/&/g, '&amp;');
  }

  /**
   * Generate secure hash for password storage
   */
  hashPassword(password, salt) {
    if (!salt) {
      salt = crypto.randomBytes(32).toString('hex');
    }

    const iterations = 100000;
    const keyLength = 64;
    const digest = 'sha512';

    const hash = crypto.pbkdf2Sync(password, salt, iterations, keyLength, digest);
    
    return {
      hash: hash.toString('hex'),
      salt,
      iterations,
      keyLength,
      digest
    };
  }

  /**
   * Verify password against hash
   */
  verifyPassword(password, storedHash) {
    const { hash, salt, iterations, keyLength, digest } = storedHash;
    const testHash = crypto.pbkdf2Sync(password, salt, iterations, keyLength, digest);
    
    return crypto.timingSafeEqual(
      Buffer.from(hash, 'hex'),
      testHash
    );
  }

  /**
   * Rate limiting validator
   */
  createRateLimiter(maxRequests, windowMs) {
    const requests = new Map();

    return {
      checkRate: (identifier) => {
        const now = Date.now();
        const windowStart = now - windowMs;

        // Clean old entries
        for (const [id, timestamps] of requests.entries()) {
          const filtered = timestamps.filter(t => t > windowStart);
          if (filtered.length === 0) {
            requests.delete(id);
          } else {
            requests.set(id, filtered);
          }
        }

        // Check current rate
        const userRequests = requests.get(identifier) || [];
        const recentRequests = userRequests.filter(t => t > windowStart);

        if (recentRequests.length >= maxRequests) {
          return {
            allowed: false,
            remainingRequests: 0,
            resetTime: Math.min(...recentRequests) + windowMs
          };
        }

        // Add current request
        recentRequests.push(now);
        requests.set(identifier, recentRequests);

        return {
          allowed: true,
          remainingRequests: maxRequests - recentRequests.length,
          resetTime: now + windowMs
        };
      }
    };
  }
}

module.exports = { SecurityValidator };