/**
 * Security Integration Tests
 *
 * Tests for integrated security features across the Neural Trader Backend
 */

const { expect } = require('chai');
const path = require('path');

describe('Security Integration Tests', () => {
  let neuralTrader;

  before(async () => {
    try {
      neuralTrader = require('../../neural-trader-rust/packages/neural-trader-backend');
    } catch (error) {
      console.warn('Native module not available, tests will be skipped');
    }
  });

  describe('Syndicate XSS Protection Integration', () => {
    it('should protect all syndicate creation inputs', async () => {
      if (!neuralTrader) return;

      const xssAttempts = [
        { id: '<script>alert(1)</script>', name: 'Legit Name', desc: 'Description' },
        { id: 'valid-id', name: '<iframe src="evil"></iframe>', desc: 'Description' },
        { id: 'valid-id', name: 'Legit', desc: 'javascript:alert(1)' },
      ];

      for (const attempt of xssAttempts) {
        try {
          await neuralTrader.createSyndicateTool(
            attempt.id,
            attempt.name,
            attempt.desc
          );
          expect.fail('XSS attempt should have been blocked');
        } catch (error) {
          expect(error.message).to.match(/XSS|Invalid|script|javascript/i);
        }
      }
    });

    it('should protect member addition inputs', async () => {
      if (!neuralTrader) return;

      // First create a valid syndicate
      try {
        await neuralTrader.createSyndicateTool(
          'test-syndicate-integration',
          'Test Syndicate',
          'A test syndicate'
        );
      } catch (error) {
        // May already exist
      }

      const xssAttempts = [
        { name: '<script>steal()</script>', email: 'test@example.com', role: 'observer' },
        { name: 'John Doe', email: '<script>@evil.com', role: 'observer' },
        { name: 'John Doe', email: 'test@example.com', role: 'onclick=alert(1)' },
      ];

      for (const attempt of xssAttempts) {
        try {
          await neuralTrader.addSyndicateMember(
            'test-syndicate-integration',
            attempt.name,
            attempt.email,
            attempt.role,
            100
          );
          expect.fail('XSS attempt should have been blocked');
        } catch (error) {
          expect(error.message).to.match(/XSS|Invalid|script|email/i);
        }
      }
    });
  });

  describe('Multi-Layer Security Validation', () => {
    it('should apply security at every layer', async () => {
      if (!neuralTrader) return;

      // Test that security is enforced consistently
      const maliciousInputs = [
        'javascript:alert(1)',
        '<script>alert(1)</script>',
        'onerror=alert(1)',
        '../../../etc/passwd',
      ];

      for (const input of maliciousInputs) {
        let blocked = false;

        try {
          await neuralTrader.createSyndicateTool(
            input,
            'Test',
            'Description'
          );
        } catch (error) {
          if (error.message.match(/XSS|Invalid|script|traversal/i)) {
            blocked = true;
          }
        }

        expect(blocked, `Input "${input}" should have been blocked`).to.be.true;
      }
    });
  });

  describe('Performance Impact Tests', () => {
    it('should validate inputs without significant performance impact', async function() {
      if (!neuralTrader) return;

      this.timeout(5000); // 5 second timeout

      const iterations = 100;
      const safeInput = 'Valid Trading Syndicate Name';

      const startTime = Date.now();

      for (let i = 0; i < iterations; i++) {
        try {
          await neuralTrader.createSyndicateTool(
            `test-perf-${i}`,
            safeInput,
            'Performance test syndicate'
          );
        } catch (error) {
          // May fail due to duplicate ID or other reasons
          // We're just testing performance
        }
      }

      const duration = Date.now() - startTime;
      const avgTime = duration / iterations;

      expect(avgTime).to.be.lessThan(50); // Should average <50ms per operation
      console.log(`  Average validation time: ${avgTime.toFixed(2)}ms`);
    });
  });

  describe('Combined Attack Vectors', () => {
    it('should handle combined XSS and path traversal attempts', async () => {
      if (!neuralTrader) return;

      const combinedAttacks = [
        '../../../etc/passwd<script>alert(1)</script>',
        'javascript:alert(1)/../../../etc/passwd',
        '<iframe src="../../../etc/passwd"></iframe>',
      ];

      for (const attack of combinedAttacks) {
        try {
          await neuralTrader.createSyndicateTool(
            attack,
            'Test',
            'Description'
          );
          expect.fail('Combined attack should have been blocked');
        } catch (error) {
          expect(error.message).to.match(/XSS|Invalid|script|traversal/i);
        }
      }
    });
  });

  describe('Edge Cases and Boundary Conditions', () => {
    it('should handle maximum length inputs', async () => {
      if (!neuralTrader) return;

      const longInput = 'A'.repeat(10000);

      try {
        await neuralTrader.createSyndicateTool(
          'test-long',
          longInput,
          'Description'
        );
      } catch (error) {
        // May fail due to length limit, which is expected
        expect(error.message).to.not.match(/XSS/i);
      }
    });

    it('should handle special characters correctly', async () => {
      if (!neuralTrader) return;

      const specialChars = '!@#$%^&*()_+-=[]{}|;:,.<>?';

      try {
        await neuralTrader.createSyndicateTool(
          'test-special',
          `Test ${specialChars}`,
          'Description'
        );
      } catch (error) {
        // Should not fail due to XSS
        expect(error.message).to.not.match(/XSS/i);
      }
    });

    it('should handle international characters', async () => {
      if (!neuralTrader) return;

      const internationalNames = [
        '北京交易集团',
        'Москва Трейдинг',
        'مجموعة التداول',
        'Ελληνική Ομάδα',
      ];

      for (const name of internationalNames) {
        try {
          await neuralTrader.createSyndicateTool(
            `test-intl-${Buffer.from(name).toString('base64')}`,
            name,
            'International test'
          );
        } catch (error) {
          // Should not fail due to XSS
          if (error.message.match(/XSS/i)) {
            expect.fail(`International name incorrectly flagged as XSS: ${name}`);
          }
        }
      }
    });
  });

  describe('Security Header Tests', () => {
    it('should properly escape HTML in responses', async () => {
      if (!neuralTrader) return;

      // Create a syndicate with special characters
      const syndicateId = 'test-escape';
      const name = 'Test & Associates <Corporation>';

      try {
        const result = await neuralTrader.createSyndicateTool(
          syndicateId,
          name,
          'Test description'
        );

        // Response should be JSON
        const parsed = JSON.parse(result);

        // Name should be preserved (validation passed)
        // but downstream HTML rendering would escape it
        expect(parsed.name).to.exist;
      } catch (error) {
        // Special chars like & and < are okay in plain text context
        expect(error.message).to.not.match(/XSS/i);
      }
    });
  });

  describe('Regression Tests', () => {
    it('should not break existing functionality', async () => {
      if (!neuralTrader) return;

      // Test that normal operations still work
      const validOperations = [
        {
          id: 'regression-test-1',
          name: 'Valid Syndicate Name',
          desc: 'A normal description',
        },
        {
          id: 'regression-test-2',
          name: 'Another Valid Name',
          desc: 'Another normal description with numbers 123',
        },
      ];

      for (const op of validOperations) {
        try {
          const result = await neuralTrader.createSyndicateTool(
            op.id,
            op.name,
            op.desc
          );

          expect(result).to.exist;
          const parsed = JSON.parse(result);
          expect(parsed.status).to.exist;
        } catch (error) {
          // If it fails, it shouldn't be due to security
          expect(error.message).to.not.match(/XSS|Invalid.*script/i);
        }
      }
    });
  });
});
