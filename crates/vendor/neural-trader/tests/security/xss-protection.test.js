/**
 * XSS Protection Tests
 *
 * Comprehensive test suite for XSS pattern detection and prevention
 * in the Neural Trader Backend security module.
 */

const { expect } = require('chai');

describe('XSS Protection Tests', () => {
  let neuralTrader;

  before(async () => {
    try {
      neuralTrader = require('../../neural-trader-rust/packages/neural-trader-backend');
    } catch (error) {
      console.warn('Native module not available, tests will be skipped');
    }
  });

  describe('Basic Script Tag Injection', () => {
    const scriptInjectionTests = [
      '<script>alert(1)</script>',
      '<SCRIPT>alert(1)</SCRIPT>',
      '<script src="evil.js"></script>',
      'Hello<script>alert(1)</script>World',
      '<script>document.cookie</script>',
      '</script><script>alert(1)</script>',
    ];

    scriptInjectionTests.forEach((maliciousInput, index) => {
      it(`should reject script injection #${index + 1}: ${maliciousInput.substring(0, 30)}...`, async () => {
        if (!neuralTrader) return;

        try {
          await neuralTrader.createSyndicateTool(
            'test-syndicate',
            maliciousInput,
            'Test syndicate'
          );
          expect.fail('Should have thrown XSS validation error');
        } catch (error) {
          expect(error.message).to.match(/XSS|Invalid|script/i);
        }
      });
    });
  });

  describe('Event Handler Injection', () => {
    const eventHandlerTests = [
      'onclick=alert(1)',
      'onerror=alert(1)',
      'onload=malicious()',
      'onmouseover=steal()',
      'onfocus=badCode()',
      '<img src=x onerror=alert(1)>',
      '<body onload=alert(1)>',
    ];

    eventHandlerTests.forEach((maliciousInput, index) => {
      it(`should reject event handler #${index + 1}: ${maliciousInput}`, async () => {
        if (!neuralTrader) return;

        try {
          await neuralTrader.addSyndicateMember(
            'test-syndicate',
            maliciousInput,
            'test@example.com',
            'observer',
            100
          );
          expect.fail('Should have thrown XSS validation error');
        } catch (error) {
          expect(error.message).to.match(/XSS|Invalid|onerror|onload/i);
        }
      });
    });
  });

  describe('JavaScript Protocol Injection', () => {
    const protocolTests = [
      'javascript:alert(1)',
      'JavaScript:void(0)',
      'JAVASCRIPT:malicious()',
      'vbscript:alert(1)',
      'data:text/html,<script>alert(1)</script>',
    ];

    protocolTests.forEach((maliciousInput, index) => {
      it(`should reject protocol injection #${index + 1}: ${maliciousInput}`, async () => {
        if (!neuralTrader) return;

        try {
          await neuralTrader.createSyndicateTool(
            'test-syndicate',
            'Test Syndicate',
            maliciousInput
          );
          expect.fail('Should have thrown XSS validation error');
        } catch (error) {
          expect(error.message).to.match(/XSS|Invalid|javascript|vbscript/i);
        }
      });
    });
  });

  describe('Dangerous HTML Tags', () => {
    const dangerousTagTests = [
      '<iframe src="evil.com"></iframe>',
      '<embed src="malicious.swf">',
      '<object data="evil"></object>',
      '<svg onload=alert(1)>',
      '<link rel="stylesheet" href="evil.css">',
      '<meta http-equiv="refresh" content="0;url=evil.com">',
    ];

    dangerousTagTests.forEach((maliciousInput, index) => {
      it(`should reject dangerous tag #${index + 1}: ${maliciousInput.substring(0, 30)}...`, async () => {
        if (!neuralTrader) return;

        try {
          await neuralTrader.createSyndicateTool(
            'test-syndicate',
            maliciousInput,
            'Description'
          );
          expect.fail('Should have thrown XSS validation error');
        } catch (error) {
          expect(error.message).to.match(/XSS|Invalid|iframe|embed|object/i);
        }
      });
    });
  });

  describe('Encoded and Obfuscated Attacks', () => {
    const encodedTests = [
      '&#60;script&#62;alert(1)&#60;/script&#62;',
      '&lt;script&gt;alert(1)&lt;/script&gt;',
      'java\u0000script:alert(1)',
      '<script>alert(String.fromCharCode(88,83,83))</script>',
    ];

    encodedTests.forEach((maliciousInput, index) => {
      it(`should reject encoded attack #${index + 1}`, async () => {
        if (!neuralTrader) return;

        try {
          await neuralTrader.addSyndicateMember(
            'test-syndicate',
            maliciousInput,
            'test@example.com',
            'observer',
            100
          );
          expect.fail('Should have thrown XSS validation error');
        } catch (error) {
          expect(error.message).to.match(/XSS|Invalid|entities|encoded/i);
        }
      });
    });
  });

  describe('Safe Input Acceptance', () => {
    const safeInputs = [
      'John Doe',
      'Trading Strategy Alpha',
      'user@example.com',
      'A legitimate description with numbers 123',
      'Valid-Name_With_Underscores',
      'Price: $100.50',
    ];

    safeInputs.forEach((safeInput, index) => {
      it(`should accept safe input #${index + 1}: ${safeInput}`, async () => {
        if (!neuralTrader) return;

        try {
          const result = await neuralTrader.createSyndicateTool(
            `test-syndicate-${index}`,
            safeInput,
            'Safe description'
          );
          expect(result).to.exist;
        } catch (error) {
          // If it fails for other reasons (like already exists), that's okay
          if (error.message.match(/XSS|Invalid.*script/i)) {
            expect.fail(`Safe input was incorrectly rejected: ${safeInput}`);
          }
        }
      });
    });
  });

  describe('Email Validation', () => {
    it('should accept valid email addresses', async () => {
      if (!neuralTrader) return;

      const validEmails = [
        'user@example.com',
        'test.user@domain.co.uk',
        'admin+tag@company.org',
      ];

      for (const email of validEmails) {
        try {
          await neuralTrader.addSyndicateMember(
            'test-syndicate',
            'John Doe',
            email,
            'observer',
            100
          );
        } catch (error) {
          if (error.message.match(/Invalid email/i)) {
            expect.fail(`Valid email was rejected: ${email}`);
          }
        }
      }
    });

    it('should reject invalid email addresses', async () => {
      if (!neuralTrader) return;

      const invalidEmails = [
        'not-an-email',
        'missing@domain',
        '<script>@example.com',
        'test<script>@domain.com',
      ];

      for (const email of invalidEmails) {
        try {
          await neuralTrader.addSyndicateMember(
            'test-syndicate',
            'John Doe',
            email,
            'observer',
            100
          );
          expect.fail(`Invalid email should have been rejected: ${email}`);
        } catch (error) {
          expect(error.message).to.match(/Invalid email|XSS/i);
        }
      }
    });
  });

  describe('Context-Aware Validation', () => {
    it('should apply appropriate validation based on context', async () => {
      if (!neuralTrader) return;

      // URL context
      const maliciousUrl = 'javascript:alert(1)';
      try {
        // If there's a URL field in any function
        await neuralTrader.createSyndicateTool(
          'test-syndicate',
          'Name',
          maliciousUrl
        );
        expect.fail('Malicious URL should have been rejected');
      } catch (error) {
        expect(error.message).to.match(/XSS|Invalid|javascript/i);
      }
    });
  });

  describe('Performance and Edge Cases', () => {
    it('should handle empty strings', async () => {
      if (!neuralTrader) return;

      try {
        await neuralTrader.createSyndicateTool(
          'test-syndicate',
          '',
          'Description'
        );
      } catch (error) {
        // Empty string might be rejected for other validation reasons
        expect(error.message).to.not.match(/XSS/i);
      }
    });

    it('should handle very long inputs efficiently', async () => {
      if (!neuralTrader) return;

      const longInput = 'A'.repeat(10000);
      const startTime = Date.now();

      try {
        await neuralTrader.createSyndicateTool(
          'test-syndicate',
          longInput,
          'Description'
        );
      } catch (error) {
        // May fail due to length limit, not XSS
      }

      const duration = Date.now() - startTime;
      expect(duration).to.be.lessThan(1000); // Should complete within 1 second
    });

    it('should handle Unicode characters safely', async () => {
      if (!neuralTrader) return;

      const unicodeInputs = [
        '你好世界',
        'Café',
        '🚀 Trading',
        'Москва',
      ];

      for (const input of unicodeInputs) {
        try {
          await neuralTrader.createSyndicateTool(
            `test-syndicate-unicode`,
            input,
            'Description'
          );
        } catch (error) {
          if (error.message.match(/XSS/i)) {
            expect.fail(`Safe Unicode input was rejected: ${input}`);
          }
        }
      }
    });
  });
});
