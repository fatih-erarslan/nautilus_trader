/**
 * Unit tests for CLI info command
 */

const { execSync } = require('child_process');
const path = require('path');

describe('CLI Info Command', () => {
  const cliPath = path.join(__dirname, '../../../../bin/cli.js');

  describe('Basic info display', () => {
    it('should display package information for trading', () => {
      const output = execSync(`node ${cliPath} info trading`, { encoding: 'utf8' });

      expect(output).toContain('Trading Strategy System');
      expect(output).toContain('Description:');
    });

    it('should display package information for backtesting', () => {
      const output = execSync(`node ${cliPath} info backtesting`, { encoding: 'utf8' });

      expect(output).toContain('Backtesting');
      expect(output).toContain('Description:');
    });

    it('should display package information for accounting', () => {
      const output = execSync(`node ${cliPath} info accounting`, { encoding: 'utf8' });

      expect(output).toContain('Agentic Accounting');
      expect(output).toContain('Description:');
    });
  });

  describe('Package details', () => {
    it('should show package category', () => {
      const output = execSync(`node ${cliPath} info trading`, { encoding: 'utf8' });

      expect(output).toContain('Category:');
      expect(output).toContain('trading');
    });

    it('should list package features', () => {
      const output = execSync(`node ${cliPath} info trading`, { encoding: 'utf8' });

      expect(output).toContain('Features:');
      expect(output).toMatch(/•/);
    });

    it('should list NPM packages', () => {
      const output = execSync(`node ${cliPath} info trading`, { encoding: 'utf8' });

      expect(output).toContain('NPM Packages:');
      expect(output).toContain('@neural-trader');
    });

    it('should show initialization command', () => {
      const output = execSync(`node ${cliPath} info trading`, { encoding: 'utf8' });

      expect(output).toContain('Initialize:');
      expect(output).toContain('neural-trader init trading');
    });
  });

  describe('Example packages', () => {
    it('should mark example packages', () => {
      const output = execSync(`node ${cliPath} info example:portfolio-optimization`, {
        encoding: 'utf8'
      });

      expect(output).toContain('example');
    });

    it('should display example package features', () => {
      const output = execSync(`node ${cliPath} info example:portfolio-optimization`, {
        encoding: 'utf8'
      });

      expect(output).toContain('Features:');
      expect(output).toContain('Portfolio Optimization');
    });
  });

  describe('Error handling', () => {
    it('should show error when package name is missing', () => {
      try {
        execSync(`node ${cliPath} info`, {
          encoding: 'utf8',
          stdio: 'pipe'
        });
        fail('Should have thrown an error');
      } catch (error) {
        expect(error.status).toBe(1);
        expect(error.stderr.toString()).toContain('Package name required');
      }
    });

    it('should show error for unknown package', () => {
      try {
        execSync(`node ${cliPath} info nonexistent-package`, {
          encoding: 'utf8',
          stdio: 'pipe'
        });
        fail('Should have thrown an error');
      } catch (error) {
        expect(error.status).toBe(1);
        expect(error.stderr.toString()).toContain('Unknown package');
      }
    });

    it('should suggest listing packages on error', () => {
      try {
        execSync(`node ${cliPath} info invalid`, {
          encoding: 'utf8',
          stdio: 'pipe'
        });
      } catch (error) {
        expect(error.stderr.toString()).toContain('neural-trader list');
      }
    });
  });

  describe('Output formatting', () => {
    it('should use consistent spacing', () => {
      const output = execSync(`node ${cliPath} info trading`, { encoding: 'utf8' });

      const lines = output.split('\n');
      expect(lines.some(line => line.trim() === '')).toBe(true);
    });

    it('should use color codes', () => {
      const output = execSync(`node ${cliPath} info trading`, { encoding: 'utf8' });

      expect(output).toMatch(/\x1b\[\d+m/);
    });

    it('should format lists with bullets', () => {
      const output = execSync(`node ${cliPath} info trading`, { encoding: 'utf8' });

      expect(output).toMatch(/[•]/);
    });
  });

  describe('Different package types', () => {
    const packages = [
      'trading',
      'backtesting',
      'portfolio',
      'accounting',
      'predictor',
      'example:portfolio-optimization'
    ];

    packages.forEach(pkg => {
      it(`should display info for ${pkg}`, () => {
        const output = execSync(`node ${cliPath} info ${pkg}`, { encoding: 'utf8' });

        expect(output).toContain('Description:');
        expect(output).toContain('Initialize:');
      });
    });
  });

  describe('Content validation', () => {
    it('should include all required sections', () => {
      const output = execSync(`node ${cliPath} info trading`, { encoding: 'utf8' });

      const requiredSections = [
        'Category:',
        'Description:',
        'Features:',
        'Initialize:'
      ];

      requiredSections.forEach(section => {
        expect(output).toContain(section);
      });
    });

    it('should not contain generic placeholder text', () => {
      const output = execSync(`node ${cliPath} info trading`, { encoding: 'utf8' });

      expect(output).not.toContain('TODO');
      expect(output).not.toContain('placeholder');
    });
  });
});
