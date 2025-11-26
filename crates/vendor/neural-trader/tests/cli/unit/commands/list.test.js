/**
 * Unit tests for CLI list command
 */

const { execSync } = require('child_process');
const path = require('path');

describe('CLI List Command', () => {
  const cliPath = path.join(__dirname, '../../../../bin/cli.js');

  describe('Basic list display', () => {
    it('should display all available packages', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      expect(output).toContain('Available Neural Trader Packages');
      expect(output).toContain('📦');
    });

    it('should show package names and descriptions', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      expect(output).toContain('Trading Strategy System');
      expect(output).toContain('Backtesting');
      expect(output).toContain('Portfolio Management');
    });

    it('should display usage instruction', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      expect(output).toContain('Use "neural-trader init <type>" to create a project');
    });
  });

  describe('Package information', () => {
    it('should list trading packages', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      expect(output).toContain('trading');
      expect(output).toContain('backtesting');
      expect(output).toContain('portfolio');
    });

    it('should list specialized packages', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      expect(output).toContain('sports-betting');
      expect(output).toContain('accounting');
      expect(output).toContain('predictor');
    });

    it('should list example packages', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      expect(output).toContain('example:portfolio-optimization');
      expect(output).toContain('example:healthcare-optimization');
    });

    it('should show NPM package names', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      expect(output).toContain('Packages:');
      expect(output).toContain('@neural-trader');
    });
  });

  describe('Package count', () => {
    it('should display multiple packages', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      // Count package entries (look for key identifiers)
      const matches = output.match(/trading|backtesting|portfolio|accounting/gi);
      expect(matches).toBeTruthy();
      expect(matches.length).toBeGreaterThan(5);
    });

    it('should include all major categories', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      const categories = ['trading', 'example', 'accounting', 'prediction'];
      const foundCategories = categories.filter(cat =>
        output.toLowerCase().includes(cat)
      );

      expect(foundCategories.length).toBeGreaterThan(2);
    });
  });

  describe('Output formatting', () => {
    it('should use consistent spacing', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      const lines = output.split('\n');
      const emptyLines = lines.filter(line => line.trim() === '');
      expect(emptyLines.length).toBeGreaterThan(0);
    });

    it('should align package information', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      // Check for consistent indentation
      const lines = output.split('\n');
      const indentedLines = lines.filter(line => line.startsWith('  '));
      expect(indentedLines.length).toBeGreaterThan(0);
    });

    it('should format package descriptions properly', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      // Each package should have key and description
      const packagePattern = /\w+\s+\w+.*\n\s+.*\n/;
      expect(output).toMatch(packagePattern);
    });
  });

  describe('Package details validation', () => {
    it('should include package descriptions', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      // Check for various descriptions
      const descriptions = [
        'algorithmic trading',
        'backtesting',
        'portfolio optimization',
        'accounting'
      ];

      const foundDescriptions = descriptions.filter(desc =>
        output.toLowerCase().includes(desc)
      );

      expect(foundDescriptions.length).toBeGreaterThan(2);
    });

    it('should show package options where available', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      // Some packages should show their sub-packages
      expect(output).toMatch(/Packages:/i);
    });
  });

  describe('Exit behavior', () => {
    it('should exit successfully', () => {
      try {
        execSync(`node ${cliPath} list`);
        expect(true).toBe(true);
      } catch (error) {
        fail('List command should not fail');
      }
    });

    it('should not output errors', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      expect(output).not.toContain('Error:');
      expect(output).not.toContain('error:');
    });
  });

  describe('Completeness', () => {
    it('should list at least 10 packages', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      // Count distinct package identifiers
      const packageMatches = output.match(/^\s+\w+[-:]?\w*/gm);
      expect(packageMatches).toBeTruthy();
      expect(packageMatches.length).toBeGreaterThan(10);
    });

    it('should provide enough information to choose a package', () => {
      const output = execSync(`node ${cliPath} list`, { encoding: 'utf8' });

      // Each package should have name and description minimum
      expect(output).toContain('Trading Strategy System');
      expect(output).toContain('algorithmic trading');
    });
  });
});
