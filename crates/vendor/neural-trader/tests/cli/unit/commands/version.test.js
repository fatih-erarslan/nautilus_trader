/**
 * Unit tests for CLI version command
 */

const { execSync } = require('child_process');
const path = require('path');

describe('CLI Version Command', () => {
  let originalConsoleLog;
  let consoleOutput;
  const cliPath = path.join(__dirname, '../../../../bin/cli.js');

  beforeEach(() => {
    consoleOutput = [];
    originalConsoleLog = console.log;
    console.log = jest.fn((...args) => {
      consoleOutput.push(args.join(' '));
    });
  });

  afterEach(() => {
    console.log = originalConsoleLog;
  });

  describe('Basic version display', () => {
    it('should display version number', () => {
      const output = execSync(`node ${cliPath} version`, {
        encoding: 'utf8',
        env: { ...process.env, NODE_ENV: 'test' }
      });

      expect(output).toContain('2.3.15');
      expect(output).toContain('Version:');
    });

    it('should display Node.js version', () => {
      const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });

      expect(output).toContain('Node:');
      expect(output).toMatch(/v\d+\.\d+\.\d+/);
    });

    it('should display banner with correct formatting', () => {
      const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });

      expect(output).toContain('Neural Trader');
      expect(output).toContain('╔═');
      expect(output).toContain('╚═');
    });
  });

  describe('NAPI bindings status', () => {
    it('should indicate NAPI bindings availability', () => {
      const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });

      expect(output).toMatch(/NAPI Bindings:.*(?:Available|Not loaded)/);
    });

    it('should show core functions count when NAPI is available', () => {
      const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });

      if (output.includes('NAPI Bindings:') && output.includes('Available')) {
        expect(output).toContain('Core Functions:');
      }
    });
  });

  describe('Package information', () => {
    it('should display total package count', () => {
      const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });

      expect(output).toContain('Available Packages:');
      expect(output).toMatch(/\d+/);
    });

    it('should display package categories', () => {
      const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });

      expect(output).toContain('Categories:');
      expect(output).toMatch(/(?:trading|example|accounting|prediction)/);
    });

    it('should show package count per category', () => {
      const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });

      const categoryMatches = output.match(/•\s+\w+:\s+\d+\s+packages/g);
      expect(categoryMatches).toBeTruthy();
      expect(categoryMatches.length).toBeGreaterThan(0);
    });
  });

  describe('Exit codes', () => {
    it('should exit with code 0 on success', () => {
      try {
        execSync(`node ${cliPath} version`);
        expect(true).toBe(true); // If we reach here, exit code was 0
      } catch (error) {
        fail('Version command should not fail');
      }
    });
  });

  describe('Environment handling', () => {
    it('should work in test environment', () => {
      const output = execSync(`node ${cliPath} version`, {
        encoding: 'utf8',
        env: { ...process.env, NODE_ENV: 'test' }
      });

      expect(output).toBeTruthy();
      expect(output).toContain('Version:');
    });

    it('should work in production environment', () => {
      const output = execSync(`node ${cliPath} version`, {
        encoding: 'utf8',
        env: { ...process.env, NODE_ENV: 'production' }
      });

      expect(output).toBeTruthy();
      expect(output).toContain('Version:');
    });
  });

  describe('Output format validation', () => {
    it('should not contain error messages', () => {
      const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });

      expect(output).not.toContain('Error:');
      expect(output).not.toContain('error');
      expect(output).not.toContain('failed');
    });

    it('should use consistent spacing', () => {
      const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });

      // Check for blank lines before/after banner
      const lines = output.split('\n');
      expect(lines.some(line => line.trim() === '')).toBe(true);
    });

    it('should include all required sections', () => {
      const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });

      const requiredSections = [
        'Neural Trader',
        'Version:',
        'Node:',
        'Available Packages:'
      ];

      requiredSections.forEach(section => {
        expect(output).toContain(section);
      });
    });
  });
});
