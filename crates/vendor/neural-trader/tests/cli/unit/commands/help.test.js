/**
 * Unit tests for CLI help command
 */

const { execSync } = require('child_process');
const path = require('path');

describe('CLI Help Command', () => {
  const cliPath = path.join(__dirname, '../../../../bin/cli.js');

  describe('Basic help display', () => {
    it('should display help when "help" command is used', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).toContain('Usage:');
      expect(output).toContain('neural-trader');
    });

    it('should display help when no command is provided', () => {
      const output = execSync(`node ${cliPath}`, { encoding: 'utf8' });

      expect(output).toContain('Usage:');
      expect(output).toContain('Commands:');
    });

    it('should display banner', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).toContain('Neural Trader');
      expect(output).toContain('╔═');
    });
  });

  describe('Commands section', () => {
    const commands = [
      'version',
      'help',
      'init',
      'list',
      'info',
      'install',
      'test',
      'doctor'
    ];

    commands.forEach(command => {
      it(`should list ${command} command`, () => {
        const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });
        expect(output).toContain(command);
      });
    });

    it('should include command descriptions', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).toContain('Show version');
      expect(output).toContain('Initialize a new project');
      expect(output).toContain('List available packages');
    });
  });

  describe('Init types section', () => {
    it('should list trading types', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).toContain('Trading:');
      expect(output).toContain('trading');
      expect(output).toContain('backtesting');
      expect(output).toContain('portfolio');
    });

    it('should list specialized types', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).toContain('Specialized:');
      expect(output).toContain('sports-betting');
      expect(output).toContain('accounting');
    });

    it('should list example types', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).toContain('Examples:');
      expect(output).toContain('example:portfolio-optimization');
    });
  });

  describe('Quick start section', () => {
    it('should include quick start examples', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).toContain('Quick Start:');
      expect(output).toContain('neural-trader init trading');
      expect(output).toContain('neural-trader list');
    });

    it('should include example commands', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).toMatch(/neural-trader init example:/);
    });
  });

  describe('Documentation links', () => {
    it('should include documentation URL', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).toContain('Documentation:');
      expect(output).toContain('github.com/ruvnet/neural-trader');
    });
  });

  describe('Formatting validation', () => {
    it('should use box drawing characters for visual structure', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).toMatch(/[┌┐└┘│─]/);
    });

    it('should have consistent indentation', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      const lines = output.split('\n');
      const indentedLines = lines.filter(line => line.startsWith('  '));
      expect(indentedLines.length).toBeGreaterThan(0);
    });

    it('should use color codes', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      // Check for ANSI color codes
      expect(output).toMatch(/\x1b\[\d+m/);
    });
  });

  describe('Exit behavior', () => {
    it('should exit successfully', () => {
      try {
        execSync(`node ${cliPath} help`);
        expect(true).toBe(true);
      } catch (error) {
        fail('Help command should not fail');
      }
    });
  });

  describe('Content completeness', () => {
    it('should include all required sections', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      const requiredSections = [
        'Usage:',
        'Commands:',
        'Init Types:',
        'Quick Start:',
        'Documentation:'
      ];

      requiredSections.forEach(section => {
        expect(output).toContain(section);
      });
    });

    it('should not contain error messages', () => {
      const output = execSync(`node ${cliPath} help`, { encoding: 'utf8' });

      expect(output).not.toContain('Error:');
      expect(output).not.toContain('error:');
    });
  });
});
