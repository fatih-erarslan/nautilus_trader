/**
 * Unit tests for CLI test command
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

describe('CLI Test Command', () => {
  const cliPath = path.join(__dirname, '../../../../bin/cli.js');
  let tempDir;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'neural-trader-test-'));
    process.chdir(tempDir);
  });

  afterEach(() => {
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  describe('NAPI bindings test', () => {
    it('should test NAPI bindings', () => {
      const output = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('Testing Neural Trader components');
      expect(output).toContain('NAPI Bindings');
    });

    it('should list expected NAPI functions', () => {
      const output = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      const expectedFunctions = [
        'fetchMarketData',
        'runStrategy',
        'backtest',
        'trainModel',
        'predict'
      ];

      expectedFunctions.forEach(fn => {
        expect(output).toContain(fn);
      });
    });

    it('should indicate if NAPI is not loaded', () => {
      const output = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      if (!output.includes('✅ NAPI Bindings')) {
        expect(output).toContain('NAPI bindings not loaded');
      }
    });
  });

  describe('Package installation test', () => {
    it('should check for installed packages', () => {
      const output = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('Installed Packages');
    });

    it('should report no packages when package.json is missing', () => {
      const output = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('No package.json');
    });

    it('should list neural-trader packages when present', () => {
      // Create package.json with dependencies
      const pkg = {
        name: 'test-project',
        version: '1.0.0',
        dependencies: {
          'neural-trader': '^2.3.15',
          '@neural-trader/core': '^1.0.1'
        }
      };
      fs.writeFileSync(
        path.join(tempDir, 'package.json'),
        JSON.stringify(pkg, null, 2)
      );

      const output = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('neural-trader');
    });
  });

  describe('Test completion', () => {
    it('should show success message', () => {
      const output = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('Tests complete');
      expect(output).toMatch(/✅|✓/);
    });
  });

  describe('Output formatting', () => {
    it('should use consistent formatting', () => {
      const output = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      const lines = output.split('\n');
      const emptyLines = lines.filter(line => line.trim() === '');
      expect(emptyLines.length).toBeGreaterThan(0);
    });

    it('should use status indicators', () => {
      const output = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toMatch(/[✓✅⚠️]/);
    });
  });

  describe('Exit behavior', () => {
    it('should exit successfully', () => {
      try {
        execSync(`node ${cliPath} test`, { cwd: tempDir });
        expect(true).toBe(true);
      } catch (error) {
        fail(`Test command should not fail: ${error.message}`);
      }
    });
  });
});
