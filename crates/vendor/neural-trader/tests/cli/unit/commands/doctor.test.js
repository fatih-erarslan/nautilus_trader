/**
 * Unit tests for CLI doctor command
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

describe('CLI Doctor Command', () => {
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

  describe('Health checks', () => {
    it('should run health check', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('Running health check');
      expect(output).toContain('Health Check Results');
    });

    it('should check NAPI bindings', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('NAPI Bindings');
      expect(output).toMatch(/✅|⚠️/);
    });

    it('should check Node.js version', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('Node.js Version');
      expect(output).toMatch(/v\d+\.\d+\.\d+/);
    });

    it('should validate Node.js version is >= 18', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      const nodeVersion = process.version;
      const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);

      if (majorVersion >= 18) {
        expect(output).toContain('✅');
      } else {
        expect(output).toContain('❌');
        expect(output).toContain('requires >=18');
      }
    });

    it('should check for package.json', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('package.json');
    });

    it('should report package.json status', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      // Without package.json
      expect(output).toContain('Not found');

      // Create package.json
      fs.writeFileSync(
        path.join(tempDir, 'package.json'),
        JSON.stringify({ name: 'test' }, null, 2)
      );

      const output2 = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output2).toContain('Found');
    });

    it('should check for config.json', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('config.json');
    });
  });

  describe('Status reporting', () => {
    it('should display overall status', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toMatch(/All systems operational|Some issues found/);
    });

    it('should report operational when no critical issues', () => {
      // Create required files
      fs.writeFileSync(
        path.join(tempDir, 'package.json'),
        JSON.stringify({ name: 'test' }, null, 2)
      );

      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      // Node version should be fine in test environment
      expect(output).toContain('✅');
    });

    it('should use status symbols', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toMatch(/[✅⚠️ℹ️❌]/);
    });
  });

  describe('Output formatting', () => {
    it('should format results in table-like structure', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      // Check for aligned columns
      const lines = output.split('\n').filter(l => l.includes('✅') || l.includes('⚠️'));
      expect(lines.length).toBeGreaterThan(0);
    });

    it('should use consistent spacing', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      const lines = output.split('\n');
      const emptyLines = lines.filter(line => line.trim() === '');
      expect(emptyLines.length).toBeGreaterThan(0);
    });
  });

  describe('Multiple checks', () => {
    it('should run at least 4 checks', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      const checkLines = output.split('\n').filter(line =>
        line.match(/[✅⚠️ℹ️❌]/)
      );

      expect(checkLines.length).toBeGreaterThanOrEqual(4);
    });

    it('should check all critical components', () => {
      const output = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      const criticalChecks = [
        'NAPI Bindings',
        'Node.js Version',
        'package.json'
      ];

      criticalChecks.forEach(check => {
        expect(output).toContain(check);
      });
    });
  });

  describe('Exit behavior', () => {
    it('should exit successfully even with warnings', () => {
      try {
        execSync(`node ${cliPath} doctor`, { cwd: tempDir });
        expect(true).toBe(true);
      } catch (error) {
        fail(`Doctor command should not fail: ${error.message}`);
      }
    });

    it('should not throw on missing optional files', () => {
      try {
        execSync(`node ${cliPath} doctor`, { cwd: tempDir });
        expect(true).toBe(true);
      } catch (error) {
        fail('Should not fail on missing optional files');
      }
    });
  });
});
