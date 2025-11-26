/**
 * Unit tests for CLI install command
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const os = require('os');

describe('CLI Install Command', () => {
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

  describe('Error handling', () => {
    it('should require package name', () => {
      try {
        execSync(`node ${cliPath} install`, {
          cwd: tempDir,
          stdio: 'pipe'
        });
        fail('Should have thrown an error');
      } catch (error) {
        expect(error.status).toBe(1);
        expect(error.stderr.toString()).toContain('Package name required');
      }
    });

    it('should require package.json', () => {
      try {
        execSync(`node ${cliPath} install @neural-trader/core`, {
          cwd: tempDir,
          stdio: 'pipe'
        });
        fail('Should have thrown an error');
      } catch (error) {
        expect(error.status).toBe(1);
        expect(error.stderr.toString()).toContain('No package.json found');
      }
    });

    it('should suggest running init first', () => {
      try {
        execSync(`node ${cliPath} install @neural-trader/core`, {
          cwd: tempDir,
          stdio: 'pipe'
        });
      } catch (error) {
        expect(error.stderr.toString()).toContain('neural-trader init');
      }
    });
  });

  describe('Installation with package.json', () => {
    beforeEach(() => {
      // Create minimal package.json
      const pkg = {
        name: 'test-project',
        version: '1.0.0',
        dependencies: {}
      };
      fs.writeFileSync(
        path.join(tempDir, 'package.json'),
        JSON.stringify(pkg, null, 2)
      );
    });

    it('should display installation message', (done) => {
      const child = spawn('node', [cliPath, 'install', 'lodash'], {
        cwd: tempDir,
        stdio: 'pipe'
      });

      let output = '';
      child.stdout.on('data', (data) => {
        output += data.toString();
      });

      child.on('exit', () => {
        expect(output).toContain('Installing');
        done();
      });
    }, 30000);

    it('should accept package names', (done) => {
      const child = spawn('node', [cliPath, 'install', '@neural-trader/core'], {
        cwd: tempDir,
        stdio: 'pipe'
      });

      let output = '';
      child.stdout.on('data', (data) => {
        output += data.toString();
      });

      child.on('exit', () => {
        expect(output).toContain('@neural-trader/core');
        done();
      });
    }, 30000);
  });

  describe('Usage information', () => {
    it('should show usage on error', () => {
      try {
        execSync(`node ${cliPath} install`, {
          cwd: tempDir,
          stdio: 'pipe'
        });
      } catch (error) {
        expect(error.stderr.toString()).toContain('Usage:');
        expect(error.stderr.toString()).toContain('neural-trader install');
      }
    });
  });
});
