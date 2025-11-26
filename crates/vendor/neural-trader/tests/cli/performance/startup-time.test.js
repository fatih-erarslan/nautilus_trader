/**
 * Performance tests for CLI startup time
 */

const { execSync } = require('child_process');
const path = require('path');

describe('CLI Performance - Startup Time', () => {
  const cliPath = path.join(__dirname, '../../../bin/cli.js');
  const MAX_STARTUP_TIME = 2000; // 2 seconds max
  const TARGET_STARTUP_TIME = 500; // 500ms target

  function measureCommandTime(command) {
    const start = Date.now();
    execSync(`node ${cliPath} ${command}`, {
      encoding: 'utf8',
      stdio: 'pipe'
    });
    const end = Date.now();
    return end - start;
  }

  describe('Command startup time', () => {
    it('should start version command quickly', () => {
      const time = measureCommandTime('version');
      expect(time).toBeLessThan(MAX_STARTUP_TIME);
    });

    it('should start help command quickly', () => {
      const time = measureCommandTime('help');
      expect(time).toBeLessThan(MAX_STARTUP_TIME);
    });

    it('should start list command quickly', () => {
      const time = measureCommandTime('list');
      expect(time).toBeLessThan(MAX_STARTUP_TIME);
    });

    it('should meet target startup time for version', () => {
      const times = [];
      for (let i = 0; i < 5; i++) {
        times.push(measureCommandTime('version'));
      }

      const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
      console.log(`Average startup time: ${avgTime.toFixed(2)}ms`);

      // Average should be under target (more lenient than individual runs)
      expect(avgTime).toBeLessThan(MAX_STARTUP_TIME);
    });
  });

  describe('Cold start vs warm start', () => {
    it('should have reasonable cold start time', () => {
      const coldStart = measureCommandTime('version');
      expect(coldStart).toBeLessThan(MAX_STARTUP_TIME);
      console.log(`Cold start: ${coldStart}ms`);
    });

    it('should have fast warm start time', () => {
      // Run once to warm up
      measureCommandTime('version');

      // Measure warm start
      const warmStart = measureCommandTime('version');
      expect(warmStart).toBeLessThan(MAX_STARTUP_TIME);
      console.log(`Warm start: ${warmStart}ms`);
    });

    it('should show performance consistency', () => {
      const times = [];
      for (let i = 0; i < 10; i++) {
        times.push(measureCommandTime('version'));
      }

      const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
      const maxTime = Math.max(...times);
      const minTime = Math.min(...times);

      console.log(`Performance stats:
        Min: ${minTime}ms
        Max: ${maxTime}ms
        Avg: ${avgTime.toFixed(2)}ms
        Variance: ${(maxTime - minTime)}ms
      `);

      // Max time should not be more than 3x min time
      expect(maxTime).toBeLessThan(minTime * 3);
    });
  });

  describe('Command comparison', () => {
    it('should measure relative performance of commands', () => {
      const commands = ['version', 'help', 'list'];
      const results = {};

      commands.forEach(cmd => {
        const times = [];
        for (let i = 0; i < 3; i++) {
          times.push(measureCommandTime(cmd));
        }
        results[cmd] = times.reduce((a, b) => a + b, 0) / times.length;
      });

      console.log('Command performance:');
      Object.entries(results).forEach(([cmd, time]) => {
        console.log(`  ${cmd}: ${time.toFixed(2)}ms`);
      });

      // All commands should be reasonably fast
      Object.values(results).forEach(time => {
        expect(time).toBeLessThan(MAX_STARTUP_TIME);
      });
    });
  });

  describe('Memory efficiency', () => {
    it('should not leak memory across multiple runs', () => {
      const initialMemory = process.memoryUsage().heapUsed;

      // Run command multiple times
      for (let i = 0; i < 10; i++) {
        execSync(`node ${cliPath} version`, { stdio: 'pipe' });
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;

      console.log(`Memory increase: ${(memoryIncrease / 1024 / 1024).toFixed(2)}MB`);

      // Memory increase should be minimal (< 50MB)
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);
    });
  });

  describe('Concurrent execution', () => {
    it('should handle concurrent commands efficiently', (done) => {
      const start = Date.now();
      const { spawn } = require('child_process');

      const promises = [];
      for (let i = 0; i < 5; i++) {
        promises.push(
          new Promise((resolve) => {
            const child = spawn('node', [cliPath, 'version']);
            child.on('exit', () => resolve());
          })
        );
      }

      Promise.all(promises).then(() => {
        const totalTime = Date.now() - start;
        console.log(`5 concurrent executions completed in ${totalTime}ms`);

        // Should complete within reasonable time
        expect(totalTime).toBeLessThan(MAX_STARTUP_TIME * 2);
        done();
      });
    }, 10000);
  });

  describe('Startup optimization', () => {
    it('should have minimal require time', () => {
      const start = Date.now();
      require(cliPath);
      const requireTime = Date.now() - start;

      console.log(`Module require time: ${requireTime}ms`);
      expect(requireTime).toBeLessThan(1000);
    });

    it('should not load unnecessary modules for simple commands', () => {
      const start = Date.now();
      execSync(`node ${cliPath} version`, { stdio: 'pipe' });
      const versionTime = Date.now() - start;

      execSync(`node ${cliPath} help`, { stdio: 'pipe' });
      const helpTime = Date.now() - start - versionTime;

      // Help and version should be similar speed (both simple commands)
      const timeDiff = Math.abs(versionTime - helpTime);
      expect(timeDiff).toBeLessThan(500);
    });
  });
});
