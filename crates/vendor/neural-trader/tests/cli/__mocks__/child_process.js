/**
 * Mock child_process for CLI testing
 */

const { EventEmitter } = require('events');

class MockChildProcess extends EventEmitter {
  constructor(command, args, options) {
    super();
    this.command = command;
    this.args = args;
    this.options = options;
    this.killed = false;

    // Simulate async process completion
    setImmediate(() => {
      if (this.command === 'npm' && this.args[0] === 'install') {
        this.emit('exit', 0);
      } else {
        this.emit('exit', 0);
      }
    });
  }

  kill() {
    this.killed = true;
    this.emit('exit', 1);
  }
}

const spawn = jest.fn((command, args, options) => {
  return new MockChildProcess(command, args, options);
});

const execSync = jest.fn((command, options) => {
  if (command.includes('npm --version')) {
    return '10.2.3';
  }
  if (command.includes('git --version')) {
    return 'git version 2.40.0';
  }
  return '';
});

module.exports = {
  spawn,
  execSync,
  MockChildProcess
};
