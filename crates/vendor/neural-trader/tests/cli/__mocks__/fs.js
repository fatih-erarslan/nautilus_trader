/**
 * Mock filesystem for CLI testing
 */

const path = require('path');

class MockFileSystem {
  constructor() {
    this.files = new Map();
    this.dirs = new Set(['/']);
  }

  reset() {
    this.files.clear();
    this.dirs.clear();
    this.dirs.add('/');
  }

  existsSync(filePath) {
    return this.files.has(filePath) || this.dirs.has(filePath);
  }

  readFileSync(filePath, encoding) {
    if (!this.files.has(filePath)) {
      throw new Error(`ENOENT: no such file or directory, open '${filePath}'`);
    }
    return this.files.get(filePath);
  }

  writeFileSync(filePath, content) {
    const dir = path.dirname(filePath);
    if (!this.dirs.has(dir)) {
      throw new Error(`ENOENT: no such file or directory, open '${filePath}'`);
    }
    this.files.set(filePath, content);
  }

  mkdirSync(dirPath, options = {}) {
    if (options.recursive) {
      const parts = dirPath.split(path.sep).filter(Boolean);
      let current = '/';
      for (const part of parts) {
        current = path.join(current, part);
        this.dirs.add(current);
      }
    } else {
      this.dirs.add(dirPath);
    }
  }

  readdirSync(dirPath) {
    if (!this.dirs.has(dirPath)) {
      throw new Error(`ENOENT: no such file or directory, scandir '${dirPath}'`);
    }
    const results = [];
    for (const file of this.files.keys()) {
      if (path.dirname(file) === dirPath) {
        results.push(path.basename(file));
      }
    }
    for (const dir of this.dirs) {
      if (path.dirname(dir) === dirPath) {
        results.push(path.basename(dir));
      }
    }
    return results;
  }

  statSync(filePath) {
    return {
      isDirectory: () => this.dirs.has(filePath),
      isFile: () => this.files.has(filePath)
    };
  }
}

const mockFS = new MockFileSystem();

module.exports = {
  existsSync: jest.fn((...args) => mockFS.existsSync(...args)),
  readFileSync: jest.fn((...args) => mockFS.readFileSync(...args)),
  writeFileSync: jest.fn((...args) => mockFS.writeFileSync(...args)),
  mkdirSync: jest.fn((...args) => mockFS.mkdirSync(...args)),
  readdirSync: jest.fn((...args) => mockFS.readdirSync(...args)),
  statSync: jest.fn((...args) => mockFS.statSync(...args)),
  __mockFS: mockFS
};
