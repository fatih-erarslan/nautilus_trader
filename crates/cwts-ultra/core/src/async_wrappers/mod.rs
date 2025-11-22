/// Production-grade async-safe wrappers for git2 and RocksDB
///
/// This module provides thread-safe, async-compatible wrappers for:
/// - git2::Repository: Async-safe Git operations
/// - RocksDB: Async-safe database operations with proper Send/Sync bounds
///
/// All wrappers maintain performance while ensuring production reliability.
pub mod git_async;
pub mod rocksdb_async;

pub use git_async::AsyncGitRepository;
pub use rocksdb_async::AsyncRocksDB;
