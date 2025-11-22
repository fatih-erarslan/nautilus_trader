use crate::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;

/// Setup logging configuration for the autopoiesis system
pub fn setup_logging() -> Result<()> {
    // Basic logging setup - in production would use tracing/log
    println!("Logging system initialized for autopoiesis framework");
    Ok(())
}

/// Log levels for the system
#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Simple logger implementation
pub struct Logger {
    level: LogLevel,
    file_path: Option<String>,
}

impl Logger {
    pub fn new(level: LogLevel) -> Self {
        Self {
            level,
            file_path: None,
        }
    }
    
    pub fn with_file<S: Into<String>>(mut self, path: S) -> Self {
        self.file_path = Some(path.into());
        self
    }
    
    pub fn log(&self, level: LogLevel, message: &str) {
        if self.should_log(level) {
            let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f");
            let formatted = format!("[{}] {:?}: {}", timestamp, level, message);
            
            println!("{}", formatted);
            
            if let Some(ref path) = self.file_path {
                if let Ok(mut file) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path) 
                {
                    let _ = writeln!(file, "{}", formatted);
                }
            }
        }
    }
    
    fn should_log(&self, level: LogLevel) -> bool {
        match (self.level, level) {
            (LogLevel::Debug, _) => true,
            (LogLevel::Info, LogLevel::Debug) => false,
            (LogLevel::Info, _) => true,
            (LogLevel::Warn, LogLevel::Debug | LogLevel::Info) => false,
            (LogLevel::Warn, _) => true,
            (LogLevel::Error, LogLevel::Error) => true,
            (LogLevel::Error, _) => false,
        }
    }
    
    pub fn debug(&self, message: &str) {
        self.log(LogLevel::Debug, message);
    }
    
    pub fn info(&self, message: &str) {
        self.log(LogLevel::Info, message);
    }
    
    pub fn warn(&self, message: &str) {
        self.log(LogLevel::Warn, message);
    }
    
    pub fn error(&self, message: &str) {
        self.log(LogLevel::Error, message);
    }
}