// CWTS Memory Safety Validation Framework
// Comprehensive memory safety testing for financial trading systems

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::collections::HashMap;
use std::backtrace::Backtrace;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Memory allocation tracking for safety validation
pub struct MemoryTracker {
    allocations: Mutex<HashMap<*mut u8, AllocationInfo>>,
    total_allocated: AtomicUsize,
    total_deallocated: AtomicUsize,
    peak_usage: AtomicUsize,
    allocation_count: AtomicUsize,
}

#[derive(Debug, Clone)]
struct AllocationInfo {
    size: usize,
    timestamp: DateTime<Utc>,
    backtrace: String,
}

static mut MEMORY_TRACKER: Option<MemoryTracker> = None;

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            allocations: Mutex::new(HashMap::new()),
            total_allocated: AtomicUsize::new(0),
            total_deallocated: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    pub fn track_allocation(&self, ptr: *mut u8, size: usize) {
        let mut allocations = self.allocations.lock().unwrap();
        let info = AllocationInfo {
            size,
            timestamp: Utc::now(),
            backtrace: format!("{:?}", Backtrace::capture()),
        };
        
        allocations.insert(ptr, info);
        
        let new_total = self.total_allocated.fetch_add(size, Ordering::AcqRel) + size;
        let current_peak = self.peak_usage.load(Ordering::Acquire);
        if new_total > current_peak {
            self.peak_usage.store(new_total, Ordering::Release);
        }
        
        self.allocation_count.fetch_add(1, Ordering::AcqRel);
    }

    pub fn track_deallocation(&self, ptr: *mut u8) -> Option<usize> {
        let mut allocations = self.allocations.lock().unwrap();
        if let Some(info) = allocations.remove(&ptr) {
            self.total_deallocated.fetch_add(info.size, Ordering::AcqRel);
            Some(info.size)
        } else {
            None
        }
    }

    pub fn get_statistics(&self) -> MemoryStatistics {
        let allocations = self.allocations.lock().unwrap();
        let current_allocated = self.total_allocated.load(Ordering::Acquire) - 
                                self.total_deallocated.load(Ordering::Acquire);
        
        MemoryStatistics {
            current_allocated,
            total_allocated: self.total_allocated.load(Ordering::Acquire),
            total_deallocated: self.total_deallocated.load(Ordering::Acquire),
            peak_usage: self.peak_usage.load(Ordering::Acquire),
            active_allocations: allocations.len(),
            allocation_count: self.allocation_count.load(Ordering::Acquire),
        }
    }

    pub fn detect_leaks(&self) -> Vec<MemoryLeak> {
        let allocations = self.allocations.lock().unwrap();
        let mut leaks = Vec::new();
        let current_time = Utc::now();
        
        for (ptr, info) in allocations.iter() {
            let age = current_time.signed_duration_since(info.timestamp);
            if age.num_seconds() > 60 { // Consider allocations older than 1 minute as potential leaks
                leaks.push(MemoryLeak {
                    ptr: *ptr as usize,
                    size: info.size,
                    age_seconds: age.num_seconds(),
                    allocation_backtrace: info.backtrace.clone(),
                });
            }
        }
        
        leaks
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    pub current_allocated: usize,
    pub total_allocated: usize,
    pub total_deallocated: usize,
    pub peak_usage: usize,
    pub active_allocations: usize,
    pub allocation_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    pub ptr: usize,
    pub size: usize,
    pub age_seconds: i64,
    pub allocation_backtrace: String,
}

/// Custom allocator for memory safety validation
pub struct SafetyValidatingAllocator;

unsafe impl GlobalAlloc for SafetyValidatingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        
        if !ptr.is_null() {
            if let Some(tracker) = &MEMORY_TRACKER {
                tracker.track_allocation(ptr, layout.size());
            }
        }
        
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        if let Some(tracker) = &MEMORY_TRACKER {
            tracker.track_deallocation(ptr);
        }
        
        System.dealloc(ptr, _layout);
    }
}

/// Unsafe code audit results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsafeCodeAudit {
    pub file_path: String,
    pub line_number: usize,
    pub unsafe_operation: String,
    pub risk_level: RiskLevel,
    pub justification: Option<String>,
    pub mitigation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Memory safety validation framework
pub struct MemorySafetyValidator {
    tracker: &'static MemoryTracker,
    unsafe_audits: Vec<UnsafeCodeAudit>,
}

impl MemorySafetyValidator {
    pub fn new() -> Self {
        unsafe {
            if MEMORY_TRACKER.is_none() {
                MEMORY_TRACKER = Some(MemoryTracker::new());
            }
        }
        
        Self {
            tracker: unsafe { MEMORY_TRACKER.as_ref().unwrap() },
            unsafe_audits: Vec::new(),
        }
    }

    /// Initialize memory tracking for validation
    pub fn initialize_tracking(&mut self) {
        // Already initialized in new()
        println!("Memory safety tracking initialized");
    }

    /// Validate memory usage patterns
    pub fn validate_memory_patterns(&self) -> Result<MemorySafetyReport, String> {
        let stats = self.tracker.get_statistics();
        let leaks = self.tracker.detect_leaks();
        
        let mut issues = Vec::new();
        
        // Check for memory leaks
        if !leaks.is_empty() {
            issues.push(format!("Memory leaks detected: {} allocations", leaks.len()));
        }
        
        // Check for excessive memory usage
        if stats.peak_usage > 1024 * 1024 * 1024 { // 1GB threshold
            issues.push("Peak memory usage exceeds 1GB threshold".to_string());
        }
        
        // Check allocation/deallocation balance
        let balance_ratio = if stats.total_allocated > 0 {
            stats.total_deallocated as f64 / stats.total_allocated as f64
        } else {
            1.0
        };
        
        if balance_ratio < 0.95 {
            issues.push(format!("Memory deallocation ratio too low: {:.2}%", balance_ratio * 100.0));
        }
        
        Ok(MemorySafetyReport {
            timestamp: Utc::now(),
            statistics: stats,
            memory_leaks: leaks,
            issues,
            overall_status: if issues.is_empty() { 
                ValidationStatus::Passed 
            } else { 
                ValidationStatus::Failed 
            },
        })
    }

    /// Audit unsafe code blocks in the codebase
    pub fn audit_unsafe_code(&mut self, codebase_path: &str) -> Result<Vec<UnsafeCodeAudit>, String> {
        use std::fs;
        use std::path::Path;
        
        let mut audits = Vec::new();
        
        // Walk through Rust files and identify unsafe blocks
        if let Ok(entries) = fs::read_dir(codebase_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                    if let Ok(content) = fs::read_to_string(&path) {
                        audits.extend(self.analyze_unsafe_blocks(&path, &content)?);
                    }
                }
            }
        }
        
        self.unsafe_audits = audits.clone();
        Ok(audits)
    }

    /// Analyze unsafe blocks in source code
    fn analyze_unsafe_blocks(&self, file_path: &Path, content: &str) -> Result<Vec<UnsafeCodeAudit>, String> {
        let mut audits = Vec::new();
        
        for (line_num, line) in content.lines().enumerate() {
            if line.contains("unsafe") {
                let risk_level = self.assess_risk_level(line);
                let audit = UnsafeCodeAudit {
                    file_path: file_path.to_string_lossy().to_string(),
                    line_number: line_num + 1,
                    unsafe_operation: line.trim().to_string(),
                    risk_level,
                    justification: None, // Would be filled by code review
                    mitigation: self.suggest_mitigation(line),
                };
                audits.push(audit);
            }
        }
        
        Ok(audits)
    }

    /// Assess risk level of unsafe operations
    fn assess_risk_level(&self, code_line: &str) -> RiskLevel {
        if code_line.contains("ptr::write") || code_line.contains("ptr::read") {
            RiskLevel::High
        } else if code_line.contains("*mut") || code_line.contains("*const") {
            RiskLevel::High
        } else if code_line.contains("transmute") {
            RiskLevel::Critical
        } else if code_line.contains("from_raw") {
            RiskLevel::High
        } else if code_line.contains("as_mut") || code_line.contains("as_ref") {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Suggest mitigation strategies for unsafe code
    fn suggest_mitigation(&self, code_line: &str) -> Option<String> {
        if code_line.contains("ptr::write") {
            Some("Consider using safe alternatives like Vec::push or validated indexing".to_string())
        } else if code_line.contains("transmute") {
            Some("Replace with safe casting or implement From/Into traits".to_string())
        } else if code_line.contains("*mut") {
            Some("Use smart pointers like Box, Arc, or Rc instead of raw pointers".to_string())
        } else {
            None
        }
    }

    /// Run comprehensive memory safety stress test
    pub fn run_stress_test(&self, duration_seconds: u64) -> Result<MemorySafetyReport, String> {
        use std::{thread, time::Duration};
        
        println!("Running memory safety stress test for {} seconds...", duration_seconds);
        
        let start_time = Utc::now();
        let test_duration = Duration::from_secs(duration_seconds);
        let start_instant = std::time::Instant::now();
        
        // Simulate high-frequency memory allocations/deallocations
        while start_instant.elapsed() < test_duration {
            // Allocate various sized blocks
            let sizes = vec![64, 128, 256, 512, 1024, 2048];
            let mut allocations = Vec::new();
            
            for &size in &sizes {
                let layout = Layout::from_size_align(size, 8).unwrap();
                let ptr = unsafe { std::alloc::alloc(layout) };
                if !ptr.is_null() {
                    allocations.push((ptr, layout));
                }
            }
            
            // Deallocate half immediately, keep half for a bit
            for (i, (ptr, layout)) in allocations.iter().enumerate() {
                if i % 2 == 0 {
                    unsafe { std::alloc::dealloc(*ptr, *layout) };
                }
            }
            
            thread::sleep(Duration::from_millis(1));
            
            // Deallocate remaining
            for (i, (ptr, layout)) in allocations.iter().enumerate() {
                if i % 2 == 1 {
                    unsafe { std::alloc::dealloc(*ptr, *layout) };
                }
            }
        }
        
        // Generate final report
        self.validate_memory_patterns()
    }

    /// Generate comprehensive safety report
    pub fn generate_comprehensive_report(&self) -> String {
        let memory_report = self.validate_memory_patterns().unwrap_or_else(|e| {
            MemorySafetyReport {
                timestamp: Utc::now(),
                statistics: MemoryStatistics {
                    current_allocated: 0,
                    total_allocated: 0,
                    total_deallocated: 0,
                    peak_usage: 0,
                    active_allocations: 0,
                    allocation_count: 0,
                },
                memory_leaks: Vec::new(),
                issues: vec![e],
                overall_status: ValidationStatus::Failed,
            }
        });
        
        let mut report = String::new();
        
        report.push_str("# CWTS Memory Safety Validation Report\n\n");
        report.push_str(&format!("Generated: {}\n\n", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        
        // Memory statistics
        report.push_str("## Memory Statistics\n");
        report.push_str(&format!("- Current Allocated: {} bytes\n", memory_report.statistics.current_allocated));
        report.push_str(&format!("- Total Allocated: {} bytes\n", memory_report.statistics.total_allocated));
        report.push_str(&format!("- Total Deallocated: {} bytes\n", memory_report.statistics.total_deallocated));
        report.push_str(&format!("- Peak Usage: {} bytes\n", memory_report.statistics.peak_usage));
        report.push_str(&format!("- Active Allocations: {}\n", memory_report.statistics.active_allocations));
        report.push_str(&format!("- Total Allocation Count: {}\n\n", memory_report.statistics.allocation_count));
        
        // Memory leaks
        if !memory_report.memory_leaks.is_empty() {
            report.push_str("## ❌ Memory Leaks Detected\n");
            for leak in &memory_report.memory_leaks {
                report.push_str(&format!("- Leak at 0x{:x}: {} bytes (age: {} seconds)\n", 
                                       leak.ptr, leak.size, leak.age_seconds));
            }
            report.push_str("\n");
        } else {
            report.push_str("## ✅ No Memory Leaks Detected\n\n");
        }
        
        // Unsafe code audit
        report.push_str("## Unsafe Code Audit\n");
        let critical_count = self.unsafe_audits.iter().filter(|a| matches!(a.risk_level, RiskLevel::Critical)).count();
        let high_count = self.unsafe_audits.iter().filter(|a| matches!(a.risk_level, RiskLevel::High)).count();
        let medium_count = self.unsafe_audits.iter().filter(|a| matches!(a.risk_level, RiskLevel::Medium)).count();
        let low_count = self.unsafe_audits.iter().filter(|a| matches!(a.risk_level, RiskLevel::Low)).count();
        
        report.push_str(&format!("- Critical Risk: {} blocks\n", critical_count));
        report.push_str(&format!("- High Risk: {} blocks\n", high_count));
        report.push_str(&format!("- Medium Risk: {} blocks\n", medium_count));
        report.push_str(&format!("- Low Risk: {} blocks\n", low_count));
        report.push_str(&format!("- Total Unsafe Blocks: {}\n\n", self.unsafe_audits.len()));
        
        // Overall status
        report.push_str("## Overall Status\n");
        match memory_report.overall_status {
            ValidationStatus::Passed => report.push_str("✅ **PASSED** - Memory safety validation successful\n"),
            ValidationStatus::Failed => report.push_str("❌ **FAILED** - Memory safety issues detected\n"),
            ValidationStatus::Warning => report.push_str("⚠️ **WARNING** - Potential memory safety concerns\n"),
        }
        
        // Issues
        if !memory_report.issues.is_empty() {
            report.push_str("\n## Issues Identified\n");
            for issue in &memory_report.issues {
                report.push_str(&format!("- {}\n", issue));
            }
        }
        
        report
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySafetyReport {
    pub timestamp: DateTime<Utc>,
    pub statistics: MemoryStatistics,
    pub memory_leaks: Vec<MemoryLeak>,
    pub issues: Vec<String>,
    pub overall_status: ValidationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Passed,
    Failed,
    Warning,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_tracker_creation() {
        let tracker = MemoryTracker::new();
        let stats = tracker.get_statistics();
        assert_eq!(stats.current_allocated, 0);
        assert_eq!(stats.active_allocations, 0);
    }
    
    #[test]
    fn test_memory_safety_validator_creation() {
        let validator = MemorySafetyValidator::new();
        assert!(validator.unsafe_audits.is_empty());
    }
    
    #[test]
    fn test_risk_level_assessment() {
        let validator = MemorySafetyValidator::new();
        
        assert!(matches!(
            validator.assess_risk_level("unsafe { transmute(x) }"), 
            RiskLevel::Critical
        ));
        
        assert!(matches!(
            validator.assess_risk_level("unsafe { ptr::write(ptr, value) }"), 
            RiskLevel::High
        ));
    }
}