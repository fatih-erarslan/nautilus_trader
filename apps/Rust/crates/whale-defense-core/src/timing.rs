//! Ultra-high precision timing for sub-microsecond whale defense
//! 
//! This module provides nanosecond-precision timing optimized for
//! performance-critical whale detection and defense operations.

use crate::{
    error::{WhaleDefenseError, Result},
    config::*,
    AtomicU64, Ordering,
};
use core::{
    arch::x86_64::_rdtsc,
    sync::atomic::compiler_fence,
    mem::MaybeUninit,
};
use serde::{Deserialize, Serialize};

/// High-precision timestamp using hardware TSC (Time Stamp Counter)
/// 
/// This provides the fastest possible timing on x86_64 systems.
/// Resolution is typically sub-nanosecond (depends on CPU frequency).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Timestamp {
    /// Raw TSC value
    tsc: u64,
}

impl Timestamp {
    /// Get current timestamp using RDTSC instruction
    /// 
    /// # Performance
    /// - Latency: ~10-30 CPU cycles
    /// - No system call overhead
    /// - Monotonic within single CPU core
    /// 
    /// # Safety
    /// Uses inline assembly for maximum performance.
    /// TSC values are only comparable on the same CPU core.
    #[inline(always)]
    pub fn now() -> Self {
        unsafe {
            // Use RDTSC for minimum latency
            // Note: This is not serializing, for maximum performance
            compiler_fence(Ordering::Acquire);
            let tsc = _rdtsc();
            compiler_fence(Ordering::Release);
            Self { tsc }
        }
    }
    
    /// Get serializing timestamp for precise measurements
    /// 
    /// Uses RDTSCP instruction which is serializing.
    /// Slightly slower but more accurate for benchmarking.
    #[inline(always)]
    pub fn now_precise() -> Self {
        unsafe {
            compiler_fence(Ordering::SeqCst);
            let tsc = core::arch::x86_64::__rdtscp(&mut 0);
            compiler_fence(Ordering::SeqCst);
            Self { tsc }
        }
    }
    
    /// Create timestamp from raw TSC value
    #[inline(always)]
    pub const fn from_tsc(tsc: u64) -> Self {
        Self { tsc }
    }
    
    /// Get raw TSC value
    #[inline(always)]
    pub const fn as_tsc(self) -> u64 {
        self.tsc
    }
    
    /// Convert to nanoseconds (approximate)
    /// 
    /// # Note
    /// This uses a calibrated TSC frequency. Accuracy depends on
    /// CPU frequency scaling and thermal throttling.
    #[inline(always)]
    pub fn as_nanos(self) -> u64 {
        // Get cached TSC frequency
        let freq_ghz = TSC_FREQUENCY.load(Ordering::Relaxed) as f64 / 1e9;
        if freq_ghz > 0.0 {
            (self.tsc as f64 / freq_ghz) as u64
        } else {
            // Fallback: assume 3 GHz
            self.tsc / 3
        }
    }
    
    /// Get elapsed time since this timestamp in nanoseconds
    #[inline(always)]
    pub fn elapsed_nanos(self) -> u64 {
        let now = Self::now();
        if now.tsc >= self.tsc {
            (now.tsc - self.tsc) / TSC_CYCLES_PER_NS.load(Ordering::Relaxed).max(1)
        } else {
            0
        }
    }
    
    /// Get elapsed time since this timestamp in TSC cycles
    #[inline(always)]
    pub fn elapsed_cycles(self) -> u64 {
        let now = Self::now();
        if now.tsc >= self.tsc {
            now.tsc - self.tsc
        } else {
            0
        }
    }
    
    /// Add duration in nanoseconds
    #[inline(always)]
    pub fn add_nanos(self, nanos: u64) -> Self {
        let cycles = nanos * TSC_CYCLES_PER_NS.load(Ordering::Relaxed);
        Self {
            tsc: self.tsc + cycles,
        }
    }
    
    /// Subtract duration in nanoseconds
    #[inline(always)]
    pub fn sub_nanos(self, nanos: u64) -> Self {
        let cycles = nanos * TSC_CYCLES_PER_NS.load(Ordering::Relaxed);
        Self {
            tsc: self.tsc.saturating_sub(cycles),
        }
    }
}

/// Cached TSC frequency in Hz
static TSC_FREQUENCY: AtomicU64 = AtomicU64::new(0);

/// Cached TSC cycles per nanosecond
static TSC_CYCLES_PER_NS: AtomicU64 = AtomicU64::new(3); // Default 3 GHz

/// Timing calibration state
static CALIBRATION_DONE: AtomicU64 = AtomicU64::new(0);

/// Calibrate TSC frequency for accurate time conversion
/// 
/// This should be called once at startup to calibrate timing.
/// Uses multiple measurement methods for accuracy.
pub fn calibrate_tsc() -> Result<()> {
    // Check if already calibrated
    if CALIBRATION_DONE.load(Ordering::Acquire) != 0 {
        return Ok(());
    }
    
    #[cfg(feature = "std")]
    {
        let freq = calibrate_tsc_with_sleep()?;
        TSC_FREQUENCY.store(freq, Ordering::Release);
        TSC_CYCLES_PER_NS.store((freq as f64 / 1e9) as u64, Ordering::Release);
        CALIBRATION_DONE.store(1, Ordering::Release);
        Ok(())
    }
    
    #[cfg(not(feature = "std"))]
    {
        // No-std environment: use CPUID if available
        let freq = estimate_tsc_frequency_cpuid().unwrap_or(3_000_000_000); // 3 GHz default
        TSC_FREQUENCY.store(freq, Ordering::Release);
        TSC_CYCLES_PER_NS.store((freq as f64 / 1e9) as u64, Ordering::Release);
        CALIBRATION_DONE.store(1, Ordering::Release);
        Ok(())
    }
}

/// Calibrate TSC using sleep (std environment only)
#[cfg(feature = "std")]
fn calibrate_tsc_with_sleep() -> Result<u64> {
    use std::time::{Duration, Instant};
    use std::thread::sleep;
    
    let mut measurements = Vec::new();
    
    // Take multiple measurements for accuracy
    for _ in 0..5 {
        let start_tsc = Timestamp::now();
        let start_time = Instant::now();
        
        // Sleep for a short duration
        sleep(Duration::from_millis(10));
        
        let end_tsc = Timestamp::now();
        let end_time = Instant::now();
        
        let tsc_cycles = end_tsc.tsc - start_tsc.tsc;
        let wall_time_ns = end_time.duration_since(start_time).as_nanos() as u64;
        
        if wall_time_ns > 0 {
            let freq = (tsc_cycles as f64 * 1e9 / wall_time_ns as f64) as u64;
            measurements.push(freq);
        }
    }
    
    if measurements.is_empty() {
        return Err(WhaleDefenseError::TimingError);
    }
    
    // Use median frequency to avoid outliers
    measurements.sort_unstable();
    let median_freq = measurements[measurements.len() / 2];
    
    Ok(median_freq)
}

/// Estimate TSC frequency using CPUID (no-std environment)
#[cfg(not(feature = "std"))]
fn estimate_tsc_frequency_cpuid() -> Option<u64> {
    // Try to get CPU base frequency from CPUID
    unsafe {
        let cpuid = core::arch::x86_64::__cpuid(0x16);
        if cpuid.eax > 0 {
            // EAX contains base frequency in MHz
            Some((cpuid.eax as u64) * 1_000_000)
        } else {
            None
        }
    }
}

/// Performance timer for measuring code execution time
/// 
/// Optimized for minimal overhead in performance-critical paths.
#[derive(Debug)]
pub struct PerfTimer {
    start_time: Timestamp,
    name: &'static str,
}

impl PerfTimer {
    /// Start new performance timer
    #[inline(always)]
    pub fn start(name: &'static str) -> Self {
        Self {
            start_time: Timestamp::now(),
            name,
        }
    }
    
    /// Get elapsed time in nanoseconds
    #[inline(always)]
    pub fn elapsed_nanos(&self) -> u64 {
        self.start_time.elapsed_nanos()
    }
    
    /// Get elapsed time in TSC cycles
    #[inline(always)]
    pub fn elapsed_cycles(&self) -> u64 {
        self.start_time.elapsed_cycles()
    }
    
    /// Stop timer and return elapsed nanoseconds
    #[inline(always)]
    pub fn stop(self) -> u64 {
        self.elapsed_nanos()
    }
}

/// Timing statistics for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    /// Operation name
    pub name: String,
    /// Sample count
    pub count: u64,
    /// Total time in nanoseconds
    pub total_nanos: u64,
    /// Minimum time in nanoseconds
    pub min_nanos: u64,
    /// Maximum time in nanoseconds
    pub max_nanos: u64,
    /// Average time in nanoseconds
    pub avg_nanos: f64,
}

impl TimingStats {
    /// Create new timing statistics
    pub fn new(name: String) -> Self {
        Self {
            name,
            count: 0,
            total_nanos: 0,
            min_nanos: u64::MAX,
            max_nanos: 0,
            avg_nanos: 0.0,
        }
    }
    
    /// Add measurement
    pub fn add_measurement(&mut self, nanos: u64) {
        self.count += 1;
        self.total_nanos += nanos;
        self.min_nanos = self.min_nanos.min(nanos);
        self.max_nanos = self.max_nanos.max(nanos);
        self.avg_nanos = self.total_nanos as f64 / self.count as f64;
    }
    
    /// Reset statistics
    pub fn reset(&mut self) {
        self.count = 0;
        self.total_nanos = 0;
        self.min_nanos = u64::MAX;
        self.max_nanos = 0;
        self.avg_nanos = 0.0;
    }
}

/// Timing profiler for whale defense operations
pub struct TimingProfiler {
    stats: std::collections::HashMap<String, TimingStats>,
}

impl TimingProfiler {
    /// Create new timing profiler
    pub fn new() -> Self {
        Self {
            stats: std::collections::HashMap::new(),
        }
    }
    
    /// Record timing measurement
    pub fn record(&mut self, name: &str, nanos: u64) {
        let stats = self.stats.entry(name.to_string())
            .or_insert_with(|| TimingStats::new(name.to_string()));
        stats.add_measurement(nanos);
    }
    
    /// Get statistics for operation
    pub fn get_stats(&self, name: &str) -> Option<&TimingStats> {
        self.stats.get(name)
    }
    
    /// Get all statistics
    pub fn get_all_stats(&self) -> &std::collections::HashMap<String, TimingStats> {
        &self.stats
    }
    
    /// Reset all statistics
    pub fn reset(&mut self) {
        for stats in self.stats.values_mut() {
            stats.reset();
        }
    }
}

/// Macro for easy performance timing
#[macro_export]
macro_rules! time_it {
    ($profiler:expr, $name:expr, $block:block) => {
        {
            let timer = PerfTimer::start($name);
            let result = $block;
            let elapsed = timer.stop();
            $profiler.record($name, elapsed);
            result
        }
    };
}

/// Sleep for specified nanoseconds using busy wait
/// 
/// More accurate than thread sleep for sub-microsecond timing.
/// Use sparingly as it consumes CPU cycles.
#[inline(always)]
pub fn busy_wait_nanos(nanos: u64) {
    let start = Timestamp::now();
    while start.elapsed_nanos() < nanos {
        core::hint::spin_loop();
    }
}

/// Sleep for specified TSC cycles using busy wait
#[inline(always)]
pub fn busy_wait_cycles(cycles: u64) {
    let start = Timestamp::now();
    while start.elapsed_cycles() < cycles {
        core::hint::spin_loop();
    }
}

/// Get current TSC frequency in Hz
pub fn get_tsc_frequency() -> u64 {
    TSC_FREQUENCY.load(Ordering::Acquire)
}

/// Get TSC cycles per nanosecond
pub fn get_tsc_cycles_per_ns() -> u64 {
    TSC_CYCLES_PER_NS.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timestamp_creation() {
        let ts1 = Timestamp::now();
        let ts2 = Timestamp::now();
        
        // Second timestamp should be later
        assert!(ts2.as_tsc() >= ts1.as_tsc());
    }
    
    #[test]
    fn test_timestamp_elapsed() {
        let start = Timestamp::now();
        
        // Small busy wait
        for _ in 0..1000 {
            core::hint::black_box(42);
        }
        
        let elapsed = start.elapsed_cycles();
        assert!(elapsed > 0);
    }
    
    #[test]
    fn test_perf_timer() {
        let timer = PerfTimer::start("test");
        
        // Small computation
        let mut sum = 0;
        for i in 0..100 {
            sum += i;
        }
        core::hint::black_box(sum);
        
        let elapsed = timer.stop();
        assert!(elapsed > 0);
    }
    
    #[test]
    fn test_timing_stats() {
        let mut stats = TimingStats::new("test".to_string());
        
        stats.add_measurement(100);
        stats.add_measurement(200);
        stats.add_measurement(150);
        
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min_nanos, 100);
        assert_eq!(stats.max_nanos, 200);
        assert_eq!(stats.avg_nanos, 150.0);
    }
    
    #[cfg(feature = "std")]
    #[test]
    fn test_tsc_calibration() {
        assert!(calibrate_tsc().is_ok());
        assert!(get_tsc_frequency() > 0);
        assert!(get_tsc_cycles_per_ns() > 0);
    }
}