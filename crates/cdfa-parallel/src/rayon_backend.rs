//! Rayon-based parallel processing backend
//!
//! Provides data-parallel implementations of CDFA algorithms using Rayon's
//! work-stealing thread pool for efficient parallelization.

use rayon::prelude::*;
use std::sync::Arc;

use cdfa_core::error::Result;
use cdfa_core::traits::{CognitiveDiversityAnalyzer, SignalProcessor};
use cdfa_core::types::{AnalysisResult, Signal};

use crate::parallel_algorithms::ParallelDiversityCalculator;
use crate::lock_free::LockFreeResultAggregator;

/// Rayon-based parallel signal processor
pub struct RayonSignalProcessor {
    /// Inner processors to apply
    processors: Vec<Arc<dyn SignalProcessor>>,
    
    /// Chunk size for parallel processing
    chunk_size: usize,
}

impl RayonSignalProcessor {
    /// Creates a new Rayon signal processor
    pub fn new(processors: Vec<Arc<dyn SignalProcessor>>, chunk_size: usize) -> Self {
        Self {
            processors,
            chunk_size,
        }
    }
    
    /// Processes signals in parallel
    pub fn process_batch(&self, signals: &[Signal]) -> Result<Vec<Signal>> {
        signals
            .par_chunks(self.chunk_size)
            .flat_map(|chunk| {
                chunk.iter().map(|signal| {
                    let mut processed = signal.clone();
                    for processor in &self.processors {
                        processed = processor.process(&processed)?;
                    }
                    Ok(processed)
                }).collect::<Result<Vec<_>>>()
            })
            .collect()
    }
}

/// Rayon-based parallel analyzer
pub struct RayonDiversityAnalyzer {
    /// Analyzers to run in parallel
    analyzers: Vec<Arc<dyn CognitiveDiversityAnalyzer>>,
    
    /// Result aggregator
    aggregator: Arc<LockFreeResultAggregator>,
    
    /// Diversity calculator
    diversity_calc: Arc<ParallelDiversityCalculator>,
}

impl RayonDiversityAnalyzer {
    /// Creates a new Rayon diversity analyzer
    pub fn new(
        analyzers: Vec<Arc<dyn CognitiveDiversityAnalyzer>>,
        num_threads: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            analyzers,
            aggregator: Arc::new(LockFreeResultAggregator::new()),
            diversity_calc: Arc::new(ParallelDiversityCalculator::new(num_threads, 64)?),
        })
    }
    
    /// Analyzes signals using all analyzers in parallel
    pub fn analyze_parallel(&self, signals: &[Signal]) -> Result<Vec<AnalysisResult>> {
        // Run all analyzers in parallel
        let results: Vec<AnalysisResult> = self.analyzers
            .par_iter()
            .filter_map(|analyzer| {
                match analyzer.analyze(signals) {
                    Ok(result) => {
                        self.aggregator.add_result(result.clone());
                        Some(result)
                    }
                    Err(_) => None,
                }
            })
            .collect();
        
        Ok(results)
    }
    
    /// Computes diversity matrix for the results
    pub fn compute_diversity(&self, results: &[AnalysisResult]) -> cdfa_core::types::DiversityMatrix {
        self.diversity_calc.compute_diversity_matrix(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cdfa_core::types::SignalId;
    
    #[test]
    fn test_rayon_signal_processor() {
        struct MockProcessor;
        
        impl SignalProcessor for MockProcessor {
            fn process(&self, signal: &Signal) -> Result<Signal> {
                let mut processed = signal.clone();
                for val in processed.as_mut_slice() {
                    *val *= 2.0;
                }
                Ok(processed)
            }
            
            fn processor_id(&self) -> &'static str {
                "mock_processor"
            }
        }
        
        let processor = RayonSignalProcessor::new(
            vec![Arc::new(MockProcessor)],
            2,
        );
        
        let signals = vec![
            Signal::new(SignalId(1), 1000, vec![1.0, 2.0, 3.0]),
            Signal::new(SignalId(2), 2000, vec![4.0, 5.0, 6.0]),
        ];
        
        let processed = processor.process_batch(&signals).unwrap();
        assert_eq!(processed.len(), 2);
        assert_eq!(processed[0].values[0], 2.0); // 1.0 * 2
        assert_eq!(processed[1].values[0], 8.0); // 4.0 * 2
    }
}