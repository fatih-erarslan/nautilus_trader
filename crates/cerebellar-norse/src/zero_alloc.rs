//! Zero-allocation neural processing for ultra-low latency trading
//! 
//! Implements stack-allocated data structures, memory pools, and allocation-free
//! hot paths to achieve sub-microsecond processing times.

use std::mem::{MaybeUninit, align_of};
use std::ptr::{NonNull, write, read};
use std::marker::PhantomData;
use anyhow::{Result, anyhow};

/// Memory pool for zero-allocation neuron processing
pub struct ZeroAllocMemoryPool {
    /// Pre-allocated memory chunks
    chunks: Vec<MemoryChunk>,
    /// Current allocation offset
    current_offset: usize,
    /// Total pool size
    total_size: usize,
    /// Alignment requirement
    alignment: usize,
}

/// Memory chunk in the pool
struct MemoryChunk {
    /// Raw memory pointer
    ptr: NonNull<u8>,
    /// Chunk size
    size: usize,
    /// Used bytes
    used: usize,
    /// Next free offset
    next_free: usize,
}

/// Stack-allocated neuron state for hot paths
#[repr(C, align(64))]
pub struct StackNeuronState<const N: usize> {
    /// Membrane potentials (cache-aligned)
    v_mem: [f32; N],
    /// Synaptic currents
    i_syn: [f32; N],
    /// Spike outputs
    spikes: [bool; N],
    /// Refractory counters
    refractory: [u8; N],
    /// Neuron parameters
    params: NeuronParams,
    /// Number of active neurons
    active_count: usize,
}

/// Neuron parameters for zero-allocation processing
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NeuronParams {
    pub decay_mem: f32,
    pub decay_syn: f32,
    pub threshold: f32,
    pub reset_potential: f32,
    pub refractory_period: u8,
}

/// Zero-allocation batch processor
pub struct ZeroAllocBatchProcessor<const BATCH_SIZE: usize, const MAX_NEURONS: usize> {
    /// Stack-allocated neuron states
    neuron_states: [StackNeuronState<MAX_NEURONS>; BATCH_SIZE],
    /// Input buffer
    input_buffer: [[f32; MAX_NEURONS]; BATCH_SIZE],
    /// Output buffer
    output_buffer: [[bool; MAX_NEURONS]; BATCH_SIZE],
    /// Working memory
    temp_buffer: [f32; MAX_NEURONS],
    /// Current batch size
    current_batch_size: usize,
}

/// Memory-mapped neuron layer for ultra-fast access
#[repr(C, align(4096))] // Page-aligned for memory mapping
pub struct MemoryMappedLayer {
    /// Layer metadata
    header: LayerHeader,
    /// Neuron data (variable size)
    data: [u8; 0],
}

/// Layer header for memory-mapped data
#[repr(C)]
struct LayerHeader {
    neuron_count: u32,
    layer_type: u8,
    params_offset: u32,
    v_mem_offset: u32,
    i_syn_offset: u32,
    spikes_offset: u32,
    checksum: u32,
}

/// Lock-free circular buffer for spike events
pub struct LockFreeSpikeBuf<const SIZE: usize> {
    /// Spike events
    events: [SpikeEvent; SIZE],
    /// Head index (producer)
    head: std::sync::atomic::AtomicUsize,
    /// Tail index (consumer)
    tail: std::sync::atomic::AtomicUsize,
    /// Capacity mask (SIZE must be power of 2)
    mask: usize,
}

/// Single spike event
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SpikeEvent {
    /// Neuron ID
    neuron_id: u32,
    /// Timestamp (nanoseconds)
    timestamp: u64,
    /// Spike weight/amplitude
    weight: f32,
    /// Layer ID
    layer_id: u8,
}

impl ZeroAllocMemoryPool {
    /// Create new memory pool with specified size
    pub fn new(total_size: usize, alignment: usize) -> Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(anyhow!("Alignment must be power of 2"));
        }
        
        // Allocate initial chunk
        let initial_chunk = MemoryChunk::new(total_size, alignment)?;
        
        Ok(Self {
            chunks: vec![initial_chunk],
            current_offset: 0,
            total_size,
            alignment,
        })
    }
    
    /// Allocate memory from pool (zero-copy when possible)
    pub fn allocate<T>(&mut self, count: usize) -> Result<*mut T> {
        let size = std::mem::size_of::<T>() * count;
        let align = align_of::<T>().max(self.alignment);
        
        // Find suitable chunk
        for chunk in &mut self.chunks {
            if let Some(ptr) = chunk.allocate(size, align) {
                return Ok(ptr.as_ptr() as *mut T);
            }
        }
        
        // Need new chunk
        let chunk_size = (size * 2).max(self.total_size / 4);
        let new_chunk = MemoryChunk::new(chunk_size, self.alignment)?;
        self.chunks.push(new_chunk);
        
        // Try allocation again
        if let Some(chunk) = self.chunks.last_mut() {
            if let Some(ptr) = chunk.allocate(size, align) {
                return Ok(ptr.as_ptr() as *mut T);
            }
        }
        
        Err(anyhow!("Failed to allocate {} bytes", size))
    }
    
    /// Reset pool (mark all memory as available)
    pub fn reset(&mut self) {
        for chunk in &mut self.chunks {
            chunk.reset();
        }
        self.current_offset = 0;
    }
    
    /// Get memory usage statistics
    pub fn usage_stats(&self) -> MemoryUsageStats {
        let total_used = self.chunks.iter().map(|c| c.used).sum();
        let total_capacity = self.chunks.iter().map(|c| c.size).sum();
        
        MemoryUsageStats {
            total_capacity,
            total_used,
            utilization: total_used as f64 / total_capacity as f64,
            chunk_count: self.chunks.len(),
        }
    }
}

impl MemoryChunk {
    fn new(size: usize, alignment: usize) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(size, alignment)
            .map_err(|e| anyhow!("Invalid layout: {}", e))?;
        
        let ptr = unsafe {
            let raw_ptr = std::alloc::alloc(layout);
            if raw_ptr.is_null() {
                return Err(anyhow!("Failed to allocate {} bytes", size));
            }
            NonNull::new_unchecked(raw_ptr)
        };
        
        Ok(Self {
            ptr,
            size,
            used: 0,
            next_free: 0,
        })
    }
    
    fn allocate(&mut self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Align the next free offset
        let aligned_offset = (self.next_free + align - 1) & !(align - 1);
        
        if aligned_offset + size <= self.size {
            let ptr = unsafe {
                NonNull::new_unchecked(self.ptr.as_ptr().add(aligned_offset))
            };
            self.next_free = aligned_offset + size;
            self.used = self.used.max(self.next_free);
            Some(ptr)
        } else {
            None
        }
    }
    
    fn reset(&mut self) {
        self.used = 0;
        self.next_free = 0;
    }
}

impl Drop for MemoryChunk {
    fn drop(&mut self) {
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(self.size, 64);
            std::alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

impl<const N: usize> StackNeuronState<N> {
    /// Create new stack-allocated neuron state
    pub const fn new(params: NeuronParams) -> Self {
        Self {
            v_mem: [0.0; N],
            i_syn: [0.0; N],
            spikes: [false; N],
            refractory: [0; N],
            params,
            active_count: N,
        }
    }
    
    /// Process neurons with zero allocations (hot path)
    #[inline(always)]
    pub fn process_step_zero_alloc(&mut self, inputs: &[f32]) -> &[bool] {
        debug_assert!(inputs.len() <= N);
        
        // Use raw pointer arithmetic for maximum performance
        unsafe {
            let v_mem_ptr = self.v_mem.as_mut_ptr();
            let i_syn_ptr = self.i_syn.as_mut_ptr();
            let spikes_ptr = self.spikes.as_mut_ptr();
            let refractory_ptr = self.refractory.as_mut_ptr();
            let inputs_ptr = inputs.as_ptr();
            
            // Unrolled loop for better performance
            let count = inputs.len().min(N);
            let mut i = 0;
            
            // Process 4 neurons at a time (manual vectorization)
            while i + 4 <= count {
                self.process_4_neurons_unrolled(
                    v_mem_ptr.add(i),
                    i_syn_ptr.add(i),
                    spikes_ptr.add(i),
                    refractory_ptr.add(i),
                    inputs_ptr.add(i),
                );
                i += 4;
            }
            
            // Handle remaining neurons
            while i < count {
                self.process_single_neuron(
                    v_mem_ptr.add(i),
                    i_syn_ptr.add(i),
                    spikes_ptr.add(i),
                    refractory_ptr.add(i),
                    *inputs_ptr.add(i),
                );
                i += 1;
            }
        }
        
        &self.spikes[..inputs.len()]
    }
    
    /// Process 4 neurons in unrolled loop
    #[inline(always)]
    unsafe fn process_4_neurons_unrolled(
        &self,
        v_mem: *mut f32,
        i_syn: *mut f32,
        spikes: *mut bool,
        refractory: *mut u8,
        inputs: *const f32,
    ) {
        // Load parameters once
        let decay_mem = self.params.decay_mem;
        let decay_syn = self.params.decay_syn;
        let threshold = self.params.threshold;
        let reset = self.params.reset_potential;
        
        // Process neuron 0
        self.process_single_neuron_raw(
            v_mem.add(0), i_syn.add(0), spikes.add(0), refractory.add(0),
            *inputs.add(0), decay_mem, decay_syn, threshold, reset
        );
        
        // Process neuron 1
        self.process_single_neuron_raw(
            v_mem.add(1), i_syn.add(1), spikes.add(1), refractory.add(1),
            *inputs.add(1), decay_mem, decay_syn, threshold, reset
        );
        
        // Process neuron 2
        self.process_single_neuron_raw(
            v_mem.add(2), i_syn.add(2), spikes.add(2), refractory.add(2),
            *inputs.add(2), decay_mem, decay_syn, threshold, reset
        );
        
        // Process neuron 3
        self.process_single_neuron_raw(
            v_mem.add(3), i_syn.add(3), spikes.add(3), refractory.add(3),
            *inputs.add(3), decay_mem, decay_syn, threshold, reset
        );
    }
    
    /// Process single neuron (called from hot path)
    #[inline(always)]
    unsafe fn process_single_neuron(
        &self,
        v_mem: *mut f32,
        i_syn: *mut f32,
        spike: *mut bool,
        refractory: *mut u8,
        input: f32,
    ) {
        self.process_single_neuron_raw(
            v_mem, i_syn, spike, refractory, input,
            self.params.decay_mem,
            self.params.decay_syn,
            self.params.threshold,
            self.params.reset_potential,
        );
    }
    
    /// Raw neuron processing (fully inlined)
    #[inline(always)]
    unsafe fn process_single_neuron_raw(
        &self,
        v_mem: *mut f32,
        i_syn: *mut f32,
        spike: *mut bool,
        refractory: *mut u8,
        input: f32,
        decay_mem: f32,
        decay_syn: f32,
        threshold: f32,
        reset: f32,
    ) {
        let ref_count = read(refractory);
        
        if ref_count > 0 {
            // In refractory period
            write(refractory, ref_count - 1);
            write(spike, false);
        } else {
            // Update synaptic current
            let i_syn_val = read(i_syn) * decay_syn + input;
            write(i_syn, i_syn_val);
            
            // Update membrane potential
            let v_mem_val = read(v_mem) * decay_mem + i_syn_val;
            
            // Check for spike
            if v_mem_val >= threshold {
                write(v_mem, reset);
                write(refractory, self.params.refractory_period);
                write(spike, true);
            } else {
                write(v_mem, v_mem_val);
                write(spike, false);
            }
        }
    }
    
    /// Reset all neurons to initial state
    pub fn reset(&mut self) {
        self.v_mem.fill(0.0);
        self.i_syn.fill(0.0);
        self.spikes.fill(false);
        self.refractory.fill(0);
    }
    
    /// Get current membrane potentials (read-only)
    pub fn membrane_potentials(&self) -> &[f32] {
        &self.v_mem[..self.active_count]
    }
    
    /// Get current spikes (read-only)
    pub fn current_spikes(&self) -> &[bool] {
        &self.spikes[..self.active_count]
    }
}

impl<const BATCH_SIZE: usize, const MAX_NEURONS: usize> ZeroAllocBatchProcessor<BATCH_SIZE, MAX_NEURONS> {
    /// Create new zero-allocation batch processor
    pub const fn new(params: NeuronParams) -> Self {
        const INIT_STATE: StackNeuronState<MAX_NEURONS> = StackNeuronState::new(params);
        
        Self {
            neuron_states: [INIT_STATE; BATCH_SIZE],
            input_buffer: [[0.0; MAX_NEURONS]; BATCH_SIZE],
            output_buffer: [[false; MAX_NEURONS]; BATCH_SIZE],
            temp_buffer: [0.0; MAX_NEURONS],
            current_batch_size: 0,
        }
    }
    
    /// Process batch with zero allocations (ultra-hot path)
    #[inline(always)]
    pub fn process_batch_zero_alloc(
        &mut self,
        batch_inputs: &[[f32; MAX_NEURONS]],
    ) -> &[[bool; MAX_NEURONS]] {
        let batch_size = batch_inputs.len().min(BATCH_SIZE);
        self.current_batch_size = batch_size;
        
        // Process each sample in the batch
        for i in 0..batch_size {
            let spikes = self.neuron_states[i].process_step_zero_alloc(&batch_inputs[i]);
            
            // Copy results to output buffer (avoid allocation)
            unsafe {
                std::ptr::copy_nonoverlapping(
                    spikes.as_ptr(),
                    self.output_buffer[i].as_mut_ptr() as *mut bool,
                    spikes.len(),
                );
            }
        }
        
        &self.output_buffer[..batch_size]
    }
    
    /// Get maximum throughput (samples per second)
    pub const fn max_throughput_estimate() -> f64 {
        // Estimate based on cycle count and target frequency
        // Assumes ~10 cycles per neuron at 3 GHz
        const CYCLES_PER_NEURON: f64 = 10.0;
        const CPU_FREQ_GHZ: f64 = 3.0;
        
        (CPU_FREQ_GHZ * 1e9) / (CYCLES_PER_NEURON * MAX_NEURONS as f64)
    }
    
    /// Reset all batch states
    pub fn reset_batch(&mut self) {
        for state in &mut self.neuron_states {
            state.reset();
        }
        self.current_batch_size = 0;
    }
}

impl<const SIZE: usize> LockFreeSpikeBuf<SIZE> {
    /// Create new lock-free spike buffer
    pub const fn new() -> Self {
        assert!(SIZE.is_power_of_two(), "SIZE must be power of 2");
        
        const INIT_EVENT: SpikeEvent = SpikeEvent {
            neuron_id: 0,
            timestamp: 0,
            weight: 0.0,
            layer_id: 0,
        };
        
        Self {
            events: [INIT_EVENT; SIZE],
            head: std::sync::atomic::AtomicUsize::new(0),
            tail: std::sync::atomic::AtomicUsize::new(0),
            mask: SIZE - 1,
        }
    }
    
    /// Push spike event (lock-free, wait-free for single producer)
    pub fn push(&self, event: SpikeEvent) -> bool {
        let head = self.head.load(std::sync::atomic::Ordering::Relaxed);
        let next_head = (head + 1) & self.mask;
        
        // Check if buffer is full
        if next_head == self.tail.load(std::sync::atomic::Ordering::Acquire) {
            return false; // Buffer full
        }
        
        // Store event
        unsafe {
            std::ptr::write_volatile(
                &self.events[head] as *const SpikeEvent as *mut SpikeEvent,
                event,
            );
        }
        
        // Update head
        self.head.store(next_head, std::sync::atomic::Ordering::Release);
        true
    }
    
    /// Pop spike event (lock-free, wait-free for single consumer)
    pub fn pop(&self) -> Option<SpikeEvent> {
        let tail = self.tail.load(std::sync::atomic::Ordering::Relaxed);
        
        // Check if buffer is empty
        if tail == self.head.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        
        // Load event
        let event = unsafe {
            std::ptr::read_volatile(&self.events[tail] as *const SpikeEvent)
        };
        
        // Update tail
        let next_tail = (tail + 1) & self.mask;
        self.tail.store(next_tail, std::sync::atomic::Ordering::Release);
        
        Some(event)
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.head.load(std::sync::atomic::Ordering::Relaxed) == 
        self.tail.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// Get current buffer usage
    pub fn len(&self) -> usize {
        let head = self.head.load(std::sync::atomic::Ordering::Relaxed);
        let tail = self.tail.load(std::sync::atomic::Ordering::Relaxed);
        (head.wrapping_sub(tail)) & self.mask
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub total_capacity: usize,
    pub total_used: usize,
    pub utilization: f64,
    pub chunk_count: usize,
}

/// Ultra-fast trading processor with zero allocations
pub struct ZeroAllocTradingProcessor {
    /// Small neuron state for ultra-fast processing
    neurons: StackNeuronState<64>, // 64 neurons max
    /// Input preprocessing buffer
    input_buffer: [f32; 64],
    /// Market data processor
    market_processor: MarketDataProcessor,
    /// Spike event buffer
    spike_buffer: LockFreeSpikeBuf<1024>,
}

/// Market data processor for zero-allocation operation
struct MarketDataProcessor {
    /// Price normalization parameters
    price_scale: f32,
    volume_scale: f32,
    /// Feature extraction buffer
    features: [f32; 16],
}

impl ZeroAllocTradingProcessor {
    /// Create new trading processor
    pub fn new() -> Self {
        let params = NeuronParams {
            decay_mem: 0.9,
            decay_syn: 0.8,
            threshold: 1.0,
            reset_potential: 0.0,
            refractory_period: 2,
        };
        
        Self {
            neurons: StackNeuronState::new(params),
            input_buffer: [0.0; 64],
            market_processor: MarketDataProcessor {
                price_scale: 0.001,
                volume_scale: 0.0001,
                features: [0.0; 16],
            },
            spike_buffer: LockFreeSpikeBuf::new(),
        }
    }
    
    /// Process market tick with zero allocations (ultra-hot path)
    /// Target: <500ns total latency
    #[inline(always)]
    pub fn process_market_tick_zero_alloc(
        &mut self,
        price: f32,
        volume: f32,
        timestamp: u64,
    ) -> &[bool] {
        // Extract features directly into buffer (no allocation)
        self.market_processor.extract_features_zero_alloc(
            price, volume, timestamp, &mut self.input_buffer
        );
        
        // Process through neurons (no allocation)
        let spikes = self.neurons.process_step_zero_alloc(&self.input_buffer[..16]);
        
        // Record significant spikes in lock-free buffer
        for (i, &spike) in spikes.iter().enumerate() {
            if spike {
                let event = SpikeEvent {
                    neuron_id: i as u32,
                    timestamp,
                    weight: 1.0,
                    layer_id: 0,
                };
                self.spike_buffer.push(event); // Ignore if buffer full
            }
        }
        
        spikes
    }
    
    /// Get trading signals from recent spikes (zero allocation)
    pub fn get_trading_signals(&self) -> TradingSignals {
        let mut signals = TradingSignals::default();
        
        // Analyze recent spikes for trading signals
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Count spikes in last microsecond
        let mut spike_count = 0;
        let time_window = 1000; // 1 microsecond in nanoseconds
        
        // This would normally iterate through recent spikes
        // For now, use dummy logic
        if self.neurons.current_spikes().iter().any(|&s| s) {
            signals.buy_strength = 0.7;
            signals.sell_strength = 0.3;
            signals.confidence = 0.8;
        }
        
        signals
    }
    
    /// Reset processor state
    pub fn reset(&mut self) {
        self.neurons.reset();
        self.input_buffer.fill(0.0);
        self.market_processor.features.fill(0.0);
    }
}

impl MarketDataProcessor {
    /// Extract features with zero allocations
    #[inline(always)]
    fn extract_features_zero_alloc(
        &mut self,
        price: f32,
        volume: f32,
        timestamp: u64,
        output: &mut [f32],
    ) {
        // Normalize inputs
        let norm_price = price * self.price_scale;
        let norm_volume = volume * self.volume_scale;
        let norm_time = (timestamp % 1000000) as f32 * 0.000001;
        
        // Extract features directly into output buffer
        if output.len() >= 16 {
            output[0] = norm_price;
            output[1] = norm_volume;
            output[2] = norm_time;
            output[3] = norm_price * norm_volume; // price-volume interaction
            output[4] = norm_price.ln(); // log price
            output[5] = norm_volume.sqrt(); // sqrt volume
            
            // Technical indicators (simplified)
            output[6] = norm_price - 0.5; // price deviation
            output[7] = norm_volume - 0.5; // volume deviation
            
            // Fill remaining with derived features
            for i in 8..16 {
                output[i] = (output[i % 8] * 1.1).tanh();
            }
        }
    }
}

/// Trading signals structure
#[derive(Debug, Clone, Copy, Default)]
pub struct TradingSignals {
    pub buy_strength: f32,
    pub sell_strength: f32,
    pub confidence: f32,
    pub urgency: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool() {
        let mut pool = ZeroAllocMemoryPool::new(1024, 64).unwrap();
        
        let ptr1 = pool.allocate::<f32>(10).unwrap();
        let ptr2 = pool.allocate::<f32>(20).unwrap();
        
        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);
        
        let stats = pool.usage_stats();
        assert!(stats.total_used > 0);
        assert!(stats.utilization > 0.0);
    }
    
    #[test]
    fn test_stack_neuron_state() {
        let params = NeuronParams {
            decay_mem: 0.9,
            decay_syn: 0.8,
            threshold: 1.0,
            reset_potential: 0.0,
            refractory_period: 2,
        };
        
        let mut state = StackNeuronState::<10>::new(params);
        let inputs = [1.5; 10];
        
        let spikes = state.process_step_zero_alloc(&inputs);
        assert_eq!(spikes.len(), 10);
        
        // Should generate spikes for strong input
        assert!(spikes.iter().any(|&s| s));
    }
    
    #[test]
    fn test_lock_free_spike_buffer() {
        let buffer = LockFreeSpikeBuf::<16>::new();
        
        let event = SpikeEvent {
            neuron_id: 5,
            timestamp: 12345,
            weight: 1.0,
            layer_id: 1,
        };
        
        assert!(buffer.push(event));
        assert_eq!(buffer.len(), 1);
        
        let popped = buffer.pop().unwrap();
        assert_eq!(popped.neuron_id, 5);
        assert_eq!(popped.timestamp, 12345);
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_zero_alloc_trading_processor() {
        let mut processor = ZeroAllocTradingProcessor::new();
        
        let spikes = processor.process_market_tick_zero_alloc(100.0, 1000.0, 12345);
        assert!(spikes.len() > 0);
        
        let signals = processor.get_trading_signals();
        assert!(signals.buy_strength >= 0.0 && signals.buy_strength <= 1.0);
        assert!(signals.sell_strength >= 0.0 && signals.sell_strength <= 1.0);
    }
    
    #[test]
    fn test_batch_processor() {
        let params = NeuronParams {
            decay_mem: 0.9,
            decay_syn: 0.8,
            threshold: 1.0,
            reset_potential: 0.0,
            refractory_period: 2,
        };
        
        let mut processor = ZeroAllocBatchProcessor::<4, 16>::new(params);
        
        let batch_inputs = [
            [1.0; 16],
            [1.5; 16],
            [0.5; 16],
            [2.0; 16],
        ];
        
        let outputs = processor.process_batch_zero_alloc(&batch_inputs);
        assert_eq!(outputs.len(), 4);
        
        // Check that different inputs produce different outputs
        assert_ne!(outputs[0], outputs[1]);
        
        // Verify throughput estimate
        let throughput = ZeroAllocBatchProcessor::<4, 16>::max_throughput_estimate();
        assert!(throughput > 1000.0); // Should be > 1000 samples/sec
    }
}