//! SIMD optimized operations for HFT systems
//! 
//! This module provides vectorized implementations of critical HFT operations
//! using AVX2/AVX-512 instructions for maximum performance.

use std::arch::x86_64::*;
use std::mem;
use std::ptr;

/// SIMD-optimized hash computation using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn parallel_hash_avx2(data: &[u8]) -> [u64; 4] {
    let mut result = [0u64; 4];
    
    if data.len() < 32 {
        // Fallback to scalar for small inputs
        return scalar_hash_fallback(data);
    }
    
    // Initialize hash state with different constants for each lane
    let mut h0 = _mm256_set_epi64x(0x9E3779B97F4A7C15_u64 as i64, 
                                   0x243F6A8885A308D3_u64 as i64,
                                   0x13198A2E03707344_u64 as i64,
                                   0xA4093822299F31D0_u64 as i64);
    
    let chunks = data.chunks_exact(32);
    let remainder = chunks.remainder();
    
    // Process 32-byte chunks with parallel hashing
    for chunk in chunks {
        // Load 32 bytes as 4 x 64-bit integers
        let data_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        
        // Multiply by prime constant
        let prime = _mm256_set1_epi64x(0x9E3779B185EBCA87_u64 as i64);
        let mul_result = _mm256_mul_epu32(data_vec, prime);
        
        // Add to running hash
        h0 = _mm256_add_epi64(h0, mul_result);
        
        // Rotate left by 13 bits for better mixing
        let rot13 = _mm256_or_si256(
            _mm256_slli_epi64(h0, 13),
            _mm256_srli_epi64(h0, 51)
        );
        h0 = rot13;
        
        // XOR with rotated value
        h0 = _mm256_xor_si256(h0, _mm256_srli_epi64(data_vec, 32));
    }
    
    // Process remainder bytes
    if !remainder.is_empty() {
        let mut tail_data = [0u8; 32];
        ptr::copy_nonoverlapping(remainder.as_ptr(), tail_data.as_mut_ptr(), remainder.len());
        
        let tail_vec = _mm256_loadu_si256(tail_data.as_ptr() as *const __m256i);
        let prime = _mm256_set1_epi64x(0x9E3779B185EBCA87_u64 as i64);
        let mul_result = _mm256_mul_epu32(tail_vec, prime);
        h0 = _mm256_add_epi64(h0, mul_result);
    }
    
    // Final mixing to improve avalanche effect
    h0 = _mm256_xor_si256(h0, _mm256_srli_epi64(h0, 33));
    h0 = _mm256_mul_epu32(h0, _mm256_set1_epi64x(0xFF51AFD7ED558CCD_u64 as i64));
    h0 = _mm256_xor_si256(h0, _mm256_srli_epi64(h0, 33));
    h0 = _mm256_mul_epu32(h0, _mm256_set1_epi64x(0xC4CEB9FE1A85EC53_u64 as i64));
    h0 = _mm256_xor_si256(h0, _mm256_srli_epi64(h0, 33));
    
    // Store results
    _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, h0);
    
    result
}

/// SIMD-optimized memory comparison using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn simd_memcmp_avx2(a: &[u8], b: &[u8]) -> i32 {
    if a.len() != b.len() {
        return a.len() as i32 - b.len() as i32;
    }
    
    let len = a.len();
    let mut offset = 0;
    
    // Process 32-byte chunks
    while offset + 32 <= len {
        let a_chunk = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
        let b_chunk = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);
        
        let cmp = _mm256_cmpeq_epi8(a_chunk, b_chunk);
        let mask = _mm256_movemask_epi8(cmp);
        
        if mask != -1 {
            // Found difference, find first differing byte
            let diff_pos = mask.trailing_ones() as usize;
            let pos = offset + diff_pos;
            if pos < len {
                return a[pos] as i32 - b[pos] as i32;
            }
        }
        
        offset += 32;
    }
    
    // Process remaining bytes
    while offset < len {
        if a[offset] != b[offset] {
            return a[offset] as i32 - b[offset] as i32;
        }
        offset += 1;
    }
    
    0
}

/// SIMD-optimized memory copy using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn simd_memcpy_avx2(dst: *mut u8, src: *const u8, count: usize) {
    let mut offset = 0;
    
    // Align destination to 32-byte boundary for better performance
    let align_offset = (32 - (dst as usize & 31)) & 31;
    if align_offset > 0 && align_offset < count {
        ptr::copy_nonoverlapping(src, dst, align_offset);
        offset += align_offset;
    }
    
    // Process 32-byte chunks with aligned stores
    while offset + 32 <= count {
        let chunk = _mm256_loadu_si256(src.add(offset) as *const __m256i);
        _mm256_store_si256(dst.add(offset) as *mut __m256i, chunk);
        offset += 32;
    }
    
    // Copy remaining bytes
    if offset < count {
        ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), count - offset);
    }
}

/// SIMD-optimized checksum calculation using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn simd_checksum_avx2(data: &[u8]) -> u64 {
    let mut checksum = _mm256_setzero_si256();
    let mut offset = 0;
    
    // Process 32-byte chunks
    while offset + 32 <= data.len() {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);
        
        // Convert bytes to 64-bit integers for accumulation
        let low_bytes = _mm256_unpacklo_epi8(chunk, _mm256_setzero_si256());
        let high_bytes = _mm256_unpackhi_epi8(chunk, _mm256_setzero_si256());
        
        let low_words = _mm256_unpacklo_epi16(low_bytes, _mm256_setzero_si256());
        let high_words = _mm256_unpackhi_epi16(low_bytes, _mm256_setzero_si256());
        
        // Accumulate
        checksum = _mm256_add_epi64(checksum, _mm256_unpacklo_epi32(low_words, _mm256_setzero_si256()));
        checksum = _mm256_add_epi64(checksum, _mm256_unpackhi_epi32(low_words, _mm256_setzero_si256()));
        checksum = _mm256_add_epi64(checksum, _mm256_unpacklo_epi32(high_words, _mm256_setzero_si256()));
        checksum = _mm256_add_epi64(checksum, _mm256_unpackhi_epi32(high_words, _mm256_setzero_si256()));
        
        // Repeat for high bytes
        let high_low_words = _mm256_unpacklo_epi16(high_bytes, _mm256_setzero_si256());
        let high_high_words = _mm256_unpackhi_epi16(high_bytes, _mm256_setzero_si256());
        
        checksum = _mm256_add_epi64(checksum, _mm256_unpacklo_epi32(high_low_words, _mm256_setzero_si256()));
        checksum = _mm256_add_epi64(checksum, _mm256_unpackhi_epi32(high_low_words, _mm256_setzero_si256()));
        checksum = _mm256_add_epi64(checksum, _mm256_unpacklo_epi32(high_high_words, _mm256_setzero_si256()));
        checksum = _mm256_add_epi64(checksum, _mm256_unpackhi_epi32(high_high_words, _mm256_setzero_si256()));
        
        offset += 32;
    }
    
    // Horizontal sum of 4 64-bit values
    let sum_vec = _mm256_add_epi64(checksum, _mm256_permute4x64_epi64(checksum, 0b01001110));
    let final_sum = _mm256_add_epi64(sum_vec, _mm256_permute4x64_epi64(sum_vec, 0b00000001));
    
    let result = _mm256_extract_epi64(final_sum, 0) as u64;
    
    // Process remaining bytes
    let mut scalar_sum = result;
    for &byte in &data[offset..] {
        scalar_sum = scalar_sum.wrapping_add(byte as u64);
    }
    
    scalar_sum
}

/// SIMD-optimized string search using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn simd_strstr_avx2(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    
    if needle.len() > haystack.len() {
        return None;
    }
    
    if needle.len() == 1 {
        return simd_strchr_avx2(haystack, needle[0]);
    }
    
    let first_char = _mm256_set1_epi8(needle[0] as i8);
    let last_char = _mm256_set1_epi8(needle[needle.len() - 1] as i8);
    
    let mut pos = 0;
    let search_len = haystack.len() - needle.len() + 1;
    
    while pos + 32 <= search_len {
        // Load 32 bytes from haystack
        let data = _mm256_loadu_si256(haystack.as_ptr().add(pos) as *const __m256i);
        
        // Compare with first character
        let first_match = _mm256_cmpeq_epi8(data, first_char);
        let first_mask = _mm256_movemask_epi8(first_match);
        
        if first_mask != 0 {
            // Found potential matches for first character
            let mut bit_pos = 0;
            let mut remaining_mask = first_mask as u32;
            
            while remaining_mask != 0 {
                let offset = remaining_mask.trailing_zeros() as usize;
                bit_pos += offset;
                
                if pos + bit_pos + needle.len() <= haystack.len() {
                    // Check if last character also matches
                    if haystack[pos + bit_pos + needle.len() - 1] == needle[needle.len() - 1] {
                        // Do full comparison
                        if haystack[pos + bit_pos..pos + bit_pos + needle.len()] == needle {
                            return Some(pos + bit_pos);
                        }
                    }
                }
                
                remaining_mask >>= offset + 1;
                bit_pos += 1;
            }
        }
        
        pos += 32;
    }
    
    // Handle remaining bytes with scalar search
    while pos < search_len {
        if haystack[pos..pos + needle.len()] == needle {
            return Some(pos);
        }
        pos += 1;
    }
    
    None
}

/// SIMD-optimized character search using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn simd_strchr_avx2(haystack: &[u8], needle: u8) -> Option<usize> {
    let needle_vec = _mm256_set1_epi8(needle as i8);
    let mut pos = 0;
    
    while pos + 32 <= haystack.len() {
        let data = _mm256_loadu_si256(haystack.as_ptr().add(pos) as *const __m256i);
        let matches = _mm256_cmpeq_epi8(data, needle_vec);
        let mask = _mm256_movemask_epi8(matches);
        
        if mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            return Some(pos + offset);
        }
        
        pos += 32;
    }
    
    // Handle remaining bytes
    for i in pos..haystack.len() {
        if haystack[i] == needle {
            return Some(i);
        }
    }
    
    None
}

/// SIMD-optimized number parsing using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn simd_parse_u64_avx2(input: &[u8]) -> Option<u64> {
    if input.is_empty() || input.len() > 20 {
        return None;
    }
    
    // Check if all characters are digits
    let mut valid_digits = true;
    let mut digit_count = 0;
    
    // Vectorized digit validation for inputs up to 32 characters
    if input.len() <= 32 {
        let mut padded = [b'0'; 32];
        ptr::copy_nonoverlapping(input.as_ptr(), padded.as_mut_ptr(), input.len());
        
        let data = _mm256_loadu_si256(padded.as_ptr() as *const __m256i);
        let zero = _mm256_set1_epi8(b'0' as i8);
        let nine = _mm256_set1_epi8(b'9' as i8);
        
        let ge_zero = _mm256_cmpgt_epi8(data, _mm256_sub_epi8(zero, _mm256_set1_epi8(1)));
        let le_nine = _mm256_cmpgt_epi8(_mm256_add_epi8(nine, _mm256_set1_epi8(1)), data);
        
        let is_digit = _mm256_and_si256(ge_zero, le_nine);
        let mask = _mm256_movemask_epi8(is_digit);
        
        // Check if first input.len() characters are all digits
        let valid_mask = (1u32 << input.len()) - 1;
        if (mask as u32 & valid_mask) != valid_mask {
            valid_digits = false;
        } else {
            digit_count = input.len();
        }
    } else {
        // Fallback for longer inputs
        for &byte in input {
            if byte >= b'0' && byte <= b'9' {
                digit_count += 1;
            } else {
                valid_digits = false;
                break;
            }
        }
    }
    
    if !valid_digits || digit_count == 0 {
        return None;
    }
    
    // Convert to number using Horner's method
    let mut result = 0u64;
    for &digit_char in input {
        let digit = (digit_char - b'0') as u64;
        result = result.checked_mul(10)?.checked_add(digit)?;
    }
    
    Some(result)
}

/// SIMD-optimized data compression preparation using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn simd_prepare_compression_avx2(data: &[u8], output: &mut [u8]) -> usize {
    if output.len() < data.len() {
        return 0;
    }
    
    let mut input_pos = 0;
    let mut output_pos = 0;
    
    // Detect runs of identical bytes for RLE compression
    while input_pos + 32 <= data.len() && output_pos + 64 < output.len() {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(input_pos) as *const __m256i);
        
        // Create comparison vectors by shifting
        let chunk_shifted = _mm256_alignr_epi8(
            chunk, 
            _mm256_permute2x128_si256(chunk, chunk, 0x08),
            15
        );
        
        let matches = _mm256_cmpeq_epi8(chunk, chunk_shifted);
        let mask = _mm256_movemask_epi8(matches);
        
        if mask == 0 {
            // No runs detected, copy as-is
            simd_memcpy_avx2(output.as_mut_ptr().add(output_pos), 
                           data.as_ptr().add(input_pos), 32);
            input_pos += 32;
            output_pos += 32;
        } else {
            // Process byte by byte to handle runs
            let mut i = 0;
            while i < 32 && input_pos < data.len() && output_pos + 1 < output.len() {
                let current_byte = data[input_pos];
                let mut run_length = 1;
                
                // Count run length
                while input_pos + run_length < data.len() && 
                      data[input_pos + run_length] == current_byte &&
                      run_length < 255 {
                    run_length += 1;
                }
                
                if run_length >= 3 {
                    // Encode as RLE: [length][byte]
                    output[output_pos] = run_length as u8;
                    output[output_pos + 1] = current_byte;
                    output_pos += 2;
                    input_pos += run_length;
                    i += run_length;
                } else {
                    // Copy literal byte
                    output[output_pos] = current_byte;
                    output_pos += 1;
                    input_pos += 1;
                    i += 1;
                }
            }
        }
    }
    
    // Handle remaining bytes
    while input_pos < data.len() && output_pos < output.len() {
        output[output_pos] = data[input_pos];
        input_pos += 1;
        output_pos += 1;
    }
    
    output_pos
}

/// Fallback hash function for small inputs
fn scalar_hash_fallback(data: &[u8]) -> [u64; 4] {
    const PRIME: u64 = 0x9E3779B185EBCA87;
    let mut hashes = [
        0x9E3779B97F4A7C15,
        0x243F6A8885A308D3,
        0x13198A2E03707344,
        0xA4093822299F31D0,
    ];
    
    for (i, &byte) in data.iter().enumerate() {
        let lane = i & 3;
        hashes[lane] = hashes[lane].wrapping_mul(PRIME);
        hashes[lane] = hashes[lane].wrapping_add(byte as u64);
        hashes[lane] = hashes[lane].rotate_left(13);
    }
    
    // Final mixing
    for hash in &mut hashes {
        *hash ^= *hash >> 33;
        *hash = hash.wrapping_mul(0xFF51AFD7ED558CCD);
        *hash ^= *hash >> 33;
        *hash = hash.wrapping_mul(0xC4CEB9FE1A85EC53);
        *hash ^= *hash >> 33;
    }
    
    hashes
}

/// High-level SIMD operation dispatcher
pub struct SIMDOperations;

impl SIMDOperations {
    /// Dispatch parallel hash computation
    pub fn parallel_hash(data: &[u8]) -> [u64; 4] {
        if is_x86_feature_detected!("avx2") {
            unsafe { parallel_hash_avx2(data) }
        } else {
            scalar_hash_fallback(data)
        }
    }
    
    /// Dispatch memory comparison
    pub fn memcmp(a: &[u8], b: &[u8]) -> i32 {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_memcmp_avx2(a, b) }
        } else {
            a.cmp(b) as i32
        }
    }
    
    /// Dispatch memory copy
    pub fn memcpy(dst: &mut [u8], src: &[u8]) {
        let len = dst.len().min(src.len());
        if len == 0 {
            return;
        }
        
        if is_x86_feature_detected!("avx2") && len >= 32 {
            unsafe { 
                simd_memcpy_avx2(dst.as_mut_ptr(), src.as_ptr(), len);
            }
        } else {
            dst[..len].copy_from_slice(&src[..len]);
        }
    }
    
    /// Dispatch checksum calculation
    pub fn checksum(data: &[u8]) -> u64 {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_checksum_avx2(data) }
        } else {
            data.iter().map(|&b| b as u64).sum()
        }
    }
    
    /// Dispatch string search
    pub fn strstr(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_strstr_avx2(haystack, needle) }
        } else {
            haystack.windows(needle.len())
                    .position(|window| window == needle)
        }
    }
    
    /// Dispatch character search
    pub fn strchr(haystack: &[u8], needle: u8) -> Option<usize> {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_strchr_avx2(haystack, needle) }
        } else {
            haystack.iter().position(|&b| b == needle)
        }
    }
    
    /// Dispatch number parsing
    pub fn parse_u64(input: &[u8]) -> Option<u64> {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_parse_u64_avx2(input) }
        } else {
            std::str::from_utf8(input).ok()?.parse().ok()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_hash() {
        let data = b"Hello, SIMD World! This is a test of parallel hashing.";
        let hashes = SIMDOperations::parallel_hash(data);
        
        // Verify we get 4 different hash values
        assert!(hashes[0] != hashes[1]);
        assert!(hashes[1] != hashes[2]);
        assert!(hashes[2] != hashes[3]);
        
        // Verify consistency
        let hashes2 = SIMDOperations::parallel_hash(data);
        assert_eq!(hashes, hashes2);
    }
    
    #[test]
    fn test_simd_memcmp() {
        let a = b"Hello, World!";
        let b = b"Hello, World!";
        let c = b"Hello, SIMD!";
        
        assert_eq!(SIMDOperations::memcmp(a, b), 0);
        assert_ne!(SIMDOperations::memcmp(a, c), 0);
    }
    
    #[test]
    fn test_simd_checksum() {
        let data = b"Testing checksum calculation with SIMD";
        let checksum1 = SIMDOperations::checksum(data);
        let checksum2 = SIMDOperations::checksum(data);
        
        assert_eq!(checksum1, checksum2);
        assert!(checksum1 > 0);
    }
    
    #[test]
    fn test_simd_strstr() {
        let haystack = b"This is a test of SIMD string search functionality";
        let needle = b"SIMD";
        
        let pos = SIMDOperations::strstr(haystack, needle);
        assert_eq!(pos, Some(18));
        
        let not_found = SIMDOperations::strstr(haystack, b"notfound");
        assert_eq!(not_found, None);
    }
    
    #[test]
    fn test_simd_strchr() {
        let haystack = b"Find the character X in this string";
        let pos = SIMDOperations::strchr(haystack, b'X');
        assert_eq!(pos, Some(19));
        
        let not_found = SIMDOperations::strchr(haystack, b'Z');
        assert_eq!(not_found, None);
    }
    
    #[test]
    fn test_simd_parse_u64() {
        assert_eq!(SIMDOperations::parse_u64(b"123456789"), Some(123456789));
        assert_eq!(SIMDOperations::parse_u64(b"0"), Some(0));
        assert_eq!(SIMDOperations::parse_u64(b"18446744073709551615"), Some(u64::MAX));
        assert_eq!(SIMDOperations::parse_u64(b"invalid"), None);
        assert_eq!(SIMDOperations::parse_u64(b"123abc"), None);
    }
    
    #[test]
    fn test_simd_memcpy() {
        let src = b"Source data for SIMD memory copy test with sufficient length";
        let mut dst = vec![0u8; src.len()];
        
        SIMDOperations::memcpy(&mut dst, src);
        assert_eq!(&dst, src);
    }
}