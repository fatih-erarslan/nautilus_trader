/**
 * @file cdfa_unified.h
 * @brief C Header for CDFA Unified Library
 * 
 * This header provides C API declarations for the CDFA Unified library,
 * enabling integration with C/C++ applications for financial analysis.
 * 
 * @version 0.1.0
 * @author CDFA Development Team
 * @date 2024
 * 
 * @section DESCRIPTION
 * 
 * The CDFA (Cross-Domain Feature Alignment) Unified library provides
 * comprehensive financial analysis capabilities including:
 * - Diversity metrics calculation
 * - Signal fusion algorithms
 * - Pattern detection
 * - Performance optimization (SIMD, parallel processing)
 * - Financial data validation and safety
 * 
 * @section USAGE
 * 
 * Basic usage pattern:
 * ```c
 * #include "cdfa_unified.h"
 * 
 * // Create CDFA instance
 * CdfaHandle* handle = cdfa_create();
 * 
 * // Create and validate data
 * CArray2D* data = cdfa_alloc_array2d(rows, cols);
 * // ... fill data ...
 * cdfa_validate_data(data);
 * 
 * // Perform analysis
 * CAnalysisResult* result = NULL;
 * cdfa_analyze(handle, data, &result);
 * 
 * // Use results...
 * 
 * // Clean up
 * cdfa_free_result(result);
 * cdfa_free_array2d(data);
 * cdfa_destroy(handle);
 * ```
 * 
 * @section THREAD_SAFETY
 * 
 * - CdfaHandle instances are thread-safe for concurrent reads
 * - Analysis operations can be performed concurrently on different handles
 * - Memory allocation/deallocation functions are thread-safe
 * - Error handling is thread-local
 * 
 * @section MEMORY_MANAGEMENT
 * 
 * - All allocated objects must be freed using corresponding cdfa_free_* functions
 * - Null pointers are handled gracefully
 * - Double-free is protected (safe no-op)
 * - Memory ownership is clearly defined by the owns_data flag
 */

#ifndef CDFA_UNIFIED_H
#define CDFA_UNIFIED_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ========================================================================== */
/*                                 CONSTANTS                                 */
/* ========================================================================== */

/** @brief Library version string */
#define CDFA_VERSION "0.1.0"

/** @brief Maximum reasonable financial value (prevents overflow) */
#define CDFA_MAX_FINANCIAL_VALUE 1e15

/** @brief Minimum array dimensions for analysis */
#define CDFA_MIN_ROWS 2
#define CDFA_MIN_COLS 2

/** @brief Default cache TTL in seconds */
#define CDFA_DEFAULT_CACHE_TTL 300

/* ========================================================================== */
/*                               ERROR CODES                                 */
/* ========================================================================== */

/**
 * @brief Error codes returned by CDFA functions
 * 
 * All CDFA functions return one of these error codes.
 * CDFA_SUCCESS (0) indicates successful operation.
 */
typedef enum {
    CDFA_SUCCESS = 0,                   /**< Operation successful */
    CDFA_INVALID_INPUT = 1,             /**< Invalid input parameters */
    CDFA_DIMENSION_MISMATCH = 2,        /**< Array dimension mismatch */
    CDFA_MATH_ERROR = 3,                /**< Mathematical computation error */
    CDFA_NUMERICAL_ERROR = 4,           /**< Numerical instability */
    CDFA_SIMD_ERROR = 5,                /**< SIMD processing error */
    CDFA_PARALLEL_ERROR = 6,            /**< Parallel processing error */
    CDFA_GPU_ERROR = 7,                 /**< GPU processing error */
    CDFA_ML_ERROR = 8,                  /**< Machine learning error */
    CDFA_DETECTION_ERROR = 9,           /**< Pattern detection error */
    CDFA_CONFIG_ERROR = 10,             /**< Configuration error */
    CDFA_RESOURCE_ERROR = 11,           /**< Resource allocation error */
    CDFA_TIMEOUT_ERROR = 12,            /**< Operation timeout */
    CDFA_UNSUPPORTED_OPERATION = 13,    /**< Unsupported operation */
    CDFA_FEATURE_NOT_ENABLED = 14,      /**< Required feature not enabled */
    CDFA_EXTERNAL_ERROR = 15,           /**< External library error */
    CDFA_UNKNOWN_ERROR = 99             /**< Unknown/unexpected error */
} CdfaErrorCode;

/* ========================================================================== */
/*                              DATA STRUCTURES                              */
/* ========================================================================== */

/**
 * @brief Opaque handle for CDFA instances
 * 
 * This handle encapsulates the internal CDFA state and configuration.
 * Create with cdfa_create(), destroy with cdfa_destroy().
 */
typedef struct CdfaHandle CdfaHandle;

/**
 * @brief 2D array structure for matrix data
 * 
 * Represents a 2D matrix with row-major layout.
 * Use cdfa_alloc_array2d() to create, cdfa_free_array2d() to destroy.
 */
typedef struct {
    double *data;           /**< Pointer to data array (row-major) */
    uint32_t rows;          /**< Number of rows */
    uint32_t cols;          /**< Number of columns */
    bool owns_data;         /**< Whether this struct owns the data memory */
} CArray2D;

/**
 * @brief 1D array structure for vector data
 * 
 * Represents a 1D vector.
 * Use cdfa_alloc_array1d() to create, cdfa_free_array1d() to destroy.
 */
typedef struct {
    double *data;           /**< Pointer to data array */
    uint32_t len;           /**< Array length */
    bool owns_data;         /**< Whether this struct owns the data memory */
} CArray1D;

/**
 * @brief Configuration structure for CDFA operations
 * 
 * Controls various aspects of CDFA behavior and performance.
 */
typedef struct {
    uint32_t num_threads;           /**< Number of threads (0 = auto-detect) */
    bool enable_simd;               /**< Enable SIMD optimizations */
    bool enable_gpu;                /**< Enable GPU acceleration */
    double tolerance;               /**< Numerical tolerance for comparisons */
    uint32_t max_iterations;        /**< Maximum iterations for iterative algorithms */
    double convergence_threshold;   /**< Convergence threshold for optimization */
    uint32_t cache_size_mb;         /**< Cache size in megabytes */
    bool enable_distributed;        /**< Enable distributed processing */
} CCdfaConfig;

/**
 * @brief Analysis result structure
 * 
 * Contains the complete results of a CDFA analysis operation.
 * Free with cdfa_free_result().
 */
typedef struct {
    CArray1D data;                  /**< Main analysis result data */
    uint32_t metrics_count;         /**< Number of computed metrics */
    char **metric_names;            /**< Names of computed metrics */
    double *metric_values;          /**< Values of computed metrics */
    uint32_t patterns_count;        /**< Number of detected patterns */
    uint64_t execution_time_us;     /**< Execution time in microseconds */
    CdfaErrorCode error_code;       /**< Error code (should be CDFA_SUCCESS) */
} CAnalysisResult;

/* ========================================================================== */
/*                            LIFECYCLE FUNCTIONS                            */
/* ========================================================================== */

/**
 * @brief Create a new CDFA instance with default configuration
 * 
 * Creates a new CDFA handle with default settings suitable for most
 * financial analysis tasks.
 * 
 * @return Valid handle on success, NULL on failure
 * 
 * @note The returned handle must be freed with cdfa_destroy()
 * @note Check cdfa_get_last_error() if NULL is returned
 * 
 * @see cdfa_create_with_config()
 * @see cdfa_destroy()
 */
CdfaHandle* cdfa_create(void);

/**
 * @brief Create a new CDFA instance with custom configuration
 * 
 * Creates a new CDFA handle with the specified configuration.
 * 
 * @param config Pointer to configuration structure (must not be NULL)
 * @return Valid handle on success, NULL on failure
 * 
 * @note The returned handle must be freed with cdfa_destroy()
 * @note The config structure is copied, so it can be freed after this call
 * @note Check cdfa_get_last_error() if NULL is returned
 * 
 * @see cdfa_create()
 * @see cdfa_destroy()
 */
CdfaHandle* cdfa_create_with_config(const CCdfaConfig* config);

/**
 * @brief Destroy a CDFA instance and free associated memory
 * 
 * Safely destroys a CDFA handle and frees all associated resources.
 * 
 * @param handle Handle to destroy (NULL is safe and ignored)
 * 
 * @note This function is thread-safe
 * @note Calling with NULL is safe (no-op)
 * @note Do not use the handle after calling this function
 * 
 * @see cdfa_create()
 * @see cdfa_create_with_config()
 */
void cdfa_destroy(CdfaHandle* handle);

/* ========================================================================== */
/*                             ERROR HANDLING                                */
/* ========================================================================== */

/**
 * @brief Get the last error message
 * 
 * Returns a human-readable description of the last error that occurred
 * in the current thread.
 * 
 * @return Null-terminated error string, or NULL if no error
 * 
 * @note The returned string is valid until the next error occurs
 * @note Error messages are thread-local
 * @note Do not free the returned string
 * 
 * @see cdfa_clear_error()
 */
const char* cdfa_get_last_error(void);

/**
 * @brief Clear the last error
 * 
 * Clears the last error message for the current thread.
 * 
 * @note This function is thread-safe
 * @note Useful for ensuring clean state before operations
 * 
 * @see cdfa_get_last_error()
 */
void cdfa_clear_error(void);

/* ========================================================================== */
/*                            ANALYSIS FUNCTIONS                             */
/* ========================================================================== */

/**
 * @brief Perform comprehensive CDFA analysis
 * 
 * Executes the complete CDFA pipeline including diversity calculation,
 * fusion algorithms, pattern detection, and specialized analysis.
 * 
 * @param handle Valid CDFA handle (must not be NULL)
 * @param data Input data matrix (must not be NULL, rows=observations, cols=features)
 * @param result Pointer to result pointer (will be allocated, must not be NULL)
 * @return Error code (CDFA_SUCCESS on success)
 * 
 * @note Input data must pass cdfa_validate_data()
 * @note Result must be freed with cdfa_free_result()
 * @note This function may take significant time for large datasets
 * 
 * @see cdfa_validate_data()
 * @see cdfa_free_result()
 * @see cdfa_calculate_diversity()
 */
CdfaErrorCode cdfa_analyze(
    CdfaHandle* handle,
    const CArray2D* data,
    CAnalysisResult** result
);

/**
 * @brief Calculate diversity metrics only
 * 
 * Computes diversity scores for the input data without performing
 * the full analysis pipeline.
 * 
 * @param handle Valid CDFA handle (must not be NULL)
 * @param data Input data matrix (must not be NULL)
 * @param result Pre-allocated result array (must not be NULL, length = data->cols)
 * @return Error code (CDFA_SUCCESS on success)
 * 
 * @note Result array must be pre-allocated with correct size
 * @note Much faster than full analysis for diversity-only needs
 * @note Input data must pass cdfa_validate_data()
 * 
 * @see cdfa_analyze()
 * @see cdfa_validate_data()
 */
CdfaErrorCode cdfa_calculate_diversity(
    CdfaHandle* handle,
    const CArray2D* data,
    CArray1D* result
);

/**
 * @brief Apply fusion algorithms
 * 
 * Combines multiple diversity scores using the configured fusion method.
 * 
 * @param handle Valid CDFA handle (must not be NULL)
 * @param scores Input diversity scores (must not be NULL)
 * @param data Original data matrix (must not be NULL)
 * @param result Pre-allocated result array (must not be NULL, length = scores->len)
 * @return Error code (CDFA_SUCCESS on success)
 * 
 * @note Result array must be pre-allocated with correct size
 * @note Scores typically come from cdfa_calculate_diversity()
 * @note Data is used for additional context in fusion algorithms
 * 
 * @see cdfa_calculate_diversity()
 * @see cdfa_analyze()
 */
CdfaErrorCode cdfa_apply_fusion(
    CdfaHandle* handle,
    const CArray1D* scores,
    const CArray2D* data,
    CArray1D* result
);

/* ========================================================================== */
/*                          CONFIGURATION FUNCTIONS                          */
/* ========================================================================== */

/**
 * @brief Get current configuration
 * 
 * Retrieves the current configuration settings from a CDFA handle.
 * 
 * @param handle Valid CDFA handle (must not be NULL)
 * @param config Pointer to configuration structure to fill (must not be NULL)
 * @return Error code (CDFA_SUCCESS on success)
 * 
 * @note The configuration is copied to the provided structure
 * @note Safe to call concurrently from multiple threads
 * 
 * @see cdfa_set_config()
 */
CdfaErrorCode cdfa_get_config(
    CdfaHandle* handle,
    CCdfaConfig* config
);

/**
 * @brief Update configuration
 * 
 * Updates the configuration settings for a CDFA handle.
 * 
 * @param handle Valid CDFA handle (must not be NULL)
 * @param config New configuration (must not be NULL)
 * @return Error code (CDFA_SUCCESS on success)
 * 
 * @note Configuration is validated before applying
 * @note Changes take effect immediately for subsequent operations
 * @note Not safe to call concurrently with analysis operations
 * 
 * @see cdfa_get_config()
 */
CdfaErrorCode cdfa_set_config(
    CdfaHandle* handle,
    const CCdfaConfig* config
);

/* ========================================================================== */
/*                            UTILITY FUNCTIONS                              */
/* ========================================================================== */

/**
 * @brief Validate input data for financial system safety
 * 
 * Performs comprehensive validation of financial data including:
 * - Dimension checks (minimum size requirements)
 * - Finite value validation (no NaN/infinity)
 * - Reasonable value ranges for financial data
 * - Variance checks (no constant data)
 * 
 * @param data Data to validate (must not be NULL)
 * @return Error code (CDFA_SUCCESS if valid)
 * 
 * @note This is automatically called by analysis functions
 * @note Useful for pre-validation in client code
 * @note Designed specifically for financial data constraints
 * 
 * @see cdfa_analyze()
 * @see cdfa_calculate_diversity()
 */
CdfaErrorCode cdfa_validate_data(const CArray2D* data);

/**
 * @brief Get library version information
 * 
 * Returns the version string of the CDFA library.
 * 
 * @return Null-terminated version string
 * 
 * @note The returned string is statically allocated
 * @note Do not free the returned string
 * @note Format: "major.minor.patch"
 */
const char* cdfa_get_version(void);

/**
 * @brief Get build information
 * 
 * Returns detailed build information including version, features,
 * build date, and compiler information.
 * 
 * @return Null-terminated build info string
 * 
 * @note The returned string is statically allocated
 * @note Do not free the returned string
 * @note Useful for debugging and support
 */
const char* cdfa_get_build_info(void);

/* ========================================================================== */
/*                           MEMORY MANAGEMENT                               */
/* ========================================================================== */

/**
 * @brief Allocate a 2D array
 * 
 * Allocates and initializes a 2D array structure with zero-filled data.
 * 
 * @param rows Number of rows (must be > 0)
 * @param cols Number of columns (must be > 0)
 * @return Allocated array on success, NULL on failure
 * 
 * @note The returned array must be freed with cdfa_free_array2d()
 * @note Data is initialized to zero
 * @note Returns NULL if rows or cols is zero
 * 
 * @see cdfa_free_array2d()
 */
CArray2D* cdfa_alloc_array2d(uint32_t rows, uint32_t cols);

/**
 * @brief Allocate a 1D array
 * 
 * Allocates and initializes a 1D array structure with zero-filled data.
 * 
 * @param len Array length (must be > 0)
 * @return Allocated array on success, NULL on failure
 * 
 * @note The returned array must be freed with cdfa_free_array1d()
 * @note Data is initialized to zero
 * @note Returns NULL if len is zero
 * 
 * @see cdfa_free_array1d()
 */
CArray1D* cdfa_alloc_array1d(uint32_t len);

/**
 * @brief Free a 2D array
 * 
 * Frees a 2D array allocated with cdfa_alloc_array2d().
 * 
 * @param array Array to free (NULL is safe and ignored)
 * 
 * @note This function is thread-safe
 * @note Calling with NULL is safe (no-op)
 * @note Only frees arrays that own their data (owns_data = true)
 * 
 * @see cdfa_alloc_array2d()
 */
void cdfa_free_array2d(CArray2D* array);

/**
 * @brief Free a 1D array
 * 
 * Frees a 1D array allocated with cdfa_alloc_array1d().
 * 
 * @param array Array to free (NULL is safe and ignored)
 * 
 * @note This function is thread-safe
 * @note Calling with NULL is safe (no-op)
 * @note Only frees arrays that own their data (owns_data = true)
 * 
 * @see cdfa_alloc_array1d()
 */
void cdfa_free_array1d(CArray1D* array);

/**
 * @brief Free an analysis result
 * 
 * Frees an analysis result structure returned by cdfa_analyze().
 * 
 * @param result Result to free (NULL is safe and ignored)
 * 
 * @note This function is thread-safe
 * @note Calling with NULL is safe (no-op)
 * @note Frees all associated memory including metric names/values
 * 
 * @see cdfa_analyze()
 */
void cdfa_free_result(CAnalysisResult* result);

/* ========================================================================== */
/*                                HELPERS                                    */
/* ========================================================================== */

/**
 * @brief Helper macro to check error codes
 * 
 * Example usage:
 * ```c
 * CDFA_CHECK(cdfa_validate_data(data));
 * ```
 */
#define CDFA_CHECK(expr) \
    do { \
        CdfaErrorCode __err = (expr); \
        if (__err != CDFA_SUCCESS) { \
            fprintf(stderr, "CDFA Error %d at %s:%d\n", __err, __FILE__, __LINE__); \
            return __err; \
        } \
    } while(0)

/**
 * @brief Helper macro for creating default configuration
 * 
 * Example usage:
 * ```c
 * CCdfaConfig config = CDFA_DEFAULT_CONFIG();
 * config.num_threads = 4;
 * ```
 */
#define CDFA_DEFAULT_CONFIG() { \
    .num_threads = 0, \
    .enable_simd = true, \
    .enable_gpu = false, \
    .tolerance = 1e-10, \
    .max_iterations = 1000, \
    .convergence_threshold = 1e-6, \
    .cache_size_mb = 100, \
    .enable_distributed = false \
}

/**
 * @brief Check if error code represents a critical error
 * 
 * @param error_code Error code to check
 * @return true if critical, false otherwise
 */
static inline bool cdfa_is_critical_error(CdfaErrorCode error_code) {
    return error_code == CDFA_RESOURCE_ERROR ||
           error_code == CDFA_NUMERICAL_ERROR ||
           error_code == CDFA_GPU_ERROR ||
           error_code == CDFA_EXTERNAL_ERROR;
}

/**
 * @brief Check if error code represents a recoverable error
 * 
 * @param error_code Error code to check
 * @return true if recoverable, false otherwise
 */
static inline bool cdfa_is_recoverable_error(CdfaErrorCode error_code) {
    return error_code == CDFA_INVALID_INPUT ||
           error_code == CDFA_DIMENSION_MISMATCH ||
           error_code == CDFA_CONFIG_ERROR ||
           error_code == CDFA_TIMEOUT_ERROR ||
           error_code == CDFA_UNSUPPORTED_OPERATION ||
           error_code == CDFA_FEATURE_NOT_ENABLED;
}

#ifdef __cplusplus
}
#endif

#endif /* CDFA_UNIFIED_H */