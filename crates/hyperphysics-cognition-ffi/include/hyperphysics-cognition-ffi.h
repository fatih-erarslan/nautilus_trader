#ifndef HYPERPHYSICS_COGNITION_FFI_H
#define HYPERPHYSICS_COGNITION_FFI_H

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

/**
 * Cognition phase enum (must match Rust enum)
 */
typedef enum CCognitionPhase {
  Perceiving = 0,
  Cognizing = 1,
  Deliberating = 2,
  Intending = 3,
  Integrating = 4,
  Acting = 5,
} CCognitionPhase;

/**
 * C-compatible cognition configuration
 */
typedef struct CCognitionConfig {
  bool enable_attention;
  bool enable_loops;
  bool enable_dream;
  bool enable_learning;
  bool enable_integration;
  double default_curvature;
  double loop_frequency;
  double dream_threshold;
} CCognitionConfig;

/**
 * Opaque pointer to CognitionSystem
 */
typedef void *CognitionSystemHandle;

/**
 * Initialize tracing (call once at startup)
 */
void hyperphysics_cognition_init_tracing(void);

/**
 * Create default configuration
 */
struct CCognitionConfig hyperphysics_cognition_config_default(void);

/**
 * Create new cognition system
 *
 * # Returns
 * Handle to cognition system, or NULL on error
 *
 * # Safety
 * The returned handle must be freed with `hyperphysics_cognition_destroy()`
 */
CognitionSystemHandle hyperphysics_cognition_create(struct CCognitionConfig config);

/**
 * Destroy cognition system
 *
 * # Safety
 * - Handle must be valid (returned from `hyperphysics_cognition_create`)
 * - Handle must not be used after this call
 * - Handle must not be freed twice
 */
void hyperphysics_cognition_destroy(CognitionSystemHandle handle);

/**
 * Get current arousal level
 *
 * # Safety
 * Handle must be valid
 */
double hyperphysics_cognition_get_arousal(CognitionSystemHandle handle);

/**
 * Set arousal level
 *
 * # Safety
 * Handle must be valid
 */
void hyperphysics_cognition_set_arousal(CognitionSystemHandle handle, double level);

/**
 * Get current cognitive load
 *
 * # Safety
 * Handle must be valid
 */
double hyperphysics_cognition_get_load(CognitionSystemHandle handle);

/**
 * Set cognitive load
 *
 * # Safety
 * Handle must be valid
 */
void hyperphysics_cognition_set_load(CognitionSystemHandle handle, double load);

/**
 * Check if system is healthy
 *
 * # Safety
 * Handle must be valid
 */
bool hyperphysics_cognition_is_healthy(CognitionSystemHandle handle);

/**
 * Get next phase in loop
 */
enum CCognitionPhase hyperphysics_cognition_phase_next(enum CCognitionPhase phase);

/**
 * Get phase name (caller must free the returned string)
 *
 * # Safety
 * The returned string must be freed with `hyperphysics_cognition_free_string()`
 */
char *hyperphysics_cognition_phase_name(enum CCognitionPhase phase);

/**
 * Free a string returned by the FFI
 *
 * # Safety
 * - ptr must have been returned by an FFI function that allocates strings
 * - ptr must not be used after this call
 * - ptr must not be freed twice
 */
void hyperphysics_cognition_free_string(char *ptr);

/**
 * Get version string (caller must free)
 *
 * # Safety
 * The returned string must be freed with `hyperphysics_cognition_free_string()`
 */
char *hyperphysics_cognition_version(void);

#endif /* HYPERPHYSICS_COGNITION_FFI_H */
