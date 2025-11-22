#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types to hide C++ classes
typedef struct JoltSystem JoltSystem;
typedef struct JoltBodyInterface JoltBodyInterface;
typedef uint32_t JoltBodyID;

// Configuration
typedef struct JoltConfig {
    uint32_t max_bodies;
    uint32_t num_body_mutexes;
    uint32_t max_body_pairs;
    uint32_t max_contact_constraints;
    float collision_tolerance;
    float penetration_tolerance;
    bool deterministic;
} JoltConfig;

// System management
JoltSystem* jolt_system_create(JoltConfig config);
void jolt_system_destroy(JoltSystem* system);
void jolt_system_optimize(JoltSystem* system);
void jolt_system_step(JoltSystem* system, float dt, int collision_steps);

// Body interface
JoltBodyInterface* jolt_system_get_body_interface(JoltSystem* system);

// Body creation (simplified)
JoltBodyID jolt_body_create_box(JoltBodyInterface* iface, float width, float height, float depth, float density, bool is_static);
JoltBodyID jolt_body_create_sphere(JoltBodyInterface* iface, float radius, float density, bool is_static);

// Body manipulation
void jolt_body_set_position(JoltBodyInterface* iface, JoltBodyID body_id, float x, float y, float z);
void jolt_body_get_position(JoltBodyInterface* iface, JoltBodyID body_id, float* x, float* y, float* z);
void jolt_body_set_velocity(JoltBodyInterface* iface, JoltBodyID body_id, float vx, float vy, float vz);
void jolt_body_get_velocity(JoltBodyInterface* iface, JoltBodyID body_id, float* vx, float* vy, float* vz);
void jolt_body_add_force(JoltBodyInterface* iface, JoltBodyID body_id, float fx, float fy, float fz);

// Body state
bool jolt_body_is_active(JoltBodyInterface* iface, JoltBodyID body_id);
void jolt_body_activate(JoltBodyInterface* iface, JoltBodyID body_id);
void jolt_body_deactivate(JoltBodyInterface* iface, JoltBodyID body_id);

#ifdef __cplusplus
}
#endif
