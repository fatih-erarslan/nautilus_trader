#include "jolt_wrapper.h"
#include <stdlib.h>
#include <stdio.h>

// Mock implementation for now since we don't have Jolt source linked yet.
// In a real integration, we would include Jolt headers here:
// #include <Jolt/Jolt.h>
// #include <Jolt/Physics/PhysicsSystem.h>

struct JoltSystem {
    JoltConfig config;
    // JPH::PhysicsSystem* physics_system;
};

struct JoltBodyInterface {
    JoltSystem* system;
    // JPH::BodyInterface* body_interface;
};

extern "C" {

JoltSystem* jolt_system_create(JoltConfig config) {
    JoltSystem* system = (JoltSystem*)malloc(sizeof(JoltSystem));
    system->config = config;
    // Initialize Jolt here
    return system;
}

void jolt_system_destroy(JoltSystem* system) {
    if (system) {
        // Teardown Jolt
        free(system);
    }
}

void jolt_system_optimize(JoltSystem* system) {
    // Optimize broadphase
}

void jolt_system_step(JoltSystem* system, float dt, int collision_steps) {
    // system->physics_system->Update(dt, collision_steps, ...);
}

JoltBodyInterface* jolt_system_get_body_interface(JoltSystem* system) {
    JoltBodyInterface* iface = (JoltBodyInterface*)malloc(sizeof(JoltBodyInterface));
    iface->system = system;
    return iface;
}

JoltBodyID jolt_body_create_box(JoltBodyInterface* iface, float width, float height, float depth, float density, bool is_static) {
    // Create box shape and body
    return 1; // Mock ID
}

JoltBodyID jolt_body_create_sphere(JoltBodyInterface* iface, float radius, float density, bool is_static) {
    // Create sphere shape and body
    return 2; // Mock ID
}

void jolt_body_set_position(JoltBodyInterface* iface, JoltBodyID body_id, float x, float y, float z) {
    // Set position
}

void jolt_body_get_position(JoltBodyInterface* iface, JoltBodyID body_id, float* x, float* y, float* z) {
    // Get position
    *x = 0.0f;
    *y = 0.0f;
    *z = 0.0f;
}

void jolt_body_set_velocity(JoltBodyInterface* iface, JoltBodyID body_id, float vx, float vy, float vz) {
    // Set velocity
}

void jolt_body_get_velocity(JoltBodyInterface* iface, JoltBodyID body_id, float* vx, float* vy, float* vz) {
    // Get velocity
    *vx = 0.0f;
    *vy = 0.0f;
    *vz = 0.0f;
}

void jolt_body_add_force(JoltBodyInterface* iface, JoltBodyID body_id, float fx, float fy, float fz) {
    // Add force
}

bool jolt_body_is_active(JoltBodyInterface* iface, JoltBodyID body_id) {
    return true;
}

void jolt_body_activate(JoltBodyInterface* iface, JoltBodyID body_id) {
    // Activate
}

void jolt_body_deactivate(JoltBodyInterface* iface, JoltBodyID body_id) {
    // Deactivate
}

} // extern "C"
