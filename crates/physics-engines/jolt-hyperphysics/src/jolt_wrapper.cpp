// JoltPhysics wrapper for HyperPhysics
// Provides C API to JoltPhysics C++ library

#include "jolt_wrapper.h"

#ifndef JOLT_MOCK_IMPLEMENTATION

// Real JoltPhysics implementation
#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/RegisterTypes.h>

#include <thread>
#include <cstdarg>

// Jolt namespace
using namespace JPH;
using namespace JPH::literals;

// Layers for collision filtering
namespace Layers {
    static constexpr ObjectLayer NON_MOVING = 0;
    static constexpr ObjectLayer MOVING = 1;
    static constexpr ObjectLayer NUM_LAYERS = 2;
}

namespace BroadPhaseLayers {
    static constexpr BroadPhaseLayer NON_MOVING(0);
    static constexpr BroadPhaseLayer MOVING(1);
    static constexpr uint NUM_LAYERS(2);
}

// BroadPhaseLayerInterface implementation
class BPLayerInterfaceImpl final : public BroadPhaseLayerInterface {
public:
    virtual uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::NUM_LAYERS; }
    virtual BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer inLayer) const override {
        static BroadPhaseLayer sMap[] = { BroadPhaseLayers::NON_MOVING, BroadPhaseLayers::MOVING };
        return sMap[inLayer];
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    virtual const char* GetBroadPhaseLayerName(BroadPhaseLayer inLayer) const override {
        switch ((BroadPhaseLayer::Type)inLayer) {
            case (BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING: return "NON_MOVING";
            case (BroadPhaseLayer::Type)BroadPhaseLayers::MOVING: return "MOVING";
            default: return "INVALID";
        }
    }
#endif
};

// ObjectLayerPairFilter implementation
class ObjectLayerPairFilterImpl : public ObjectLayerPairFilter {
public:
    virtual bool ShouldCollide(ObjectLayer inObject1, ObjectLayer inObject2) const override {
        switch (inObject1) {
            case Layers::NON_MOVING:
                return inObject2 == Layers::MOVING;
            case Layers::MOVING:
                return true;
            default:
                return false;
        }
    }
};

// ObjectVsBroadPhaseLayerFilter implementation
class ObjectVsBroadPhaseLayerFilterImpl : public ObjectVsBroadPhaseLayerFilter {
public:
    virtual bool ShouldCollide(ObjectLayer inLayer1, BroadPhaseLayer inLayer2) const override {
        switch (inLayer1) {
            case Layers::NON_MOVING:
                return inLayer2 == BroadPhaseLayers::MOVING;
            case Layers::MOVING:
                return true;
            default:
                return false;
        }
    }
};

// Contact listener for events
class ContactListenerImpl : public ContactListener {
public:
    virtual ValidateResult OnContactValidate(const Body &inBody1, const Body &inBody2,
                                             RVec3Arg inBaseOffset,
                                             const CollideShapeResult &inCollisionResult) override {
        return ValidateResult::AcceptAllContactsForThisBodyPair;
    }
    virtual void OnContactAdded(const Body &inBody1, const Body &inBody2,
                               const ContactManifold &inManifold, ContactSettings &ioSettings) override {}
    virtual void OnContactPersisted(const Body &inBody1, const Body &inBody2,
                                    const ContactManifold &inManifold, ContactSettings &ioSettings) override {}
    virtual void OnContactRemoved(const SubShapeIDPair &inSubShapePair) override {}
};

// Body activation listener
class BodyActivationListenerImpl : public BodyActivationListener {
public:
    virtual void OnBodyActivated(const BodyID &inBodyID, uint64 inBodyUserData) override {}
    virtual void OnBodyDeactivated(const BodyID &inBodyID, uint64 inBodyUserData) override {}
};

// Internal system structure
struct JoltSystem {
    JoltConfig config;
    std::unique_ptr<TempAllocatorImpl> temp_allocator;
    std::unique_ptr<JobSystemThreadPool> job_system;
    std::unique_ptr<PhysicsSystem> physics_system;
    BPLayerInterfaceImpl broad_phase_layer_interface;
    ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
    ObjectLayerPairFilterImpl object_layer_pair_filter;
    ContactListenerImpl contact_listener;
    BodyActivationListenerImpl body_activation_listener;
};

struct JoltBodyInterface {
    JoltSystem* system;
    BodyInterface* body_interface;
};

// Trace callback for Jolt debugging
static void JoltTraceImpl(const char *inFMT, ...) {
    va_list list;
    va_start(list, inFMT);
    vfprintf(stderr, inFMT, list);
    va_end(list);
    fprintf(stderr, "\n");
}

#ifdef JPH_ENABLE_ASSERTS
static bool JoltAssertFailed(const char *inExpression, const char *inMessage,
                              const char *inFile, uint inLine) {
    fprintf(stderr, "Jolt Assert: %s:%d: (%s) %s\n", inFile, inLine, inExpression,
            inMessage ? inMessage : "");
    return true; // Break into debugger
}
#endif

extern "C" {

JoltSystem* jolt_system_create(JoltConfig config) {
    // Register default allocator and install callbacks
    RegisterDefaultAllocator();
    Trace = JoltTraceImpl;
    #ifdef JPH_ENABLE_ASSERTS
    AssertFailed = JoltAssertFailed;
    #endif

    // Create a factory (required for types)
    Factory::sInstance = new Factory();

    // Register physics types
    RegisterTypes();

    auto* system = new JoltSystem();
    system->config = config;

    // Allocators
    system->temp_allocator = std::make_unique<TempAllocatorImpl>(10 * 1024 * 1024); // 10 MB
    system->job_system = std::make_unique<JobSystemThreadPool>(
        cMaxPhysicsJobs, cMaxPhysicsBarriers,
        std::thread::hardware_concurrency() - 1
    );

    // Create physics system
    system->physics_system = std::make_unique<PhysicsSystem>();
    system->physics_system->Init(
        config.max_bodies,
        config.num_body_mutexes,
        config.max_body_pairs,
        config.max_contact_constraints,
        system->broad_phase_layer_interface,
        system->object_vs_broadphase_layer_filter,
        system->object_layer_pair_filter
    );

    // Set up listeners
    system->physics_system->SetBodyActivationListener(&system->body_activation_listener);
    system->physics_system->SetContactListener(&system->contact_listener);

    return system;
}

void jolt_system_destroy(JoltSystem* system) {
    if (system) {
        // Unregister types
        UnregisterTypes();

        // Destroy factory
        delete Factory::sInstance;
        Factory::sInstance = nullptr;

        delete system;
    }
}

void jolt_system_optimize(JoltSystem* system) {
    if (system && system->physics_system) {
        system->physics_system->OptimizeBroadPhase();
    }
}

void jolt_system_step(JoltSystem* system, float dt, int collision_steps) {
    if (system && system->physics_system) {
        system->physics_system->Update(
            dt,
            collision_steps,
            system->temp_allocator.get(),
            system->job_system.get()
        );
    }
}

JoltBodyInterface* jolt_system_get_body_interface(JoltSystem* system) {
    if (!system) return nullptr;

    auto* iface = new JoltBodyInterface();
    iface->system = system;
    iface->body_interface = &system->physics_system->GetBodyInterface();
    return iface;
}

JoltBodyID jolt_body_create_box(JoltBodyInterface* iface, float width, float height, float depth,
                                 float density, bool is_static) {
    if (!iface || !iface->body_interface) return 0;

    BoxShapeSettings shape_settings(Vec3(width / 2.0f, height / 2.0f, depth / 2.0f));
    shape_settings.SetDensity(density);

    ShapeSettings::ShapeResult shape_result = shape_settings.Create();
    if (shape_result.HasError()) return 0;

    BodyCreationSettings body_settings(
        shape_result.Get(),
        RVec3::sZero(),
        Quat::sIdentity(),
        is_static ? EMotionType::Static : EMotionType::Dynamic,
        is_static ? Layers::NON_MOVING : Layers::MOVING
    );

    Body* body = iface->body_interface->CreateBody(body_settings);
    if (!body) return 0;

    iface->body_interface->AddBody(body->GetID(), is_static ? EActivation::DontActivate : EActivation::Activate);
    return body->GetID().GetIndexAndSequenceNumber();
}

JoltBodyID jolt_body_create_sphere(JoltBodyInterface* iface, float radius, float density, bool is_static) {
    if (!iface || !iface->body_interface) return 0;

    SphereShapeSettings shape_settings(radius);
    shape_settings.SetDensity(density);

    ShapeSettings::ShapeResult shape_result = shape_settings.Create();
    if (shape_result.HasError()) return 0;

    BodyCreationSettings body_settings(
        shape_result.Get(),
        RVec3::sZero(),
        Quat::sIdentity(),
        is_static ? EMotionType::Static : EMotionType::Dynamic,
        is_static ? Layers::NON_MOVING : Layers::MOVING
    );

    Body* body = iface->body_interface->CreateBody(body_settings);
    if (!body) return 0;

    iface->body_interface->AddBody(body->GetID(), is_static ? EActivation::DontActivate : EActivation::Activate);
    return body->GetID().GetIndexAndSequenceNumber();
}

void jolt_body_set_position(JoltBodyInterface* iface, JoltBodyID body_id, float x, float y, float z) {
    if (!iface || !iface->body_interface) return;
    BodyID id(body_id);
    iface->body_interface->SetPosition(id, RVec3(x, y, z), EActivation::Activate);
}

void jolt_body_get_position(JoltBodyInterface* iface, JoltBodyID body_id, float* x, float* y, float* z) {
    if (!iface || !iface->body_interface || !x || !y || !z) return;
    BodyID id(body_id);
    RVec3 pos = iface->body_interface->GetCenterOfMassPosition(id);
    *x = (float)pos.GetX();
    *y = (float)pos.GetY();
    *z = (float)pos.GetZ();
}

void jolt_body_set_velocity(JoltBodyInterface* iface, JoltBodyID body_id, float vx, float vy, float vz) {
    if (!iface || !iface->body_interface) return;
    BodyID id(body_id);
    iface->body_interface->SetLinearVelocity(id, Vec3(vx, vy, vz));
}

void jolt_body_get_velocity(JoltBodyInterface* iface, JoltBodyID body_id, float* vx, float* vy, float* vz) {
    if (!iface || !iface->body_interface || !vx || !vy || !vz) return;
    BodyID id(body_id);
    Vec3 vel = iface->body_interface->GetLinearVelocity(id);
    *vx = vel.GetX();
    *vy = vel.GetY();
    *vz = vel.GetZ();
}

void jolt_body_add_force(JoltBodyInterface* iface, JoltBodyID body_id, float fx, float fy, float fz) {
    if (!iface || !iface->body_interface) return;
    BodyID id(body_id);
    iface->body_interface->AddForce(id, Vec3(fx, fy, fz));
}

bool jolt_body_is_active(JoltBodyInterface* iface, JoltBodyID body_id) {
    if (!iface || !iface->body_interface) return false;
    BodyID id(body_id);
    return iface->body_interface->IsActive(id);
}

void jolt_body_activate(JoltBodyInterface* iface, JoltBodyID body_id) {
    if (!iface || !iface->body_interface) return;
    BodyID id(body_id);
    iface->body_interface->ActivateBody(id);
}

void jolt_body_deactivate(JoltBodyInterface* iface, JoltBodyID body_id) {
    if (!iface || !iface->body_interface) return;
    BodyID id(body_id);
    iface->body_interface->DeactivateBody(id);
}

} // extern "C"

#else // JOLT_MOCK_IMPLEMENTATION

// Mock implementation when JoltPhysics is not available
#include <stdlib.h>
#include <stdio.h>

struct JoltSystem {
    JoltConfig config;
};

struct JoltBodyInterface {
    JoltSystem* system;
};

extern "C" {

JoltSystem* jolt_system_create(JoltConfig config) {
    JoltSystem* system = (JoltSystem*)malloc(sizeof(JoltSystem));
    system->config = config;
    fprintf(stderr, "Warning: Using mock JoltPhysics implementation\n");
    return system;
}

void jolt_system_destroy(JoltSystem* system) {
    if (system) free(system);
}

void jolt_system_optimize(JoltSystem* system) {}

void jolt_system_step(JoltSystem* system, float dt, int collision_steps) {}

JoltBodyInterface* jolt_system_get_body_interface(JoltSystem* system) {
    JoltBodyInterface* iface = (JoltBodyInterface*)malloc(sizeof(JoltBodyInterface));
    iface->system = system;
    return iface;
}

JoltBodyID jolt_body_create_box(JoltBodyInterface* iface, float width, float height, float depth,
                                 float density, bool is_static) {
    return 1;
}

JoltBodyID jolt_body_create_sphere(JoltBodyInterface* iface, float radius, float density, bool is_static) {
    return 2;
}

void jolt_body_set_position(JoltBodyInterface* iface, JoltBodyID body_id, float x, float y, float z) {}
void jolt_body_get_position(JoltBodyInterface* iface, JoltBodyID body_id, float* x, float* y, float* z) {
    if (x) *x = 0.0f;
    if (y) *y = 0.0f;
    if (z) *z = 0.0f;
}

void jolt_body_set_velocity(JoltBodyInterface* iface, JoltBodyID body_id, float vx, float vy, float vz) {}
void jolt_body_get_velocity(JoltBodyInterface* iface, JoltBodyID body_id, float* vx, float* vy, float* vz) {
    if (vx) *vx = 0.0f;
    if (vy) *vy = 0.0f;
    if (vz) *vz = 0.0f;
}

void jolt_body_add_force(JoltBodyInterface* iface, JoltBodyID body_id, float fx, float fy, float fz) {}
bool jolt_body_is_active(JoltBodyInterface* iface, JoltBodyID body_id) { return true; }
void jolt_body_activate(JoltBodyInterface* iface, JoltBodyID body_id) {}
void jolt_body_deactivate(JoltBodyInterface* iface, JoltBodyID body_id) {}

} // extern "C"

#endif // JOLT_MOCK_IMPLEMENTATION
