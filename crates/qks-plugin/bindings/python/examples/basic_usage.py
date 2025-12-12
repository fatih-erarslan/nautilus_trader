"""
QKS Plugin - Python Basic Usage Example

Demonstrates all three cognitive layers:
- Layer 6: Consciousness (IIT + Global Workspace)
- Layer 7: Metacognition (Introspection)
- Layer 8: Integration (Cognitive Cycles)
"""

import numpy as np
from qks import QKSPlugin, QKSConfig, SensoryInput

def main():
    print("=== QKS Plugin - Python Example ===\n")

    # Initialize plugin with GPU acceleration
    config = QKSConfig(
        num_qubits=20,
        use_gpu=True,
        enable_consciousness=True,
        enable_metacognition=True
    )

    with QKSPlugin(config) as plugin:
        print(f"QKS Version: {plugin.version()}")
        print(f"Configuration: {plugin.info()}\n")

        # ====================================================================
        # LAYER 6: CONSCIOUSNESS
        # ====================================================================
        print("=== Layer 6: Consciousness ===")

        # IIT 3.0: Compute integrated information Φ
        network_state = np.array([0.8, 0.6, 0.9, 0.7, 0.5])
        print(f"Network state: {network_state}")

        phi_result = plugin.consciousness.compute_phi(network_state)
        print(f"\nΦ = {phi_result.phi:.3f}")
        print(f"Conscious: {phi_result.is_conscious}")
        print(f"Level: {phi_result.consciousness_level}")
        print(f"MIP size: {phi_result.mip_size}")
        print(f"Computation time: {phi_result.computation_time_ms:.2f}ms")

        # Global Workspace: Broadcast content
        broadcast_result = plugin.consciousness.broadcast(
            content={
                "type": "visual",
                "scene": "red triangle",
                "salience": 0.9
            },
            priority=0.8
        )
        print(f"\nBroadcast successful: {broadcast_result.success}")
        print(f"Recipients: {broadcast_result.recipients}")

        # ====================================================================
        # LAYER 7: METACOGNITION
        # ====================================================================
        print("\n=== Layer 7: Metacognition ===")

        # Add beliefs
        plugin.metacognition.add_belief(
            "The network exhibits consciousness",
            confidence=0.85,
            evidence=["Phi > 1.0", "Global integration present"]
        )

        plugin.metacognition.add_belief(
            "Visual processing is active",
            confidence=0.92,
            evidence=["High salience visual input"]
        )

        # Add goals
        plugin.metacognition.add_goal(
            "Optimize quantum circuit depth",
            priority=0.9
        )

        plugin.metacognition.add_goal(
            "Maintain metacognitive accuracy",
            priority=0.7
        )

        # Introspection
        report = plugin.metacognition.introspect()
        print(f"\nIntrospection Report:")
        print(f"  Confidence: {report.confidence:.2%}")
        print(f"  Beliefs: {len(report.beliefs)}")
        print(f"  Goals: {len(report.goals)} ({report.num_active_goals} active)")
        print(f"  Capabilities: {len(report.capabilities)}")

        print(f"\nHigh-confidence beliefs:")
        for belief in report.high_confidence_beliefs:
            print(f"  - {belief.content} ({belief.confidence:.2f})")

        # Monitor performance
        plugin.metacognition.monitor_performance({
            "accuracy": 0.92,
            "latency_ms": 145,
            "memory_mb": 512,
        })

        # ====================================================================
        # LAYER 8: INTEGRATION
        # ====================================================================
        print("\n=== Layer 8: Integration ===")

        # Execute cognitive cycle
        sensory_input = SensoryInput(
            modality="visual",
            data={
                "scene": "moving red triangle",
                "velocity": 2.5,
                "salience": 0.9
            },
            priority=0.8
        )

        output = plugin.integration.cognitive_cycle(sensory_input)
        print(f"\nCognitive Cycle Output:")
        print(f"  Action: {output.action}")
        print(f"  Confidence: {output.confidence:.2%}")
        print(f"\nReasoning trace:")
        for step in output.reasoning:
            print(f"  - {step}")

        print(f"\nPhase timings (ms):")
        for phase, time_ms in output.phase_timings.items():
            print(f"  {phase.value}: {time_ms:.2f}ms")

        # Batch processing
        print("\n=== Batch Processing ===")
        batch_inputs = [
            SensoryInput("visual", {"scene": "cat"}, priority=0.6),
            SensoryInput("audio", {"sound": "meow"}, priority=0.7),
            SensoryInput("visual", {"scene": "dog"}, priority=0.5),
        ]

        batch_outputs = plugin.integration.batch_process(batch_inputs)
        print(f"Processed {len(batch_outputs)} inputs:")
        for i, out in enumerate(batch_outputs):
            print(f"  {i+1}. {out.action} (confidence: {out.confidence:.2f})")

        # Statistics
        stats = plugin.integration.get_statistics()
        print(f"\nCognitive Cycle Statistics:")
        print(f"  Total cycles: {stats.total_cycles}")
        print(f"  Success rate: {stats.success_rate:.2%}")
        print(f"  Avg cycle time: {stats.avg_cycle_time_ms:.1f}ms")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
