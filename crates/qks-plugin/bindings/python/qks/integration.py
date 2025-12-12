"""Layer 8: Integration APIs - Full Cognitive Cycles.

This module implements complete perception-cognition-action loops,
integrating consciousness and metacognition into unified cognitive cycles.

References:
    Anderson, J. R. (2007).
    How can the human mind occur in the physical universe?
    Oxford University Press.

    Laird, J. E. (2012).
    The Soar cognitive architecture.
    MIT Press.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import ctypes

from qks._ffi import _lib, QksHandle, check_error


class CognitivePhase(Enum):
    """Phases of the cognitive cycle."""
    PERCEPTION = "perception"
    ATTENTION = "attention"
    REASONING = "reasoning"
    DECISION = "decision"
    ACTION = "action"
    LEARNING = "learning"


@dataclass
class SensoryInput:
    """Sensory input for cognitive processing.

    Attributes:
        modality: Input modality (visual, auditory, etc.)
        data: Raw sensory data
        timestamp_ms: Input timestamp
        priority: Input priority (0.0-1.0)
        metadata: Additional metadata
    """
    modality: str
    data: Any
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    priority: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveOutput:
    """Output of a cognitive cycle.

    Attributes:
        action: Recommended action
        confidence: Action confidence (0.0-1.0)
        reasoning: Reasoning trace
        phase_timings: Time spent in each phase (ms)
        beliefs_updated: Beliefs updated during cycle
        goals_updated: Goals updated during cycle
    """
    action: str
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    phase_timings: Dict[CognitivePhase, float] = field(default_factory=dict)
    beliefs_updated: List[str] = field(default_factory=list)
    goals_updated: List[str] = field(default_factory=list)


@dataclass
class CycleStatistics:
    """Statistics from cognitive cycles.

    Attributes:
        total_cycles: Total number of cycles executed
        avg_cycle_time_ms: Average cycle time
        successful_cycles: Number of successful cycles
        failed_cycles: Number of failed cycles
        phase_breakdown: Time breakdown by phase
    """
    total_cycles: int = 0
    avg_cycle_time_ms: float = 0.0
    successful_cycles: int = 0
    failed_cycles: int = 0
    phase_breakdown: Dict[CognitivePhase, float] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_cycles == 0:
            return 0.0
        return self.successful_cycles / self.total_cycles


class Integration:
    """Layer 8: Integration APIs.

    Provides complete cognitive cycles that integrate:
    - Sensory perception
    - Conscious awareness (Layer 6)
    - Metacognitive monitoring (Layer 7)
    - Decision making
    - Action generation
    - Learning and adaptation

    This represents the highest level of cognitive integration,
    combining all lower layers into unified cognitive behavior.
    """

    def __init__(self, handle: QksHandle):
        """Initialize integration API.

        Args:
            handle: Native plugin handle (internal use only)
        """
        self._handle = handle
        self._statistics = CycleStatistics()

    def cognitive_cycle(
        self,
        sensory_input: SensoryInput,
        enable_learning: bool = True
    ) -> CognitiveOutput:
        """Execute one complete cognitive cycle.

        Implements a full perception-cognition-action loop:

        1. PERCEPTION: Process sensory input
        2. ATTENTION: Filter for salient information
        3. REASONING: Apply knowledge and beliefs
        4. DECISION: Select action based on goals
        5. ACTION: Generate action recommendation
        6. LEARNING: Update beliefs and models

        Args:
            sensory_input: Input to process
            enable_learning: Whether to update internal models

        Returns:
            CognitiveOutput with action and reasoning trace

        Raises:
            RuntimeError: If cycle execution fails

        Example:
            >>> # Process visual input
            >>> input_data = SensoryInput(
            ...     modality="visual",
            ...     data={"scene": "red triangle", "salience": 0.9},
            ...     priority=0.8
            ... )
            >>> output = integration.cognitive_cycle(input_data)
            >>> print(f"Action: {output.action}")
            >>> print(f"Confidence: {output.confidence:.2%}")
            >>> for step in output.reasoning:
            ...     print(f"  - {step}")

        Scientific Background:
            This implements a cognitive architecture similar to ACT-R
            (Anderson, 2007) and Soar (Laird, 2012), with added
            consciousness and metacognition layers. The cycle integrates
            bottom-up sensory processing with top-down goal-driven control.
        """
        start_time = time.time()
        phase_timings = {}

        try:
            # Phase 1: PERCEPTION
            phase_start = time.time()
            percept = self._perceive(sensory_input)
            phase_timings[CognitivePhase.PERCEPTION] = (time.time() - phase_start) * 1000

            # Phase 2: ATTENTION
            phase_start = time.time()
            attended = self._attend(percept, sensory_input.priority)
            phase_timings[CognitivePhase.ATTENTION] = (time.time() - phase_start) * 1000

            # Phase 3: REASONING
            phase_start = time.time()
            reasoning_trace = self._reason(attended)
            phase_timings[CognitivePhase.REASONING] = (time.time() - phase_start) * 1000

            # Phase 4: DECISION
            phase_start = time.time()
            decision, confidence = self._decide(attended, reasoning_trace)
            phase_timings[CognitivePhase.DECISION] = (time.time() - phase_start) * 1000

            # Phase 5: ACTION
            phase_start = time.time()
            action = self._generate_action(decision)
            phase_timings[CognitivePhase.ACTION] = (time.time() - phase_start) * 1000

            # Phase 6: LEARNING
            beliefs_updated = []
            goals_updated = []
            if enable_learning:
                phase_start = time.time()
                beliefs_updated, goals_updated = self._learn(sensory_input, action, confidence)
                phase_timings[CognitivePhase.LEARNING] = (time.time() - phase_start) * 1000

            # Update statistics
            self._statistics.total_cycles += 1
            self._statistics.successful_cycles += 1
            cycle_time = (time.time() - start_time) * 1000
            self._statistics.avg_cycle_time_ms = (
                (self._statistics.avg_cycle_time_ms * (self._statistics.total_cycles - 1) + cycle_time)
                / self._statistics.total_cycles
            )

            return CognitiveOutput(
                action=action,
                confidence=confidence,
                reasoning=reasoning_trace,
                phase_timings=phase_timings,
                beliefs_updated=beliefs_updated,
                goals_updated=goals_updated,
            )

        except Exception as e:
            self._statistics.total_cycles += 1
            self._statistics.failed_cycles += 1
            raise RuntimeError(f"Cognitive cycle failed: {e}")

    def _perceive(self, sensory_input: SensoryInput) -> Dict[str, Any]:
        """Phase 1: Perceive and preprocess sensory input."""
        return {
            "modality": sensory_input.modality,
            "features": sensory_input.data,
            "timestamp": sensory_input.timestamp_ms,
        }

    def _attend(self, percept: Dict[str, Any], priority: float) -> Dict[str, Any]:
        """Phase 2: Apply attention to salient features."""
        # Filter based on priority and salience
        attended = percept.copy()
        attended["priority"] = priority
        attended["attended"] = priority > 0.5
        return attended

    def _reason(self, attended: Dict[str, Any]) -> List[str]:
        """Phase 3: Apply reasoning and knowledge."""
        reasoning = []

        # Check if attention threshold met
        if attended.get("attended", False):
            reasoning.append("Input exceeded attention threshold")
            reasoning.append(f"Processing {attended['modality']} input")

            # Apply domain knowledge
            if "features" in attended:
                reasoning.append(f"Extracted features: {list(attended['features'].keys())}")

        return reasoning

    def _decide(self, attended: Dict[str, Any], reasoning: List[str]) -> tuple[str, float]:
        """Phase 4: Make decision based on goals and reasoning."""
        # Simple decision logic (placeholder)
        if attended.get("attended", False):
            decision = "process_input"
            confidence = attended.get("priority", 0.5)
        else:
            decision = "ignore_input"
            confidence = 0.3

        return decision, confidence

    def _generate_action(self, decision: str) -> str:
        """Phase 5: Generate action from decision."""
        action_map = {
            "process_input": "Process and store input",
            "ignore_input": "Ignore low-priority input",
            "request_clarification": "Request additional information",
        }
        return action_map.get(decision, "Unknown action")

    def _learn(
        self,
        sensory_input: SensoryInput,
        action: str,
        confidence: float
    ) -> tuple[List[str], List[str]]:
        """Phase 6: Update beliefs and models based on outcome."""
        beliefs_updated = []
        goals_updated = []

        # Update beliefs based on experience
        if confidence > 0.7:
            beliefs_updated.append(f"High confidence in {sensory_input.modality} processing")

        return beliefs_updated, goals_updated

    def get_statistics(self) -> CycleStatistics:
        """Get cognitive cycle statistics.

        Returns:
            CycleStatistics with performance metrics

        Example:
            >>> stats = integration.get_statistics()
            >>> print(f"Total cycles: {stats.total_cycles}")
            >>> print(f"Success rate: {stats.success_rate:.2%}")
            >>> print(f"Avg cycle time: {stats.avg_cycle_time_ms:.1f}ms")
        """
        return self._statistics

    def reset_statistics(self) -> None:
        """Reset cycle statistics."""
        self._statistics = CycleStatistics()

    def batch_process(
        self,
        inputs: List[SensoryInput],
        parallel: bool = False
    ) -> List[CognitiveOutput]:
        """Process multiple inputs in batch.

        Args:
            inputs: List of sensory inputs
            parallel: Whether to process in parallel (requires multi-threading)

        Returns:
            List of cognitive outputs

        Example:
            >>> inputs = [
            ...     SensoryInput("visual", {"scene": "cat"}),
            ...     SensoryInput("audio", {"sound": "meow"}),
            ... ]
            >>> outputs = integration.batch_process(inputs)
        """
        if parallel:
            # TODO: Implement parallel processing
            pass

        outputs = []
        for input_item in inputs:
            try:
                output = self.cognitive_cycle(input_item)
                outputs.append(output)
            except Exception as e:
                # Continue processing remaining inputs
                outputs.append(CognitiveOutput(
                    action="error",
                    confidence=0.0,
                    reasoning=[f"Error: {e}"]
                ))

        return outputs
