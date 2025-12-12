"""Layer 7: Metacognition APIs - Introspection and Self-Monitoring.

This module implements metacognitive capabilities - "thinking about thinking":
- Real-time introspection of cognitive state
- Performance monitoring and adjustment
- Belief and goal tracking
- Confidence estimation

References:
    Fleming, S. M., & Dolan, R. J. (2012).
    The neural basis of metacognitive ability.
    Philosophical Transactions of the Royal Society B, 367(1594), 1338-1349.

    Nelson, T. O., & Narens, L. (1990).
    Metamemory: A theoretical framework and new findings.
    Psychology of Learning and Motivation, 26, 125-173.
"""

import ctypes
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

from qks._ffi import _lib, QksHandle, IntrospectionReportC, check_error


class ConfidenceLevel(Enum):
    """Metacognitive confidence levels."""
    VERY_LOW = "very_low"    # < 20%
    LOW = "low"               # 20-40%
    MODERATE = "moderate"     # 40-60%
    HIGH = "high"             # 60-80%
    VERY_HIGH = "very_high"   # > 80%


@dataclass
class BeliefState:
    """Represents a belief in the cognitive system.

    Attributes:
        content: Belief content (proposition)
        confidence: Confidence level (0.0-1.0)
        evidence: Supporting evidence
        timestamp: Creation timestamp
    """
    content: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get categorical confidence level."""
        if self.confidence < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.6:
            return ConfidenceLevel.MODERATE
        elif self.confidence < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH


@dataclass
class GoalState:
    """Represents a goal in the cognitive system.

    Attributes:
        description: Goal description
        priority: Priority level (0.0-1.0)
        progress: Progress toward goal (0.0-1.0)
        active: Whether goal is currently active
    """
    description: str
    priority: float
    progress: float = 0.0
    active: bool = True


@dataclass
class IntrospectionReport:
    """Complete introspection report of cognitive state.

    This report provides a snapshot of the system's current cognitive state,
    including beliefs, goals, capabilities, and overall confidence.

    Attributes:
        beliefs: Current belief states
        goals: Active goals
        capabilities: Known capabilities
        confidence: Overall confidence level (0.0-1.0)
        timestamp_ms: Report generation timestamp
        performance_metrics: Recent performance metrics
    """
    beliefs: List[BeliefState] = field(default_factory=list)
    goals: List[GoalState] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp_ms: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def num_active_goals(self) -> int:
        """Count active goals."""
        return sum(1 for g in self.goals if g.active)

    @property
    def high_confidence_beliefs(self) -> List[BeliefState]:
        """Get beliefs with high confidence (> 0.7)."""
        return [b for b in self.beliefs if b.confidence > 0.7]


class Metacognition:
    """Layer 7: Metacognition APIs.

    Provides tools for introspection and self-monitoring - the ability
    of the system to monitor and reason about its own cognitive processes.

    Key capabilities:
    - Real-time state introspection
    - Performance monitoring
    - Confidence estimation
    - Belief and goal management
    """

    def __init__(self, handle: QksHandle):
        """Initialize metacognition API.

        Args:
            handle: Native plugin handle (internal use only)
        """
        self._handle = handle
        self._belief_store: List[BeliefState] = []
        self._goal_store: List[GoalState] = []

    def introspect(self) -> IntrospectionReport:
        """Perform real-time introspection of cognitive state.

        Examines the current state of the cognitive system and generates
        a comprehensive report including beliefs, goals, capabilities,
        and confidence levels.

        Returns:
            IntrospectionReport with complete cognitive state snapshot.

        Raises:
            RuntimeError: If introspection fails

        Example:
            >>> report = metacognition.introspect()
            >>> print(f"Confidence: {report.confidence:.2%}")
            >>> print(f"Active goals: {report.num_active_goals}")
            >>> for belief in report.high_confidence_beliefs:
            ...     print(f"- {belief.content} ({belief.confidence:.2f})")

        Scientific Background:
            Metacognition allows agents to monitor their own cognition,
            enabling self-correction, learning from mistakes, and
            explaining their reasoning. This is a key component of
            human-level intelligence.
        """
        # Get native introspection data
        report_c = IntrospectionReportC()
        ret = _lib.qks_introspect(self._handle, ctypes.byref(report_c))
        check_error(ret)

        # Build report
        report = IntrospectionReport(
            beliefs=self._belief_store.copy(),
            goals=self._goal_store.copy(),
            capabilities=[
                "consciousness_measurement",
                "metacognitive_monitoring",
                "cognitive_cycles",
                "quantum_simulation",
            ],
            confidence=report_c.confidence,
            timestamp_ms=report_c.timestamp_ms,
            performance_metrics={
                "num_beliefs": len(self._belief_store),
                "num_goals": len(self._goal_store),
                "avg_belief_confidence": sum(b.confidence for b in self._belief_store) / max(len(self._belief_store), 1),
            }
        )

        return report

    def add_belief(self, content: str, confidence: float, evidence: Optional[List[str]] = None) -> BeliefState:
        """Add a new belief to the cognitive state.

        Args:
            content: Belief content (proposition)
            confidence: Confidence level (0.0-1.0)
            evidence: Supporting evidence items

        Returns:
            Created BeliefState

        Raises:
            ValueError: If confidence not in [0, 1]

        Example:
            >>> belief = metacognition.add_belief(
            ...     "The network exhibits consciousness",
            ...     confidence=0.85,
            ...     evidence=["Phi > 1.0", "Global integration present"]
            ... )
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        belief = BeliefState(
            content=content,
            confidence=confidence,
            evidence=evidence or [],
        )
        self._belief_store.append(belief)
        return belief

    def update_belief(self, content: str, new_confidence: float) -> Optional[BeliefState]:
        """Update confidence of an existing belief.

        Args:
            content: Belief content to update
            new_confidence: New confidence level (0.0-1.0)

        Returns:
            Updated BeliefState if found, None otherwise
        """
        for belief in self._belief_store:
            if belief.content == content:
                belief.confidence = new_confidence
                belief.timestamp = time.time()
                return belief
        return None

    def add_goal(self, description: str, priority: float) -> GoalState:
        """Add a new goal to the cognitive state.

        Args:
            description: Goal description
            priority: Priority level (0.0-1.0)

        Returns:
            Created GoalState

        Example:
            >>> goal = metacognition.add_goal(
            ...     "Optimize quantum circuit depth",
            ...     priority=0.9
            ... )
        """
        if not 0.0 <= priority <= 1.0:
            raise ValueError(f"priority must be in [0, 1], got {priority}")

        goal = GoalState(
            description=description,
            priority=priority,
        )
        self._goal_store.append(goal)
        return goal

    def update_goal_progress(self, description: str, progress: float) -> Optional[GoalState]:
        """Update progress of an existing goal.

        Args:
            description: Goal description
            progress: New progress value (0.0-1.0)

        Returns:
            Updated GoalState if found, None otherwise
        """
        for goal in self._goal_store:
            if goal.description == description:
                goal.progress = progress
                if progress >= 1.0:
                    goal.active = False
                return goal
        return None

    def monitor_performance(self, metrics: Dict[str, float]) -> None:
        """Monitor performance metrics.

        Tracks performance over time and adjusts cognitive strategies
        based on observed performance.

        Args:
            metrics: Dictionary of metric_name -> value

        Example:
            >>> metacognition.monitor_performance({
            ...     "accuracy": 0.92,
            ...     "latency_ms": 145,
            ...     "memory_mb": 512,
            ... })
        """
        # Convert to C array
        metric_array = (ctypes.c_double * len(metrics))(*metrics.values())

        ret = _lib.qks_monitor_performance(
            self._handle,
            metric_array,
            len(metrics)
        )
        check_error(ret)

    def estimate_confidence(self, task_description: str) -> float:
        """Estimate confidence for a given task.

        Uses historical performance and current state to estimate
        confidence in successfully completing a task.

        Args:
            task_description: Description of the task

        Returns:
            Estimated confidence level (0.0-1.0)

        Example:
            >>> confidence = metacognition.estimate_confidence(
            ...     "Solve 20-qubit VQE problem"
            ... )
            >>> print(f"Confidence: {confidence:.2%}")
        """
        # Placeholder - needs implementation with ML model
        return 0.5
