"""
Execution planning and optimization for quantum workloads.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import asyncio
import networkx as nx
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ExecutionTask:
    """Represents a single execution task."""
    task_id: str
    operation: str
    resource_requirements: Dict[str, Any]
    dependencies: List[str]
    priority: float
    estimated_runtime: float
    deadline: Optional[datetime] = None

@dataclass
class ExecutionPlan:
    """Complete execution plan."""
    plan_id: str
    tasks: List[ExecutionTask]
    schedule: Dict[str, Tuple[float, float]]  # task_id -> (start_time, end_time)
    resource_allocation: Dict[str, Dict[str, Any]]
    estimated_total_time: float
    optimization_score: float

class ExecutionPlanner:
    """
    Plans and optimizes execution of quantum and classical workloads.
    """

    def __init__(self, resource_manager: Any, hw_optimizer: Any):
        """
        Initialize execution planner.

        Args:
            resource_manager: Resource manager instance
            hw_optimizer: Hardware optimizer instance
        """
        self.resource_manager = resource_manager
        self.hw_optimizer = hw_optimizer

        # Planning configuration
        self.planning_horizon = 300  # 5 minutes
        self.replan_interval = 30    # 30 seconds

        # Execution tracking
        self.active_plans = {}
        self.execution_history = deque(maxlen=1000)
        self.task_queue = asyncio.PriorityQueue()

        # Performance metrics
        self.planning_stats = {
            'plans_created': 0,
            'tasks_scheduled': 0,
            'optimization_improvements': [],
            'deadline_violations': 0
        }

    async def create_execution_plan(self, workload: Dict[str, Any]) -> ExecutionPlan:
        """
        Create optimized execution plan for workload.

        Args:
            workload: Workload specification

        Returns:
            Optimized execution plan
        """
        # Extract tasks from workload
        tasks = self._extract_tasks(workload)

        # Build dependency graph
        dep_graph = self._build_dependency_graph(tasks)

        # Estimate resource requirements
        resource_estimates = await self._estimate_resources(tasks)

        # Optimize execution order
        optimized_schedule = self._optimize_schedule(
            tasks, dep_graph, resource_estimates
        )

        # Allocate resources
        resource_allocation = await self._allocate_resources(
            optimized_schedule, resource_estimates
        )

        # Create execution plan
        plan = ExecutionPlan(
            plan_id=f"plan_{datetime.utcnow().timestamp()}",
            tasks=tasks,
            schedule=optimized_schedule,
            resource_allocation=resource_allocation,
            estimated_total_time=self._calculate_total_time(optimized_schedule),
            optimization_score=self._calculate_optimization_score(optimized_schedule, tasks)
        )

        # Update statistics
        self.planning_stats['plans_created'] += 1
        self.planning_stats['tasks_scheduled'] += len(tasks)

        logger.info(f"Created execution plan {plan.plan_id} with {len(tasks)} tasks")

        return plan

    def _extract_tasks(self, workload: Dict[str, Any]) -> List[ExecutionTask]:
        """Extract execution tasks from workload specification."""
        tasks = []

        # Quantum circuit executions
        if 'quantum_circuits' in workload:
            for i, circuit in enumerate(workload['quantum_circuits']):
                task = ExecutionTask(
                    task_id=f"quantum_{i}",
                    operation='quantum_circuit',
                    resource_requirements={
                        'num_qubits': circuit.get('num_qubits', 16),
                        'circuit_depth': circuit.get('depth', 100),
                        'shots': circuit.get('shots', 1024),
                        'gpu_memory_mb': 2048
                    },
                    dependencies=circuit.get('dependencies', []),
                    priority=circuit.get('priority', 0.5),
                    estimated_runtime=self._estimate_quantum_runtime(circuit)
                )
                tasks.append(task)

        # Nash equilibrium computations
        if 'nash_computations' in workload:
            for i, nash in enumerate(workload['nash_computations']):
                task = ExecutionTask(
                    task_id=f"nash_{i}",
                    operation='nash_equilibrium',
                    resource_requirements={
                        'matrix_size': nash.get('matrix_size', 100),
                        'iterations': nash.get('iterations', 200),
                        'cpu_cores': 4,
                        'memory_mb': 1024
                    },
                    dependencies=nash.get('dependencies', []),
                    priority=nash.get('priority', 0.5),
                    estimated_runtime=self._estimate_nash_runtime(nash)
                )
                tasks.append(task)

        # Pattern matching tasks
        if 'pattern_matching' in workload:
            for i, pattern in enumerate(workload['pattern_matching']):
                task = ExecutionTask(
                    task_id=f"pattern_{i}",
                    operation='pattern_matching',
                    resource_requirements={
                        'num_patterns': pattern.get('num_patterns', 1000),
                        'pattern_size': pattern.get('pattern_size', 128),
                        'cpu_cores': 8,
                        'memory_mb': 512
                    },
                    dependencies=pattern.get('dependencies', []),
                    priority=pattern.get('priority', 0.3),
                    estimated_runtime=self._estimate_pattern_runtime(pattern)
                )
                tasks.append(task)

        return tasks

    def _build_dependency_graph(self, tasks: List[ExecutionTask]) -> nx.DiGraph:
        """Build directed acyclic graph of task dependencies."""
        graph = nx.DiGraph()

        # Add nodes
        for task in tasks:
            graph.add_node(task.task_id, task=task)

        # Add edges
        for task in tasks:
            for dep in task.dependencies:
                if dep in graph:
                    graph.add_edge(dep, task.task_id)

        # Verify DAG
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Task dependencies contain cycles")

        return graph

    async def _estimate_resources(self, tasks: List[ExecutionTask]) -> Dict[str, Dict[str, Any]]:
        """Estimate resource requirements for each task."""
        estimates = {}

        for task in tasks:
            if task.operation == 'quantum_circuit':
                # Use hardware optimizer for quantum resource estimation
                config = self.hw_optimizer.optimize_quantum_circuit_execution(
                    task.resource_requirements['num_qubits'],
                    task.resource_requirements['circuit_depth']
                )
                estimates[task.task_id] = {
                    'memory_mb': config['memory_required_mb'],
                    'gpu_required': 'gpu' in config['backend_config']['backend'],
                    'estimated_time': config['estimated_time_ms'] / 1000
                }

            else:
                # Use task requirements directly
                estimates[task.task_id] = {
                    'memory_mb': task.resource_requirements.get('memory_mb', 1024),
                    'cpu_cores': task.resource_requirements.get('cpu_cores', 1),
                    'gpu_required': task.resource_requirements.get('gpu_memory_mb', 0) > 0,
                    'estimated_time': task.estimated_runtime
                }

        return estimates

    def _optimize_schedule(self, tasks: List[ExecutionTask],
                         dep_graph: nx.DiGraph,
                         resource_estimates: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """
        Optimize task execution schedule.

        Uses critical path method with resource constraints.
        """
        # Topological sort for valid execution order
        topo_order = list(nx.topological_sort(dep_graph))

        # Initialize schedule
        schedule = {}
        resource_timeline = {
            'cpu': [],
            'gpu': [],
            'memory': []
        }

        # Schedule tasks in topological order
        for task_id in topo_order:
            task = next(t for t in tasks if t.task_id == task_id)
            estimates = resource_estimates[task_id]

            # Find earliest start time considering dependencies
            earliest_start = 0.0
            for dep in task.dependencies:
                if dep in schedule:
                    earliest_start = max(earliest_start, schedule[dep][1])

            # Find earliest time slot with available resources
            start_time = self._find_resource_slot(
                earliest_start,
                estimates['estimated_time'],
                estimates,
                resource_timeline
            )

            end_time = start_time + estimates['estimated_time']
            schedule[task_id] = (start_time, end_time)

            # Update resource timeline
            self._update_resource_timeline(
                resource_timeline,
                start_time,
                end_time,
                estimates
            )

        # Apply optimizations
        optimized_schedule = self._apply_schedule_optimizations(
            schedule, tasks, dep_graph, resource_estimates
        )

        return optimized_schedule

    def _find_resource_slot(self, earliest_start: float, duration: float,
                          requirements: Dict[str, Any],
                          timeline: Dict[str, List[Tuple[float, float]]]) -> float:
        """Find earliest available resource slot."""
        start_time = earliest_start

        while True:
            # Check if resources available at start_time
            if self._check_resource_availability(start_time, duration, requirements, timeline):
                return start_time

            # Find next potential start time
            next_times = []

            # Check CPU timeline
            if requirements.get('cpu_cores', 0) > 0:
                for slot_start, slot_end in timeline['cpu']:
                    if slot_end > start_time:
                        next_times.append(slot_end)

            # Check GPU timeline
            if requirements.get('gpu_required', False):
                for slot_start, slot_end in timeline['gpu']:
                    if slot_end > start_time:
                        next_times.append(slot_end)

            if next_times:
                start_time = min(next_times)
            else:
                break

        return start_time

    def _check_resource_availability(self, start_time: float, duration: float,
                                   requirements: Dict[str, Any],
                                   timeline: Dict[str, List[Tuple[float, float]]]) -> bool:
        """Check if resources are available for time slot."""
        end_time = start_time + duration

        # Check CPU conflicts
        if requirements.get('cpu_cores', 0) > 0:
            for slot_start, slot_end in timeline['cpu']:
                if not (end_time <= slot_start or start_time >= slot_end):
                    return False

        # Check GPU conflicts
        if requirements.get('gpu_required', False):
            for slot_start, slot_end in timeline['gpu']:
                if not (end_time <= slot_start or start_time >= slot_end):
                    return False

        return True

    def _update_resource_timeline(self, timeline: Dict[str, List[Tuple[float, float]]],
                                start_time: float, end_time: float,
                                requirements: Dict[str, Any]):
        """Update resource timeline with new allocation."""
        if requirements.get('cpu_cores', 0) > 0:
            timeline['cpu'].append((start_time, end_time))

        if requirements.get('gpu_required', False):
            timeline['gpu'].append((start_time, end_time))

        if requirements.get('memory_mb', 0) > 0:
            timeline['memory'].append((start_time, end_time))

    def _apply_schedule_optimizations(self, schedule: Dict[str, Tuple[float, float]],
                                    tasks: List[ExecutionTask],
                                    dep_graph: nx.DiGraph,
                                    resource_estimates: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """Apply additional schedule optimizations."""
        optimized = schedule.copy()

        # Optimization 1: Task reordering within constraints
        # Try to minimize total execution time by reordering independent tasks
        independent_groups = self._find_independent_task_groups(dep_graph)

        for group in independent_groups:
            if len(group) > 1:
                # Sort by priority and estimated runtime
                group_tasks = [t for t in tasks if t.task_id in group]
                group_tasks.sort(key=lambda t: (-t.priority, t.estimated_runtime))

                # Reschedule group
                group_start = min(optimized[tid][0] for tid in group)
                current_time = group_start

                for task in group_tasks:
                    duration = resource_estimates[task.task_id]['estimated_time']
                    optimized[task.task_id] = (current_time, current_time + duration)
                    current_time += duration

        # Optimization 2: Gap filling
        # Try to fill gaps in schedule with smaller tasks
        gaps = self._find_schedule_gaps(optimized)

        for gap_start, gap_end in gaps:
            gap_duration = gap_end - gap_start

            # Find tasks that could fit in gap
            for task in tasks:
                if task.task_id not in optimized:
                    continue

                task_duration = resource_estimates[task.task_id]['estimated_time']
                if task_duration <= gap_duration * 0.9:  # 90% to leave buffer
                    # Check if moving task doesn't violate dependencies
                    if self._can_move_task(task.task_id, gap_start, dep_graph, optimized):
                        optimized[task.task_id] = (gap_start, gap_start + task_duration)
                        break

        return optimized

    def _find_independent_task_groups(self, dep_graph: nx.DiGraph) -> List[List[str]]:
        """Find groups of tasks that can execute independently."""
        groups = []
        remaining = set(dep_graph.nodes())

        while remaining:
            # Find tasks with no dependencies in remaining set
            independent = []
            for task in remaining:
                predecessors = set(dep_graph.predecessors(task))
                if not predecessors.intersection(remaining):
                    independent.append(task)

            if independent:
                groups.append(independent)
                remaining -= set(independent)
            else:
                # Remaining tasks have circular dependencies (shouldn't happen with DAG)
                break

        return groups

    def _find_schedule_gaps(self, schedule: Dict[str, Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Find gaps in execution schedule."""
        if not schedule:
            return []

        # Sort tasks by start time
        sorted_tasks = sorted(schedule.items(), key=lambda x: x[1][0])

        gaps = []
        prev_end = 0.0

        for task_id, (start, end) in sorted_tasks:
            if start > prev_end + 0.1:  # Minimum gap of 0.1 seconds
                gaps.append((prev_end, start))
            prev_end = max(prev_end, end)

        return gaps

    def _can_move_task(self, task_id: str, new_start: float,
                      dep_graph: nx.DiGraph,
                      schedule: Dict[str, Tuple[float, float]]) -> bool:
        """Check if task can be moved to new start time."""
        # Check predecessors
        for pred in dep_graph.predecessors(task_id):
            if pred in schedule and schedule[pred][1] > new_start:
                return False

        # Check successors
        task_end = new_start + (schedule[task_id][1] - schedule[task_id][0])
        for succ in dep_graph.successors(task_id):
            if succ in schedule and schedule[succ][0] < task_end:
                return False

        return True

    async def _allocate_resources(self, schedule: Dict[str, Tuple[float, float]],
                                resource_estimates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Allocate resources for scheduled tasks."""
        allocations = {}

        for task_id, (start_time, end_time) in schedule.items():
            estimates = resource_estimates[task_id]

            # Request resources from resource manager
            resource_req = {
                'operation': task_id,
                'memory_mb': estimates.get('memory_mb', 1024),
                'cpu_cores': estimates.get('cpu_cores', 1),
                'gpu_memory_mb': estimates.get('memory_mb', 0) if estimates.get('gpu_required') else 0,
                'priority': 'normal',
                'duration_seconds': end_time - start_time
            }

            # Simulate resource allocation (in practice would use resource manager)
            allocations[task_id] = {
                'allocated_memory_mb': resource_req['memory_mb'],
                'allocated_cpu_cores': resource_req['cpu_cores'],
                'allocated_gpu': estimates.get('gpu_required', False),
                'allocation_time': datetime.utcnow().isoformat()
            }

        return allocations

    def _calculate_total_time(self, schedule: Dict[str, Tuple[float, float]]) -> float:
        """Calculate total execution time for schedule."""
        if not schedule:
            return 0.0

        start_times = [s[0] for s in schedule.values()]
        end_times = [s[1] for s in schedule.values()]

        return max(end_times) - min(start_times)

    def _calculate_optimization_score(self, schedule: Dict[str, Tuple[float, float]],
                                    tasks: List[ExecutionTask]) -> float:
        """Calculate optimization score for schedule."""
        if not schedule:
            return 0.0

        # Factors:
        # 1. Resource utilization
        total_time = self._calculate_total_time(schedule)
        task_time = sum(s[1] - s[0] for s in schedule.values())
        utilization = task_time / (total_time * 2) if total_time > 0 else 0  # Assume 2 resources

        # 2. Priority compliance
        priority_score = 0.0
        for task in tasks:
            if task.task_id in schedule:
                # Higher priority tasks should start earlier
                relative_start = schedule[task.task_id][0] / total_time if total_time > 0 else 0
                priority_score += task.priority * (1.0 - relative_start)
        priority_score /= len(tasks)

        # 3. Deadline compliance
        deadline_score = 1.0
        violations = 0
        for task in tasks:
            if task.deadline and task.task_id in schedule:
                end_time = schedule[task.task_id][1]
                if end_time > task.deadline.timestamp():
                    violations += 1

        if tasks:
            deadline_score = 1.0 - (violations / len(tasks))

        # Combined score
        optimization_score = (utilization * 0.4 +
                            priority_score * 0.3 +
                            deadline_score * 0.3)

        return optimization_score

    def _estimate_quantum_runtime(self, circuit: Dict[str, Any]) -> float:
        """Estimate runtime for quantum circuit."""
        num_qubits = circuit.get('num_qubits', 16)
        depth = circuit.get('depth', 100)
        shots = circuit.get('shots', 1024)

        # Simple estimation model
        base_time = depth * (2 ** min(num_qubits, 20)) / 1e6  # Gate operations
        shot_time = shots * 0.001  # 1ms per shot

        return base_time + shot_time

    def _estimate_nash_runtime(self, nash: Dict[str, Any]) -> float:
        """Estimate runtime for Nash equilibrium computation."""
        matrix_size = nash.get('matrix_size', 100)
        iterations = nash.get('iterations', 200)

        # O(n^2) per iteration
        return (matrix_size ** 2) * iterations / 1e6

    def _estimate_pattern_runtime(self, pattern: Dict[str, Any]) -> float:
        """Estimate runtime for pattern matching."""
        num_patterns = pattern.get('num_patterns', 1000)
        pattern_size = pattern.get('pattern_size', 128)

        # O(n * m) complexity
        return num_patterns * pattern_size / 1e6

    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Execute an execution plan.

        Args:
            plan: Execution plan to execute

        Returns:
            Execution results
        """
        self.active_plans[plan.plan_id] = plan
        results = {}

        try:
            # Execute tasks according to schedule
            for task in plan.tasks:
                start_time, end_time = plan.schedule[task.task_id]

                # Wait until start time
                current_time = asyncio.get_event_loop().time()
                if start_time > current_time:
                    await asyncio.sleep(start_time - current_time)

                # Execute task (simulated)
                result = await self._execute_task(task, plan.resource_allocation[task.task_id])
                results[task.task_id] = result

                # Update execution history
                self.execution_history.append({
                    'task_id': task.task_id,
                    'plan_id': plan.plan_id,
                    'start_time': start_time,
                    'end_time': asyncio.get_event_loop().time(),
                    'result': result['status']
                })

            return {
                'plan_id': plan.plan_id,
                'status': 'completed',
                'results': results,
                'actual_runtime': asyncio.get_event_loop().time()
            }

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            return {
                'plan_id': plan.plan_id,
                'status': 'failed',
                'error': str(e)
            }
        finally:
            del self.active_plans[plan.plan_id]

    async def _execute_task(self, task: ExecutionTask,
                          allocation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual task."""
        # Simulated execution
        await asyncio.sleep(task.estimated_runtime * 0.1)  # Simulate 10% of estimated time

        return {
            'status': 'completed',
            'runtime': task.estimated_runtime * 0.1,
            'resource_usage': allocation
        }

    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get execution planning statistics."""
        stats = self.planning_stats.copy()

        # Calculate average optimization improvement
        if stats['optimization_improvements']:
            stats['average_optimization'] = np.mean(stats['optimization_improvements'])
        else:
            stats['average_optimization'] = 0.0

        # Success rate
        completed = sum(1 for e in self.execution_history if e['result'] == 'completed')
        total = len(self.execution_history)
        stats['success_rate'] = completed / total if total > 0 else 0.0

        return stats
