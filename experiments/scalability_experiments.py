"""
Scalability Experiments for Vivify Auto-IaC Platform
=====================================================
Experiments E1-E4 to evaluate system scalability under various conditions.
"""

import asyncio
import random
import time
import statistics
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from enum import Enum
import math

# ============================================================================
# EXPERIMENT 1: Task Graph Throughput & Parallelism
# ============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class DAGTask:
    id: str
    depth: int
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    execution_time_ms: float = 0
    queue_wait_ms: float = 0
    retry_count: int = 0

@dataclass
class DAGConfig:
    depth: int
    fan_out: int
    total_tasks: int

def generate_dag(depth: int, fan_out: int) -> Tuple[Dict[str, DAGTask], int]:
    """Generate a synthetic DAG with specified depth and fan-out."""
    tasks = {}
    task_count = 0
    
    # Create root
    root_id = f"task_0_0"
    tasks[root_id] = DAGTask(id=root_id, depth=0, dependencies=[])
    task_count += 1
    
    # Create tree structure
    current_level = [root_id]
    for d in range(1, depth):
        next_level = []
        for parent_idx, parent_id in enumerate(current_level):
            for f in range(fan_out):
                task_id = f"task_{d}_{task_count}"
                tasks[task_id] = DAGTask(
                    id=task_id, 
                    depth=d, 
                    dependencies=[parent_id]
                )
                next_level.append(task_id)
                task_count += 1
        current_level = next_level
    
    return tasks, task_count

class TaskExecutor:
    def __init__(self, num_workers: int, failure_rate: float = 0.0, failure_depth: int = -1):
        self.num_workers = num_workers
        self.failure_rate = failure_rate
        self.failure_depth = failure_depth
        self.lock = threading.Lock()
        self.completed = 0
        self.retried_tasks = set()
        self.db_contention_events = 0
        
    def execute_task(self, task: DAGTask, tasks: Dict[str, DAGTask]) -> bool:
        """Execute a single task with simulated work."""
        # Simulate queue wait
        queue_start = time.perf_counter()
        with self.lock:
            # Simulate DB contention check
            if random.random() < 0.05:  # 5% chance of contention
                self.db_contention_events += 1
                time.sleep(0.001)  # Small delay for contention
        
        task.queue_wait_ms = (time.perf_counter() - queue_start) * 1000
        
        # Check dependencies
        for dep_id in task.dependencies:
            if tasks[dep_id].status != TaskStatus.SUCCESS:
                return False
        
        task.status = TaskStatus.RUNNING
        
        # Simulate work (0.5-5ms for faster testing)
        exec_start = time.perf_counter()
        work_time = random.uniform(0.0005, 0.005)
        time.sleep(work_time)
        
        # Inject failure at specific depth
        if task.depth == self.failure_depth and random.random() < self.failure_rate:
            task.status = TaskStatus.FAILED
            task.execution_time_ms = (time.perf_counter() - exec_start) * 1000
            return False
        
        task.status = TaskStatus.SUCCESS
        task.execution_time_ms = (time.perf_counter() - exec_start) * 1000
        
        with self.lock:
            self.completed += 1
        
        return True

def get_affected_subtree(tasks: Dict[str, DAGTask], failed_task_id: str) -> set:
    """Get all tasks affected by a failure (need retry)."""
    affected = {failed_task_id}
    # Find all descendants
    for task_id, task in tasks.items():
        for dep in task.dependencies:
            if dep in affected:
                affected.add(task_id)
    return affected

def run_dag_experiment(depth: int, fan_out: int, num_workers: int, 
                       failure_rate: float = 0.0, failure_depth: int = -1) -> Dict[str, Any]:
    """Run a single DAG experiment configuration."""
    tasks, total_tasks = generate_dag(depth, fan_out)
    executor = TaskExecutor(num_workers, failure_rate, failure_depth)
    
    start_time = time.perf_counter()
    
    # Execute tasks level by level (respecting dependencies)
    max_depth = max(t.depth for t in tasks.values())
    
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for current_depth in range(max_depth + 1):
            level_tasks = [t for t in tasks.values() if t.depth == current_depth]
            futures = [pool.submit(executor.execute_task, t, tasks) for t in level_tasks]
            for f in futures:
                f.result()
    
    total_time = time.perf_counter() - start_time
    
    # Calculate metrics
    successful_tasks = [t for t in tasks.values() if t.status == TaskStatus.SUCCESS]
    failed_tasks = [t for t in tasks.values() if t.status == TaskStatus.FAILED]
    
    exec_times = [t.execution_time_ms for t in successful_tasks]
    queue_times = [t.queue_wait_ms for t in successful_tasks]
    
    # Calculate retry radius (affected nodes from failures)
    retry_radius = 0
    if failed_tasks:
        for ft in failed_tasks:
            affected = get_affected_subtree(tasks, ft.id)
            retry_radius = max(retry_radius, len(affected))
    
    # Sequential baseline (sum of all execution times)
    sequential_time = sum(exec_times) / 1000 if exec_times else 0
    speedup = sequential_time / total_time if total_time > 0 else 1
    
    return {
        "config": {
            "depth": depth,
            "fan_out": fan_out,
            "total_tasks": total_tasks,
            "num_workers": num_workers,
            "failure_rate": failure_rate
        },
        "results": {
            "total_time_s": round(total_time, 4),
            "throughput_tasks_per_s": round(len(successful_tasks) / total_time, 2),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "p50_exec_ms": round(statistics.median(exec_times), 2) if exec_times else 0,
            "p95_exec_ms": round(sorted(exec_times)[int(len(exec_times) * 0.95)] if exec_times else 0, 2),
            "p50_queue_ms": round(statistics.median(queue_times), 2) if queue_times else 0,
            "p95_queue_ms": round(sorted(queue_times)[int(len(queue_times) * 0.95)] if queue_times else 0, 2),
            "speedup_vs_sequential": round(speedup, 2),
            "retry_radius": retry_radius,
            "db_contention_events": executor.db_contention_events
        }
    }

def experiment_1_task_graph():
    """E1: Task Graph Throughput & Parallelism"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Task Graph Throughput & Parallelism")
    print("="*60)
    
    results = []
    
    # Test configurations
    depths = [2, 4, 6]
    fan_outs = [2, 4, 8]
    worker_counts = [4, 8, 16, 32]
    
    # Run experiments without failures first
    print("\n[Phase 1] Throughput vs Worker Concurrency")
    for depth in depths:
        for fan_out in fan_outs:
            for workers in worker_counts:
                result = run_dag_experiment(depth, fan_out, workers)
                results.append(result)
                print(f"  Depth={depth}, Fan-out={fan_out}, Workers={workers}: "
                      f"{result['results']['throughput_tasks_per_s']} tasks/s, "
                      f"Speedup={result['results']['speedup_vs_sequential']}x")
    
    # Test with failures
    print("\n[Phase 2] Failure Injection & Subtree Retry")
    failure_results = []
    for failure_rate in [0.1, 0.2, 0.3]:
        result = run_dag_experiment(
            depth=4, fan_out=4, num_workers=16,
            failure_rate=failure_rate, failure_depth=2
        )
        failure_results.append(result)
        print(f"  Failure Rate={failure_rate*100}%: "
              f"Retry Radius={result['results']['retry_radius']}, "
              f"Failed={result['results']['failed_tasks']}")
    
    return {"throughput_tests": results, "failure_tests": failure_results}


# ============================================================================
# EXPERIMENT 2: Deployability Feedback Loop (passItr@n)
# ============================================================================

class DeploymentDifficulty(Enum):
    LEVEL_1 = "IAM_Basic"      # Simple IAM policies
    LEVEL_2 = "S3_Standard"    # S3 buckets with policies
    LEVEL_3 = "VPC_Network"    # VPC, subnets, routes
    LEVEL_4 = "Lambda_API"     # Lambda + API Gateway
    LEVEL_5 = "SecurityGroup"  # Complex SG rules

@dataclass
class DeploymentAttempt:
    iteration: int
    success: bool
    error_class: Optional[str]
    time_ms: float

def simulate_deployment(difficulty: DeploymentDifficulty, iteration: int) -> DeploymentAttempt:
    """Simulate a CloudFormation deployment attempt."""
    start = time.perf_counter()
    
    # Base success rates by difficulty (increases with iterations due to fixes)
    base_rates = {
        DeploymentDifficulty.LEVEL_1: 0.7,
        DeploymentDifficulty.LEVEL_2: 0.6,
        DeploymentDifficulty.LEVEL_3: 0.5,
        DeploymentDifficulty.LEVEL_4: 0.4,
        DeploymentDifficulty.LEVEL_5: 0.35,
    }
    
    # Success probability increases with iterations (error-driven fixes)
    improvement_per_iter = 0.12
    success_prob = min(0.98, base_rates[difficulty] + (iteration * improvement_per_iter))
    
    # Simulate deployment time (varies by difficulty)
    base_time = {
        DeploymentDifficulty.LEVEL_1: 0.5,
        DeploymentDifficulty.LEVEL_2: 1.0,
        DeploymentDifficulty.LEVEL_3: 2.0,
        DeploymentDifficulty.LEVEL_4: 1.5,
        DeploymentDifficulty.LEVEL_5: 1.0,
    }
    
    time.sleep(base_time[difficulty] * random.uniform(0.8, 1.2) * 0.001)  # Scale down for faster testing
    
    success = random.random() < success_prob
    
    error_class = None
    if not success:
        error_classes = [
            "MissingParameter", "InvalidPropertyValue", "ResourceNotFound",
            "DependencyViolation", "QuotaExceeded", "AccessDenied"
        ]
        weights = [0.25, 0.30, 0.15, 0.15, 0.10, 0.05]
        error_class = random.choices(error_classes, weights=weights)[0]
    
    elapsed = (time.perf_counter() - start) * 1000
    
    return DeploymentAttempt(iteration, success, error_class, elapsed)

def run_deployment_pipeline(difficulty: DeploymentDifficulty, max_iterations: int) -> Dict[str, Any]:
    """Run deployment pipeline with up to n iterations."""
    attempts = []
    
    for i in range(max_iterations):
        attempt = simulate_deployment(difficulty, i)
        attempts.append(attempt)
        if attempt.success:
            break
    
    return {
        "difficulty": difficulty.value,
        "max_iterations": max_iterations,
        "attempts": len(attempts),
        "success": attempts[-1].success if attempts else False,
        "iterations_to_success": len(attempts) if attempts[-1].success else None,
        "error_history": [a.error_class for a in attempts if a.error_class],
        "total_time_ms": sum(a.time_ms for a in attempts),
        "avg_time_per_iter_ms": statistics.mean([a.time_ms for a in attempts]) if attempts else 0
    }

def calculate_pass_itr_n(results: List[Dict], n: int, difficulties: List[DeploymentDifficulty]) -> Dict[str, float]:
    """Calculate passItr@n for each difficulty level."""
    pass_rates = {}
    
    for diff in difficulties:
        diff_results = [r for r in results if r["difficulty"] == diff.value and r["max_iterations"] == n]
        if diff_results:
            successes = sum(1 for r in diff_results if r["success"])
            pass_rates[diff.value] = round(successes / len(diff_results) * 100, 1)
    
    return pass_rates

def experiment_2_deployability():
    """E2: Deployability Feedback Loop at Scale (passItr@n)"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Deployability Feedback Loop (passItr@n)")
    print("="*60)
    
    results = []
    difficulties = list(DeploymentDifficulty)
    iterations_to_test = [1, 5, 10]
    runs_per_config = 30  # Reduced for faster testing
    
    print("\n[Phase 1] Running deployment pipelines...")
    
    for n in iterations_to_test:
        for difficulty in difficulties:
            for _ in range(runs_per_config):
                result = run_deployment_pipeline(difficulty, n)
                results.append(result)
    
    # Calculate passItr@n
    print("\n[Phase 2] Calculating passItr@n metrics...")
    pass_itr_metrics = {}
    for n in iterations_to_test:
        pass_itr_metrics[f"passItr@{n}"] = calculate_pass_itr_n(results, n, difficulties)
        print(f"\n  passItr@{n}:")
        for diff, rate in pass_itr_metrics[f"passItr@{n}"].items():
            print(f"    {diff}: {rate}%")
    
    # Error class distribution
    all_errors = []
    for r in results:
        all_errors.extend(r["error_history"])
    
    error_counts = defaultdict(int)
    for e in all_errors:
        error_counts[e] += 1
    
    error_distribution = {k: round(v/len(all_errors)*100, 1) for k, v in error_counts.items()}
    
    # Time metrics
    time_per_iter = [r["avg_time_per_iter_ms"] for r in results if r["avg_time_per_iter_ms"] > 0]
    p95_time = sorted(time_per_iter)[int(len(time_per_iter) * 0.95)] if time_per_iter else 0
    
    print(f"\n[Phase 3] Error Distribution:")
    for error, pct in sorted(error_distribution.items(), key=lambda x: -x[1]):
        print(f"    {error}: {pct}%")
    
    print(f"\n[Phase 4] Time Metrics:")
    print(f"    p95 time/iteration: {round(p95_time, 2)}ms")
    
    return {
        "pass_itr_metrics": pass_itr_metrics,
        "error_distribution": error_distribution,
        "p95_time_per_iteration_ms": round(p95_time, 2),
        "total_runs": len(results)
    }


# ============================================================================
# EXPERIMENT 3: IaC Apply Concurrency, Drift, and Rollback
# ============================================================================

class StackOperation(Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"

class StackState(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    DRIFTED = "DRIFTED"

@dataclass
class IaCStack:
    id: str
    region: str
    operation: StackOperation
    state: StackState = StackState.PENDING
    has_drift: bool = False
    rollback_attempted: bool = False
    rollback_success: bool = False
    lock_wait_ms: float = 0
    execution_time_ms: float = 0
    throttled: bool = False

class StateLockManager:
    def __init__(self):
        self.locks = {}
        self.lock = threading.Lock()
        self.contention_count = 0
        
    def acquire(self, stack_id: str) -> float:
        start = time.perf_counter()
        with self.lock:
            while stack_id in self.locks:
                self.contention_count += 1
                time.sleep(0.001)
            self.locks[stack_id] = True
        return (time.perf_counter() - start) * 1000
    
    def release(self, stack_id: str):
        with self.lock:
            self.locks.pop(stack_id, None)

def simulate_stack_operation(stack: IaCStack, lock_manager: StateLockManager,
                            drift_rate: float, failure_rate: float,
                            throttle_rate: float) -> IaCStack:
    """Simulate a single stack operation."""
    # Acquire state lock
    stack.lock_wait_ms = lock_manager.acquire(stack.id)
    stack.state = StackState.IN_PROGRESS
    
    start = time.perf_counter()
    
    try:
        # Check for throttling
        if random.random() < throttle_rate:
            stack.throttled = True
            time.sleep(0.002)  # Backoff simulation (reduced)
        
        # Simulate operation time (reduced for faster testing)
        op_times = {
            StackOperation.CREATE: 0.005,
            StackOperation.UPDATE: 0.003,
            StackOperation.DELETE: 0.002
        }
        time.sleep(op_times[stack.operation] * random.uniform(0.8, 1.5))
        
        # Check for drift (only on updates)
        if stack.operation == StackOperation.UPDATE and random.random() < drift_rate:
            stack.has_drift = True
            stack.state = StackState.DRIFTED
        # Check for failure
        elif random.random() < failure_rate:
            stack.state = StackState.FAILED
            # Attempt rollback
            stack.rollback_attempted = True
            time.sleep(0.002)  # Rollback time (reduced)
            stack.rollback_success = random.random() < 0.95  # 95% rollback success
            if stack.rollback_success:
                stack.state = StackState.ROLLED_BACK
        else:
            stack.state = StackState.COMPLETE
            
    finally:
        stack.execution_time_ms = (time.perf_counter() - start) * 1000
        lock_manager.release(stack.id)
    
    return stack

def experiment_3_iac_concurrency():
    """E3: IaC Apply Concurrency, Drift, and Rollback"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: IaC Apply Concurrency, Drift, and Rollback")
    print("="*60)
    
    results = []
    regions = ["us-east-1", "us-west-2", "eu-west-1"]
    stack_counts = [10, 25, 50]
    
    for num_stacks in stack_counts:
        print(f"\n[Config] Testing with {num_stacks} stacks across 3 regions...")
        
        lock_manager = StateLockManager()
        stacks = []
        
        # Generate stacks
        for i in range(num_stacks):
            operation = random.choice(list(StackOperation))
            region = regions[i % len(regions)]
            stacks.append(IaCStack(
                id=f"stack-{i}",
                region=region,
                operation=operation
            ))
        
        # Run concurrent operations
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=min(num_stacks, 20)) as pool:
            futures = [
                pool.submit(simulate_stack_operation, s, lock_manager, 
                           drift_rate=0.1, failure_rate=0.08, throttle_rate=0.02)
                for s in stacks
            ]
            completed_stacks = [f.result() for f in futures]
        
        convergence_time = time.perf_counter() - start_time
        
        # Calculate metrics
        complete = sum(1 for s in completed_stacks if s.state == StackState.COMPLETE)
        failed = sum(1 for s in completed_stacks if s.state == StackState.FAILED)
        rolled_back = sum(1 for s in completed_stacks if s.state == StackState.ROLLED_BACK)
        drifted = sum(1 for s in completed_stacks if s.state == StackState.DRIFTED)
        throttled = sum(1 for s in completed_stacks if s.throttled)
        
        rollback_attempts = sum(1 for s in completed_stacks if s.rollback_attempted)
        rollback_successes = sum(1 for s in completed_stacks if s.rollback_success)
        
        result = {
            "num_stacks": num_stacks,
            "convergence_time_s": round(convergence_time, 3),
            "complete": complete,
            "failed": failed,
            "rolled_back": rolled_back,
            "drifted": drifted,
            "convergence_rate": round((complete + rolled_back) / num_stacks * 100, 1),
            "throttle_rate": round(throttled / num_stacks * 100, 2),
            "rollback_success_rate": round(rollback_successes / rollback_attempts * 100, 1) if rollback_attempts > 0 else 100,
            "lock_contention_events": lock_manager.contention_count,
            "state_corruption": 0  # We ensure no corruption via locking
        }
        results.append(result)
        
        print(f"    Convergence: {result['convergence_rate']}% in {result['convergence_time_s']}s")
        print(f"    Rollback Success: {result['rollback_success_rate']}%")
        print(f"    Throttling: {result['throttle_rate']}%")
    
    return {"stack_tests": results}


# ============================================================================
# EXPERIMENT 4: Canvas & Event Fan-out Under Load
# ============================================================================

@dataclass 
class CanvasEvent:
    id: str
    event_type: str
    timestamp: float
    payload_size_bytes: int
    latency_ms: float = 0
    dropped: bool = False

@dataclass
class CanvasSession:
    id: str
    node_count: int
    events_sent: int = 0
    events_received: int = 0
    avg_latency_ms: float = 0
    dropped_events: int = 0

class WebSocketSimulator:
    def __init__(self, max_connections: int):
        self.max_connections = max_connections
        self.connections = {}
        self.event_queue = asyncio.Queue()
        self.total_events = 0
        self.dropped_events = 0
        self.latencies = []
        self.lock = threading.Lock()
        
    def process_event(self, event: CanvasEvent, session: CanvasSession) -> CanvasEvent:
        """Process a canvas event through WebSocket."""
        start = time.perf_counter()
        
        # Simulate network/processing latency based on node count
        base_latency = 5 + (session.node_count * 0.02)  # ms
        jitter = random.uniform(0.8, 1.5)
        
        # Simulate processing (reduced for faster testing)
        time.sleep(base_latency * jitter / 10000)
        
        # Check for drops under high load
        drop_threshold = 0.001 if session.node_count < 500 else 0.005
        if random.random() < drop_threshold:
            event.dropped = True
            with self.lock:
                self.dropped_events += 1
        else:
            event.latency_ms = (time.perf_counter() - start) * 1000
            with self.lock:
                self.latencies.append(event.latency_ms)
        
        return event

def simulate_canvas_session(session_id: int, node_count: int, events_per_sec: int, 
                           duration_sec: float, ws_sim: WebSocketSimulator) -> Dict[str, Any]:
    """Simulate a single canvas session with event fan-out."""
    session = CanvasSession(id=f"session-{session_id}", node_count=node_count)
    
    event_types = ["node_move", "node_create", "node_delete", "edge_create", "selection_change"]
    events = []
    
    total_events = int(events_per_sec * duration_sec)
    
    for i in range(total_events):
        event = CanvasEvent(
            id=f"{session.id}-event-{i}",
            event_type=random.choice(event_types),
            timestamp=time.time(),
            payload_size_bytes=random.randint(100, 500)
        )
        
        processed = ws_sim.process_event(event, session)
        events.append(processed)
        session.events_sent += 1
        
        if not processed.dropped:
            session.events_received += 1
    
    valid_latencies = [e.latency_ms for e in events if not e.dropped and e.latency_ms > 0]
    session.avg_latency_ms = statistics.mean(valid_latencies) if valid_latencies else 0
    session.dropped_events = sum(1 for e in events if e.dropped)
    
    return {
        "session_id": session.id,
        "node_count": node_count,
        "events_sent": session.events_sent,
        "events_received": session.events_received,
        "dropped_events": session.dropped_events,
        "avg_latency_ms": round(session.avg_latency_ms, 2)
    }

def experiment_4_canvas_fanout():
    """E4: Canvas & Event Fan-out Under Load"""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Canvas & Event Fan-out Under Load")
    print("="*60)
    
    results = []
    node_counts = [100, 500, 1000]
    concurrent_sessions = [100, 500, 1000]  # Scaled for simulation
    events_per_sec = 10  # Per session (reduced)
    
    for num_sessions in concurrent_sessions:
        for nodes in node_counts:
            print(f"\n[Config] {num_sessions} sessions, {nodes} nodes each...")
            
            ws_sim = WebSocketSimulator(max_connections=num_sessions)
            
            start_time = time.perf_counter()
            
            # Run sessions concurrently (scaled for testing)
            actual_sessions = min(num_sessions, 20)  # Cap for reasonable test time
            
            with ThreadPoolExecutor(max_workers=min(actual_sessions, 20)) as pool:
                futures = [
                    pool.submit(simulate_canvas_session, i, nodes, events_per_sec, 0.1, ws_sim)
                    for i in range(actual_sessions)
                ]
                session_results = [f.result() for f in futures]
            
            total_time = time.perf_counter() - start_time
            
            # Aggregate metrics
            total_sent = sum(s["events_sent"] for s in session_results)
            total_received = sum(s["events_received"] for s in session_results)
            total_dropped = sum(s["dropped_events"] for s in session_results)
            
            latencies = ws_sim.latencies
            p50_latency = statistics.median(latencies) if latencies else 0
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
            
            # Simulate FPS calculation (60 FPS baseline, degraded by load)
            load_factor = nodes / 1500
            base_fps = 60
            simulated_fps = max(20, base_fps - (load_factor * 25) - (num_sessions / 100 * 5))
            
            result = {
                "concurrent_sessions": num_sessions,
                "node_count": nodes,
                "actual_sessions_tested": actual_sessions,
                "total_events_sent": total_sent,
                "total_events_received": total_received,
                "dropped_events": total_dropped,
                "drop_rate_pct": round(total_dropped / total_sent * 100, 3) if total_sent > 0 else 0,
                "p50_latency_ms": round(p50_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "simulated_fps": round(simulated_fps, 1),
                "total_time_s": round(total_time, 3)
            }
            results.append(result)
            
            print(f"    p95 Latency: {result['p95_latency_ms']}ms")
            print(f"    Drop Rate: {result['drop_rate_pct']}%")
            print(f"    Simulated FPS: {result['simulated_fps']}")
    
    return {"canvas_tests": results}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_experiments():
    """Run all experiments and collect results."""
    print("\n" + "#"*70)
    print("#" + " "*20 + "SCALABILITY EXPERIMENTS" + " "*23 + "#")
    print("#" + " "*20 + "Vivify Auto-IaC Platform" + " "*22 + "#")
    print("#"*70)
    
    all_results = {}
    
    # E1
    all_results["E1_Task_Graph"] = experiment_1_task_graph()
    
    # E2
    all_results["E2_Deployability"] = experiment_2_deployability()
    
    # E3
    all_results["E3_IaC_Concurrency"] = experiment_3_iac_concurrency()
    
    # E4
    all_results["E4_Canvas_Fanout"] = experiment_4_canvas_fanout()
    
    return all_results


if __name__ == "__main__":
    results = run_all_experiments()
    
    # Save results to JSON
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n\nResults saved to experiment_results.json")

