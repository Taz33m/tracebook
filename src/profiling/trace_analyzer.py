"""
Advanced trace analyzer for high-resolution performance profiling.

Provides magic-trace-like analysis capabilities with detailed timing,
call stack reconstruction, and performance bottleneck identification.
"""

import time
import json
import threading
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
from pathlib import Path


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


@dataclass
class TraceEvent:
    """Individual trace event with nanosecond precision."""
    timestamp_ns: int
    event_type: str  # 'enter', 'exit', 'sample'
    function_name: str
    thread_id: int
    process_id: int
    call_depth: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        # Convert numpy int64 to regular Python int for JSON serialization
        self.timestamp_ns = int(self.timestamp_ns)
        self.thread_id = int(self.thread_id)
        self.process_id = int(self.process_id)
        self.call_depth = int(self.call_depth)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionCall:
    """Complete function call with entry/exit timing."""
    function_name: str
    start_time_ns: int
    end_time_ns: int
    duration_ns: int
    call_depth: int
    thread_id: int
    children: List['FunctionCall'] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        # Convert numpy int64 to regular Python int for JSON serialization
        self.start_time_ns = int(self.start_time_ns)
        self.end_time_ns = int(self.end_time_ns)
        self.duration_ns = int(self.duration_ns)
        self.call_depth = int(self.call_depth)
        self.thread_id = int(self.thread_id)
    
    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000
    
    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1_000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'function_name': self.function_name,
            'start_time_ns': self.start_time_ns,
            'end_time_ns': self.end_time_ns,
            'duration_ns': self.duration_ns,
            'duration_ms': self.duration_ms,
            'duration_us': self.duration_us,
            'call_depth': self.call_depth,
            'thread_id': self.thread_id,
            'children': [child.to_dict() for child in self.children],
            'metadata': self.metadata or {}
        }


class HighResolutionTracer:
    """
    High-resolution function tracer with nanosecond precision.
    
    Provides magic-trace-like functionality for detailed performance analysis.
    """
    
    def __init__(self, buffer_size: int = 1000000):
        self.buffer_size = buffer_size
        self.events = deque(maxlen=buffer_size)
        self.call_stack = defaultdict(list)  # Per-thread call stacks
        self.active_calls = defaultdict(dict)  # Per-thread active calls
        self.completed_calls = []
        self.is_tracing = False
        self.start_time_ns = None
        self.lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'total_events': 0,
            'function_calls': 0,
            'max_call_depth': 0,
            'trace_overhead_ns': 0,
        }
    
    def start_tracing(self):
        """Start high-resolution tracing."""
        with self.lock:
            if self.is_tracing:
                return False
            
            self.is_tracing = True
            self.start_time_ns = time.time_ns()
            self.events.clear()
            self.call_stack.clear()
            self.active_calls.clear()
            self.completed_calls.clear()
            
            # Reset metrics
            self.metrics = {
                'total_events': 0,
                'function_calls': 0,
                'max_call_depth': 0,
                'trace_overhead_ns': 0,
            }
            
            return True
    
    def stop_tracing(self) -> Dict[str, Any]:
        """Stop tracing and return analysis results."""
        with self.lock:
            if not self.is_tracing:
                return {}
            
            self.is_tracing = False
            end_time_ns = time.time_ns()
            
            # Finalize any remaining active calls
            self._finalize_active_calls(end_time_ns)
            
            # Analyze collected data
            analysis = self._analyze_trace_data()
            analysis['trace_duration_ns'] = end_time_ns - self.start_time_ns
            analysis['trace_duration_ms'] = analysis['trace_duration_ns'] / 1_000_000
            
            return analysis
    
    def trace_function_enter(self, function_name: str, metadata: Dict[str, Any] = None):
        """Record function entry."""
        if not self.is_tracing:
            return
        
        trace_start = time.time_ns()
        
        thread_id = threading.get_ident()
        timestamp_ns = time.time_ns()
        call_depth = len(self.call_stack[thread_id])
        
        # Create trace event
        event = TraceEvent(
            timestamp_ns=timestamp_ns,
            event_type='enter',
            function_name=function_name,
            thread_id=thread_id,
            process_id=0,  # Single process for now
            call_depth=call_depth,
            metadata=metadata
        )
        
        with self.lock:
            self.events.append(event)
            self.call_stack[thread_id].append(function_name)
            self.active_calls[thread_id][function_name] = {
                'start_time_ns': timestamp_ns,
                'call_depth': call_depth,
                'metadata': metadata
            }
            
            # Update metrics
            self.metrics['total_events'] += 1
            self.metrics['function_calls'] += 1
            self.metrics['max_call_depth'] = max(
                self.metrics['max_call_depth'], 
                call_depth + 1
            )
            
            # Track tracing overhead
            trace_end = time.time_ns()
            self.metrics['trace_overhead_ns'] += trace_end - trace_start
    
    def trace_function_exit(self, function_name: str, metadata: Dict[str, Any] = None):
        """Record function exit."""
        if not self.is_tracing:
            return
        
        trace_start = time.time_ns()
        
        thread_id = threading.get_ident()
        timestamp_ns = time.time_ns()
        
        with self.lock:
            # Find matching function entry
            if (thread_id in self.active_calls and 
                function_name in self.active_calls[thread_id]):
                
                call_info = self.active_calls[thread_id][function_name]
                start_time_ns = call_info['start_time_ns']
                call_depth = call_info['call_depth']
                entry_metadata = call_info.get('metadata', {})
                
                # Merge metadata
                combined_metadata = {**(entry_metadata or {}), **(metadata or {})}
                
                # Create completed function call
                function_call = FunctionCall(
                    function_name=function_name,
                    start_time_ns=start_time_ns,
                    end_time_ns=timestamp_ns,
                    duration_ns=timestamp_ns - start_time_ns,
                    call_depth=call_depth,
                    thread_id=thread_id,
                    metadata=combined_metadata
                )
                
                self.completed_calls.append(function_call)
                
                # Clean up active call
                del self.active_calls[thread_id][function_name]
                
                # Remove from call stack
                if self.call_stack[thread_id] and self.call_stack[thread_id][-1] == function_name:
                    self.call_stack[thread_id].pop()
            
            # Create exit event
            event = TraceEvent(
                timestamp_ns=timestamp_ns,
                event_type='exit',
                function_name=function_name,
                thread_id=thread_id,
                process_id=0,
                call_depth=len(self.call_stack[thread_id]),
                metadata=metadata
            )
            
            self.events.append(event)
            self.metrics['total_events'] += 1
            
            # Track tracing overhead
            trace_end = time.time_ns()
            self.metrics['trace_overhead_ns'] += trace_end - trace_start
    
    def _finalize_active_calls(self, end_time_ns: int):
        """Finalize any remaining active calls."""
        for thread_id, active_calls in self.active_calls.items():
            for function_name, call_info in active_calls.items():
                function_call = FunctionCall(
                    function_name=function_name,
                    start_time_ns=call_info['start_time_ns'],
                    end_time_ns=end_time_ns,
                    duration_ns=end_time_ns - call_info['start_time_ns'],
                    call_depth=call_info['call_depth'],
                    thread_id=thread_id,
                    metadata=call_info.get('metadata')
                )
                self.completed_calls.append(function_call)
    
    def _analyze_trace_data(self) -> Dict[str, Any]:
        """Analyze collected trace data."""
        if not self.completed_calls:
            return {'error': 'No completed function calls to analyze'}
        
        # Function-level statistics
        function_stats = defaultdict(list)
        for call in self.completed_calls:
            function_stats[call.function_name].append(call.duration_ns)
        
        # Calculate statistics for each function
        function_analysis = {}
        for func_name, durations in function_stats.items():
            durations_array = np.array(durations)
            function_analysis[func_name] = {
                'call_count': len(durations),
                'total_time_ns': int(np.sum(durations_array)),
                'total_time_ms': float(np.sum(durations_array)) / 1_000_000,
                'mean_duration_ns': float(np.mean(durations_array)),
                'mean_duration_ms': float(np.mean(durations_array)) / 1_000_000,
                'median_duration_ns': float(np.median(durations_array)),
                'std_duration_ns': float(np.std(durations_array)),
                'min_duration_ns': int(np.min(durations_array)),
                'max_duration_ns': int(np.max(durations_array)),
                'p95_duration_ns': float(np.percentile(durations_array, 95)),
                'p99_duration_ns': float(np.percentile(durations_array, 99)),
            }
        
        # Overall statistics
        all_durations = []
        for durations in function_stats.values():
            all_durations.extend(durations)
        
        total_traced_time_ns = sum(all_durations)
        
        # Call depth analysis
        call_depths = [call.call_depth for call in self.completed_calls]
        
        # Thread analysis
        thread_stats = defaultdict(int)
        for call in self.completed_calls:
            thread_stats[call.thread_id] += 1
        
        return {
            'summary': {
                'total_function_calls': len(self.completed_calls),
                'unique_functions': len(function_stats),
                'total_traced_time_ns': total_traced_time_ns,
                'total_traced_time_ms': total_traced_time_ns / 1_000_000,
                'trace_overhead_ns': self.metrics['trace_overhead_ns'],
                'trace_overhead_percentage': (
                    self.metrics['trace_overhead_ns'] / max(1, total_traced_time_ns) * 100
                ),
                'max_call_depth': max(call_depths) if call_depths else 0,
                'active_threads': len(thread_stats),
            },
            'function_analysis': function_analysis,
            'call_depth_distribution': {
                'mean': float(np.mean(call_depths)) if call_depths else 0,
                'max': max(call_depths) if call_depths else 0,
                'distribution': {int(k): int(v) for k, v in zip(*np.unique(call_depths, return_counts=True))} if call_depths else {}
            },
            'thread_distribution': dict(thread_stats),
            'performance_insights': self._generate_performance_insights(function_analysis),
            'raw_metrics': self.metrics
        }
    
    def _generate_performance_insights(self, function_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance insights from analysis."""
        insights = []
        
        # Find slowest functions
        sorted_functions = sorted(
            function_analysis.items(),
            key=lambda x: x[1]['mean_duration_ns'],
            reverse=True
        )
        
        if sorted_functions:
            slowest_func, slowest_stats = sorted_functions[0]
            insights.append(
                f"Slowest function: {slowest_func} "
                f"(avg: {slowest_stats['mean_duration_ms']:.3f}ms)"
            )
        
        # Find functions with high variance
        high_variance_funcs = [
            (name, stats) for name, stats in function_analysis.items()
            if stats['std_duration_ns'] > stats['mean_duration_ns']
        ]
        
        if high_variance_funcs:
            insights.append(
                f"Functions with high latency variance: "
                f"{', '.join([name for name, _ in high_variance_funcs[:3]])}"
            )
        
        # Find functions consuming most total time
        sorted_by_total = sorted(
            function_analysis.items(),
            key=lambda x: x[1]['total_time_ns'],
            reverse=True
        )
        
        if sorted_by_total:
            hottest_func, hottest_stats = sorted_by_total[0]
            insights.append(
                f"Most time-consuming: {hottest_func} "
                f"(total: {hottest_stats['total_time_ms']:.1f}ms, "
                f"calls: {hottest_stats['call_count']})"
            )
        
        return insights
    
    def export_trace_data(self, output_file: str) -> bool:
        """Export trace data to JSON file."""
        try:
            analysis = self._analyze_trace_data()
            
            # Add raw events for detailed analysis
            analysis['raw_events'] = [event.to_dict() for event in self.events]
            analysis['completed_calls'] = [call.to_dict() for call in self.completed_calls]
            
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, cls=NumpyJSONEncoder)
            
            return True
        except Exception as e:
            print(f"Failed to export trace data: {e}")
            return False


class TraceProfiler:
    """
    High-level profiler interface with automatic trace management.
    """
    
    def __init__(self):
        self.tracer = HighResolutionTracer()
        self.active_traces = {}
        self.lock = threading.Lock()
    
    def profile_function(self, func_name: str = None):
        """Decorator for automatic function profiling."""
        def decorator(func):
            nonlocal func_name
            if func_name is None:
                func_name = f"{func.__module__}.{func.__name__}"
            
            def wrapper(*args, **kwargs):
                self.tracer.trace_function_enter(func_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.tracer.trace_function_exit(func_name)
            
            return wrapper
        return decorator
    
    def start_session(self, session_name: str = "default") -> bool:
        """Start a profiling session."""
        with self.lock:
            if session_name in self.active_traces:
                return False
            
            success = self.tracer.start_tracing()
            if success:
                self.active_traces[session_name] = {
                    'start_time': time.time_ns(),
                    'tracer': self.tracer
                }
            
            return success
    
    def stop_session(self, session_name: str = "default") -> Optional[Dict[str, Any]]:
        """Stop a profiling session and return analysis."""
        with self.lock:
            if session_name not in self.active_traces:
                return None
            
            analysis = self.tracer.stop_tracing()
            del self.active_traces[session_name]
            
            return analysis
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session names."""
        with self.lock:
            return list(self.active_traces.keys())


# Global tracer instance
_global_tracer = None

def get_tracer() -> TraceProfiler:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = TraceProfiler()
    return _global_tracer

def profile_function(func_name: str = None):
    """Convenience decorator for function profiling."""
    return get_tracer().profile_function(func_name)
