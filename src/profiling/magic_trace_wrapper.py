"""
Magic-trace integration wrapper for Jane Street's profiling tool.

This module provides a high-level interface for integrating magic-trace
profiling into the order book simulator with minimal performance overhead.
"""

import os
import subprocess
import time
import json
import tempfile
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
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
        return super().default(obj)
import threading
import signal


class MagicTraceConfig:
    """Configuration for magic-trace profiling."""
    
    def __init__(self):
        # Default configuration
        self.output_dir = "traces"
        self.trace_format = "fxt"  # Jane Street's trace format
        self.buffer_size_mb = 64
        self.sample_rate_hz = 1000
        self.enable_kernel_events = False
        self.enable_user_events = True
        self.max_trace_duration_seconds = 60
        self.auto_analyze = True
        
        # Function-level profiling
        self.profile_functions = [
            "add_order",
            "match_orders_fifo", 
            "match_orders_pro_rata",
            "execute_matches_at_level",
            "_match_buy_order",
            "_match_sell_order"
        ]
        
        # Performance thresholds for alerts
        self.latency_threshold_ns = 1_000_000  # 1ms
        self.throughput_threshold_ops_per_sec = 1000
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'output_dir': self.output_dir,
            'trace_format': self.trace_format,
            'buffer_size_mb': self.buffer_size_mb,
            'sample_rate_hz': self.sample_rate_hz,
            'enable_kernel_events': self.enable_kernel_events,
            'enable_user_events': self.enable_user_events,
            'max_trace_duration_seconds': self.max_trace_duration_seconds,
            'auto_analyze': self.auto_analyze,
            'profile_functions': self.profile_functions,
            'latency_threshold_ns': self.latency_threshold_ns,
            'throughput_threshold_ops_per_sec': self.throughput_threshold_ops_per_sec,
        }


class MagicTraceSession:
    """
    A magic-trace profiling session.
    
    Manages the lifecycle of a profiling session including
    starting/stopping traces, collecting data, and analysis.
    """
    
    def __init__(self, config: MagicTraceConfig, session_name: str = None):
        self.config = config
        self.session_name = session_name or f"orderbook_trace_{int(time.time())}"
        self.session_id = None
        self.start_time = None
        self.end_time = None
        self.trace_file = None
        self.is_active = False
        self.process = None
        
        # Create output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Session-specific files
        self.trace_file = self.output_path / f"{self.session_name}.{config.trace_format}"
        self.metadata_file = self.output_path / f"{self.session_name}_metadata.json"
        self.analysis_file = self.output_path / f"{self.session_name}_analysis.json"
    
    def start(self) -> bool:
        """Start the magic-trace profiling session."""
        if self.is_active:
            return False
        
        try:
            # Check if magic-trace is available
            if not self._check_magic_trace_available():
                print("Warning: magic-trace not found. Using fallback profiling.")
                return self._start_fallback_profiling()
            
            # Build magic-trace command
            cmd = self._build_magic_trace_command()
            
            # Start magic-trace process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            self.start_time = time.time_ns()
            self.is_active = True
            self.session_id = self.process.pid
            
            # Save session metadata
            self._save_metadata()
            
            print(f"Magic-trace session started: {self.session_name} (PID: {self.session_id})")
            return True
            
        except Exception as e:
            print(f"Failed to start magic-trace session: {e}")
            return self._start_fallback_profiling()
    
    def stop(self) -> bool:
        """Stop the magic-trace profiling session."""
        if not self.is_active:
            return False
        
        try:
            self.end_time = time.time_ns()
            
            if self.process:
                # Send SIGTERM to magic-trace process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                
                # Wait for process to terminate
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()
            
            # Handle fallback tracer
            if hasattr(self, 'fallback_tracer') and self.fallback_tracer:
                analysis = self.fallback_tracer.stop_session(self.session_name)
                if analysis:
                    # Save analysis to file
                    analysis_file = self.output_path / f"{self.session_name}_fallback_analysis.json"
                    with open(analysis_file, 'w') as f:
                        json.dump(analysis, f, indent=2, cls=NumpyJSONEncoder)
                    print(f"Fallback trace analysis saved to: {analysis_file}")
            
            self.is_active = False
            
            # Update metadata
            self._save_metadata()
            
            # Auto-analyze if enabled
            if self.config.auto_analyze:
                self._analyze_trace()
            
            print(f"Magic-trace session stopped: {self.session_name}")
            return True
            
        except Exception as e:
            print(f"Error stopping magic-trace session: {e}")
            return False
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        duration_ns = 0
        if self.start_time:
            end_time = self.end_time or time.time_ns()
            duration_ns = end_time - self.start_time
        
        return {
            'session_name': self.session_name,
            'session_id': self.session_id,
            'is_active': self.is_active,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ns': duration_ns,
            'duration_seconds': duration_ns / 1_000_000_000,
            'trace_file': str(self.trace_file) if self.trace_file else None,
            'config': self.config.to_dict(),
        }
    
    def _check_magic_trace_available(self) -> bool:
        """Check if magic-trace is available on the system."""
        try:
            result = subprocess.run(['magic-trace', '--version'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _build_magic_trace_command(self) -> List[str]:
        """Build the magic-trace command line."""
        cmd = [
            'magic-trace',
            'attach',
            '-p', str(os.getpid()),  # Attach to current process
            '-o', str(self.trace_file),
        ]
        
        # Add buffer size
        if self.config.buffer_size_mb:
            cmd.extend(['-buffer-size', f"{self.config.buffer_size_mb}MB"])
        
        # Add sampling rate
        if self.config.sample_rate_hz:
            cmd.extend(['-sample-rate', str(self.config.sample_rate_hz)])
        
        # Add duration limit
        if self.config.max_trace_duration_seconds:
            cmd.extend(['-duration', f"{self.config.max_trace_duration_seconds}s"])
        
        return cmd
    
    def _start_fallback_profiling(self):
        """Start fallback profiling when magic-trace is not available."""
        print("Starting fallback profiling mode...")
        
        # Import and start high-resolution tracer
        try:
            from .trace_analyzer import get_tracer
            self.fallback_tracer = get_tracer()
            success = self.fallback_tracer.start_session(self.session_name)
            
            if success:
                print(f"High-resolution fallback profiling started for session: {self.session_name}")
                self.is_active = True
                self.start_time = time.time_ns()
                return True
            else:
                print("Failed to start fallback profiling")
                return False
                
        except ImportError as e:
            print(f"Failed to import trace analyzer: {e}")
            return False
    
    def _save_metadata(self):
        """Save session metadata to file."""
        metadata = self.get_session_info()
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Failed to save metadata: {e}")
    
    def _analyze_trace(self):
        """Analyze the collected trace data."""
        if not self.trace_file.exists():
            print("No trace file found for analysis")
            return
        
        try:
            # Basic trace analysis
            analysis = {
                'session_name': self.session_name,
                'trace_file_size': self.trace_file.stat().st_size,
                'analysis_timestamp': time.time_ns(),
                'summary': 'Trace analysis would be performed here with magic-trace tools',
                'recommendations': []
            }
            
            # Save analysis results
            with open(self.analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            print(f"Trace analysis saved to: {self.analysis_file}")
            
        except Exception as e:
            print(f"Failed to analyze trace: {e}")


class MagicTraceProfiler:
    """
    High-level interface for magic-trace profiling in the order book.
    
    Provides context managers, decorators, and utilities for
    seamless integration of profiling into the trading system.
    """
    
    def __init__(self, config: MagicTraceConfig = None):
        self.config = config or MagicTraceConfig()
        self.active_sessions = {}
        self.session_counter = 0
        self._lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_trace_time_ns': 0,
            'function_call_counts': {},
            'function_latencies': {},
        }
    
    @contextmanager
    def profile_session(self, session_name: str = None):
        """Context manager for profiling sessions."""
        session = self.create_session(session_name)
        
        try:
            if session.start():
                yield session
            else:
                print("Failed to start profiling session")
                yield None
        finally:
            if session.is_active:
                session.stop()
            self._cleanup_session(session)
    
    def create_session(self, session_name: str = None) -> MagicTraceSession:
        """Create a new profiling session."""
        with self._lock:
            if session_name is None:
                self.session_counter += 1
                session_name = f"session_{self.session_counter}"
            
            session = MagicTraceSession(self.config, session_name)
            self.active_sessions[session_name] = session
            self.metrics['total_sessions'] += 1
            self.metrics['active_sessions'] += 1
            
            return session
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session names."""
        with self._lock:
            return [name for name, session in self.active_sessions.items() 
                   if session.is_active]
    
    def stop_all_sessions(self):
        """Stop all active profiling sessions."""
        with self._lock:
            for session in self.active_sessions.values():
                if session.is_active:
                    session.stop()
    
    def profile_function(self, func_name: str = None):
        """Decorator for profiling individual functions."""
        def decorator(func: Callable):
            name = func_name or func.__name__
            
            def wrapper(*args, **kwargs):
                start_time = time.time_ns()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time_ns()
                    latency = end_time - start_time
                    
                    # Update metrics
                    with self._lock:
                        if name not in self.metrics['function_call_counts']:
                            self.metrics['function_call_counts'][name] = 0
                            self.metrics['function_latencies'][name] = []
                        
                        self.metrics['function_call_counts'][name] += 1
                        self.metrics['function_latencies'][name].append(latency)
                        
                        # Keep only recent latencies to avoid memory growth
                        if len(self.metrics['function_latencies'][name]) > 1000:
                            self.metrics['function_latencies'][name] = \
                                self.metrics['function_latencies'][name][-1000:]
                        
                        # Check for performance alerts
                        if latency > self.config.latency_threshold_ns:
                            self._trigger_latency_alert(name, latency)
            
            return wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            summary = {
                'session_metrics': dict(self.metrics),
                'function_performance': {},
                'alerts': [],
                'recommendations': []
            }
            
            # Calculate function performance statistics
            for func_name, latencies in self.metrics['function_latencies'].items():
                if latencies:
                    import numpy as np
                    latencies_array = np.array(latencies)
                    
                    summary['function_performance'][func_name] = {
                        'call_count': self.metrics['function_call_counts'][func_name],
                        'mean_latency_ns': float(np.mean(latencies_array)),
                        'median_latency_ns': float(np.median(latencies_array)),
                        'p95_latency_ns': float(np.percentile(latencies_array, 95)),
                        'p99_latency_ns': float(np.percentile(latencies_array, 99)),
                        'max_latency_ns': float(np.max(latencies_array)),
                        'std_latency_ns': float(np.std(latencies_array)),
                    }
            
            # Generate recommendations
            summary['recommendations'] = self._generate_performance_recommendations(summary)
            
            return summary
    
    def export_traces(self, output_dir: str = "exported_traces") -> bool:
        """Export all trace files for external analysis."""
        try:
            export_path = Path(output_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            
            trace_dir = Path(self.config.output_dir)
            if not trace_dir.exists():
                return False
            
            # Copy all trace files
            import shutil
            for trace_file in trace_dir.glob("*"):
                if trace_file.is_file():
                    shutil.copy2(trace_file, export_path / trace_file.name)
            
            print(f"Traces exported to: {export_path}")
            return True
            
        except Exception as e:
            print(f"Failed to export traces: {e}")
            return False
    
    def _cleanup_session(self, session: MagicTraceSession):
        """Clean up a finished session."""
        with self._lock:
            if session.session_name in self.active_sessions:
                del self.active_sessions[session.session_name]
                self.metrics['active_sessions'] -= 1
                
                # Update total trace time
                if session.start_time and session.end_time:
                    duration = session.end_time - session.start_time
                    self.metrics['total_trace_time_ns'] += duration
    
    def _trigger_latency_alert(self, func_name: str, latency_ns: int):
        """Trigger alert for high latency function call."""
        alert = {
            'type': 'high_latency',
            'function': func_name,
            'latency_ns': latency_ns,
            'latency_ms': latency_ns / 1_000_000,
            'threshold_ns': self.config.latency_threshold_ns,
            'timestamp': time.time_ns(),
        }
        
        print(f"LATENCY ALERT: {func_name} took {latency_ns/1_000_000:.2f}ms "
              f"(threshold: {self.config.latency_threshold_ns/1_000_000:.2f}ms)")
    
    def _generate_performance_recommendations(self, summary: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        func_perf = summary.get('function_performance', {})
        
        for func_name, stats in func_perf.items():
            p99_latency = stats.get('p99_latency_ns', 0)
            mean_latency = stats.get('mean_latency_ns', 0)
            
            if p99_latency > 10_000_000:  # > 10ms
                recommendations.append(
                    f"Function '{func_name}' has high P99 latency ({p99_latency/1_000_000:.1f}ms). "
                    "Consider optimization or async processing."
                )
            
            if mean_latency > 1_000_000:  # > 1ms
                recommendations.append(
                    f"Function '{func_name}' has high average latency ({mean_latency/1_000_000:.1f}ms). "
                    "Review algorithm efficiency."
                )
            
            # Check for high variance
            std_latency = stats.get('std_latency_ns', 0)
            if std_latency > mean_latency:
                recommendations.append(
                    f"Function '{func_name}' has inconsistent performance. "
                    "Investigate causes of latency spikes."
                )
        
        return recommendations


# Global profiler instance
_global_profiler = None

def get_profiler() -> MagicTraceProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = MagicTraceProfiler()
    return _global_profiler

def profile_function(func_name: str = None):
    """Convenience decorator for function profiling."""
    return get_profiler().profile_function(func_name)

@contextmanager
def profile_block(block_name: str = "code_block"):
    """Context manager for profiling code blocks."""
    profiler = get_profiler()
    with profiler.profile_session(f"block_{block_name}") as session:
        yield session
