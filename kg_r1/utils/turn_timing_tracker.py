"""
Turn-level timing instrumentation for KG-R1 generation pipeline.
Provides detailed component-level timing analysis.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class TurnTiming:
    """Timing data for a single turn."""
    turn_id: int
    sample_id: int
    
    # Main timing components
    forward_pass_time: float = 0.0
    generation_time: float = 0.0
    kg_server_time: float = 0.0
    environment_time: float = 0.0
    total_turn_time: float = 0.0
    
    # Detailed KG breakdown
    kg_requests_count: int = 0
    kg_network_time: float = 0.0
    kg_processing_time: float = 0.0
    kg_timeout_count: int = 0
    
    # Detailed generation breakdown
    token_count: int = 0
    tokens_per_second: float = 0.0
    
    # Timestamps for debugging
    turn_start_time: float = field(default_factory=time.time)
    turn_end_time: Optional[float] = None

class TurnTimingTracker:
    """
    Comprehensive turn-level timing tracker for KG-R1 pipeline.
    
    Designed to be lightweight and non-intrusive to the main generation loop.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.current_timings: Dict[int, TurnTiming] = {}  # sample_id -> current turn timing
        self.completed_timings: List[TurnTiming] = []
        self.logger = logging.getLogger(f"{__name__}.TurnTimingTracker")
        
        # Aggregated statistics
        self.stats = {
            'total_turns': 0,
            'total_time': 0.0,
            'avg_forward_pass': 0.0,
            'avg_generation': 0.0,
            'avg_kg_server': 0.0,
            'avg_environment': 0.0,
            'kg_timeout_rate': 0.0,
            'avg_tokens_per_sec': 0.0
        }
        
    def start_turn(self, sample_id: int, turn_id: int) -> None:
        """Start timing a new turn."""
        if not self.enabled:
            return
            
        self.current_timings[sample_id] = TurnTiming(
            turn_id=turn_id,
            sample_id=sample_id,
            turn_start_time=time.time()
        )
        
    def time_forward_pass_start(self, sample_id: int) -> None:
        """Mark the start of VLLM forward pass."""
        if not self.enabled or sample_id not in self.current_timings:
            return
        self.current_timings[sample_id]._forward_start = time.time()
        
    def time_forward_pass_end(self, sample_id: int) -> None:
        """Mark the end of VLLM forward pass."""
        if not self.enabled or sample_id not in self.current_timings:
            return
        timing = self.current_timings[sample_id]
        if hasattr(timing, '_forward_start'):
            timing.forward_pass_time = time.time() - timing._forward_start
            
    def time_generation_start(self, sample_id: int) -> None:
        """Mark the start of text generation."""
        if not self.enabled or sample_id not in self.current_timings:
            return
        self.current_timings[sample_id]._generation_start = time.time()
        
    def time_generation_end(self, sample_id: int, token_count: int = 0) -> None:
        """Mark the end of text generation."""
        if not self.enabled or sample_id not in self.current_timings:
            return
        timing = self.current_timings[sample_id]
        if hasattr(timing, '_generation_start'):
            generation_time = time.time() - timing._generation_start
            timing.generation_time = generation_time
            timing.token_count = token_count
            if generation_time > 0 and token_count > 0:
                timing.tokens_per_second = token_count / generation_time
                
    def time_kg_server_start(self, sample_id: int) -> None:
        """Mark the start of KG server requests."""
        if not self.enabled or sample_id not in self.current_timings:
            return
        self.current_timings[sample_id]._kg_start = time.time()
        
    def time_kg_server_end(self, sample_id: int, request_count: int = 0, timeout_count: int = 0) -> None:
        """Mark the end of KG server requests."""
        if not self.enabled or sample_id not in self.current_timings:
            return
        timing = self.current_timings[sample_id]
        if hasattr(timing, '_kg_start'):
            timing.kg_server_time = time.time() - timing._kg_start
            timing.kg_requests_count = request_count
            timing.kg_timeout_count = timeout_count
            
    def time_environment_start(self, sample_id: int) -> None:
        """Mark the start of environment processing."""
        if not self.enabled or sample_id not in self.current_timings:
            return
        self.current_timings[sample_id]._env_start = time.time()
        
    def time_environment_end(self, sample_id: int) -> None:
        """Mark the end of environment processing."""
        if not self.enabled or sample_id not in self.current_timings:
            return
        timing = self.current_timings[sample_id]
        if hasattr(timing, '_env_start'):
            timing.environment_time = time.time() - timing._env_start
            
    def end_turn(self, sample_id: int) -> Optional[TurnTiming]:
        """Complete timing for a turn and return the timing data."""
        if not self.enabled or sample_id not in self.current_timings:
            return None
            
        timing = self.current_timings.pop(sample_id)
        timing.turn_end_time = time.time()
        timing.total_turn_time = timing.turn_end_time - timing.turn_start_time
        
        # Store completed timing
        self.completed_timings.append(timing)
        
        # Update statistics
        self._update_stats(timing)
        
        # Log timing breakdown
        self._log_turn_timing(timing)
        
        return timing
        
    def _update_stats(self, timing: TurnTiming) -> None:
        """Update aggregated statistics."""
        self.stats['total_turns'] += 1
        self.stats['total_time'] += timing.total_turn_time
        
        # Running averages
        n = self.stats['total_turns']
        self.stats['avg_forward_pass'] = ((n-1) * self.stats['avg_forward_pass'] + timing.forward_pass_time) / n
        self.stats['avg_generation'] = ((n-1) * self.stats['avg_generation'] + timing.generation_time) / n
        self.stats['avg_kg_server'] = ((n-1) * self.stats['avg_kg_server'] + timing.kg_server_time) / n
        self.stats['avg_environment'] = ((n-1) * self.stats['avg_environment'] + timing.environment_time) / n
        
        if timing.tokens_per_second > 0:
            current_avg = self.stats['avg_tokens_per_sec']
            self.stats['avg_tokens_per_sec'] = ((n-1) * current_avg + timing.tokens_per_second) / n
            
        # Timeout rate
        total_timeouts = sum(t.kg_timeout_count for t in self.completed_timings)
        total_requests = sum(t.kg_requests_count for t in self.completed_timings)
        if total_requests > 0:
            self.stats['kg_timeout_rate'] = total_timeouts / total_requests
            
    def _log_turn_timing(self, timing: TurnTiming) -> None:
        """Log detailed timing breakdown for a turn."""
        self.logger.info(
            f"🔍 TURN-TIMING [Sample {timing.sample_id}, Turn {timing.turn_id}] "
            f"Total: {timing.total_turn_time:.3f}s | "
            f"Forward: {timing.forward_pass_time*1000:.1f}ms | "
            f"Generation: {timing.generation_time*1000:.1f}ms ({timing.tokens_per_second:.0f} tok/s) | "
            f"KG: {timing.kg_server_time:.3f}s ({timing.kg_requests_count} reqs, {timing.kg_timeout_count} timeouts) | "
            f"Env: {timing.environment_time*1000:.1f}ms"
        )
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive timing statistics."""
        if not self.completed_timings:
            return {"error": "No timing data collected"}
            
        return {
            "overview": {
                "total_turns_processed": self.stats['total_turns'],
                "total_evaluation_time": f"{self.stats['total_time']:.2f}s",
                "average_turn_time": f"{self.stats['total_time'] / max(1, self.stats['total_turns']):.3f}s"
            },
            "component_breakdown": {
                "avg_forward_pass_ms": f"{self.stats['avg_forward_pass'] * 1000:.1f}",
                "avg_generation_ms": f"{self.stats['avg_generation'] * 1000:.1f}",
                "avg_kg_server_s": f"{self.stats['avg_kg_server']:.3f}",
                "avg_environment_ms": f"{self.stats['avg_environment'] * 1000:.1f}"
            },
            "performance_metrics": {
                "avg_tokens_per_second": f"{self.stats['avg_tokens_per_sec']:.0f}",
                "kg_timeout_rate": f"{self.stats['kg_timeout_rate']:.1%}"
            },
            "bottleneck_analysis": self._analyze_bottlenecks()
        }
        
    def _analyze_bottlenecks(self) -> Dict[str, str]:
        """Analyze which components are bottlenecks."""
        components = {
            "forward_pass": self.stats['avg_forward_pass'],
            "generation": self.stats['avg_generation'], 
            "kg_server": self.stats['avg_kg_server'],
            "environment": self.stats['avg_environment']
        }
        
        total_time = sum(components.values())
        if total_time == 0:
            return {"analysis": "No timing data available"}
            
        percentages = {k: (v / total_time) * 100 for k, v in components.items()}
        
        # Find primary bottleneck
        bottleneck = max(percentages, key=percentages.get)
        bottleneck_pct = percentages[bottleneck]
        
        analysis = {
            "primary_bottleneck": f"{bottleneck} ({bottleneck_pct:.1f}% of turn time)",
            "component_percentages": {k: f"{v:.1f}%" for k, v in percentages.items()},
        }
        
        # Recommendations
        if bottleneck == "kg_server" and bottleneck_pct > 50:
            analysis["recommendation"] = "KG server requests dominate timing - consider async processing or caching"
        elif bottleneck == "generation" and bottleneck_pct > 40:
            analysis["recommendation"] = "Text generation is slow - check VLLM configuration and batching"
        elif bottleneck == "forward_pass" and bottleneck_pct > 30:
            analysis["recommendation"] = "Model inference is slow - consider model optimization or hardware upgrade"
        else:
            analysis["recommendation"] = "Timing is well-balanced across components"
            
        return analysis
        
    def export_detailed_timings(self, filepath: str) -> None:
        """Export all timing data to JSON file."""
        if not self.completed_timings:
            self.logger.warning("No timing data to export")
            return
            
        export_data = {
            "summary_stats": self.get_summary_stats(),
            "detailed_timings": [
                {
                    "sample_id": t.sample_id,
                    "turn_id": t.turn_id,
                    "total_turn_time": t.total_turn_time,
                    "forward_pass_time": t.forward_pass_time,
                    "generation_time": t.generation_time,
                    "kg_server_time": t.kg_server_time,
                    "environment_time": t.environment_time,
                    "kg_requests_count": t.kg_requests_count,
                    "kg_timeout_count": t.kg_timeout_count,
                    "token_count": t.token_count,
                    "tokens_per_second": t.tokens_per_second
                }
                for t in self.completed_timings
            ]
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"📊 Detailed timing data exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export timing data: {e}")

# Global timing tracker instance
_global_tracker: Optional[TurnTimingTracker] = None

def get_timing_tracker() -> TurnTimingTracker:
    """Get the global timing tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TurnTimingTracker()
    return _global_tracker

def enable_turn_timing(enabled: bool = True) -> None:
    """Enable or disable turn timing globally."""
    tracker = get_timing_tracker()
    tracker.enabled = enabled
    if enabled:
        tracker.logger.info("🔍 Turn-level timing instrumentation ENABLED")
    else:
        tracker.logger.info("🔍 Turn-level timing instrumentation DISABLED")