"""
Real-time token consumption tracking and efficiency monitoring.

Provides fine-grained tracking of token usage with automatic alerts
for anomalous consumption patterns and efficiency degradation.
"""

import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Set
from datetime import datetime, timedelta
import logging
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ConsumptionEvent:
    """Single token consumption event."""
    timestamp: float
    agent_id: str
    tokens: int
    operation: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class EfficiencyMetric:
    """Point-in-time efficiency measurement."""
    timestamp: float
    agent_id: str
    tokens_used: int
    value_generated: float
    efficiency: float  # value per token
    operation_count: int


class TokenConsumptionMonitor:
    """
    Real-time monitoring of token consumption with anomaly detection.
    
    Tracks consumption patterns, detects spikes, and can automatically
    throttle or terminate agents exhibiting runaway consumption.
    """
    
    def __init__(self,
                 window_size: int = 100,
                 spike_threshold: float = 3.0,
                 alert_callback: Optional[Callable] = None):
        """
        Initialize consumption monitor.
        
        Args:
            window_size: Number of events to track per agent
            spike_threshold: Standard deviations for spike detection
            alert_callback: Function to call on alerts
        """
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.alert_callback = alert_callback
        
        # Tracking data structures
        self.consumption_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.consumption_rates: Dict[str, List[float]] = defaultdict(list)
        self.total_consumption: Dict[str, int] = defaultdict(int)
        self.alert_history: List[Dict] = []
        
        # Throttling state
        self.throttled_agents: Dict[str, float] = {}  # agent_id -> throttle_until
        self.terminated_agents: Set[str] = set()
        
        # Start monitoring loop
        self._monitor_task = None
        self._running = False
    
    def track_consumption(self, event: ConsumptionEvent) -> None:
        """
        Track a token consumption event.
        
        Automatically checks for anomalies and triggers alerts.
        """
        if event.agent_id in self.terminated_agents:
            logger.warning(f"Ignoring consumption from terminated agent: {event.agent_id}")
            return
        
        # Check throttling
        if event.agent_id in self.throttled_agents:
            if time.time() < self.throttled_agents[event.agent_id]:
                logger.info(f"Agent {event.agent_id} is throttled, delaying operation")
                time.sleep(0.5)  # Simple throttling
        
        # Record consumption
        self.consumption_windows[event.agent_id].append(event)
        self.total_consumption[event.agent_id] += event.tokens
        
        # Calculate consumption rate
        window = list(self.consumption_windows[event.agent_id])
        if len(window) >= 2:
            time_span = window[-1].timestamp - window[0].timestamp
            if time_span > 0:
                rate = sum(e.tokens for e in window) / time_span
                self.consumption_rates[event.agent_id].append(rate)
                
                # Check for spikes
                self._check_consumption_spike(event.agent_id, rate)
    
    def _check_consumption_spike(self, agent_id: str, current_rate: float) -> None:
        """Detect anomalous consumption spikes."""
        rates = self.consumption_rates[agent_id]
        if len(rates) < 10:
            return  # Not enough data
        
        # Calculate statistics
        recent_rates = rates[-10:-1]  # Exclude current
        mean_rate = sum(recent_rates) / len(recent_rates)
        variance = sum((r - mean_rate) ** 2 for r in recent_rates) / len(recent_rates)
        std_dev = variance ** 0.5
        
        # Check for spike
        if std_dev > 0 and (current_rate - mean_rate) > (self.spike_threshold * std_dev):
            self._trigger_alert(
                AlertLevel.WARNING,
                f"Consumption spike detected for agent {agent_id}: "
                f"{current_rate:.2f} tokens/sec (normal: {mean_rate:.2f})",
                {
                    'agent_id': agent_id,
                    'current_rate': current_rate,
                    'mean_rate': mean_rate,
                    'std_dev': std_dev
                }
            )
            
            # Auto-throttle on severe spikes
            if current_rate > mean_rate * 5:
                self.throttle_agent(agent_id, duration=30)
    
    def throttle_agent(self, agent_id: str, duration: float) -> None:
        """Temporarily throttle an agent's token consumption."""
        self.throttled_agents[agent_id] = time.time() + duration
        self._trigger_alert(
            AlertLevel.INFO,
            f"Agent {agent_id} throttled for {duration} seconds",
            {'agent_id': agent_id, 'duration': duration}
        )
    
    def terminate_agent_consumption(self, agent_id: str) -> None:
        """Permanently block an agent from consuming tokens."""
        self.terminated_agents.add(agent_id)
        self._trigger_alert(
            AlertLevel.CRITICAL,
            f"Agent {agent_id} terminated for consumption violations",
            {'agent_id': agent_id}
        )
    
    def get_consumption_stats(self, agent_id: str) -> Dict:
        """Get detailed consumption statistics for an agent."""
        if agent_id not in self.consumption_windows:
            return {}
        
        window = list(self.consumption_windows[agent_id])
        if not window:
            return {}
        
        total = self.total_consumption[agent_id]
        time_span = window[-1].timestamp - window[0].timestamp if len(window) > 1 else 0
        
        return {
            'total_consumed': total,
            'window_size': len(window),
            'time_span': time_span,
            'average_rate': total / time_span if time_span > 0 else 0,
            'recent_operations': [e.operation for e in window[-5:]],
            'is_throttled': agent_id in self.throttled_agents,
            'is_terminated': agent_id in self.terminated_agents
        }
    
    def _trigger_alert(self, level: AlertLevel, message: str, data: Dict) -> None:
        """Trigger an alert and notify callback."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'message': message,
            'data': data
        }
        
        self.alert_history.append(alert)
        logger.log(
            logging.WARNING if level in [AlertLevel.WARNING, AlertLevel.CRITICAL] else logging.INFO,
            message
        )
        
        if self.alert_callback:
            self.alert_callback(alert)
    
    async def start_monitoring(self) -> None:
        """Start the background monitoring loop."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop the background monitoring loop."""
        self._running = False
        if self._monitor_task:
            await self._monitor_task
    
    async def _monitor_loop(self) -> None:
        """Background monitoring for patterns and cleanup."""
        while self._running:
            try:
                # Clean up expired throttles
                current_time = time.time()
                expired = [
                    agent_id for agent_id, until in self.throttled_agents.items()
                    if current_time >= until
                ]
                for agent_id in expired:
                    del self.throttled_agents[agent_id]
                    logger.info(f"Throttle expired for agent {agent_id}")
                
                # Check for sustained high consumption
                for agent_id, rates in self.consumption_rates.items():
                    if len(rates) >= 20:
                        recent_avg = sum(rates[-20:]) / 20
                        if recent_avg > 1000:  # High sustained rate
                            self._trigger_alert(
                                AlertLevel.WARNING,
                                f"Sustained high consumption for agent {agent_id}: "
                                f"{recent_avg:.2f} tokens/sec",
                                {'agent_id': agent_id, 'rate': recent_avg}
                            )
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)


class EfficiencyTracker:
    """
    Tracks value generation efficiency over time.
    
    Monitors how effectively agents convert tokens into value,
    identifying trends and degradation patterns.
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize efficiency tracker.
        
        Args:
            history_size: Number of metrics to retain per agent
        """
        self.history_size = history_size
        self.efficiency_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.value_totals: Dict[str, float] = defaultdict(float)
        self.token_totals: Dict[str, int] = defaultdict(int)
    
    def record_operation(self,
                        agent_id: str,
                        tokens_used: int,
                        value_generated: float,
                        operation_type: str = "unknown") -> float:
        """
        Record an operation and calculate efficiency.
        
        Returns:
            Current efficiency ratio
        """
        efficiency = value_generated / tokens_used if tokens_used > 0 else 0.0
        
        metric = EfficiencyMetric(
            timestamp=time.time(),
            agent_id=agent_id,
            tokens_used=tokens_used,
            value_generated=value_generated,
            efficiency=efficiency,
            operation_count=self.operation_counts[agent_id] + 1
        )
        
        self.efficiency_history[agent_id].append(metric)
        self.operation_counts[agent_id] += 1
        self.value_totals[agent_id] += value_generated
        self.token_totals[agent_id] += tokens_used
        
        # Check for efficiency degradation
        self._check_efficiency_trend(agent_id)
        
        return efficiency
    
    def _check_efficiency_trend(self, agent_id: str) -> None:
        """Detect efficiency degradation trends."""
        history = list(self.efficiency_history[agent_id])
        if len(history) < 10:
            return
        
        # Compare recent vs historical efficiency
        recent = history[-5:]
        historical = history[-20:-10] if len(history) >= 20 else history[:5]
        
        recent_avg = sum(m.efficiency for m in recent) / len(recent)
        historical_avg = sum(m.efficiency for m in historical) / len(historical)
        
        # Significant degradation detection
        if historical_avg > 0 and recent_avg < historical_avg * 0.5:
            logger.warning(
                f"Efficiency degradation detected for agent {agent_id}: "
                f"{recent_avg:.3f} (recent) vs {historical_avg:.3f} (historical)"
            )
    
    def get_efficiency_stats(self, agent_id: str, window: Optional[int] = None) -> Dict:
        """
        Get comprehensive efficiency statistics.
        
        Args:
            agent_id: Agent to analyze
            window: Number of recent operations to consider
            
        Returns:
            Dictionary of efficiency metrics
        """
        if agent_id not in self.efficiency_history:
            return {}
        
        history = list(self.efficiency_history[agent_id])
        if window:
            history = history[-window:]
        
        if not history:
            return {}
        
        efficiencies = [m.efficiency for m in history]
        
        return {
            'agent_id': agent_id,
            'total_operations': self.operation_counts[agent_id],
            'total_value': self.value_totals[agent_id],
            'total_tokens': self.token_totals[agent_id],
            'lifetime_efficiency': (
                self.value_totals[agent_id] / self.token_totals[agent_id]
                if self.token_totals[agent_id] > 0 else 0.0
            ),
            'window_size': len(history),
            'current_efficiency': efficiencies[-1] if efficiencies else 0.0,
            'average_efficiency': sum(efficiencies) / len(efficiencies),
            'min_efficiency': min(efficiencies),
            'max_efficiency': max(efficiencies),
            'efficiency_trend': self._calculate_trend(efficiencies)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple linear regression
        x = list(range(len(values)))
        x_mean = sum(x) / len(x)
        y_mean = sum(values) / len(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(len(values)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(values)))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"
    
    def export_efficiency_report(self, filepath: str) -> None:
        """Export detailed efficiency report for all agents."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'agents': {}
        }
        
        for agent_id in self.efficiency_history:
            report['agents'][agent_id] = self.get_efficiency_stats(agent_id)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported efficiency report to {filepath}")