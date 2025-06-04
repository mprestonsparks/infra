"""
Detection of gaming behaviors and metric manipulation.

Identifies when agents are optimizing for metrics rather than
true performance, distinguishing between beneficial innovations
and exploitative behaviors.
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
import logging
from datetime import datetime

from .pattern_detector import Pattern, PatternType

logger = logging.getLogger(__name__)


@dataclass
class GamingIndicator:
    """Indicator of potential gaming behavior."""
    indicator_type: str
    severity: float  # 0-1, higher = more severe
    description: str
    evidence: Dict[str, Any]
    timestamp: datetime = datetime.now()


class GamingDetector:
    """
    Detects metric gaming and exploitation behaviors.
    
    Distinguishes between legitimate optimization and
    behaviors that game the system without providing value.
    """
    
    def __init__(self,
                 sensitivity: float = 0.7,
                 history_window: int = 100):
        """
        Initialize gaming detector.
        
        Args:
            sensitivity: Detection sensitivity (0-1)
            history_window: Actions to consider for pattern analysis
        """
        self.sensitivity = sensitivity
        self.history_window = history_window
        
        # Known gaming patterns
        self.gaming_signatures = {
            'metric_fixation': self._detect_metric_fixation,
            'reward_hacking': self._detect_reward_hacking,
            'edge_case_exploitation': self._detect_edge_exploitation,
            'repetitive_exploitation': self._detect_repetitive_exploitation,
            'goodhart_optimization': self._detect_goodhart_optimization
        }
        
        # Tracking
        self.agent_histories: Dict[str, List[Dict]] = defaultdict(list)
        self.gaming_incidents: Dict[str, List[GamingIndicator]] = defaultdict(list)
    
    def detect_gaming(self,
                     agent_id: str,
                     actions: List[Dict],
                     performance: float,
                     patterns: List[Pattern]) -> List[GamingIndicator]:
        """
        Detect gaming behaviors in agent actions.
        
        Returns list of gaming indicators found.
        """
        # Update history
        self.agent_histories[agent_id].extend(actions)
        if len(self.agent_histories[agent_id]) > self.history_window:
            self.agent_histories[agent_id] = self.agent_histories[agent_id][-self.history_window:]
        
        indicators = []
        
        # Run each detection method
        for name, detector in self.gaming_signatures.items():
            result = detector(agent_id, actions, performance, patterns)
            if result:
                indicators.extend(result)
        
        # Filter by severity threshold
        threshold = 1.0 - self.sensitivity
        significant_indicators = [i for i in indicators if i.severity >= threshold]
        
        # Record incidents
        if significant_indicators:
            self.gaming_incidents[agent_id].extend(significant_indicators)
            logger.warning(f"Gaming behavior detected for agent {agent_id}: "
                         f"{len(significant_indicators)} indicators")
        
        return significant_indicators
    
    def _detect_metric_fixation(self,
                               agent_id: str,
                               actions: List[Dict],
                               performance: float,
                               patterns: List[Pattern]) -> List[GamingIndicator]:
        """Detect fixation on specific metrics without holistic improvement."""
        indicators = []
        
        # Check if agent is optimizing single metric excessively
        if actions:
            action_types = [a.get('action_type', 'unknown') for a in actions]
            type_counts = Counter(action_types)
            
            # High concentration on single action type
            if type_counts:
                most_common = type_counts.most_common(1)[0]
                concentration = most_common[1] / len(actions)
                
                if concentration > 0.8:  # 80% of actions are same type
                    indicators.append(GamingIndicator(
                        indicator_type='metric_fixation',
                        severity=concentration,
                        description=f"Excessive focus on {most_common[0]} actions",
                        evidence={
                            'dominant_action': most_common[0],
                            'concentration': concentration,
                            'total_actions': len(actions)
                        }
                    ))
        
        # Check patterns for metric fixation
        optimization_patterns = [p for p in patterns if p.pattern_type == PatternType.OPTIMIZATION]
        if len(optimization_patterns) > 3:
            # Multiple optimization patterns on same metric
            optimization_targets = defaultdict(int)
            for pattern in optimization_patterns:
                if 'optimization_target' in pattern.context:
                    optimization_targets[pattern.context['optimization_target']] += 1
            
            if optimization_targets:
                most_targeted = max(optimization_targets.values())
                if most_targeted >= 3:
                    indicators.append(GamingIndicator(
                        indicator_type='metric_fixation',
                        severity=0.7,
                        description="Multiple optimizations targeting same metric",
                        evidence={
                            'pattern_count': len(optimization_patterns),
                            'target_concentration': most_targeted
                        }
                    ))
        
        return indicators
    
    def _detect_reward_hacking(self,
                              agent_id: str,
                              actions: List[Dict],
                              performance: float,
                              patterns: List[Pattern]) -> List[GamingIndicator]:
        """Detect behaviors that hack the reward system."""
        indicators = []
        
        # Sudden performance spikes without corresponding value
        history = self.agent_histories[agent_id]
        if len(history) >= 10:
            # Look for performance anomalies
            recent_actions = history[-10:]
            action_value_ratios = []
            
            for action in recent_actions:
                if 'reward' in action and 'cost' in action:
                    ratio = action['reward'] / (action['cost'] + 1)
                    action_value_ratios.append(ratio)
            
            if action_value_ratios:
                avg_ratio = np.mean(action_value_ratios)
                max_ratio = max(action_value_ratios)
                
                # Suspicious if max is way higher than average
                if max_ratio > avg_ratio * 5:
                    indicators.append(GamingIndicator(
                        indicator_type='reward_hacking',
                        severity=0.8,
                        description="Anomalous reward/cost ratio detected",
                        evidence={
                            'avg_ratio': avg_ratio,
                            'max_ratio': max_ratio,
                            'spike_factor': max_ratio / avg_ratio
                        }
                    ))
        
        # Check for loophole exploitation patterns
        behavioral_patterns = [p for p in patterns if p.pattern_type == PatternType.BEHAVIORAL]
        for pattern in behavioral_patterns:
            if pattern.effectiveness > 2.0:  # Unusually high effectiveness
                # Check if it's a simple repetitive pattern
                if len(set(pattern.sequence)) == 1:  # All same action
                    indicators.append(GamingIndicator(
                        indicator_type='reward_hacking',
                        severity=0.9,
                        description="Simple repetitive pattern with high reward",
                        evidence={
                            'pattern': pattern.sequence,
                            'effectiveness': pattern.effectiveness
                        }
                    ))
        
        return indicators
    
    def _detect_edge_exploitation(self,
                                 agent_id: str,
                                 actions: List[Dict],
                                 performance: float,
                                 patterns: List[Pattern]) -> List[GamingIndicator]:
        """Detect exploitation of edge cases or boundary conditions."""
        indicators = []
        
        # Check for boundary value exploitation
        for action in actions:
            if 'parameters' in action:
                params = action['parameters']
                
                # Check for extreme values
                extreme_count = 0
                for key, value in params.items():
                    if isinstance(value, (int, float)):
                        # Common boundary values
                        if value in [0, 1, -1, float('inf'), float('-inf'), 
                                   1e-10, 1e10, 2**31-1, -2**31]:
                            extreme_count += 1
                
                if extreme_count >= 2:
                    indicators.append(GamingIndicator(
                        indicator_type='edge_case_exploitation',
                        severity=0.6,
                        description="Multiple boundary values in parameters",
                        evidence={
                            'action_type': action.get('action_type', 'unknown'),
                            'extreme_params': extreme_count
                        }
                    ))
        
        # Check for edge case patterns
        edge_patterns = []
        for pattern in patterns:
            if any(term in pattern.description.lower() 
                   for term in ['boundary', 'edge', 'limit', 'extreme']):
                edge_patterns.append(pattern)
        
        if len(edge_patterns) >= 2:
            indicators.append(GamingIndicator(
                indicator_type='edge_case_exploitation',
                severity=0.7,
                description="Multiple patterns exploiting edge cases",
                evidence={
                    'pattern_count': len(edge_patterns),
                    'patterns': [p.description for p in edge_patterns]
                }
            ))
        
        return indicators
    
    def _detect_repetitive_exploitation(self,
                                       agent_id: str,
                                       actions: List[Dict],
                                       performance: float,
                                       patterns: List[Pattern]) -> List[GamingIndicator]:
        """Detect repetitive exploitation of same mechanic."""
        indicators = []
        
        if len(actions) < 5:
            return indicators
        
        # Check for repetitive sequences
        sequence_length = 3
        sequences = []
        
        for i in range(len(actions) - sequence_length + 1):
            seq = tuple(
                actions[j].get('action_type', 'unknown') 
                for j in range(i, i + sequence_length)
            )
            sequences.append(seq)
        
        seq_counts = Counter(sequences)
        
        # Find overly repeated sequences
        for seq, count in seq_counts.items():
            repetition_rate = count / len(sequences)
            if repetition_rate > 0.5:  # More than 50% repetition
                indicators.append(GamingIndicator(
                    indicator_type='repetitive_exploitation',
                    severity=repetition_rate,
                    description=f"Excessive repetition of sequence: {' -> '.join(seq)}",
                    evidence={
                        'sequence': seq,
                        'count': count,
                        'rate': repetition_rate
                    }
                ))
        
        # Check for lack of exploration
        unique_actions = len(set(a.get('action_type', 'unknown') for a in actions))
        exploration_rate = unique_actions / len(actions)
        
        if exploration_rate < 0.1:  # Less than 10% unique actions
            indicators.append(GamingIndicator(
                indicator_type='repetitive_exploitation',
                severity=0.8,
                description="Lack of exploration - too few unique actions",
                evidence={
                    'unique_actions': unique_actions,
                    'total_actions': len(actions),
                    'exploration_rate': exploration_rate
                }
            ))
        
        return indicators
    
    def _detect_goodhart_optimization(self,
                                     agent_id: str,
                                     actions: List[Dict],
                                     performance: float,
                                     patterns: List[Pattern]) -> List[GamingIndicator]:
        """Detect Goodhart's Law violations - optimizing metric instead of goal."""
        indicators = []
        
        # Check if performance metrics diverge from value metrics
        value_sum = sum(a.get('value_generated', 0) for a in actions)
        reward_sum = sum(a.get('reward', 0) for a in actions)
        
        if reward_sum > 0 and value_sum > 0:
            # High reward but low value is suspicious
            value_reward_ratio = value_sum / reward_sum
            if value_reward_ratio < 0.2:  # Value is less than 20% of reward
                indicators.append(GamingIndicator(
                    indicator_type='goodhart_optimization',
                    severity=0.85,
                    description="High reward with disproportionately low value",
                    evidence={
                        'value_sum': value_sum,
                        'reward_sum': reward_sum,
                        'ratio': value_reward_ratio
                    }
                ))
        
        # Check if agent found patterns that maximize metric but not goal
        high_reward_patterns = [
            p for p in patterns 
            if p.effectiveness > 1.0 and p.pattern_type == PatternType.OPTIMIZATION
        ]
        
        for pattern in high_reward_patterns:
            # Check if pattern description suggests metric gaming
            gaming_terms = ['maximize', 'exploit', 'hack', 'loophole', 'bypass']
            if any(term in pattern.description.lower() for term in gaming_terms):
                indicators.append(GamingIndicator(
                    indicator_type='goodhart_optimization',
                    severity=0.7,
                    description=f"Pattern suggests metric optimization: {pattern.description}",
                    evidence={
                        'pattern_id': pattern.pattern_id,
                        'effectiveness': pattern.effectiveness
                    }
                ))
        
        return indicators
    
    def get_gaming_report(self, agent_id: Optional[str] = None) -> Dict:
        """Generate report on gaming behaviors."""
        if agent_id:
            incidents = self.gaming_incidents.get(agent_id, [])
            
            return {
                'agent_id': agent_id,
                'total_incidents': len(incidents),
                'incident_types': Counter(i.indicator_type for i in incidents),
                'avg_severity': np.mean([i.severity for i in incidents]) if incidents else 0,
                'recent_incidents': [
                    {
                        'type': i.indicator_type,
                        'severity': i.severity,
                        'description': i.description,
                        'timestamp': i.timestamp.isoformat()
                    }
                    for i in incidents[-5:]
                ]
            }
        else:
            # Global report
            all_incidents = sum(len(inc) for inc in self.gaming_incidents.values())
            
            return {
                'total_incidents': all_incidents,
                'affected_agents': len(self.gaming_incidents),
                'incident_breakdown': self._get_incident_breakdown(),
                'top_offenders': self._get_top_gaming_agents(5)
            }
    
    def _get_incident_breakdown(self) -> Dict[str, int]:
        """Get breakdown of incidents by type."""
        breakdown = defaultdict(int)
        
        for incidents in self.gaming_incidents.values():
            for incident in incidents:
                breakdown[incident.indicator_type] += 1
        
        return dict(breakdown)
    
    def _get_top_gaming_agents(self, limit: int) -> List[Dict]:
        """Get agents with most gaming incidents."""
        agent_scores = []
        
        for agent_id, incidents in self.gaming_incidents.items():
            if incidents:
                score = len(incidents) * np.mean([i.severity for i in incidents])
                agent_scores.append({
                    'agent_id': agent_id,
                    'incident_count': len(incidents),
                    'gaming_score': score
                })
        
        agent_scores.sort(key=lambda x: x['gaming_score'], reverse=True)
        return agent_scores[:limit]