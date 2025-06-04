"""
Emergent behavior monitoring system.

Captures and catalogs novel agent strategies that arise
through evolution but were not explicitly programmed.
"""

import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import logging
from datetime import datetime

from ..economy import TokenEconomyManager
from ..diversity import AgentGenome
from .pattern_detector import PatternDetector, Pattern, PatternType
from .gaming_detector import GamingDetector

logger = logging.getLogger(__name__)


@dataclass
class BehaviorMetrics:
    """Metrics for agent behavior analysis."""
    agent_id: str
    generation: int
    total_actions: int
    unique_actions: int
    action_diversity: float  # Shannon entropy of action distribution
    performance_trend: str  # 'improving', 'stable', 'degrading'
    efficiency_ratio: float  # Value per action
    innovation_score: float  # Novel patterns discovered
    gaming_risk: float  # Risk of metric gaming
    emergent_behaviors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EmergentStrategy:
    """Represents a discovered emergent strategy."""
    strategy_id: str
    description: str
    first_observed: datetime
    agent_count: int  # Number of agents using this strategy
    effectiveness: float
    replicability: float  # How easily other agents can adopt it
    components: List[Pattern]  # Patterns that make up this strategy
    prerequisites: Dict[str, Any]  # Required conditions
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmergentBehaviorMonitor:
    """
    Captures and catalogs novel agent strategies.
    
    Monitors agent behavior to identify emergent strategies that
    arise from evolutionary pressure but were not explicitly programmed.
    """
    
    def __init__(self,
                 metrics_db: Optional[Any] = None,
                 pattern_detector: Optional[PatternDetector] = None,
                 gaming_detector: Optional[GamingDetector] = None):
        """
        Initialize behavior monitor.
        
        Args:
            metrics_db: Database for storing discovered patterns
            pattern_detector: Pattern detection engine
            gaming_detector: Detector for gaming behaviors
        """
        self.metrics_db = metrics_db
        self.pattern_detector = pattern_detector or PatternDetector()
        self.gaming_detector = gaming_detector or GamingDetector()
        
        # Behavior tracking
        self.agent_behaviors: Dict[str, List[Dict]] = defaultdict(list)
        self.behavior_metrics: Dict[str, List[BehaviorMetrics]] = defaultdict(list)
        self.emergent_strategies: Dict[str, EmergentStrategy] = {}
        
        # Innovation tracking
        self.innovation_timeline: List[Dict] = []
        self.strategy_adoption: Dict[str, Set[str]] = defaultdict(set)  # strategy -> agents
        
        # Performance baselines for comparison
        self.performance_baselines: Dict[str, float] = {}
        self.expected_behaviors: Set[str] = set()  # Explicitly programmed behaviors
    
    def set_expected_behaviors(self, behaviors: List[str]) -> None:
        """Define behaviors that are explicitly programmed (not emergent)."""
        self.expected_behaviors = set(behaviors)
    
    def analyze_agent_behavior(self, 
                             agent_id: str,
                             genome: AgentGenome,
                             actions: List[Dict],
                             performance: float,
                             token_usage: int) -> List[Pattern]:
        """
        Analyze agent behavior to identify emergent strategies.
        
        Returns list of discovered patterns.
        """
        # Track all actions
        for action in actions:
            self._track_action(agent_id, action)
        
        # Update pattern detector
        self.pattern_detector.track_performance(agent_id, performance)
        self.pattern_detector.track_genome_state(agent_id, {
            'generation': genome.generation,
            'gene_count': len(genome.genes),
            'gene_types': [g.gene_type.value for g in genome.genes.values()]
        })
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(agent_id)
        
        # Filter for emergent patterns
        emergent_patterns = self._identify_emergent_patterns(patterns)
        
        # Check for gaming behaviors
        gaming_indicators = self.gaming_detector.detect_gaming(
            agent_id, actions, performance, patterns
        )
        
        # Calculate behavior metrics
        metrics = self._calculate_behavior_metrics(
            agent_id, genome, actions, performance, 
            token_usage, emergent_patterns, gaming_indicators
        )
        
        self.behavior_metrics[agent_id].append(metrics)
        
        # Identify emergent strategies from patterns
        strategies = self._extract_emergent_strategies(emergent_patterns, metrics)
        
        # Record innovations
        if emergent_patterns:
            self._record_innovation(agent_id, emergent_patterns, strategies)
        
        # Store in metrics database if available
        if self.metrics_db and emergent_patterns:
            self._store_patterns(emergent_patterns)
        
        return emergent_patterns
    
    def _track_action(self, agent_id: str, action: Dict) -> None:
        """Track individual agent action."""
        timestamped_action = {
            **action,
            'timestamp': time.time(),
            'agent_id': agent_id
        }
        self.agent_behaviors[agent_id].append(timestamped_action)
        
        # Keep bounded
        if len(self.agent_behaviors[agent_id]) > 1000:
            self.agent_behaviors[agent_id] = self.agent_behaviors[agent_id][-1000:]
    
    def _identify_emergent_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Filter patterns to identify truly emergent behaviors."""
        emergent = []
        
        for pattern in patterns:
            # Check if pattern represents unexpected behavior
            is_emergent = True
            
            # Not emergent if it's an expected behavior
            pattern_desc = pattern.description.lower()
            for expected in self.expected_behaviors:
                if expected.lower() in pattern_desc:
                    is_emergent = False
                    break
            
            # Additional heuristics for emergent detection
            if is_emergent:
                # High effectiveness patterns that weren't designed
                if pattern.effectiveness > 0.5 and pattern.confidence > 0.8:
                    emergent.append(pattern)
                
                # Unusual combinations
                elif pattern.pattern_type == PatternType.STRUCTURAL and pattern.occurrences > 5:
                    emergent.append(pattern)
                
                # Novel optimization strategies
                elif (pattern.pattern_type == PatternType.OPTIMIZATION and 
                      pattern.effectiveness > 0.3):
                    emergent.append(pattern)
        
        return emergent
    
    def _calculate_behavior_metrics(self,
                                   agent_id: str,
                                   genome: AgentGenome,
                                   actions: List[Dict],
                                   performance: float,
                                   token_usage: int,
                                   emergent_patterns: List[Pattern],
                                   gaming_indicators: List[Any]) -> BehaviorMetrics:
        """Calculate comprehensive behavior metrics."""
        # Action diversity
        action_types = [a.get('action_type', 'unknown') for a in actions]
        unique_actions = len(set(action_types))
        
        # Shannon entropy for action diversity
        if action_types:
            action_counts = defaultdict(int)
            for action in action_types:
                action_counts[action] += 1
            
            total = len(action_types)
            entropy = -sum(
                (count/total) * np.log2(count/total) 
                for count in action_counts.values() if count > 0
            )
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(action_counts)) if len(action_counts) > 1 else 1
            action_diversity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            action_diversity = 0
        
        # Performance trend
        if agent_id in self.pattern_detector.performance_history:
            perf_history = self.pattern_detector.performance_history[agent_id]
            if len(perf_history) >= 5:
                recent = np.mean(perf_history[-5:])
                older = np.mean(perf_history[-10:-5]) if len(perf_history) >= 10 else perf_history[0]
                
                if recent > older * 1.1:
                    performance_trend = "improving"
                elif recent < older * 0.9:
                    performance_trend = "degrading"
                else:
                    performance_trend = "stable"
            else:
                performance_trend = "stable"
        else:
            performance_trend = "unknown"
        
        # Efficiency ratio
        efficiency_ratio = performance / len(actions) if actions else 0
        
        # Innovation score
        innovation_score = len(emergent_patterns) * 0.1
        for pattern in emergent_patterns:
            innovation_score += pattern.effectiveness * pattern.confidence
        
        # Gaming risk
        gaming_risk = len(gaming_indicators) / 10.0  # Normalize to 0-1
        
        return BehaviorMetrics(
            agent_id=agent_id,
            generation=genome.generation,
            total_actions=len(actions),
            unique_actions=unique_actions,
            action_diversity=action_diversity,
            performance_trend=performance_trend,
            efficiency_ratio=efficiency_ratio,
            innovation_score=innovation_score,
            gaming_risk=min(1.0, gaming_risk),
            emergent_behaviors=[p.description for p in emergent_patterns]
        )
    
    def _extract_emergent_strategies(self,
                                    patterns: List[Pattern],
                                    metrics: BehaviorMetrics) -> List[EmergentStrategy]:
        """Extract high-level strategies from patterns."""
        strategies = []
        
        # Group related patterns
        pattern_groups = self._group_related_patterns(patterns)
        
        for group in pattern_groups:
            if len(group) >= 2:  # Strategy needs multiple patterns
                # Calculate combined effectiveness
                combined_effectiveness = np.mean([p.effectiveness for p in group])
                
                # Generate strategy description
                pattern_types = set(p.pattern_type for p in group)
                if PatternType.OPTIMIZATION in pattern_types:
                    strategy_type = "optimization"
                elif PatternType.BEHAVIORAL in pattern_types:
                    strategy_type = "behavioral"
                else:
                    strategy_type = "mixed"
                
                strategy = EmergentStrategy(
                    strategy_id=f"strategy_{hash(tuple(p.pattern_id for p in group))}",
                    description=f"Emergent {strategy_type} strategy combining {len(group)} patterns",
                    first_observed=datetime.now(),
                    agent_count=1,
                    effectiveness=combined_effectiveness,
                    replicability=self._estimate_replicability(group),
                    components=group,
                    prerequisites=self._extract_prerequisites(group)
                )
                
                strategies.append(strategy)
                
                # Update strategy registry
                if strategy.strategy_id not in self.emergent_strategies:
                    self.emergent_strategies[strategy.strategy_id] = strategy
                    logger.info(f"New emergent strategy discovered: {strategy.description}")
                else:
                    # Update existing strategy
                    existing = self.emergent_strategies[strategy.strategy_id]
                    existing.agent_count += 1
                    existing.effectiveness = (
                        (existing.effectiveness * (existing.agent_count - 1) + 
                         strategy.effectiveness) / existing.agent_count
                    )
                
                # Track adoption
                self.strategy_adoption[strategy.strategy_id].add(metrics.agent_id)
        
        return strategies
    
    def _group_related_patterns(self, patterns: List[Pattern]) -> List[List[Pattern]]:
        """Group patterns that likely form a coherent strategy."""
        if len(patterns) < 2:
            return [[p] for p in patterns]
        
        groups = []
        used = set()
        
        for i, pattern1 in enumerate(patterns):
            if i in used:
                continue
                
            group = [pattern1]
            used.add(i)
            
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if patterns are related
                if self._patterns_related(pattern1, pattern2):
                    group.append(pattern2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _patterns_related(self, p1: Pattern, p2: Pattern) -> bool:
        """Determine if two patterns are related."""
        # Same agent
        if p1.agent_ids and p2.agent_ids:
            if p1.agent_ids & p2.agent_ids:  # Intersection
                return True
        
        # Temporal proximity
        time_diff = abs((p1.last_seen - p2.last_seen).total_seconds())
        if time_diff < 60:  # Within 1 minute
            return True
        
        # Similar context
        shared_context = set(p1.context.keys()) & set(p2.context.keys())
        if len(shared_context) > len(p1.context) * 0.5:
            return True
        
        return False
    
    def _estimate_replicability(self, patterns: List[Pattern]) -> float:
        """Estimate how easily a strategy can be replicated."""
        # Factors affecting replicability
        factors = []
        
        # Complexity - fewer patterns = more replicable
        factors.append(1.0 / (1 + len(patterns)))
        
        # Consistency - higher occurrence = more replicable
        avg_occurrences = np.mean([p.occurrences for p in patterns])
        factors.append(min(1.0, avg_occurrences / 10))
        
        # Effectiveness consistency
        if len(patterns) > 1:
            effectiveness_std = np.std([p.effectiveness for p in patterns])
            factors.append(1.0 / (1 + effectiveness_std))
        
        return np.mean(factors)
    
    def _extract_prerequisites(self, patterns: List[Pattern]) -> Dict[str, Any]:
        """Extract prerequisites for a strategy."""
        prerequisites = {}
        
        # Gene type requirements
        required_genes = set()
        for pattern in patterns:
            if pattern.pattern_type == PatternType.STRUCTURAL:
                required_genes.update(pattern.sequence)
        
        if required_genes:
            prerequisites['required_gene_types'] = list(required_genes)
        
        # Performance threshold
        min_effectiveness = min(p.effectiveness for p in patterns)
        if min_effectiveness > 0:
            prerequisites['min_performance'] = min_effectiveness
        
        # Context requirements
        shared_context = None
        for pattern in patterns:
            if shared_context is None:
                shared_context = set(pattern.context.keys())
            else:
                shared_context &= set(pattern.context.keys())
        
        if shared_context:
            prerequisites['context_requirements'] = list(shared_context)
        
        return prerequisites
    
    def _record_innovation(self,
                          agent_id: str,
                          patterns: List[Pattern],
                          strategies: List[EmergentStrategy]) -> None:
        """Record innovation in timeline."""
        innovation = {
            'timestamp': datetime.now(),
            'agent_id': agent_id,
            'patterns': [p.pattern_id for p in patterns],
            'strategies': [s.strategy_id for s in strategies],
            'description': f"Agent {agent_id} discovered {len(patterns)} patterns"
        }
        
        self.innovation_timeline.append(innovation)
        logger.info(f"Innovation recorded: {innovation['description']}")
    
    def _store_patterns(self, patterns: List[Pattern]) -> None:
        """Store patterns in metrics database."""
        if not self.metrics_db:
            return
        
        # This would interface with the actual database
        # For now, just log
        for pattern in patterns:
            logger.debug(f"Storing pattern {pattern.pattern_id} in metrics database")
    
    def get_innovation_report(self) -> Dict:
        """Generate report on discovered innovations."""
        return {
            'total_emergent_strategies': len(self.emergent_strategies),
            'total_innovations': len(self.innovation_timeline),
            'top_strategies': [
                {
                    'id': s.strategy_id,
                    'description': s.description,
                    'effectiveness': s.effectiveness,
                    'adoption': len(self.strategy_adoption[s.strategy_id]),
                    'replicability': s.replicability
                }
                for s in sorted(
                    self.emergent_strategies.values(),
                    key=lambda x: x.effectiveness * x.agent_count,
                    reverse=True
                )[:10]
            ],
            'recent_innovations': [
                {
                    'timestamp': i['timestamp'].isoformat(),
                    'agent_id': i['agent_id'],
                    'pattern_count': len(i['patterns'])
                }
                for i in self.innovation_timeline[-10:]
            ],
            'adoption_rates': {
                sid: len(agents) 
                for sid, agents in self.strategy_adoption.items()
            }
        }
    
    def export_strategies(self, filepath: str) -> None:
        """Export discovered strategies for reuse."""
        import json
        
        strategies_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_strategies': len(self.emergent_strategies)
            },
            'strategies': [
                {
                    'id': s.strategy_id,
                    'description': s.description,
                    'effectiveness': s.effectiveness,
                    'replicability': s.replicability,
                    'agent_count': s.agent_count,
                    'prerequisites': s.prerequisites,
                    'components': [p.pattern_id for p in s.components]
                }
                for s in self.emergent_strategies.values()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(strategies_data, f, indent=2)
        
        logger.info(f"Exported {len(self.emergent_strategies)} strategies to {filepath}")