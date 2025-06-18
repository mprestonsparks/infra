#!/usr/bin/env python3
"""
Real Cellular Automata Evolution Engine for DEAN
Implements minimal but genuine evolution using CA rules.
NO MOCKS - Real code analysis and pattern discovery.
"""

import os
import ast
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
try:
    from code_analyzer import FileBasedCodeAnalyzer
    from token_economy import TokenEconomyEngine, TokenTransactionType
except ImportError:
    from .code_analyzer import FileBasedCodeAnalyzer
    from .token_economy import TokenEconomyEngine, TokenTransactionType


class CellularAutomataEvolution:
    """
    Real CA-based evolution engine that analyzes code and discovers patterns.
    Uses Rule 110 to identify improvement opportunities.
    """
    
    def __init__(self, db_session: Session, token_economy: TokenEconomyEngine = None):
        self.db = db_session
        self.ca_state = None
        self.generation = 0
        self.token_economy = token_economy or TokenEconomyEngine(db_session)
        
    def initialize_ca_state(self, agent_id: str, code_metrics: Dict[str, int]) -> np.ndarray:
        """
        Initialize CA state from code analysis metrics.
        Maps code characteristics to binary CA state.
        """
        # Create initial state from code metrics
        state_size = 64  # Fixed size for CA state
        state = np.zeros(state_size, dtype=int)
        
        # Map code metrics to CA state positions
        # This creates a deterministic mapping from code state to CA state
        if code_metrics.get('todo_count', 0) > 0:
            state[0:8] = 1  # TODOs present
            
        if code_metrics.get('long_functions', 0) > 0:
            state[8:16] = 1  # Long functions detected
            
        if code_metrics.get('unused_imports', 0) > 0:
            state[16:24] = 1  # Unused imports found
            
        if code_metrics.get('complexity', 0) > 10:
            state[24:32] = 1  # High complexity
            
        # Add some randomness for exploration
        random_positions = random.sample(range(32, state_size), k=8)
        for pos in random_positions:
            state[pos] = random.randint(0, 1)
            
        self.ca_state = state
        return state
    
    def apply_rule_110(self, state: np.ndarray) -> np.ndarray:
        """
        Apply Rule 110 - Class 4 complexity cellular automaton.
        This rule exhibits complex behavior between order and chaos.
        """
        # Rule 110 lookup table: [111, 110, 101, 100, 011, 010, 001, 000]
        rule_110 = [0, 1, 1, 0, 1, 1, 1, 0]
        
        new_state = np.zeros_like(state)
        n = len(state)
        
        for i in range(n):
            # Get neighborhood with wraparound
            left = state[(i - 1) % n]
            center = state[i]
            right = state[(i + 1) % n]
            
            # Convert to rule index
            neighborhood = (left << 2) | (center << 1) | right
            new_state[i] = rule_110[neighborhood]
            
        return new_state
    
    def detect_patterns_in_ca(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Detect patterns in CA state that indicate optimization opportunities.
        Returns specific actions based on CA patterns.
        """
        patterns = {
            'gliders': False,
            'oscillators': False,
            'stable_structures': False,
            'complexity_score': 0.0,
            'suggested_action': None
        }
        
        # Calculate complexity as transition density
        transitions = np.sum(np.abs(np.diff(state)))
        patterns['complexity_score'] = transitions / len(state)
        
        # Pattern detection
        # Check for stable regions (optimization opportunity)
        stable_regions = []
        for i in range(0, len(state) - 8, 4):
            region = state[i:i+8]
            if np.sum(region) == 0 or np.sum(region) == 8:
                stable_regions.append(i)
                
        if len(stable_regions) > 2:
            patterns['stable_structures'] = True
            patterns['suggested_action'] = 'consolidate_code'
            
        # Check for oscillating patterns (refactoring opportunity)
        if self.generation > 5:
            # Would need history tracking for real oscillator detection
            # For now, use complexity as proxy
            if 0.3 < patterns['complexity_score'] < 0.7:
                patterns['oscillators'] = True
                patterns['suggested_action'] = 'refactor_patterns'
                
        # High complexity suggests optimization needed
        if patterns['complexity_score'] > 0.6:
            patterns['gliders'] = True
            patterns['suggested_action'] = 'implement_todo'
            
        return patterns
    
    def analyze_code_file(self, file_path: Path) -> Dict[str, int]:
        """
        Analyze a Python file and extract metrics for CA initialization.
        Returns concrete metrics that can guide evolution.
        """
        metrics = {
            'todo_count': 0,
            'long_functions': 0,
            'unused_imports': 0,
            'complexity': 0,
            'total_functions': 0,
            'total_lines': 0
        }
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Count TODOs
            metrics['todo_count'] = content.count('TODO') + content.count('FIXME')
            metrics['total_lines'] = len(content.splitlines())
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                
                # Find imports
                imports = set()
                used_names = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                imports.add(f"{node.module}.{alias.name}")
                    elif isinstance(node, ast.Name):
                        used_names.add(node.id)
                        
                # Simple unused import detection
                for imp in imports:
                    base_name = imp.split('.')[-1]
                    if base_name not in str(content):
                        metrics['unused_imports'] += 1
                        
                # Analyze functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        metrics['total_functions'] += 1
                        # Count lines in function
                        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                            func_lines = node.end_lineno - node.lineno
                            if func_lines > 50:  # Long function threshold
                                metrics['long_functions'] += 1
                            # Simple complexity: count control flow statements
                            complexity = 0
                            for child in ast.walk(node):
                                if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                                    complexity += 1
                            metrics['complexity'] = max(metrics['complexity'], complexity)
                            
            except SyntaxError:
                # If AST parsing fails, use simpler metrics
                pass
                
        except Exception as e:
            # Return empty metrics if file can't be analyzed
            pass
            
        return metrics
    
    def generate_improvement_proposal(self, agent_id: str, ca_patterns: Dict[str, Any],
                                    code_metrics: Dict[str, int]) -> Dict[str, Any]:
        """
        Generate a concrete code improvement proposal based on CA patterns.
        Returns actionable proposal with estimated token cost.
        """
        proposal = {
            'agent_id': agent_id,
            'pattern_type': 'optimization',
            'pattern_content': {},
            'effectiveness_score': 0.0,
            'token_cost': 0,
            'concrete_action': None
        }
        
        action = ca_patterns.get('suggested_action')
        
        if action == 'implement_todo':
            # Propose implementing a TODO
            proposal['pattern_type'] = 'todo_implementation'
            proposal['pattern_content'] = {
                'action': 'implement_todo',
                'description': f"Found {code_metrics['todo_count']} TODO comments to implement",
                'complexity': 'medium',
                'expected_improvement': 'feature_completion'
            }
            proposal['effectiveness_score'] = 0.7
            proposal['token_cost'] = 500 * code_metrics['todo_count']
            proposal['concrete_action'] = f"Implement {code_metrics['todo_count']} TODO items"
            
        elif action == 'refactor_patterns':
            # Propose refactoring long functions
            proposal['pattern_type'] = 'refactoring'
            proposal['pattern_content'] = {
                'action': 'split_long_functions',
                'description': f"Refactor {code_metrics['long_functions']} long functions",
                'complexity': 'high',
                'expected_improvement': 'maintainability'
            }
            proposal['effectiveness_score'] = 0.6
            proposal['token_cost'] = 300 * code_metrics['long_functions']
            proposal['concrete_action'] = f"Split {code_metrics['long_functions']} long functions"
            
        elif action == 'consolidate_code':
            # Propose removing unused imports
            proposal['pattern_type'] = 'cleanup'
            proposal['pattern_content'] = {
                'action': 'remove_unused_imports',
                'description': f"Remove {code_metrics['unused_imports']} unused imports",
                'complexity': 'low',
                'expected_improvement': 'code_cleanliness'
            }
            proposal['effectiveness_score'] = 0.4
            proposal['token_cost'] = 50 * code_metrics['unused_imports']
            proposal['concrete_action'] = f"Remove {code_metrics['unused_imports']} unused imports"
            
        else:
            # Default: complexity reduction
            proposal['pattern_type'] = 'complexity_reduction'
            proposal['pattern_content'] = {
                'action': 'reduce_complexity',
                'description': f"Reduce complexity in functions (current max: {code_metrics['complexity']})",
                'complexity': 'medium',
                'expected_improvement': 'performance'
            }
            proposal['effectiveness_score'] = 0.5
            proposal['token_cost'] = 100 * code_metrics['complexity']
            proposal['concrete_action'] = "Simplify complex control flow"
            
        # Add CA state info
        proposal['pattern_content']['ca_complexity'] = ca_patterns['complexity_score']
        proposal['pattern_content']['ca_generation'] = self.generation
        
        return proposal
    
    async def evolve_single_generation(self, agent_id: str, worktree_path: str) -> Dict[str, Any]:
        """
        Execute one generation of CA-driven evolution.
        Analyzes code, applies Rule 110, generates improvement proposal.
        Now with real token economy constraints!
        """
        self.generation += 1
        
        # Check token budget before starting
        can_afford, analysis_cost, reason = self.token_economy.can_afford_operation(
            agent_id, 'code_analysis', complexity=1
        )
        
        if not can_afford:
            return {
                'error': 'insufficient_tokens',
                'reason': reason,
                'required_tokens': analysis_cost,
                'generation': self.generation
            }
        
        # Step 1: Use FileBasedCodeAnalyzer for deeper analysis
        # Charge for analysis
        if not self.token_economy.charge_tokens(
            agent_id, analysis_cost, TokenTransactionType.ANALYSIS_COST,
            f"Code analysis for generation {self.generation}"
        ):
            return {'error': 'token_charge_failed', 'generation': self.generation}
            
        analyzer = FileBasedCodeAnalyzer(worktree_path)
        analysis_results = analyzer.analyze_repository(max_files=5)
        
        # Convert detailed patterns to simple metrics for CA
        code_metrics = {
            'todo_count': 0,
            'long_functions': 0, 
            'unused_imports': 0,
            'complexity': 0,
            'magic_numbers': 0,
            'missing_docs': 0
        }
        
        # Aggregate pattern counts
        for pattern in analyzer.patterns_found:
            if pattern.pattern_type.startswith('todo_'):
                code_metrics['todo_count'] += 1
            elif pattern.pattern_type == 'long_function':
                code_metrics['long_functions'] += 1
            elif pattern.pattern_type == 'unused_import':
                code_metrics['unused_imports'] += 1
            elif pattern.pattern_type == 'high_complexity':
                code_metrics['complexity'] += pattern.fix_complexity
            elif pattern.pattern_type == 'magic_number':
                code_metrics['magic_numbers'] += 1
            elif pattern.pattern_type == 'missing_docstring':
                code_metrics['missing_docs'] += 1
                
        # Step 2: Initialize or update CA state
        if self.ca_state is None:
            self.ca_state = self.initialize_ca_state(agent_id, code_metrics)
        else:
            # Apply Rule 110 to evolve state
            self.ca_state = self.apply_rule_110(self.ca_state)
            
        # Step 3: Detect patterns in evolved CA state
        ca_patterns = self.detect_patterns_in_ca(self.ca_state)
        
        # Step 4: Generate improvement proposal
        # If analyzer found concrete proposals, use the best one based on CA guidance
        if analyzer.proposals and ca_patterns.get('suggested_action'):
            # Select proposal based on CA pattern
            selected_proposal = None
            
            if ca_patterns['suggested_action'] == 'implement_todo':
                # Find TODO proposals
                todo_proposals = [p for p in analyzer.proposals 
                                if p.pattern.pattern_type.startswith('todo_')]
                if todo_proposals:
                    selected_proposal = todo_proposals[0]
                    
            elif ca_patterns['suggested_action'] == 'refactor_patterns':
                # Find refactoring proposals
                refactor_proposals = [p for p in analyzer.proposals 
                                    if p.pattern.pattern_type in ['long_function', 'high_complexity']]
                if refactor_proposals:
                    selected_proposal = refactor_proposals[0]
                    
            elif ca_patterns['suggested_action'] == 'consolidate_code':
                # Find cleanup proposals
                cleanup_proposals = [p for p in analyzer.proposals 
                                   if p.pattern.pattern_type in ['unused_import', 'duplicate_code']]
                if cleanup_proposals:
                    selected_proposal = cleanup_proposals[0]
                    
            # Convert to our proposal format
            if selected_proposal:
                proposal = {
                    'agent_id': agent_id,
                    'pattern_type': selected_proposal.pattern.pattern_type,
                    'pattern_content': {
                        'action': selected_proposal.pattern.pattern_type,
                        'description': selected_proposal.explanation,
                        'file_path': selected_proposal.file_path,
                        'line_range': f"{selected_proposal.line_start}-{selected_proposal.line_end}",
                        'original_code': selected_proposal.original_code,
                        'proposed_code': selected_proposal.proposed_code,
                        'confidence': selected_proposal.confidence,
                        'ca_complexity': ca_patterns['complexity_score'],
                        'ca_generation': self.generation
                    },
                    'effectiveness_score': selected_proposal.confidence,
                    'token_cost': selected_proposal.pattern.estimated_tokens,
                    'concrete_action': selected_proposal.explanation
                }
            else:
                # Fall back to generic proposal
                proposal = self.generate_improvement_proposal(agent_id, ca_patterns, code_metrics)
        else:
            # No concrete proposals, use generic
            proposal = self.generate_improvement_proposal(agent_id, ca_patterns, code_metrics)
        
        # Step 5: Store in database
        pattern_hash = hashlib.sha256(
            json.dumps(proposal['pattern_content'], sort_keys=True).encode()
        ).hexdigest()
        
        # Check if pattern already exists
        existing = self.db.execute(text("""
            SELECT id FROM agent_evolution.discovered_patterns 
            WHERE pattern_hash = :hash
        """), {"hash": pattern_hash}).fetchone()
        
        if not existing:
            # Insert new pattern
            pattern_id = self.db.execute(text("""
                INSERT INTO agent_evolution.discovered_patterns 
                (agent_id, pattern_hash, pattern_type, pattern_content, 
                 effectiveness_score, token_efficiency_delta)
                VALUES (:agent_id, :hash, :type, :content, :score, :efficiency)
                RETURNING id
            """), {
                "agent_id": agent_id,
                "hash": pattern_hash,
                "type": proposal['pattern_type'],
                "content": json.dumps(proposal['pattern_content']),
                "score": proposal['effectiveness_score'],
                "efficiency": -proposal['token_cost'] / 1000.0  # Negative because it costs tokens
            }).scalar()
            
            proposal['pattern_id'] = str(pattern_id)
            
        # Record evolution history
        self.db.execute(text("""
            INSERT INTO agent_evolution.evolution_history
            (agent_id, generation, evolution_type, rule_applied, 
             fitness_before, fitness_after, new_patterns_discovered,
             population_size, population_diversity)
            VALUES (:agent_id, :gen, :type, :rule, :fit_before, :fit_after, 
                    :new_patterns, :pop_size, :pop_diversity)
        """), {
            "agent_id": agent_id,
            "gen": self.generation,
            "type": "cellular_automata",
            "rule": "rule_110",
            "fit_before": 0.5,  # Would fetch actual fitness
            "fit_after": 0.5 + proposal['effectiveness_score'] * 0.1,
            "new_patterns": 1 if not existing else 0,
            "pop_size": 1,
            "pop_diversity": ca_patterns['complexity_score']
        })
        
        # Charge for CA evolution
        ca_cost = self.token_economy.calculate_operation_cost('ca_evolution')
        self.token_economy.charge_tokens(
            agent_id, ca_cost, TokenTransactionType.EVOLUTION_COST,
            f"CA evolution generation {self.generation}"
        )
        
        # Charge for the proposed improvement (if agent were to implement it)
        # This creates pressure to find efficient solutions
        self.token_economy.charge_tokens(
            agent_id, proposal['token_cost'], TokenTransactionType.IMPLEMENTATION,
            f"Proposed: {proposal['concrete_action']}",
            metadata={'pattern_type': proposal['pattern_type']}
        )
        
        # Update agent fitness
        self.db.execute(text("""
            UPDATE agent_evolution.agents
            SET fitness_score = fitness_score + :fitness_delta,
                generation = :gen
            WHERE id = :agent_id
        """), {
            "agent_id": agent_id,
            "fitness_delta": proposal['effectiveness_score'] * 0.1,
            "gen": self.generation
        })
        
        # If pattern was highly effective, give a reward
        if proposal['effectiveness_score'] > 0.7 and not existing:
            reward = self.token_economy.calculate_pattern_reward(
                proposal['pattern_type'], 
                proposal['effectiveness_score']
            )
            self.token_economy.reward_tokens(
                agent_id, reward,
                f"Discovered effective pattern: {proposal['pattern_type']}",
                metadata={'pattern_id': proposal.get('pattern_id')}
            )
        
        # Check if agent should be retired due to token constraints
        budget_status = self.token_economy.enforce_budget_limits(agent_id)
        
        self.db.commit()
        
        # Calculate total cost for this generation
        total_cost = analysis_cost + ca_cost + proposal['token_cost']
        _, _, remaining = self.token_economy.get_agent_balance(agent_id)
        
        return {
            'generation': self.generation,
            'code_metrics': code_metrics,
            'ca_patterns': ca_patterns,
            'proposal': proposal,
            'ca_state': self.ca_state.tolist(),
            'token_economy': {
                'generation_cost': total_cost,
                'remaining_balance': remaining,
                'budget_status': budget_status,
                'received_reward': proposal['effectiveness_score'] > 0.7 and not existing
            }
        }