"""DEAN agent implementation with cellular automata rules."""

from .fractal_agent import FractalAgent, AgentState, EvolutionResult
from .cellular_automata import CellularAutomataEngine, CARule, Neighborhood, CAState
from .worktree_manager import GitWorktreeManager, WorktreeConstraints
from .agent_factory import AgentFactory, AgentConfig

__all__ = [
    'FractalAgent',
    'AgentState', 
    'EvolutionResult',
    'CellularAutomataEngine',
    'CARule',
    'Neighborhood',
    'CAState',
    'GitWorktreeManager',
    'WorktreeConstraints',
    'AgentFactory',
    'AgentConfig'
]