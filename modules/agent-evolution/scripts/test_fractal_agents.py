#!/usr/bin/env python3
"""
Test the complete FractalAgent implementation with cellular automata.

Demonstrates Phase 5: Agent Implementation with CA rules, token economics,
and git worktree isolation.
"""

import asyncio
import sys
import tempfile
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Force use of simple database
from dean.repository.simple_metrics_db import SimpleMetricsDatabase
import dean.repository.repository_manager
dean.repository.repository_manager.MetricsDatabase = SimpleMetricsDatabase

# Import DEAN components
from dean.agents import (
    FractalAgent, AgentFactory, AgentConfig, 
    CellularAutomataEngine, GitWorktreeManager,
    AgentState, EvolutionResult, WorktreeConstraints
)
from dean.economy import TokenEconomyManager
from dean.diversity import GeneticDiversityManager, AgentGenome
from dean.patterns import PatternDetector, EmergentBehaviorMonitor
from dean.repository import RepositoryManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def test_complete_fractal_agent_system():
    """Test the complete FractalAgent system with all components."""
    print("Testing Complete FractalAgent System with Cellular Automata")
    print("=" * 60)
    
    # Create temporary database and git repo
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    temp_git_dir = tempfile.mkdtemp(prefix='dean_test_repo_')
    
    try:
        # Initialize git repository
        await _setup_test_git_repo(temp_git_dir)
        
        # Initialize core managers
        print("🔧 Initializing core managers...")
        
        token_manager = TokenEconomyManager(global_budget=10000)
        diversity_manager = GeneticDiversityManager()
        repository = RepositoryManager(f"sqlite:///{db_path}", auto_learn=False)
        
        # Initialize behavior monitoring
        behavior_monitor = EmergentBehaviorMonitor(repository.metrics_db)
        pattern_detector = PatternDetector()
        
        # Initialize CA engine
        ca_engine = CellularAutomataEngine(
            token_manager=token_manager,
            diversity_manager=diversity_manager,
            repository=repository,
            behavior_monitor=behavior_monitor
        )
        
        # Initialize worktree manager
        worktree_manager = GitWorktreeManager(
            base_repo_path=Path(temp_git_dir),
            max_worktrees=10
        )
        
        # Initialize agent factory
        agent_factory = AgentFactory(
            token_manager=token_manager,
            diversity_manager=diversity_manager,
            ca_engine=ca_engine,
            pattern_detector=pattern_detector,
            behavior_monitor=behavior_monitor,
            repository=repository,
            worktree_manager=worktree_manager
        )
        
        print("✓ All managers initialized successfully")
        
        # Test 1: Create individual agents
        print("\n📦 Test 1: Creating individual agents...")
        
        agent_config = AgentConfig(
            agent_id="test_agent_001",
            token_budget=1500,
            initial_genes=[
                {
                    'type': 'strategy',
                    'name': 'test_strategy',
                    'value': {'approach': 'experimental'}
                }
            ]
        )
        
        agent1 = await agent_factory.create_agent(agent_config)
        print(f"✓ Created agent: {agent1.agent_id} (efficiency: {agent1.get_efficiency():.3f})")
        
        # Test 2: Create agent population
        print("\n🏘️  Test 2: Creating diverse agent population...")
        
        base_config = AgentConfig(
            token_budget=1000,
            worktree_enabled=True
        )
        
        population = await agent_factory.create_population(
            population_size=5,
            base_config=base_config,
            diversity_factor=0.4
        )
        
        print(f"✓ Created population of {len(population)} agents")
        for agent in population:
            print(f"  - {agent.agent_id}: level {agent.level}, "
                  f"tokens {agent.token_budget.total}, "
                  f"efficiency {agent.get_efficiency():.3f}")
        
        # Test 3: Evolution cycles with CA rules
        print("\n🧬 Test 3: Running evolution cycles with CA rules...")
        
        evolution_results = []
        for cycle in range(3):
            print(f"\n--- Evolution Cycle {cycle + 1} ---")
            
            # Create environment for evolution
            environment = {
                'population': population,
                'generation': cycle,
                'global_metrics': {
                    'average_efficiency': sum(a.get_efficiency() for a in population) / len(population)
                }
            }
            
            # Evolve each agent
            cycle_results = []
            for agent in population[:]:  # Copy list since it may be modified
                try:
                    result = await agent.evolve(environment)
                    cycle_results.append(result)
                    
                    print(f"  {agent.agent_id}: "
                          f"efficiency {result.efficiency_before:.3f} → {result.efficiency_after:.3f}, "
                          f"tokens used: {result.tokens_consumed}, "
                          f"patterns: {len(result.patterns_discovered)}, "
                          f"rules: {[r.value for r in result.ca_rules_activated]}")
                    
                    if result.children_created:
                        print(f"    → Created children: {result.children_created}")
                    
                    if result.meta_agent_created:
                        print(f"    → Created meta-agent!")
                    
                except Exception as e:
                    print(f"  {agent.agent_id}: Evolution failed - {e}")
            
            evolution_results.extend(cycle_results)
            
            # Update population with any new agents from CA rules
            if hasattr(ca_engine, 'ca_state') and ca_engine.ca_state.agents:
                new_agents = [a for a in ca_engine.ca_state.agents if a not in population]
                population.extend(new_agents)
                if new_agents:
                    print(f"  + Added {len(new_agents)} new agents from CA evolution")
        
        # Test 4: Child agent creation
        print("\n👶 Test 4: Creating child agents...")
        
        if population:
            parent = population[0]
            child = await agent_factory.create_child_agent(
                parent=parent,
                mutation_intensity=0.2,
                token_budget=500
            )
            
            print(f"✓ Created child {child.agent_id} from parent {parent.agent_id}")
            print(f"  Parent children: {parent.children}")
            print(f"  Child efficiency: {child.get_efficiency():.3f}")
        
        # Test 5: Meta-agent creation
        print("\n🧠 Test 5: Creating meta-agent...")
        
        if len(population) >= 2:
            meta_agent = await agent_factory.create_meta_agent(
                base_agents=population[:3],
                meta_level=1,
                token_budget=2000
            )
            
            print(f"✓ Created meta-agent {meta_agent.agent_id} at level {meta_agent.level}")
            print(f"  Meta-agent genes: {len(meta_agent.genome.genes)}")
            print(f"  Meta-agent efficiency: {meta_agent.get_efficiency():.3f}")
        
        # Test 6: Worktree management
        print("\n🌳 Test 6: Testing git worktree management...")
        
        worktree_stats = await worktree_manager.list_active_worktrees()
        print(f"✓ Active worktrees: {len(worktree_stats)}")
        
        for wt in worktree_stats[:3]:  # Show first 3
            print(f"  - Agent {wt['agent_id']}: {wt['file_count']} files, "
                  f"age {wt['age_seconds']:.0f}s")
        
        # Test 7: Performance metrics and repository
        print("\n📊 Test 7: Analyzing performance metrics...")
        
        factory_stats = agent_factory.get_factory_stats()
        print(f"✓ Factory created {factory_stats['agents_created']} agents total")
        
        repo_stats = repository.get_repository_stats(use_cache=False)
        print(f"✓ Repository: {repo_stats.total_agents} agents, "
              f"{repo_stats.total_patterns} patterns recorded")
        
        # Test 8: Evolution analysis
        print("\n🔍 Test 8: Evolution progress analysis...")
        
        if evolution_results:
            successful_evolutions = [r for r in evolution_results if r.success]
            avg_efficiency_gain = sum(
                r.efficiency_after - r.efficiency_before 
                for r in successful_evolutions
            ) / max(len(successful_evolutions), 1)
            
            total_patterns = sum(len(r.patterns_discovered) for r in evolution_results)
            total_children = sum(len(r.children_created) for r in evolution_results)
            
            print(f"✓ Evolution summary:")
            print(f"  - Successful evolutions: {len(successful_evolutions)}/{len(evolution_results)}")
            print(f"  - Average efficiency gain: {avg_efficiency_gain:.4f}")
            print(f"  - Total patterns discovered: {total_patterns}")
            print(f"  - Total children created: {total_children}")
            
            # Show CA rule activations
            rule_counts = {}
            for result in evolution_results:
                for rule in result.ca_rules_activated:
                    rule_counts[rule.value] = rule_counts.get(rule.value, 0) + 1
            
            if rule_counts:
                print(f"  - CA rule activations:")
                for rule, count in rule_counts.items():
                    print(f"    * {rule}: {count} times")
        
        # Cleanup
        print("\n🧹 Cleaning up agents...")
        cleanup_count = 0
        for agent in population + ([child] if 'child' in locals() else []) + ([meta_agent] if 'meta_agent' in locals() else []):
            try:
                await agent.cleanup()
                cleanup_count += 1
            except Exception as e:
                print(f"Warning: Failed to cleanup {agent.agent_id}: {e}")
        
        worktree_cleanup_count = await worktree_manager.cleanup_all_worktrees()
        
        print(f"✓ Cleaned up {cleanup_count} agents and {worktree_cleanup_count} worktrees")
        
        print("\n" + "=" * 60)
        print("✅ All FractalAgent system tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ FractalAgent system test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
        
        try:
            import shutil
            shutil.rmtree(temp_git_dir)
        except:
            pass


async def _setup_test_git_repo(repo_path: str) -> None:
    """Set up a test git repository."""
    repo_path = Path(repo_path)
    
    # Initialize git repo
    proc = await asyncio.create_subprocess_exec(
        'git', 'init',
        cwd=str(repo_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await proc.wait()
    
    # Configure git
    await asyncio.create_subprocess_exec(
        'git', 'config', 'user.name', 'DEAN Test',
        cwd=str(repo_path)
    )
    await asyncio.create_subprocess_exec(
        'git', 'config', 'user.email', 'dean@test.com',
        cwd=str(repo_path)
    )
    
    # Create initial commit
    test_file = repo_path / 'README.md'
    test_file.write_text('# DEAN Test Repository\n\nTest repository for FractalAgent development.\n')
    
    await asyncio.create_subprocess_exec(
        'git', 'add', 'README.md',
        cwd=str(repo_path)
    )
    await asyncio.create_subprocess_exec(
        'git', 'commit', '-m', 'Initial commit',
        cwd=str(repo_path)
    )


if __name__ == "__main__":
    success = asyncio.run(test_complete_fractal_agent_system())
    exit(0 if success else 1)