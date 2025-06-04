"""
Test token enforcement mechanisms.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dean.economy import TokenEconomyManager, TokenBudget


class TestTokenLimits:
    """Test suite for token limit enforcement."""
    
    def test_budget_initialization(self):
        """Test token budget initialization."""
        budget = TokenBudget(total=1000)
        assert budget.total == 1000
        assert budget.used == 0
        assert budget.reserved == 0
        assert budget.available == 1000
        assert budget.utilization_rate == 0.0
    
    def test_token_consumption(self):
        """Test basic token consumption."""
        budget = TokenBudget(total=1000)
        
        # Normal consumption
        budget.consume(100)
        assert budget.used == 100
        assert budget.available == 900
        assert budget.utilization_rate == 0.1
        
        # Multiple consumptions
        budget.consume(200)
        assert budget.used == 300
        assert budget.available == 700
    
    def test_insufficient_tokens(self):
        """Test handling of insufficient tokens."""
        budget = TokenBudget(total=100)
        budget.consume(90)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Insufficient tokens"):
            budget.consume(20)
    
    def test_token_reservation(self):
        """Test token reservation mechanism."""
        budget = TokenBudget(total=1000)
        
        # Reserve tokens
        budget.reserve(300)
        assert budget.reserved == 300
        assert budget.available == 700
        
        # Try to consume more than available
        with pytest.raises(ValueError):
            budget.consume(800)
        
        # Release reservation
        budget.release_reservation(100)
        assert budget.reserved == 200
        assert budget.available == 800
    
    def test_economy_manager_initialization(self):
        """Test TokenEconomyManager initialization."""
        manager = TokenEconomyManager(global_budget=10000)
        
        assert manager.global_budget == 10000
        assert manager.safety_reserve == 1000  # 10% default
        assert manager.available_budget == 9000
    
    def test_agent_allocation(self):
        """Test token allocation to agents."""
        manager = TokenEconomyManager(global_budget=10000)
        
        # First agent gets base allocation
        allocation1 = manager.allocate_tokens("agent1")
        assert allocation1 > 0
        assert allocation1 <= manager.available_budget
        
        # Second agent with good efficiency
        allocation2 = manager.allocate_tokens("agent2", historical_efficiency=0.8)
        assert allocation2 > 0
        
        # Verify budgets were created
        assert "agent1" in manager.agent_allocations
        assert "agent2" in manager.agent_allocations
    
    def test_consumption_tracking(self):
        """Test consumption tracking and efficiency calculation."""
        manager = TokenEconomyManager(global_budget=10000)
        manager.allocate_tokens("agent1")
        
        # Track consumption
        manager.track_consumption("agent1", tokens_used=100, value_generated=150)
        
        metrics = manager.get_agent_metrics("agent1")
        assert metrics['tokens_used'] == 100
        assert len(manager.efficiency_history["agent1"]) == 1
        assert manager.efficiency_history["agent1"][0] == 1.5  # 150/100
    
    def test_budget_exceeded_termination(self):
        """Test agent termination on budget exceeded."""
        manager = TokenEconomyManager(global_budget=1000)
        manager.allocate_tokens("agent1")
        
        budget = manager.agent_allocations["agent1"]
        total_budget = budget.total
        
        # Use most of the budget
        manager.track_consumption("agent1", tokens_used=total_budget - 10, value_generated=100)
        
        # Exceed budget
        manager.track_consumption("agent1", tokens_used=20, value_generated=10)
        
        # Agent should be terminated
        assert "agent1" in manager._terminated_agents
    
    def test_inefficiency_termination(self):
        """Test agent termination for persistent inefficiency."""
        manager = TokenEconomyManager(global_budget=10000)
        manager.allocate_tokens("agent1")
        
        # Track multiple inefficient operations
        for i in range(5):
            manager.track_consumption("agent1", tokens_used=100, value_generated=5)
        
        # Check if terminated for inefficiency
        metrics = manager.get_agent_metrics("agent1")
        avg_efficiency = metrics['average_efficiency']
        assert avg_efficiency < 0.1
        
        # Should be terminated if utilization > 50%
        if metrics['utilization_rate'] > 0.5:
            assert "agent1" in manager._terminated_agents
    
    def test_global_metrics(self):
        """Test global metrics calculation."""
        manager = TokenEconomyManager(global_budget=10000)
        
        # Allocate to multiple agents
        manager.allocate_tokens("agent1")
        manager.allocate_tokens("agent2", historical_efficiency=0.9)
        
        # Track some consumption
        manager.track_consumption("agent1", tokens_used=100, value_generated=120)
        manager.track_consumption("agent2", tokens_used=200, value_generated=400)
        
        metrics = manager.get_global_metrics()
        assert metrics['global_budget'] == 10000
        assert metrics['active_agents'] == 2
        assert metrics['total_used'] == 300
        assert metrics['average_efficiency'] > 0
    
    def test_roi_calculation(self):
        """Test ROI calculation."""
        manager = TokenEconomyManager(global_budget=10000)
        
        assert manager.calculate_roi(100, 150) == 1.5
        assert manager.calculate_roi(0, 100) == 0.0
        assert manager.calculate_roi(50, 0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])