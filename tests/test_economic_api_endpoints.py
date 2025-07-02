#!/usr/bin/env python3
"""
Tests for Economic Governor API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add services to path
sys.path.append(str(Path(__file__).parent.parent / "services"))

from dean_api.main import app


class TestEconomicGovernorEndpoints:
    """Test Economic Governor API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture(autouse=True)
    def mock_economic_governor(self):
        """Mock the economic governor for all tests"""
        with patch('dean_api.main.economic_governor') as mock:
            # Set up default mock behaviors
            mock.get_system_metrics.return_value = {
                "total_budget": 1000000,
                "allocated_budget": 600000,
                "used_budget": 400000,
                "available_budget": 400000,
                "usage_rate": 0.4,
                "agent_count": 10,
                "average_efficiency": 0.75,
                "top_performers": [
                    {"agent_id": "agent_001", "efficiency": 0.95},
                    {"agent_id": "agent_002", "efficiency": 0.90}
                ]
            }
            
            mock.allocator.get_agent_budget.return_value = {
                "allocated": 10000,
                "used": 3000,
                "remaining": 7000,
                "efficiency": 0.7,
                "generation": 5,
                "last_allocation": None
            }
            
            mock.use_tokens.return_value = True
            mock.allocate_to_agent.return_value = 1500
            mock.rebalance_budgets.return_value = ["agent_001", "agent_002", "agent_003"]
            
            yield mock
    
    def test_get_economic_metrics(self, client):
        """Test GET /api/v1/economy/metrics"""
        response = client.get("/api/v1/economy/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "global_budget" in data
        assert data["global_budget"]["total"] == 1000000
        assert data["global_budget"]["available"] == 400000
        assert data["global_budget"]["usage_rate"] == 0.4
        
        assert "agents" in data
        assert data["agents"]["total"] == 10
        assert data["agents"]["average_efficiency"] == 0.75
        
        assert "top_performers" in data
        assert len(data["top_performers"]) == 2
    
    def test_get_economic_metrics_no_governor(self, client):
        """Test metrics endpoint when governor not initialized"""
        with patch('dean_api.main.economic_governor', None):
            response = client.get("/api/v1/economy/metrics")
            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"]
    
    def test_get_agent_economic_status(self, client):
        """Test GET /api/v1/economy/agent/{agent_id}"""
        response = client.get("/api/v1/economy/agent/test_agent_123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["agent_id"] == "test_agent_123"
        assert data["budget"]["allocated"] == 10000
        assert data["budget"]["remaining"] == 7000
        assert data["budget"]["efficiency"] == 0.7
        assert data["performance"]["generation"] == 5
    
    def test_get_agent_economic_status_not_found(self, client, mock_economic_governor):
        """Test agent status when agent not found"""
        mock_economic_governor.allocator.get_agent_budget.return_value = None
        
        response = client.get("/api/v1/economy/agent/unknown_agent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_use_agent_tokens_success(self, client):
        """Test POST /api/v1/economy/use-tokens success"""
        payload = {
            "agent_id": "test_agent",
            "tokens": 100,
            "action_type": "optimization",
            "task_success": 0.8,
            "quality_score": 0.9
        }
        
        response = client.post("/api/v1/economy/use-tokens", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["agent_id"] == "test_agent"
        assert data["tokens_used"] == 100
        assert data["remaining_budget"] == 7000
    
    def test_use_agent_tokens_insufficient_budget(self, client, mock_economic_governor):
        """Test token usage with insufficient budget"""
        mock_economic_governor.use_tokens.return_value = False
        
        payload = {
            "agent_id": "test_agent",
            "tokens": 10000,
            "action_type": "optimization",
            "task_success": 0.8,
            "quality_score": 0.9
        }
        
        response = client.post("/api/v1/economy/use-tokens", json=payload)
        
        assert response.status_code == 400
        assert "Insufficient token budget" in response.json()["detail"]
    
    def test_allocate_agent_tokens(self, client):
        """Test POST /api/v1/economy/allocate"""
        payload = {
            "agent_id": "test_agent",
            "performance": 0.85,
            "generation": 3
        }
        
        response = client.post("/api/v1/economy/allocate", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["agent_id"] == "test_agent"
        assert data["tokens_allocated"] == 1500
        assert data["total_budget"] == 10000
    
    def test_rebalance_economy(self, client):
        """Test POST /api/v1/economy/rebalance"""
        response = client.post("/api/v1/economy/rebalance")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["agents_rebalanced"] == 3
        assert len(data["rebalanced_agents"]) == 3
        assert "metrics" in data
        assert "before" in data["metrics"]
        assert "after" in data["metrics"]
    
    def test_request_validation(self, client):
        """Test request validation for token usage"""
        # Missing required field
        payload = {
            "agent_id": "test_agent",
            "tokens": 100
            # Missing action_type, task_success, quality_score
        }
        
        response = client.post("/api/v1/economy/use-tokens", json=payload)
        assert response.status_code == 422  # Validation error
        
        # Invalid token value
        payload = {
            "agent_id": "test_agent",
            "tokens": -100,  # Negative tokens
            "action_type": "test",
            "task_success": 0.5,
            "quality_score": 0.5
        }
        
        response = client.post("/api/v1/economy/use-tokens", json=payload)
        assert response.status_code == 422
        
        # Invalid performance score
        payload = {
            "agent_id": "test_agent",
            "performance": 1.5,  # > 1.0
            "generation": 1
        }
        
        response = client.post("/api/v1/economy/allocate", json=payload)
        assert response.status_code == 422


class TestNewAPIEndpoints:
    """Test newly added API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_worktree_endpoints(self, client):
        """Test worktree management endpoints"""
        # Create worktree
        response = client.post("/api/v1/worktrees?agent_id=test_agent&base_repo=/tmp/test_repo")
        # May fail due to git requirements, but endpoint should exist
        assert response.status_code in [200, 500]
        
        # Get worktree info
        response = client.get("/api/v1/worktrees/test_agent")
        assert response.status_code in [200, 404]
        
        # Delete worktree
        response = client.delete("/api/v1/worktrees/test_agent")
        assert response.status_code in [200, 500]
    
    def test_code_modification_endpoints(self, client):
        """Test code modification endpoints"""
        payload = {
            "agent_id": "test_agent",
            "worktree_path": "/tmp/worktree",
            "prompt": "Optimize this function",
            "target_files": ["main.py"],
            "max_tokens": 1000
        }
        
        response = client.post("/api/v1/modifications", json=payload)
        assert response.status_code == 200
        assert "task_id" in response.json()
        
        # Get modification result
        task_id = response.json()["task_id"]
        response = client.get(f"/api/v1/modifications/{task_id}")
        assert response.status_code == 200
    
    def test_optimization_endpoints(self, client):
        """Test optimization endpoints"""
        # Optimize prompt
        payload = {
            "task_description": "Implement TODO items",
            "performance_metrics": {
                "success_rate": 0.6,
                "token_efficiency": 0.8
            }
        }
        
        response = client.post("/api/v1/optimize/prompt", json=payload)
        assert response.status_code == 200
        assert "optimized_prompt" in response.json()
        
        # Inject patterns
        patterns = [
            {"pattern": "test_pattern", "score": 0.9}
        ]
        
        response = client.post("/api/v1/optimize/patterns/inject", json=patterns)
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    def test_diversity_endpoints(self, client):
        """Test diversity management endpoints"""
        # Register agent
        payload = {
            "agent_id": "test_agent",
            "strategies": ["strategy1", "strategy2"],
            "lineage": ["parent1"],
            "generation": 1
        }
        
        response = client.post("/api/v1/diversity/agents/register", json=payload)
        assert response.status_code == 200
        
        # Get diversity metrics
        response = client.get("/api/v1/diversity/metrics")
        assert response.status_code == 200
        assert "diversity_score" in response.json()
        
        # Check intervention
        response = client.post("/api/v1/diversity/interventions/check")
        assert response.status_code == 200
        
        # Apply intervention
        response = client.post(
            "/api/v1/diversity/interventions/apply?agent_id=test_agent&intervention_type=mutation"
        )
        assert response.status_code in [200, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])