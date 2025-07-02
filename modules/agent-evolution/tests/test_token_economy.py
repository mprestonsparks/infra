"""
Test suite for Token Economy Service - Validates hard enforcement of budget limits
"""

import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import redis.asyncio as redis
from datetime import datetime

from services.token_economy import app, TokenEconomyService, Base, TokenAllocation


@pytest.fixture
async def test_db():
    """Create test database"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    yield async_session
    
    await engine.dispose()


@pytest.fixture
async def redis_client():
    """Create test Redis client"""
    client = await redis.from_url("redis://localhost:6379/15", decode_responses=True)
    await client.flushdb()
    yield client
    await client.close()


@pytest.fixture
async def token_service(redis_client):
    """Create token service with test budget"""
    service = TokenEconomyService(global_budget=1000.0)
    service.redis_client = redis_client
    await service.redis_client.set("global_budget_remaining", 1000.0)
    yield service


class TestTokenEconomy:
    """Test cases demonstrating hard budget enforcement"""
    
    @pytest.mark.asyncio
    async def test_basic_allocation(self, token_service, test_db):
        """Test basic token allocation with efficiency adjustment"""
        async with test_db() as session:
            # First allocation for new agent (0.5 baseline efficiency)
            request = {
                "agent_id": "agent-001",
                "requested_tokens": 1000.0,
                "task_complexity": 0.5
            }
            
            response = await token_service.allocate_tokens(
                TokenRequest(**request),
                session
            )
            
            # Base allocation = min(1000, 1000) = 1000
            # Efficiency multiplier = 0.5 + 0.5 = 1.0
            # Expected = 1000 * 1.0 = 1000
            assert response.allocated_tokens == 1000.0
            assert response.efficiency_multiplier == 1.0
            assert response.remaining_global_budget == 0.0  # All budget allocated
    
    @pytest.mark.asyncio
    async def test_efficiency_based_allocation(self, token_service, test_db):
        """Test allocation with high efficiency agent"""
        async with test_db() as session:
            # Create historical high-efficiency record
            efficient_agent = TokenAllocation(
                agent_id="agent-002",
                allocated_tokens=500,
                consumed_tokens=500,
                value_generated=10.0,  # Efficiency = 10/500 = 0.02
                efficiency=0.02,
                is_active=False
            )
            session.add(efficient_agent)
            await session.commit()
            
            # Request new allocation
            request = {
                "agent_id": "agent-002",
                "requested_tokens": 800.0
            }
            
            response = await token_service.allocate_tokens(
                TokenRequest(**request),
                session
            )
            
            # Efficiency multiplier = 0.5 + 0.02 = 0.52
            # Expected = 800 * 0.52 = 416
            assert response.allocated_tokens == pytest.approx(416.0, rel=0.01)
            assert response.efficiency_multiplier == pytest.approx(0.52, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_budget_exhaustion_rejection(self, token_service, test_db):
        """Test that allocation is rejected when budget exhausted"""
        async with test_db() as session:
            # Set budget to nearly exhausted
            await token_service.redis_client.set("global_budget_remaining", 50.0)
            
            request = {
                "agent_id": "agent-003",
                "requested_tokens": 1000.0
            }
            
            # Should raise HTTPException with 403 status
            with pytest.raises(Exception) as exc_info:
                await token_service.allocate_tokens(
                    TokenRequest(**request),
                    session
                )
            
            assert "Cannot allocate sufficient tokens" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_hard_stop_on_budget_exhaustion(self, token_service, test_db):
        """Test that agents are ACTUALLY TERMINATED when budget exhausted"""
        async with test_db() as session:
            # First allocate tokens
            await token_service.redis_client.set("global_budget_remaining", 1000.0)
            
            allocation_request = {
                "agent_id": "agent-004",
                "requested_tokens": 100.0
            }
            
            response = await token_service.allocate_tokens(
                TokenRequest(**allocation_request),
                session
            )
            
            assert response.allocated_tokens == 100.0
            
            # Consume most of the budget
            consumption1 = {
                "agent_id": "agent-004",
                "tokens_consumed": 90.0,
                "value_generated": 1.0
            }
            
            result1 = await token_service.update_consumption(
                ConsumptionUpdate(**consumption1),
                session,
                background_tasks=None
            )
            
            assert result1["remaining"] == 10.0
            assert result1["warning"] is not None  # Should warn about low budget
            
            # Try to consume more than remaining - should TERMINATE
            consumption2 = {
                "agent_id": "agent-004",
                "tokens_consumed": 15.0,  # More than 10 remaining
                "value_generated": 0.5
            }
            
            with pytest.raises(Exception) as exc_info:
                await token_service.update_consumption(
                    ConsumptionUpdate(**consumption2),
                    session,
                    background_tasks=None
                )
            
            assert "terminated: Token budget exhausted" in str(exc_info.value)
            
            # Verify agent is marked as terminated in database
            result = await session.execute(
                select(TokenAllocation)
                .where(TokenAllocation.agent_id == "agent-004")
            )
            allocation = result.scalar_one()
            
            assert allocation.is_active == False
            assert "Token budget exhausted" in allocation.termination_reason
            
            # Verify Redis state is cleaned up
            remaining_allocation = await token_service.redis_client.hgetall(
                f"agent_allocation:agent-004"
            )
            assert remaining_allocation == {}  # Should be deleted
    
    @pytest.mark.asyncio
    async def test_global_constraint_enforcement(self, token_service, test_db):
        """Test that global budget constraint is enforced across multiple agents"""
        async with test_db() as session:
            await token_service.redis_client.set("global_budget_remaining", 1000.0)
            
            # Allocate to first agent
            request1 = {
                "agent_id": "agent-005",
                "requested_tokens": 600.0
            }
            
            response1 = await token_service.allocate_tokens(
                TokenRequest(**request1),
                session
            )
            
            assert response1.allocated_tokens == 600.0
            assert response1.remaining_global_budget == 400.0
            
            # Try to allocate more than remaining to second agent
            request2 = {
                "agent_id": "agent-006",
                "requested_tokens": 800.0
            }
            
            response2 = await token_service.allocate_tokens(
                TokenRequest(**request2),
                session
            )
            
            # Should only get what's remaining (400), adjusted by efficiency
            # New agent efficiency = 0.5, so 400 * 1.0 = 400
            assert response2.allocated_tokens == 400.0
            assert response2.remaining_global_budget == 0.0
            assert response2.warning is not None
    
    @pytest.mark.asyncio
    async def test_efficiency_tracking(self, token_service, test_db):
        """Test that efficiency is correctly calculated and tracked"""
        async with test_db() as session:
            await token_service.redis_client.set("global_budget_remaining", 1000.0)
            
            # Allocate tokens
            allocation_request = {
                "agent_id": "agent-007",
                "requested_tokens": 200.0
            }
            
            await token_service.allocate_tokens(
                TokenRequest(**allocation_request),
                session
            )
            
            # Update consumption with value
            consumption = {
                "agent_id": "agent-007",
                "tokens_consumed": 100.0,
                "value_generated": 5.0  # Efficiency = 5/100 = 0.05
            }
            
            result = await token_service.update_consumption(
                ConsumptionUpdate(**consumption),
                session,
                background_tasks=None
            )
            
            assert result["efficiency"] == pytest.approx(0.05, rel=0.01)
            
            # Verify database update
            db_result = await session.execute(
                select(TokenAllocation)
                .where(TokenAllocation.agent_id == "agent-007")
            )
            allocation = db_result.scalar_one()
            
            assert allocation.efficiency == pytest.approx(0.05, rel=0.01)
            assert allocation.value_generated == 5.0
            assert allocation.consumed_tokens == 100.0


class TestAPIEndpoints:
    """Test the FastAPI endpoints"""
    
    @pytest.mark.asyncio
    async def test_allocation_endpoint(self):
        """Test the allocation API endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            request_data = {
                "agent_id": "test-agent-001",
                "requested_tokens": 500.0,
                "task_complexity": 0.7
            }
            
            # Note: This would need proper test setup with database and Redis
            # This is a template for integration testing
            
    @pytest.mark.asyncio
    async def test_budget_status_endpoint(self):
        """Test the budget status endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # This would need proper test setup
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])