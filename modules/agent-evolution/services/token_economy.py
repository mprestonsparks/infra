"""
Token Economy Service - Enforces strict token budget constraints with real-time tracking.

This service implements the economic governance layer for DEAN agents, ensuring:
- Hard enforcement of token budgets (agents STOP when budget exhausted)
- Performance-based allocation with efficiency metrics
- Real-time Prometheus metrics and database tracking
- Global budget constraints across all agents
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple
from decimal import Decimal
import os

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, select, update
from sqlalchemy.orm import declarative_base
import redis.asyncio as redis
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@postgres:5432/agent_evolution")
engine = create_async_engine(DATABASE_URL, echo=False, pool_size=20, max_overflow=30)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

Base = declarative_base()

# Redis setup for real-time state
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Prometheus Metrics
tokens_allocated_total = Counter(
    'dean_tokens_allocated_total',
    'Total tokens allocated to agents',
    ['agent_id']
)

tokens_consumed_total = Counter(
    'dean_tokens_consumed_total',
    'Total tokens consumed by agents',
    ['agent_id']
)

token_efficiency_histogram = Histogram(
    'dean_token_efficiency',
    'Token efficiency distribution (value per token)',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

active_token_allocation = Gauge(
    'dean_active_token_allocation',
    'Current active token allocation',
    ['agent_id']
)

global_budget_remaining = Gauge(
    'dean_global_budget_remaining',
    'Remaining tokens in global budget'
)

agents_terminated_budget = Counter(
    'dean_agents_terminated_budget',
    'Agents terminated due to budget exhaustion'
)

allocation_rejections = Counter(
    'dean_allocation_rejections',
    'Token allocation requests rejected',
    ['reason']
)


class TokenAllocation(Base):
    """Database model for token allocations"""
    __tablename__ = "token_allocations"
    __table_args__ = {"schema": "agent_evolution"}
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String, nullable=False, index=True)
    allocated_tokens = Column(Float, nullable=False)
    consumed_tokens = Column(Float, default=0.0)
    value_generated = Column(Float, default=0.0)
    efficiency = Column(Float, default=0.0)
    allocation_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_update = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    termination_reason = Column(String, nullable=True)


# Pydantic models
class TokenRequest(BaseModel):
    agent_id: str
    requested_tokens: float = Field(gt=0, le=10000)
    task_complexity: float = Field(ge=0.1, le=1.0, default=0.5)
    
    @validator('requested_tokens')
    def validate_tokens(cls, v):
        if v <= 0:
            raise ValueError("Requested tokens must be positive")
        return v


class TokenResponse(BaseModel):
    agent_id: str
    allocated_tokens: float
    efficiency_multiplier: float
    remaining_global_budget: float
    warning: Optional[str] = None


class ConsumptionUpdate(BaseModel):
    agent_id: str
    tokens_consumed: float = Field(ge=0)
    value_generated: float = Field(ge=0)


class BudgetStatus(BaseModel):
    global_budget: float
    allocated_total: float
    consumed_total: float
    remaining_budget: float
    active_agents: int
    average_efficiency: float
    top_performers: List[Dict]
    budget_utilization_percent: float


class TokenEconomyService:
    """Main service class managing token economy with hard enforcement"""
    
    def __init__(self, global_budget: float):
        self.global_budget = global_budget
        self.redis_client: Optional[redis.Redis] = None
        self._allocation_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize Redis connection and set initial budget"""
        self.redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
        await self.redis_client.set("global_budget_remaining", self.global_budget)
        global_budget_remaining.set(self.global_budget)
        logger.info(f"Token economy initialized with budget: {self.global_budget}")
        
    async def close(self):
        """Cleanup connections"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def allocate_tokens(
        self, 
        request: TokenRequest, 
        session: AsyncSession
    ) -> TokenResponse:
        """
        Allocate tokens with strict enforcement and efficiency-based adjustment.
        
        Algorithm:
        1. Base allocation = min(requested, 1000)
        2. Get agent efficiency from history
        3. Adjusted = base * (0.5 + efficiency)
        4. Final = min(adjusted, remaining_global_budget)
        5. REJECT if final < 10% of requested (prevents ineffective agents)
        """
        async with self._allocation_lock:
            # Get remaining budget
            remaining_str = await self.redis_client.get("global_budget_remaining")
            remaining_budget = float(remaining_str) if remaining_str else 0.0
            
            if remaining_budget <= 0:
                allocation_rejections.inc(reason="budget_exhausted")
                raise HTTPException(
                    status_code=403,
                    detail="Global token budget exhausted. No new allocations possible."
                )
            
            # Calculate base allocation
            base_allocation = min(request.requested_tokens, 1000.0)
            
            # Get agent efficiency
            efficiency = await self._get_agent_efficiency(request.agent_id, session)
            efficiency_multiplier = 0.5 + efficiency  # Range: 0.5 to 1.5+
            
            # Apply efficiency adjustment
            adjusted_allocation = base_allocation * efficiency_multiplier
            
            # Apply global constraint
            final_allocation = min(adjusted_allocation, remaining_budget)
            
            # Reject if allocation is too small to be useful
            if final_allocation < request.requested_tokens * 0.1:
                allocation_rejections.inc(reason="insufficient_allocation")
                await self._terminate_agent(
                    request.agent_id, 
                    "Insufficient budget for meaningful allocation",
                    session
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Cannot allocate sufficient tokens. Available: {final_allocation:.0f}, Minimum needed: {request.requested_tokens * 0.1:.0f}"
                )
            
            # Update global budget
            new_remaining = remaining_budget - final_allocation
            await self.redis_client.set("global_budget_remaining", new_remaining)
            global_budget_remaining.set(new_remaining)
            
            # Record allocation in database
            allocation = TokenAllocation(
                agent_id=request.agent_id,
                allocated_tokens=final_allocation,
                efficiency=efficiency
            )
            session.add(allocation)
            await session.commit()
            
            # Update metrics
            tokens_allocated_total.labels(agent_id=request.agent_id).inc(final_allocation)
            active_token_allocation.labels(agent_id=request.agent_id).set(final_allocation)
            
            # Store active allocation in Redis for fast lookup
            await self.redis_client.hset(
                f"agent_allocation:{request.agent_id}",
                mapping={
                    "allocated": final_allocation,
                    "consumed": 0,
                    "remaining": final_allocation
                }
            )
            
            warning = None
            if final_allocation < request.requested_tokens:
                warning = f"Reduced allocation due to constraints. Requested: {request.requested_tokens}, Allocated: {final_allocation:.0f}"
            
            logger.info(f"Allocated {final_allocation:.0f} tokens to agent {request.agent_id} (efficiency: {efficiency:.3f})")
            
            return TokenResponse(
                agent_id=request.agent_id,
                allocated_tokens=final_allocation,
                efficiency_multiplier=efficiency_multiplier,
                remaining_global_budget=new_remaining,
                warning=warning
            )
    
    async def update_consumption(
        self,
        update: ConsumptionUpdate,
        session: AsyncSession,
        background_tasks: BackgroundTasks
    ) -> Dict:
        """
        Update token consumption and ENFORCE HARD STOP when budget exhausted.
        
        CRITICAL: This method must actually stop agents, not just log warnings!
        """
        # Get current allocation from Redis
        allocation_data = await self.redis_client.hgetall(f"agent_allocation:{update.agent_id}")
        
        if not allocation_data:
            raise HTTPException(
                status_code=404,
                detail=f"No active allocation found for agent {update.agent_id}"
            )
        
        allocated = float(allocation_data["allocated"])
        previously_consumed = float(allocation_data["consumed"])
        new_total_consumed = previously_consumed + update.tokens_consumed
        remaining = allocated - new_total_consumed
        
        # HARD ENFORCEMENT: Check if budget exhausted
        if remaining <= 0:
            # Agent has exhausted budget - TERMINATE IT
            await self._terminate_agent(
                update.agent_id,
                f"Token budget exhausted. Allocated: {allocated:.0f}, Consumed: {new_total_consumed:.0f}",
                session
            )
            
            # Update metrics
            agents_terminated_budget.inc()
            tokens_consumed_total.labels(agent_id=update.agent_id).inc(update.tokens_consumed)
            
            # Clear active allocation
            await self.redis_client.delete(f"agent_allocation:{update.agent_id}")
            active_token_allocation.labels(agent_id=update.agent_id).set(0)
            
            # Schedule cleanup in background
            background_tasks.add_task(self._cleanup_agent_resources, update.agent_id)
            
            raise HTTPException(
                status_code=403,
                detail=f"Agent {update.agent_id} terminated: Token budget exhausted"
            )
        
        # Update consumption in Redis
        await self.redis_client.hset(
            f"agent_allocation:{update.agent_id}",
            mapping={
                "consumed": new_total_consumed,
                "remaining": remaining
            }
        )
        
        # Update database
        result = await session.execute(
            select(TokenAllocation)
            .where(TokenAllocation.agent_id == update.agent_id)
            .where(TokenAllocation.is_active == True)
            .order_by(TokenAllocation.allocation_time.desc())
            .limit(1)
        )
        allocation = result.scalar_one_or_none()
        
        if allocation:
            allocation.consumed_tokens = new_total_consumed
            allocation.value_generated += update.value_generated
            
            # Calculate new efficiency
            if new_total_consumed > 0:
                new_efficiency = allocation.value_generated / new_total_consumed
                allocation.efficiency = new_efficiency
                token_efficiency_histogram.observe(new_efficiency)
            
            allocation.last_update = datetime.now(timezone.utc)
            await session.commit()
        
        # Update Prometheus metrics
        tokens_consumed_total.labels(agent_id=update.agent_id).inc(update.tokens_consumed)
        active_token_allocation.labels(agent_id=update.agent_id).set(remaining)
        
        # Warn if approaching limit
        warning = None
        if remaining < allocated * 0.2:  # Less than 20% remaining
            warning = f"Low token budget: {remaining:.0f} tokens remaining ({(remaining/allocated*100):.1f}%)"
        
        return {
            "agent_id": update.agent_id,
            "consumed": new_total_consumed,
            "remaining": remaining,
            "efficiency": allocation.efficiency if allocation else 0.0,
            "warning": warning
        }
    
    async def get_budget_status(self, session: AsyncSession) -> BudgetStatus:
        """Get comprehensive budget status"""
        # Get remaining budget
        remaining_str = await self.redis_client.get("global_budget_remaining")
        remaining = float(remaining_str) if remaining_str else 0.0
        
        # Get active allocations
        result = await session.execute(
            select(
                TokenAllocation.agent_id,
                TokenAllocation.allocated_tokens,
                TokenAllocation.consumed_tokens,
                TokenAllocation.value_generated,
                TokenAllocation.efficiency
            ).where(TokenAllocation.is_active == True)
        )
        
        active_allocations = result.fetchall()
        
        allocated_total = sum(a.allocated_tokens for a in active_allocations)
        consumed_total = sum(a.consumed_tokens for a in active_allocations)
        
        # Calculate average efficiency
        efficiencies = [a.efficiency for a in active_allocations if a.efficiency > 0]
        avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
        
        # Get top performers
        top_performers = sorted(
            [
                {
                    "agent_id": a.agent_id,
                    "efficiency": a.efficiency,
                    "value_generated": a.value_generated,
                    "tokens_consumed": a.consumed_tokens
                }
                for a in active_allocations
            ],
            key=lambda x: x["efficiency"],
            reverse=True
        )[:5]
        
        utilization = ((self.global_budget - remaining) / self.global_budget * 100) if self.global_budget > 0 else 0
        
        return BudgetStatus(
            global_budget=self.global_budget,
            allocated_total=allocated_total,
            consumed_total=consumed_total,
            remaining_budget=remaining,
            active_agents=len(active_allocations),
            average_efficiency=avg_efficiency,
            top_performers=top_performers,
            budget_utilization_percent=utilization
        )
    
    async def _get_agent_efficiency(self, agent_id: str, session: AsyncSession) -> float:
        """Get historical efficiency for an agent"""
        # Look for recent performance
        result = await session.execute(
            select(TokenAllocation.efficiency)
            .where(TokenAllocation.agent_id == agent_id)
            .where(TokenAllocation.efficiency > 0)
            .order_by(TokenAllocation.allocation_time.desc())
            .limit(5)
        )
        
        efficiencies = [row[0] for row in result.fetchall()]
        
        if not efficiencies:
            # New agent - start with baseline
            return 0.5
        
        # Average recent efficiency, capped at 1.0 for multiplier calculation
        return min(sum(efficiencies) / len(efficiencies), 1.0)
    
    async def _terminate_agent(self, agent_id: str, reason: str, session: AsyncSession):
        """Terminate an agent and record the reason"""
        # Update database
        result = await session.execute(
            update(TokenAllocation)
            .where(TokenAllocation.agent_id == agent_id)
            .where(TokenAllocation.is_active == True)
            .values(
                is_active=False,
                termination_reason=reason,
                last_update=datetime.now(timezone.utc)
            )
        )
        await session.commit()
        
        # Log termination
        logger.warning(f"TERMINATED Agent {agent_id}: {reason}")
        
        # Send termination signal via Redis pub/sub
        await self.redis_client.publish(
            f"agent_termination",
            f"{agent_id}:{reason}"
        )
    
    async def _cleanup_agent_resources(self, agent_id: str):
        """Background task to cleanup agent resources"""
        try:
            # Remove from active allocations
            await self.redis_client.delete(f"agent_allocation:{agent_id}")
            
            # Additional cleanup can be added here
            # e.g., notify other services, cleanup worktrees, etc.
            
            logger.info(f"Cleaned up resources for terminated agent {agent_id}")
        except Exception as e:
            logger.error(f"Error cleaning up agent {agent_id}: {e}")


# FastAPI app setup
app = FastAPI(title="DEAN Token Economy Service")
token_service: Optional[TokenEconomyService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global token_service
    
    # Get global budget from environment
    global_budget = float(os.getenv("GLOBAL_TOKEN_BUDGET", "100000"))
    
    # Initialize service
    token_service = TokenEconomyService(global_budget)
    await token_service.initialize()
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Cleanup
    await token_service.close()
    await engine.dispose()


app = FastAPI(title="DEAN Token Economy Service", lifespan=lifespan)


async def get_db():
    """Database session dependency"""
    async with AsyncSessionLocal() as session:
        yield session


@app.post("/api/v1/tokens/allocate", response_model=TokenResponse)
async def allocate_tokens(
    request: TokenRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Allocate tokens to an agent with efficiency-based adjustment.
    
    Enforces hard limits - will reject allocation if budget insufficient.
    """
    if not token_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await token_service.allocate_tokens(request, db)


@app.post("/api/v1/tokens/consume")
async def update_consumption(
    update: ConsumptionUpdate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Update token consumption for an agent.
    
    CRITICAL: Will TERMINATE agent if budget exhausted!
    """
    if not token_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await token_service.update_consumption(update, db, background_tasks)


@app.get("/api/v1/tokens/status", response_model=BudgetStatus)
async def get_budget_status(db: AsyncSession = Depends(get_db)):
    """Get comprehensive budget status"""
    if not token_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await token_service.get_budget_status(db)


@app.get("/api/v1/tokens/agent/{agent_id}")
async def get_agent_status(agent_id: str):
    """Get current token status for a specific agent"""
    if not token_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Get from Redis for real-time data
    allocation_data = await token_service.redis_client.hgetall(f"agent_allocation:{agent_id}")
    
    if not allocation_data:
        raise HTTPException(status_code=404, detail=f"No active allocation for agent {agent_id}")
    
    return {
        "agent_id": agent_id,
        "allocated": float(allocation_data["allocated"]),
        "consumed": float(allocation_data["consumed"]),
        "remaining": float(allocation_data["remaining"]),
        "percent_used": (float(allocation_data["consumed"]) / float(allocation_data["allocated"]) * 100)
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not token_service or not token_service.redis_client:
        raise HTTPException(status_code=503, detail="Service not healthy")
    
    try:
        # Check Redis
        await token_service.redis_client.ping()
        
        # Check database
        async with AsyncSessionLocal() as session:
            await session.execute(select(1))
        
        return {"status": "healthy", "service": "token-economy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091)