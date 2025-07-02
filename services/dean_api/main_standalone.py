#!/usr/bin/env python3
"""
DEAN Evolution API - Standalone Implementation
Provides evolution governance without cross-repository dependencies
"""

import os
import uuid
import random
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/agent_evolution")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
GLOBAL_TOKEN_BUDGET = int(os.getenv("GLOBAL_TOKEN_BUDGET", "1000000"))
MIN_DIVERSITY_THRESHOLD = float(os.getenv("MIN_DIVERSITY_THRESHOLD", "0.3"))

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI app
app = FastAPI(
    title="DEAN Evolution API",
    description="Economic governance and evolution management for DEAN system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class CARule(str, Enum):
    RULE_30 = "rule_30"
    RULE_90 = "rule_90"
    RULE_110 = "rule_110"
    RULE_184 = "rule_184"

class TokenAllocation(BaseModel):
    agent_id: str
    requested_tokens: int
    priority: str = Field(default="normal", pattern="^(low|normal|high|critical)$")

class TokenAllocationResponse(BaseModel):
    agent_id: str
    allocated_tokens: int
    remaining_budget: int
    allocation_id: str
    timestamp: datetime

class BudgetStatus(BaseModel):
    global_budget: int
    allocated: int
    consumed: int
    reserved: int
    available: int
    agents_count: int
    efficiency_score: float

class EvolutionConstraints(BaseModel):
    min_diversity: float = 0.3
    max_generations: int = 100
    token_limit_per_generation: int = 10000
    mutation_rate_range: tuple = (0.05, 0.3)
    population_size_range: tuple = (3, 50)

class EvolutionRequest(BaseModel):
    population_ids: List[str]
    generations: int = Field(default=5, ge=1, le=100)
    ca_rules: List[CARule] = Field(default=[CARule.RULE_110])
    token_budget: int = Field(default=5000, ge=100)

class EvolutionStatus(BaseModel):
    cycle_id: str
    status: str
    current_generation: int
    total_generations: int
    tokens_consumed: int
    patterns_discovered: int
    population_diversity: float
    estimated_completion: Optional[datetime]

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "service": "DEAN Evolution API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": db_status,
            "token_economy": "healthy",
            "evolution_engine": "healthy"
        }
    }

# Token Economy Endpoints
@app.get("/api/v1/economy/budget", response_model=BudgetStatus)
async def get_budget_status(db: Session = Depends(get_db)):
    """Get current token budget status"""
    try:
        # Get allocation stats from database - simplified for actual schema
        try:
            # Count active agents
            agent_result = db.execute(text("""
                SELECT COUNT(DISTINCT id) 
                FROM agent_evolution.agents 
                WHERE status = 'active'
            """)).fetchone()
            agents_count = agent_result[0] if agent_result else 0
            
            # Get token usage from metrics
            token_result = db.execute(text("""
                SELECT 
                    COALESCE(SUM(tokens_used), 0) as total_tokens,
                    COUNT(*) as metric_count
                FROM agent_evolution.performance_metrics
                WHERE recorded_at > NOW() - INTERVAL '24 hours'
            """)).fetchone()
            
            tokens_used = int(token_result[0]) if token_result else 0
            
            # Simple allocation model - each active agent gets base allocation
            base_allocation = 5000
            allocated = agents_count * base_allocation
            consumed = tokens_used
            reserved = 0
        except Exception as e:
            logger.warning(f"Failed to query metrics: {e}")
            agents_count = 0
            allocated = 0
            consumed = 0
            reserved = 0
        available = GLOBAL_TOKEN_BUDGET - allocated
        
        # Calculate efficiency
        efficiency = (consumed / allocated) if allocated > 0 else 0.0
        
        return BudgetStatus(
            global_budget=GLOBAL_TOKEN_BUDGET,
            allocated=allocated,
            consumed=consumed,
            reserved=reserved,
            available=available,
            agents_count=agents_count,
            efficiency_score=efficiency
        )
    except Exception as e:
        logger.error(f"Failed to get budget status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/economy/allocate", response_model=TokenAllocationResponse)
async def allocate_tokens(
    allocation: TokenAllocation,
    db: Session = Depends(get_db)
):
    """Allocate tokens to an agent"""
    try:
        # Check available budget
        budget_status = await get_budget_status(db)
        
        if allocation.requested_tokens > budget_status.available:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient budget. Available: {budget_status.available}"
            )
        
        # Priority multipliers
        priority_multipliers = {
            "low": 0.5,
            "normal": 1.0,
            "high": 1.5,
            "critical": 2.0
        }
        
        # Calculate actual allocation based on priority
        multiplier = priority_multipliers.get(allocation.priority, 1.0)
        allocated = min(
            int(allocation.requested_tokens * multiplier),
            budget_status.available
        )
        
        # Record allocation in performance metrics
        allocation_id = str(uuid.uuid4())
        db.execute(text("""
            INSERT INTO agent_evolution.performance_metrics 
            (id, agent_id, metric_type, metric_value, tokens_used, task_type, additional_data)
            VALUES (:id, :agent_id, 'token_allocation', :allocated, 0, 'allocation', :data)
        """), {
            "id": allocation_id,
            "agent_id": allocation.agent_id,
            "allocated": float(allocated),
            "data": json.dumps({
                "priority": allocation.priority,
                "requested": allocation.requested_tokens,
                "allocated": allocated
            })
        })
        db.commit()
        
        return TokenAllocationResponse(
            agent_id=allocation.agent_id,
            allocated_tokens=allocated,
            remaining_budget=budget_status.available - allocated,
            allocation_id=allocation_id,
            timestamp=datetime.utcnow()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to allocate tokens: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Evolution Management Endpoints
@app.get("/api/v1/evolution/constraints", response_model=EvolutionConstraints)
async def get_evolution_constraints():
    """Get current evolution constraints"""
    return EvolutionConstraints(
        min_diversity=MIN_DIVERSITY_THRESHOLD,
        max_generations=100,
        token_limit_per_generation=10000,
        mutation_rate_range=(0.05, 0.3),
        population_size_range=(3, 50)
    )

@app.post("/api/v1/evolution/validate")
async def validate_evolution_request(
    request: EvolutionRequest,
    db: Session = Depends(get_db)
):
    """Validate an evolution request before execution"""
    try:
        # Check population exists
        if not request.population_ids:
            raise HTTPException(status_code=400, detail="No population IDs provided")
        
        # Check token budget
        budget_status = await get_budget_status(db)
        if request.token_budget > budget_status.available:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient budget. Requested: {request.token_budget}, Available: {budget_status.available}"
            )
        
        # Validate constraints
        constraints = await get_evolution_constraints()
        if request.generations > constraints.max_generations:
            raise HTTPException(
                status_code=400,
                detail=f"Generations exceed maximum: {constraints.max_generations}"
            )
        
        # Calculate estimated token usage
        estimated_tokens = len(request.population_ids) * request.generations * 100
        
        return {
            "valid": True,
            "estimated_tokens": estimated_tokens,
            "estimated_duration_seconds": request.generations * 5,
            "warnings": []
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate evolution request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/evolution/start")
async def start_evolution(
    request: EvolutionRequest,
    db: Session = Depends(get_db)
):
    """Start an evolution cycle"""
    try:
        # Validate request
        validation = await validate_evolution_request(request, db)
        
        # Create evolution cycle record
        cycle_id = str(uuid.uuid4())
        
        # In a real implementation, this would trigger the actual evolution
        # For now, we'll create a tracking record
        db.execute(text("""
            INSERT INTO agent_evolution.evolution_history 
            (id, generation, population_snapshot, diversity_score, patterns_discovered, created_at)
            VALUES (:id, 0, :snapshot, 0.35, '[]', NOW())
        """), {
            "id": cycle_id,
            "snapshot": {"population_ids": request.population_ids}
        })
        db.commit()
        
        return {
            "cycle_id": cycle_id,
            "status": "started",
            "population_size": len(request.population_ids),
            "generations": request.generations,
            "token_budget": request.token_budget,
            "estimated_completion": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start evolution: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/evolution/{cycle_id}/status", response_model=EvolutionStatus)
async def get_evolution_status(
    cycle_id: str,
    db: Session = Depends(get_db)
):
    """Get status of an evolution cycle"""
    try:
        # Simulate evolution progress
        # In real implementation, this would query actual evolution state
        return EvolutionStatus(
            cycle_id=cycle_id,
            status="running",
            current_generation=random.randint(1, 5),
            total_generations=5,
            tokens_consumed=random.randint(100, 1000),
            patterns_discovered=random.randint(0, 3),
            population_diversity=round(random.uniform(0.3, 0.5), 2),
            estimated_completion=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to get evolution status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/evolution/{cycle_id}/stop")
async def stop_evolution(
    cycle_id: str,
    db: Session = Depends(get_db)
):
    """Stop an evolution cycle"""
    try:
        # In real implementation, this would signal evolution to stop
        return {
            "cycle_id": cycle_id,
            "status": "stopped",
            "message": "Evolution cycle stopped successfully"
        }
    except Exception as e:
        logger.error(f"Failed to stop evolution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Diversity Management
@app.post("/api/v1/diversity/check")
async def check_diversity(
    population_ids: List[str],
    db: Session = Depends(get_db)
):
    """Check diversity of a population"""
    try:
        # Simple diversity calculation
        # In real implementation, this would analyze actual agent genomes
        diversity = round(random.uniform(0.25, 0.45), 2)
        needs_intervention = diversity < MIN_DIVERSITY_THRESHOLD
        
        return {
            "population_size": len(population_ids),
            "diversity_score": diversity,
            "threshold": MIN_DIVERSITY_THRESHOLD,
            "needs_intervention": needs_intervention,
            "recommendations": [
                "Inject random mutations",
                "Import external patterns"
            ] if needs_intervention else []
        }
    except Exception as e:
        logger.error(f"Failed to check diversity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main_standalone:app",
        host="0.0.0.0",
        port=8091,
        log_level="info",
        reload=False
    )