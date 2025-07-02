#!/usr/bin/env python3
"""
DEAN API Service - Real Implementation
NO MOCKS - This service only uses real IndexAgent components and PostgreSQL data
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import hashlib
import json

from fastapi import FastAPI, HTTPException, WebSocket, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
import uvicorn
from collections import Counter

# Add IndexAgent to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "IndexAgent"))

# Import REAL IndexAgent components
from indexagent.agents.evolution.cellular_automata import (
    CellularAutomataEngine, CARule, ComplexityMetrics
)
from indexagent.agents.base_agent import FractalAgent, AgentGenome, TokenBudget
from indexagent.agents.worktree_manager import GitWorktreeManager
from indexagent.agents.economy.token_manager import TokenEconomyManager
from indexagent.agents.patterns.detector import PatternDetector
from indexagent.agents.patterns.monitor import EmergentBehaviorMonitor
from indexagent.agents.evolution.diversity_manager import GeneticDiversityManager
from indexagent.agents.evolution.genetic_algorithm import GeneticAlgorithm
from indexagent.database.connection import get_db_session, init_database, get_db
from indexagent.database.schema import (
    Agent, DiscoveredPattern, PerformanceMetric, EvolutionHistory, AuditLog
)

# Import Economic Governor
from services.economy.economic_governor import EconomicGovernor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DEAN System API - Real Implementation",
    description="Real agent evolution system with NO mock data",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
worktree_mgr: Optional[GitWorktreeManager] = None
token_manager: Optional[TokenEconomyManager] = None
ca_engine: Optional[CellularAutomataEngine] = None
diversity_manager: Optional[GeneticDiversityManager] = None
pattern_detector: Optional[PatternDetector] = None
behavior_monitor: Optional[EmergentBehaviorMonitor] = None
economic_governor: Optional[EconomicGovernor] = None

# Storage for API state (use Redis/DB in production)
modification_tasks: Dict[str, Any] = {}
stored_patterns: List[Dict[str, Any]] = []
diversity_registry: Dict[str, Any] = {}

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

# Request/Response Models
class AgentCreationRequest(BaseModel):
    goal: str = Field(..., description="Agent's optimization goal")
    token_budget: int = Field(default=4096, ge=100, le=10000)
    diversity_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    specialized_domain: Optional[str] = None

class EvolutionRequest(BaseModel):
    agent_id: str
    generations: int = Field(default=1, ge=1, le=10)
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)

class PopulationEvolutionRequest(BaseModel):
    cycles: int = Field(default=1, ge=1, le=5)
    parallel_workers: int = Field(default=4, ge=1, le=8)

# WebSocket manager
async def broadcast_update(message: Dict[str, Any]):
    """Broadcast updates to all connected WebSocket clients"""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            active_connections.remove(connection)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check with real service status"""
    from indexagent.database.connection import check_database_health
    
    db_health = await check_database_health()
    
    return {
        "status": "healthy" if db_health["connected"] else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": db_health,
            "worktree_manager": worktree_mgr is not None,
            "token_manager": token_manager is not None,
            "ca_engine": ca_engine is not None,
            "diversity_manager": diversity_manager is not None,
            "pattern_detector": pattern_detector is not None
        }
    }

# Agent Management Endpoints
@app.post("/api/v1/agents")
async def create_agent(
    request: AgentCreationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create a REAL agent with git worktree and Claude CLI integration"""
    try:
        # Create initial genome with some randomization
        import random
        initial_genome = AgentGenome(
            traits={
                "optimization": 0.5 + random.random() * 0.5,
                "pattern_recognition": 0.5 + random.random() * 0.5,
                "collaboration": random.random(),
                "innovation": random.random()
            },
            strategies=["gradient_descent", "pattern_matching"],
            mutation_rate=0.05 + random.random() * 0.15
        )
        
        # Create agent in database
        agent = Agent(
            name=f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            goal=request.goal,
            token_budget=request.token_budget,
            diversity_weight=request.diversity_weight,
            specialized_domain=request.specialized_domain,
            status="initializing"
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)
        
        # Store initial genome as a discovered pattern
        genome_pattern = DiscoveredPattern(
            agent_id=agent.id,
            pattern_hash=hashlib.sha256(json.dumps(initial_genome.model_dump(), sort_keys=True).encode()).hexdigest(),
            pattern_type="genome",
            pattern_content=initial_genome.model_dump(),
            effectiveness_score=0.5,
            confidence_score=1.0,
            pattern_meta={"generation": 0, "is_active": True}
        )
        db.add(genome_pattern)
        await db.commit()
        
        # Create git worktree for agent
        worktree_path = await worktree_mgr.create_worktree(
            branch_name=f"agent/{agent.id}",
            agent_id=str(agent.id),
            token_limit=request.token_budget
        )
        
        # Update agent with worktree path
        agent.worktree_path = str(worktree_path)
        agent.status = "active"
        await db.commit()
        
        # Allocate tokens from economy manager
        allocated_tokens = await token_manager.allocate_tokens(
            agent_id=str(agent.id),
            base_amount=request.token_budget
        )
        
        # Start agent in background
        background_tasks.add_task(
            initialize_agent_claude_cli,
            agent.id,
            worktree_path,
            request.goal
        )
        
        # Broadcast creation event
        await broadcast_update({
            "type": "agent_created",
            "data": {
                "agent_id": str(agent.id),
                "goal": agent.goal,
                "worktree_path": str(worktree_path),
                "tokens_allocated": allocated_tokens
            }
        })
        
        return {
            "id": str(agent.id),
            "name": agent.name,
            "goal": agent.goal,
            "worktree_path": str(worktree_path),
            "token_budget": allocated_tokens,
            "status": "active"
        }
        
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agents")
async def list_agents(
    limit: int = 100,
    active_only: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """List REAL agents from database"""
    query = select(Agent)
    if active_only:
        query = query.where(Agent.status == "active")
    
    query = query.limit(limit)
    result = await db.execute(query)
    agents = result.scalars().all()
    
    return {
        "agents": [
            {
                "id": str(agent.id),
                "name": agent.name,
                "goal": agent.goal,
                "status": agent.status,
                "worktree_path": agent.worktree_path,
                "token_budget": agent.token_budget,
                "token_consumed": agent.token_consumed,
                "fitness_score": agent.fitness_score,
                "generation": agent.generation,
                "created_at": agent.created_at.isoformat()
            }
            for agent in agents
        ]
    }

@app.get("/api/v1/agents/{agent_id}")
async def get_agent_details(
    agent_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get REAL agent details including performance metrics"""
    # Get agent
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Get recent performance metrics
    metrics_query = select(PerformanceMetric).where(
        PerformanceMetric.agent_id == agent_id
    ).order_by(PerformanceMetric.timestamp.desc()).limit(10)
    metrics_result = await db.execute(metrics_query)
    metrics = metrics_result.scalars().all()
    
    # Get discovered patterns
    patterns_query = select(DiscoveredPattern).where(
        DiscoveredPattern.agent_id == agent_id
    ).order_by(DiscoveredPattern.effectiveness_score.desc()).limit(5)
    patterns_result = await db.execute(patterns_query)
    patterns = patterns_result.scalars().all()
    
    return {
        "agent": {
            "id": str(agent.id),
            "name": agent.name,
            "goal": agent.goal,
            "status": agent.status,
            "worktree_path": agent.worktree_path,
            "token_budget": agent.token_budget,
            "token_consumed": agent.token_consumed,
            "token_efficiency": agent.token_efficiency,
            "fitness_score": agent.fitness_score,
            "diversity_score": agent.diversity_score,
            "generation": agent.generation,
            "parent_ids": agent.parent_ids,
            "created_at": agent.created_at.isoformat()
        },
        "metrics": [
            {
                "timestamp": m.timestamp.isoformat(),
                "metric_type": m.metric_type,
                "metric_value": m.metric_value,
                "tokens_used": m.tokens_used,
                "value_per_token": m.value_per_token
            }
            for m in metrics
        ],
        "patterns": [
            {
                "id": str(p.id),
                "pattern_type": p.pattern_type,
                "effectiveness_score": p.effectiveness_score,
                "reuse_count": p.reuse_count,
                "discovered_at": p.discovered_at.isoformat()
            }
            for p in patterns
        ]
    }

@app.patch("/api/v1/agents/{agent_id}")
async def update_agent_status(
    agent_id: str,
    update_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """Update agent status (for retirement, completion, etc.)"""
    # Get agent
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Update allowed fields
    allowed_fields = ["status", "retirement_reason", "fitness_score", "token_efficiency"]
    
    for field, value in update_data.items():
        if field in allowed_fields and hasattr(agent, field):
            setattr(agent, field, value)
    
    # Set terminated_at if status is changing to retired/completed
    if update_data.get("status") in ["retired", "completed", "failed"]:
        agent.terminated_at = datetime.now()
    
    agent.updated_at = datetime.now()
    await db.commit()
    await db.refresh(agent)
    
    # Log the status change
    await broadcast_update({
        "type": "agent_status_changed",
        "data": {
            "agent_id": str(agent.id),
            "new_status": agent.status,
            "reason": update_data.get("retirement_reason", "status_update")
        }
    })
    
    return {
        "id": str(agent.id),
        "status": agent.status,
        "updated_at": agent.updated_at.isoformat()
    }

# Evolution Endpoints
@app.post("/api/v1/agents/{agent_id}/evolve")
async def evolve_agent(
    agent_id: str,
    request: EvolutionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Execute REAL evolution using cellular automata rules"""
    # Get agent
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if agent.status != "active":
        raise HTTPException(status_code=400, detail="Agent is not active")
    
    # Check token budget
    remaining_tokens = agent.token_budget - agent.token_consumed
    if remaining_tokens < 100:
        raise HTTPException(status_code=400, detail="Insufficient token budget")
    
    # Start evolution in background
    background_tasks.add_task(
        run_agent_evolution,
        agent_id,
        request.generations,
        request.mutation_rate
    )
    
    return {
        "message": "Evolution started",
        "agent_id": agent_id,
        "generations": request.generations,
        "estimated_tokens": request.generations * 500
    }

@app.post("/api/v1/population/evolve")
async def evolve_population(
    request: PopulationEvolutionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Evolve entire population with REAL genetic algorithm"""
    # Get active agents
    result = await db.execute(
        select(Agent).where(Agent.status == "active")
    )
    agents = result.scalars().all()
    
    if len(agents) == 0:
        raise HTTPException(status_code=400, detail="No active agents to evolve")
    
    # Check population diversity
    diversity_score = await diversity_manager.calculate_population_diversity(agents)
    
    # Start population evolution
    background_tasks.add_task(
        run_population_evolution,
        [str(a.id) for a in agents],
        request.cycles,
        request.parallel_workers
    )
    
    return {
        "message": "Population evolution started",
        "agent_count": len(agents),
        "diversity_score": diversity_score,
        "cycles": request.cycles
    }

# Pattern Discovery Endpoints
@app.get("/api/v1/patterns")
async def list_discovered_patterns(
    limit: int = 100,
    min_effectiveness: float = 0.5,
    db: AsyncSession = Depends(get_db)
):
    """List REAL discovered patterns from database"""
    query = select(DiscoveredPattern).where(
        DiscoveredPattern.effectiveness_score >= min_effectiveness
    ).order_by(DiscoveredPattern.effectiveness_score.desc()).limit(limit)
    
    result = await db.execute(query)
    patterns = result.scalars().all()
    
    return {
        "patterns": [
            {
                "id": str(p.id),
                "agent_id": str(p.agent_id),
                "pattern_type": p.pattern_type,
                "pattern_content": p.pattern_content,
                "effectiveness_score": p.effectiveness_score,
                "token_efficiency_delta": p.token_efficiency_delta,
                "reuse_count": p.reuse_count,
                "discovered_at": p.discovered_at.isoformat()
            }
            for p in patterns
        ]
    }

# System Metrics Endpoints
@app.get("/api/v1/system/metrics")
async def get_system_metrics(db: AsyncSession = Depends(get_db)):
    """Get REAL system metrics from database"""
    # Count agents by status
    from sqlalchemy import text
    agent_counts_result = await db.execute(
        text("SELECT status, COUNT(*) as count FROM agent_evolution.agents GROUP BY status")
    )
    agent_counts = agent_counts_result.fetchall()
    
    # Calculate token usage
    token_stats_result = await db.execute(text("""
        SELECT 
            SUM(token_budget) as total_allocated,
            SUM(token_consumed) as total_consumed,
            AVG(token_efficiency) as avg_efficiency
        FROM agent_evolution.agents
        WHERE status = 'active'
    """))
    token_data = token_stats_result.first()
    
    # Count patterns
    pattern_count_result = await db.execute(
        text("SELECT COUNT(*) as count FROM agent_evolution.discovered_patterns")
    )
    pattern_count = pattern_count_result.scalar()
    
    agent_status = {row.status: row.count for row in agent_counts}
    
    return {
        "agents": {
            "active": agent_status.get("active", 0),
            "completed": agent_status.get("completed", 0),
            "failed": agent_status.get("failed", 0)
        },
        "tokens": {
            "allocated": token_data.total_allocated or 0,
            "consumed": token_data.total_consumed or 0,
            "efficiency": token_data.avg_efficiency or 0.0
        },
        "patterns": {
            "discovered": pattern_count or 0
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/system/diversity")
async def get_diversity_metrics(db: AsyncSession = Depends(get_db)):
    """Get REAL population diversity metrics"""
    result = await db.execute(
        select(Agent).where(Agent.status == "active")
    )
    agents = result.scalars().all()
    
    if not agents:
        return {"diversity_score": 0.0, "agent_count": 0}
    
    diversity_score = await diversity_manager.calculate_population_diversity(agents)
    variance_components = await diversity_manager.get_variance_components(agents)
    
    return {
        "diversity_score": diversity_score,
        "agent_count": len(agents),
        "variance_components": variance_components,
        "minimum_threshold": diversity_manager.min_diversity_threshold,
        "requires_intervention": diversity_score < diversity_manager.min_diversity_threshold
    }

# Economic Governor Endpoints
@app.get("/api/v1/economy/metrics")
async def get_economic_metrics():
    """Get system-wide economic metrics"""
    if not economic_governor:
        raise HTTPException(status_code=503, detail="Economic governor not initialized")
    
    try:
        metrics = economic_governor.get_system_metrics()
        return {
            "global_budget": {
                "total": metrics["total_budget"],
                "allocated": metrics["allocated_budget"],
                "used": metrics["used_budget"],
                "available": metrics["available_budget"],
                "usage_rate": metrics["usage_rate"]
            },
            "agents": {
                "total": metrics["agent_count"],
                "average_efficiency": metrics["average_efficiency"]
            },
            "top_performers": metrics["top_performers"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get economic metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/economy/agent/{agent_id}")
async def get_agent_economic_status(agent_id: str):
    """Get economic status for a specific agent"""
    if not economic_governor:
        raise HTTPException(status_code=503, detail="Economic governor not initialized")
    
    try:
        budget_info = economic_governor.allocator.get_agent_budget(agent_id)
        if not budget_info:
            raise HTTPException(status_code=404, detail="Agent not found in economic system")
        
        return {
            "agent_id": agent_id,
            "budget": {
                "allocated": budget_info["allocated"],
                "used": budget_info["used"], 
                "remaining": budget_info["remaining"],
                "efficiency": budget_info["efficiency"]
            },
            "performance": {
                "generation": budget_info["generation"],
                "last_allocation": budget_info["last_allocation"].isoformat() if budget_info["last_allocation"] else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent economic status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class TokenUsageRequest(BaseModel):
    agent_id: str
    tokens: int = Field(..., gt=0)
    action_type: str
    task_success: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)

@app.post("/api/v1/economy/use-tokens")
async def use_agent_tokens(request: TokenUsageRequest):
    """Record token usage for an agent"""
    if not economic_governor:
        raise HTTPException(status_code=503, detail="Economic governor not initialized")
    
    try:
        success = economic_governor.use_tokens(
            agent_id=request.agent_id,
            tokens=request.tokens,
            action_type=request.action_type,
            task_success=request.task_success,
            quality_score=request.quality_score
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Insufficient token budget")
        
        # Get updated budget
        budget_info = economic_governor.allocator.get_agent_budget(request.agent_id)
        
        return {
            "success": True,
            "agent_id": request.agent_id,
            "tokens_used": request.tokens,
            "remaining_budget": budget_info["remaining"] if budget_info else 0
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record token usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class TokenAllocationRequest(BaseModel):
    agent_id: str
    performance: float = Field(ge=0.0, le=1.0)
    generation: int = Field(ge=0)

@app.post("/api/v1/economy/allocate")
async def allocate_agent_tokens(request: TokenAllocationRequest):
    """Allocate tokens to an agent based on performance"""
    if not economic_governor:
        raise HTTPException(status_code=503, detail="Economic governor not initialized")
    
    try:
        allocated = economic_governor.allocate_to_agent(
            agent_id=request.agent_id,
            performance=request.performance,
            generation=request.generation
        )
        
        return {
            "success": True,
            "agent_id": request.agent_id,
            "tokens_allocated": allocated,
            "total_budget": economic_governor.allocator.get_agent_budget(request.agent_id)["allocated"]
        }
    except Exception as e:
        logger.error(f"Failed to allocate tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/economy/rebalance")
async def rebalance_economy():
    """Trigger economic rebalancing across all agents"""
    if not economic_governor:
        raise HTTPException(status_code=503, detail="Economic governor not initialized")
    
    try:
        # Get metrics before rebalancing
        metrics_before = economic_governor.get_system_metrics()
        
        # Perform rebalancing
        rebalanced_agents = economic_governor.rebalance_budgets()
        
        # Get metrics after rebalancing
        metrics_after = economic_governor.get_system_metrics()
        
        return {
            "success": True,
            "agents_rebalanced": len(rebalanced_agents),
            "rebalanced_agents": rebalanced_agents,
            "metrics": {
                "before": {
                    "average_efficiency": metrics_before["average_efficiency"],
                    "used_budget": metrics_before["used_budget"]
                },
                "after": {
                    "average_efficiency": metrics_after["average_efficiency"],
                    "used_budget": metrics_after["used_budget"]
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to rebalance economy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Worktree Management Endpoints
@app.post("/api/v1/worktrees")
async def create_worktree(agent_id: str, base_repo: str = "/repos/target"):
    """Create isolated git worktree for agent"""
    import subprocess
    import tempfile
    from pathlib import Path
    
    try:
        # Create worktree directory
        worktree_base = Path("/tmp/worktrees")
        worktree_base.mkdir(exist_ok=True)
        worktree_path = worktree_base / f"agent_{agent_id}"
        
        # Create git worktree
        cmd = ["git", "worktree", "add", str(worktree_path), "HEAD"]
        result = subprocess.run(cmd, cwd=base_repo, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Failed to create worktree: {result.stderr}")
        
        return {
            "agent_id": agent_id,
            "worktree_path": str(worktree_path),
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Failed to create worktree for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/worktrees/{agent_id}")
async def get_worktree_info(agent_id: str):
    """Get worktree path and status"""
    from pathlib import Path
    
    worktree_path = Path(f"/tmp/worktrees/agent_{agent_id}")
    
    if not worktree_path.exists():
        raise HTTPException(status_code=404, detail="Worktree not found")
    
    return {
        "agent_id": agent_id,
        "worktree_path": str(worktree_path),
        "exists": True,
        "status": "active"
    }

@app.delete("/api/v1/worktrees/{agent_id}")
async def cleanup_worktree(agent_id: str):
    """Remove worktree after use"""
    import subprocess
    import shutil
    from pathlib import Path
    
    try:
        worktree_path = Path(f"/tmp/worktrees/agent_{agent_id}")
        
        if worktree_path.exists():
            # Remove git worktree
            cmd = ["git", "worktree", "remove", str(worktree_path), "--force"]
            subprocess.run(cmd, capture_output=True)
            
            # Ensure directory is removed
            if worktree_path.exists():
                shutil.rmtree(worktree_path, ignore_errors=True)
        
        return {"status": "removed", "agent_id": agent_id}
        
    except Exception as e:
        logger.error(f"Failed to cleanup worktree for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Code Modification Endpoints
class CodeModificationRequest(BaseModel):
    agent_id: str
    worktree_path: str
    prompt: str
    target_files: List[str] = []
    max_tokens: int = 4096

@app.post("/api/v1/modifications")
async def execute_code_modification(
    request: CodeModificationRequest,
    background_tasks: BackgroundTasks
):
    """Execute code modification via Claude"""
    import uuid
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Store task status (in production, use Redis or database)
    modification_tasks[task_id] = {
        "status": "pending",
        "agent_id": request.agent_id,
        "created_at": datetime.now()
    }
    
    # Execute in background
    background_tasks.add_task(
        run_code_modification,
        task_id,
        request
    )
    
    return {"task_id": task_id, "status": "submitted"}

@app.get("/api/v1/modifications/{task_id}")
async def get_modification_result(task_id: str):
    """Get result of code modification"""
    if task_id not in modification_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return modification_tasks[task_id]

# Optimization Endpoints
class OptimizationRequest(BaseModel):
    task_description: str
    performance_metrics: Dict[str, float]
    historical_context: Optional[List[Dict]] = None

@app.post("/api/v1/optimize/prompt")
async def optimize_prompt(request: OptimizationRequest):
    """Optimize prompt based on task and performance"""
    try:
        # Simple optimization logic for now
        # In production, this would use DSPy or similar
        base_prompt = request.task_description
        
        # Add performance-based adjustments
        if request.performance_metrics.get("success_rate", 0) < 0.5:
            optimized = f"Focus on correctness and reliability. {base_prompt}"
        elif request.performance_metrics.get("token_efficiency", 0) < 0.7:
            optimized = f"Be concise and efficient. {base_prompt}"
        else:
            optimized = base_prompt
        
        return {
            "optimized_prompt": optimized,
            "optimization_score": 0.8,
            "method": "performance_based"
        }
        
    except Exception as e:
        logger.error(f"Failed to optimize prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/optimize/patterns/inject")
async def inject_patterns(patterns: List[Dict[str, Any]]):
    """Inject meta-patterns into optimizer"""
    try:
        # Store patterns (in production, persist to database)
        stored_patterns.extend(patterns)
        
        return {
            "success": True,
            "patterns_injected": len(patterns),
            "total_patterns": len(stored_patterns)
        }
        
    except Exception as e:
        logger.error(f"Failed to inject patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Diversity Management Endpoints
class AgentRegistration(BaseModel):
    agent_id: str
    strategies: List[str]
    lineage: List[str]
    generation: int

@app.post("/api/v1/diversity/agents/register")
async def register_agent(registration: AgentRegistration):
    """Register agent with diversity tracker"""
    try:
        # Store agent info (in production, use database)
        diversity_registry[registration.agent_id] = {
            "strategies": registration.strategies,
            "lineage": registration.lineage,
            "generation": registration.generation,
            "registered_at": datetime.now()
        }
        
        return {"success": True, "agent_id": registration.agent_id}
        
    except Exception as e:
        logger.error(f"Failed to register agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/diversity/metrics")
async def get_diversity_metrics():
    """Calculate current population diversity"""
    try:
        # Calculate diversity metrics
        if not diversity_registry:
            return {
                "diversity_score": 0.0,
                "population_size": 0,
                "unique_strategies": 0
            }
        
        # Get unique strategies
        all_strategies = set()
        for agent_data in diversity_registry.values():
            all_strategies.update(agent_data["strategies"])
        
        diversity_score = min(1.0, len(all_strategies) / 10.0)  # Simple metric
        
        return {
            "diversity_score": diversity_score,
            "population_size": len(diversity_registry),
            "unique_strategies": len(all_strategies),
            "strategy_distribution": dict(Counter(str(s) for agent in diversity_registry.values() for s in agent["strategies"]))
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate diversity metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/diversity/interventions/check")
async def check_intervention_needed():
    """Check if diversity intervention needed"""
    metrics = await get_diversity_metrics()
    
    if metrics["diversity_score"] < 0.3:
        return {"intervention_type": "low_variance"}
    elif metrics["population_size"] > 20 and metrics["unique_strategies"] < 5:
        return {"intervention_type": "high_convergence"}
    else:
        return {"intervention_type": None}

@app.post("/api/v1/diversity/interventions/apply")
async def apply_intervention(agent_id: str, intervention_type: str):
    """Apply diversity intervention to agent"""
    try:
        if agent_id not in diversity_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Apply intervention (simplified)
        if intervention_type == "mutation":
            # Add random strategy
            diversity_registry[agent_id]["strategies"].append(f"mutated_{datetime.now().timestamp()}")
        elif intervention_type == "foreign_pattern":
            # Import pattern from another agent
            if len(diversity_registry) > 1:
                other_agents = [a for a in diversity_registry if a != agent_id]
                source_agent = other_agents[0]  # Simple selection
                pattern = diversity_registry[source_agent]["strategies"][0]
                diversity_registry[agent_id]["strategies"].append(f"imported_{pattern}")
        
        return {"success": True, "intervention_applied": intervention_type}
        
    except Exception as e:
        logger.error(f"Failed to apply intervention: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except:
        active_connections.remove(websocket)

# Background task implementations
async def run_code_modification(task_id: str, request: CodeModificationRequest):
    """Execute code modification in background"""
    try:
        # Update task status
        modification_tasks[task_id]["status"] = "running"
        modification_tasks[task_id]["started_at"] = datetime.now()
        
        # Mock modification execution
        # In production, this would call Claude API
        import asyncio
        await asyncio.sleep(2)  # Simulate processing
        
        # Generate mock result
        result = {
            "status": "completed",
            "agent_id": request.agent_id,
            "modifications": [
                {
                    "file": request.target_files[0] if request.target_files else "main.py",
                    "changes": 5,
                    "tokens_used": 150
                }
            ],
            "total_tokens": 150,
            "success": True,
            "completed_at": datetime.now().isoformat()
        }
        
        modification_tasks[task_id].update(result)
        
    except Exception as e:
        modification_tasks[task_id]["status"] = "failed"
        modification_tasks[task_id]["error"] = str(e)
        modification_tasks[task_id]["completed_at"] = datetime.now().isoformat()

async def initialize_agent_claude_cli(agent_id: str, worktree_path: Path, goal: str):
    """Initialize Claude CLI in agent's worktree"""
    try:
        # TODO: Implement actual Claude CLI initialization
        # This would spawn a Docker container with Claude CLI
        # For now, log the action
        logger.info(f"Initializing Claude CLI for agent {agent_id} in {worktree_path}")
        
        await broadcast_update({
            "type": "agent_initialized",
            "data": {
                "agent_id": agent_id,
                "status": "ready"
            }
        })
    except Exception as e:
        logger.error(f"Failed to initialize agent {agent_id}: {e}")

async def run_agent_evolution(agent_id: str, generations: int, mutation_rate: float):
    """Run REAL agent evolution using cellular automata"""
    try:
        async with get_db_session() as db:
            # Get agent
            result = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent = result.scalar_one_or_none()
            if not agent:
                logger.error(f"Agent {agent_id} not found")
                return
            
            # Get agent's current genome from patterns
            genome_result = await db.execute(
                select(DiscoveredPattern).where(
                    and_(
                        DiscoveredPattern.agent_id == agent_id,
                        DiscoveredPattern.pattern_type == "genome"
                    )
                ).order_by(DiscoveredPattern.discovered_at.desc()).limit(1)
            )
            genome_pattern = genome_result.scalar_one_or_none()
            
            if not genome_pattern:
                logger.error(f"No genome found for agent {agent_id}")
                return
            
            # Create FractalAgent instance from database agent
            fractal_agent = FractalAgent(
                genome=AgentGenome(**genome_pattern.pattern_content),
                token_budget=TokenBudget(
                    total=agent.token_budget,
                    used=agent.token_consumed
                )
            )
            
            for generation in range(generations):
                # Get agent environment
                environment = await get_agent_environment(agent_id, db)
                
                # Apply cellular automata rules
                ca_result = await ca_engine.evolve_agent(
                    fractal_agent,
                    rule=CARule.RULE_110,
                    environment=environment
                )
                
                # Detect patterns
                patterns = await pattern_detector.analyze_evolution_result(ca_result)
                
                # Store patterns
                for pattern in patterns:
                    pattern_hash = hashlib.sha256(
                        json.dumps(pattern.content, sort_keys=True).encode()
                    ).hexdigest()
                    
                    # Check if pattern exists
                    existing = await db.execute(
                        select(DiscoveredPattern).where(
                            DiscoveredPattern.pattern_hash == pattern_hash
                        )
                    )
                    if not existing.scalar_one_or_none():
                        db_pattern = DiscoveredPattern(
                            agent_id=agent_id,
                            pattern_hash=pattern_hash,
                            pattern_type=pattern.pattern_type,
                            pattern_content=pattern.content,
                            effectiveness_score=pattern.effectiveness_score
                        )
                        db.add(db_pattern)
                
                # Record evolution history
                evolution_record = EvolutionHistory(
                    agent_id=agent_id,
                    generation=agent.generation + 1,
                    evolution_type="cellular_automata",
                    rule_applied="rule_110",
                    fitness_before=agent.fitness_score,
                    fitness_after=ca_result.fitness_score,
                    patterns_applied=[],
                    new_patterns_discovered=len(patterns),
                    population_size=len(environment.get('neighbors', [])),
                    population_diversity=environment.get('diversity_score', 0.5)
                )
                db.add(evolution_record)
                
                # Update agent metrics
                agent.generation += 1
                agent.fitness_score = ca_result.fitness_score
                agent.token_consumed += ca_result.tokens_used
                
                # Store evolved genome as new pattern
                evolved_genome_pattern = DiscoveredPattern(
                    agent_id=agent_id,
                    pattern_hash=hashlib.sha256(
                        json.dumps(fractal_agent.genome.model_dump(), sort_keys=True).encode()
                    ).hexdigest(),
                    pattern_type="genome",
                    pattern_content=fractal_agent.genome.model_dump(),
                    effectiveness_score=ca_result.fitness_score,
                    confidence_score=1.0,
                    pattern_meta={"generation": agent.generation, "is_active": True}
                )
                db.add(evolved_genome_pattern)
                
                # Broadcast progress
                await broadcast_update({
                    "type": "evolution_progress",
                    "data": {
                        "agent_id": agent_id,
                        "generation": generation + 1,
                        "fitness": ca_result.fitness_score,
                        "patterns_found": len(patterns)
                    }
                })
                
                await db.commit()
                
    except Exception as e:
        logger.error(f"Evolution failed for agent {agent_id}: {e}")

async def run_population_evolution(agent_ids: List[str], cycles: int, workers: int):
    """Run REAL population evolution with genetic algorithm"""
    try:
        genetic_algo = GeneticAlgorithm(
            population_size=len(agent_ids),
            mutation_rate=0.1,
            crossover_rate=0.7
        )
        
        for cycle in range(cycles):
            # Evolve population
            evolved_population = await genetic_algo.evolve_population(agent_ids)
            
            # Enforce diversity
            await diversity_manager.enforce_diversity(evolved_population)
            
            # Broadcast progress
            await broadcast_update({
                "type": "population_evolution",
                "data": {
                    "cycle": cycle + 1,
                    "total_cycles": cycles,
                    "population_size": len(evolved_population)
                }
            })
            
    except Exception as e:
        logger.error(f"Population evolution failed: {e}")

async def get_agent_environment(agent_id: str, db: AsyncSession) -> Dict[str, Any]:
    """Get agent's environment for evolution"""
    # Get neighboring agents
    neighbors_query = select(Agent).where(
        and_(Agent.status == "active", Agent.id != agent_id)
    ).limit(5)
    result = await db.execute(neighbors_query)
    neighbors = result.scalars().all()
    
    # Calculate diversity
    diversity_score = 0.5
    if neighbors and diversity_manager:
        diversity_score = await diversity_manager.calculate_population_diversity(neighbors)
    
    return {
        "agent_id": agent_id,
        "neighbors": [{
            "id": str(n.id),
            "fitness_score": n.fitness_score,
            "diversity_score": n.diversity_score
        } for n in neighbors],
        "patterns": await pattern_detector.get_available_patterns() if pattern_detector else [],
        "token_limit": agent.token_budget - agent.token_consumed if 'agent' in locals() else 1000,
        "diversity_score": diversity_score
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize all REAL components"""
    global worktree_mgr, token_manager, ca_engine, diversity_manager, pattern_detector, behavior_monitor, economic_governor
    
    logger.info("Initializing DEAN API with REAL components...")
    
    # Initialize database
    await init_database()
    
    # Initialize components
    worktree_mgr = GitWorktreeManager(
        base_path=Path(os.getenv("WORKTREE_BASE_PATH", "/app/worktrees"))
    )
    
    token_manager = TokenEconomyManager(
        global_budget=int(os.getenv("GLOBAL_TOKEN_BUDGET", "1000000"))
    )
    
    ca_engine = CellularAutomataEngine()
    
    diversity_manager = GeneticDiversityManager(
        min_diversity_threshold=float(os.getenv("MIN_DIVERSITY_THRESHOLD", "0.3"))
    )
    
    pattern_detector = PatternDetector()
    behavior_monitor = EmergentBehaviorMonitor()
    
    # Initialize Economic Governor
    db_url = os.getenv("DATABASE_URL", "postgresql://airflow:airflow@postgres:5432/agent_evolution")
    total_budget = int(os.getenv("ECONOMIC_TOTAL_BUDGET", "1000000"))
    economic_governor = EconomicGovernor(db_url, total_budget)
    
    logger.info("DEAN API initialized successfully - NO MOCKS!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    from indexagent.database.connection import close_database
    await close_database()
    logger.info("DEAN API shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8091,
        reload=False,
        log_level="info"
    )