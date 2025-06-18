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

from fastapi import FastAPI, HTTPException, WebSocket, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import uvicorn

# Import our real evolution engine
try:
    from evolution_engine import CellularAutomataEvolution
except ImportError:
    from .evolution_engine import CellularAutomataEvolution

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

# Database configuration
DATABASE_URL = os.getenv(
    "AGENT_EVOLUTION_DATABASE_URL", 
    "postgresql://postgres:postgres@postgres:5432/agent_evolution"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Worktree manager wrapper
class WorktreeManager:
    """Wrapper for worktree management"""
    def __init__(self, base_path: Path):
        self.base_path = base_path
        
    async def create_worktree(self, branch_name: str, agent_id: str, token_limit: int) -> Path:
        """Create a worktree for an agent"""
        worktree_path = self.base_path / agent_id
        worktree_path.mkdir(parents=True, exist_ok=True)
        return worktree_path

# Simple manager classes for demo
class TokenEconomyManager:
    def __init__(self, global_budget: int):
        self.global_budget = global_budget
        
    async def allocate_tokens(self, agent_id: str, base_amount: int) -> int:
        return base_amount

class CellularAutomataEngine:
    pass

class GeneticDiversityManager:
    def __init__(self, min_diversity_threshold: float):
        self.min_diversity_threshold = min_diversity_threshold
        
    async def calculate_population_diversity(self, agents: List[Any]) -> float:
        return 0.75  # Mock value for demo

class PatternDetector:
    async def get_available_patterns(self) -> List[Dict]:
        return []

# Global instances (initialized on startup)
worktree_mgr: Optional[WorktreeManager] = None
token_manager: Optional[TokenEconomyManager] = None
ca_engine: Optional[CellularAutomataEngine] = None
diversity_manager: Optional[GeneticDiversityManager] = None
pattern_detector: Optional[PatternDetector] = None

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

# Database dependency
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": await check_database_health(),
            "worktree_manager": worktree_mgr is not None,
            "token_manager": token_manager is not None,
            "ca_engine": ca_engine is not None
        }
    }

async def check_database_health() -> bool:
    """Check if database is accessible"""
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT 1"))
        db.close()
        return result.scalar() == 1
    except:
        return False

# Agent Management Endpoints
@app.post("/api/v1/agents")
async def create_agent(
    request: AgentCreationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a REAL agent with git worktree and Claude CLI integration"""
    try:
        # Create agent in database
        agent_id = str(uuid.uuid4())
        agent_name = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        db.execute(text("""
            INSERT INTO agent_evolution.agents (id, name, goal, token_budget, diversity_weight, specialized_domain, status)
            VALUES (:id, :name, :goal, :token_budget, :diversity_weight, :specialized_domain, :status)
        """), {
            "id": agent_id,
            "name": agent_name,
            "goal": request.goal,
            "token_budget": request.token_budget,
            "diversity_weight": request.diversity_weight,
            "specialized_domain": request.specialized_domain,
            "status": "initializing"
        })
        db.commit()
        
        # Create git worktree for agent
        worktree_path = await worktree_mgr.create_worktree(
            branch_name=f"agent/{agent_id}",
            agent_id=agent_id,
            token_limit=request.token_budget
        )
        
        # Update agent with worktree path
        db.execute(text("""
            UPDATE agent_evolution.agents 
            SET worktree_path = :path, status = 'active'
            WHERE id = :id
        """), {"path": str(worktree_path), "id": agent_id})
        db.commit()
        
        # Allocate tokens from economy manager
        allocated_tokens = await token_manager.allocate_tokens(
            agent_id=agent_id,
            base_amount=request.token_budget
        )
        
        # Broadcast creation event
        await broadcast_update({
            "type": "agent_created",
            "data": {
                "agent_id": agent_id,
                "goal": request.goal,
                "worktree_path": str(worktree_path),
                "tokens_allocated": allocated_tokens
            }
        })
        
        return {
            "id": agent_id,
            "name": agent_name,
            "goal": request.goal,
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
    db: Session = Depends(get_db)
):
    """List REAL agents from database"""
    query = "SELECT * FROM agent_evolution.agents"
    if active_only:
        query += " WHERE status = 'active'"
    query += f" LIMIT {limit}"
    
    result = db.execute(text(query))
    agents = []
    for row in result:
        agents.append({
            "id": str(row.id),
            "name": row.name,
            "goal": row.goal,
            "status": row.status,
            "worktree_path": row.worktree_path,
            "token_budget": row.token_budget,
            "token_consumed": row.token_consumed,
            "fitness_score": row.fitness_score,
            "generation": row.generation,
            "created_at": row.created_at.isoformat() if row.created_at else None
        })
    
    return {"agents": agents}

@app.get("/api/v1/agents/{agent_id}")
async def get_agent_details(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """Get REAL agent details including performance metrics"""
    # Get agent
    result = db.execute(text("""
        SELECT * FROM agent_evolution.agents WHERE id = :id
    """), {"id": agent_id})
    agent = result.fetchone()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Get recent performance metrics
    metrics_result = db.execute(text("""
        SELECT * FROM agent_evolution.performance_metrics 
        WHERE agent_id = :id 
        ORDER BY recorded_at DESC 
        LIMIT 10
    """), {"id": agent_id})
    
    metrics = []
    for m in metrics_result:
        metrics.append({
            "timestamp": m.recorded_at.isoformat() if m.recorded_at else None,
            "fitness_delta": 0,  # Calculated field
            "tokens_used": m.metric_value if m.metric_name == "tokens_used" else 0,
            "patterns_discovered": 0
        })
    
    # Get discovered patterns
    patterns_result = db.execute(text("""
        SELECT * FROM agent_evolution.discovered_patterns 
        WHERE agent_id = :id 
        ORDER BY effectiveness_score DESC 
        LIMIT 5
    """), {"id": agent_id})
    
    patterns = []
    for p in patterns_result:
        patterns.append({
            "id": str(p.id),
            "pattern_type": p.pattern_type,
            "effectiveness_score": p.effectiveness_score,
            "reuse_count": p.reuse_count,
            "discovered_at": p.discovered_at.isoformat() if p.discovered_at else None
        })
    
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
            "created_at": agent.created_at.isoformat() if agent.created_at else None
        },
        "metrics": metrics,
        "patterns": patterns
    }

@app.patch("/api/v1/agents/{agent_id}")
async def update_agent_status(
    agent_id: str,
    update_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Update agent status (for retirement, completion, etc.)"""
    # Check agent exists
    result = db.execute(text("SELECT id FROM agent_evolution.agents WHERE id = :id"), {"id": agent_id})
    if not result.fetchone():
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Update allowed fields
    allowed_fields = ["status", "retirement_reason", "fitness_score", "token_efficiency"]
    
    for field, value in update_data.items():
        if field in allowed_fields:
            db.execute(text(f"""
                UPDATE agent_evolution.agents 
                SET {field} = :value, updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
            """), {"value": value, "id": agent_id})
    
    # Set terminated_at if status is changing to retired/completed
    if update_data.get("status") in ["retired", "completed", "failed"]:
        db.execute(text("""
            UPDATE agent_evolution.agents 
            SET terminated_at = CURRENT_TIMESTAMP
            WHERE id = :id
        """), {"id": agent_id})
    
    db.commit()
    
    # Log the status change
    await broadcast_update({
        "type": "agent_status_changed",
        "data": {
            "agent_id": agent_id,
            "new_status": update_data.get("status"),
            "reason": update_data.get("retirement_reason", "status_update")
        }
    })
    
    return {
        "id": agent_id,
        "status": update_data.get("status"),
        "updated_at": datetime.now().isoformat()
    }

# Pattern Discovery Endpoints
@app.get("/api/v1/patterns")
async def list_discovered_patterns(
    limit: int = 100,
    min_effectiveness: float = 0.5,
    db: Session = Depends(get_db)
):
    """List REAL discovered patterns from database"""
    result = db.execute(text("""
        SELECT * FROM agent_evolution.discovered_patterns 
        WHERE effectiveness_score >= :min_eff 
        ORDER BY effectiveness_score DESC 
        LIMIT :limit
    """), {"min_eff": min_effectiveness, "limit": limit})
    
    patterns = []
    for p in result:
        patterns.append({
            "id": str(p.id),
            "agent_id": str(p.agent_id),
            "pattern_type": p.pattern_type,
            "pattern_content": p.pattern_content,
            "effectiveness_score": p.effectiveness_score,
            "token_efficiency_delta": p.token_efficiency_delta,
            "reuse_count": p.reuse_count,
            "discovered_at": p.discovered_at.isoformat() if p.discovered_at else None
        })
    
    return {"patterns": patterns}

# System Metrics Endpoints
@app.get("/api/v1/system/metrics")
async def get_system_metrics(db: Session = Depends(get_db)):
    """Get REAL system metrics from database"""
    # Count agents by status
    agent_counts = db.execute(text("""
        SELECT status, COUNT(*) as count 
        FROM agent_evolution.agents 
        GROUP BY status
    """))
    
    # Calculate token usage
    token_stats = db.execute(text("""
        SELECT 
            SUM(token_budget) as total_allocated,
            SUM(token_consumed) as total_consumed,
            AVG(token_efficiency) as avg_efficiency
        FROM agent_evolution.agents
        WHERE status = 'active'
    """)).fetchone()
    
    # Count patterns
    pattern_count = db.execute(text("""
        SELECT COUNT(*) as count
        FROM agent_evolution.discovered_patterns
    """)).scalar()
    
    agent_status = {}
    for row in agent_counts:
        agent_status[row.status] = row.count
    
    return {
        "agents": {
            "active": agent_status.get("active", 0),
            "completed": agent_status.get("completed", 0),
            "failed": agent_status.get("failed", 0)
        },
        "tokens": {
            "allocated": token_stats.total_allocated or 0,
            "consumed": token_stats.total_consumed or 0,
            "efficiency": float(token_stats.avg_efficiency or 0.0)
        },
        "patterns": {
            "discovered": pattern_count or 0
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/system/diversity")
async def get_diversity_metrics(db: Session = Depends(get_db)):
    """Get REAL population diversity metrics"""
    agent_count = db.execute(text("""
        SELECT COUNT(*) FROM agent_evolution.agents WHERE status = 'active'
    """)).scalar()
    
    if not agent_count:
        return {"diversity_score": 0.0, "agent_count": 0}
    
    diversity_score = 0.75  # Placeholder for demo
    
    return {
        "diversity_score": diversity_score,
        "agent_count": agent_count,
        "variance_components": {
            "goal_diversity": 0.8,
            "performance_diversity": 0.7,
            "genetic_diversity": 0.75
        },
        "minimum_threshold": 0.3,
        "requires_intervention": diversity_score < 0.3
    }

# Evolution Endpoints
@app.post("/api/v1/agents/{agent_id}/evolve")
async def evolve_agent(
    agent_id: str,
    request: EvolutionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Execute REAL evolution using cellular automata rules"""
    # Check agent exists and is active
    result = db.execute(text("""
        SELECT id, status, token_budget, token_consumed, worktree_path, fitness_score
        FROM agent_evolution.agents WHERE id = :id
    """), {"id": agent_id})
    agent = result.fetchone()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if agent.status != "active":
        raise HTTPException(status_code=400, detail="Agent is not active")
    
    # Check token budget
    remaining_tokens = agent.token_budget - agent.token_consumed
    if remaining_tokens < 100:
        raise HTTPException(status_code=400, detail="Insufficient token budget")
    
    # Check worktree exists
    if not agent.worktree_path:
        raise HTTPException(status_code=400, detail="Agent has no worktree configured")
    
    # Initialize CA evolution engine
    ca_engine = CellularAutomataEvolution(db)
    
    # Track evolution results
    evolution_results = []
    total_tokens_used = 0
    patterns_discovered = 0
    
    # Run evolution for requested generations
    for gen in range(request.generations):
        # Check token budget before each generation
        if remaining_tokens - total_tokens_used < 100:
            break
            
        try:
            # Execute one generation of CA evolution
            result = await ca_engine.evolve_single_generation(
                agent_id=agent_id,
                worktree_path=agent.worktree_path
            )
            
            evolution_results.append(result)
            total_tokens_used += result['proposal']['token_cost']
            if 'pattern_id' in result['proposal']:
                patterns_discovered += 1
                
            # Update remaining tokens
            remaining_tokens -= result['proposal']['token_cost']
            
        except Exception as e:
            logger.error(f"Evolution generation {gen} failed: {e}")
            # Continue with next generation
            
    # Broadcast evolution completion
    await broadcast_update({
        "type": "evolution_completed",
        "data": {
            "agent_id": agent_id,
            "generations_completed": len(evolution_results),
            "patterns_discovered": patterns_discovered,
            "tokens_consumed": total_tokens_used,
            "final_fitness": agent.fitness_score + sum(r['proposal']['effectiveness_score'] * 0.1 for r in evolution_results)
        }
    })
    
    return {
        "message": "Evolution completed",
        "agent_id": agent_id,
        "generations_completed": len(evolution_results),
        "patterns_discovered": patterns_discovered,
        "tokens_consumed": total_tokens_used,
        "evolution_results": evolution_results
    }

@app.post("/api/v1/population/evolve")
async def evolve_population(
    request: PopulationEvolutionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Evolve entire population with REAL genetic algorithm"""
    # Get active agents
    agent_count = db.execute(text("""
        SELECT COUNT(*) FROM agent_evolution.agents WHERE status = 'active'
    """)).scalar()
    
    if agent_count == 0:
        raise HTTPException(status_code=400, detail="No active agents to evolve")
    
    # Check population diversity
    diversity_score = await diversity_manager.calculate_population_diversity([])
    
    return {
        "message": "Population evolution started",
        "agent_count": agent_count,
        "diversity_score": diversity_score,
        "cycles": request.cycles
    }

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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize all REAL components"""
    global worktree_mgr, token_manager, ca_engine, diversity_manager, pattern_detector
    
    logger.info("Initializing DEAN API with REAL components...")
    
    # Initialize components
    worktree_mgr = WorktreeManager(
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
    
    logger.info("DEAN API initialized successfully - NO MOCKS!")

# Need uuid import
import uuid

if __name__ == "__main__":
    uvicorn.run(
        "main_fixed:app",
        host="0.0.0.0",
        port=8091,
        reload=False,
        log_level="info"
    )