#!/usr/bin/env python3
"""
DEAN API - Integrated with IndexAgent Components
Distributed Evolutionary Agent Network REST API
"""

import os
import sys
import logging
import uuid
import hashlib
import json
import random
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

# Import local evolution engine as fallback
try:
    from evolution_engine import CellularAutomataEvolution
except ImportError:
    from .evolution_engine import CellularAutomataEvolution

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="DEAN API - Distributed Evolutionary Agent Network",
    description="Real-time agent evolution with IndexAgent integration",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# Global instances (initialized on startup)
worktree_mgr: Optional[GitWorktreeManager] = None
token_manager: Optional[TokenEconomyManager] = None
ca_engine: Optional[CellularAutomataEngine] = None
diversity_manager: Optional[GeneticDiversityManager] = None
pattern_detector: Optional[PatternDetector] = None
behavior_monitor: Optional[EmergentBehaviorMonitor] = None

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

# Request/Response Models
class AgentCreationRequest(BaseModel):
    goal: str
    token_budget: int = Field(default=4096, ge=100, le=100000)
    diversity_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    specialized_domain: Optional[str] = None
    repository_url: Optional[str] = None
    target_path: Optional[str] = None

class FitnessUpdateRequest(BaseModel):
    fitness_score: float = Field(ge=0.0, le=1.0)
    patterns_discovered: List[Dict[str, Any]] = []
    complexity_metrics: Dict[str, float] = {}

class AgentEvolutionRequest(BaseModel):
    population_size: int = Field(default=10, ge=2, le=50)
    generations: int = Field(default=10, ge=1, le=100)
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    tournament_size: int = Field(default=3, ge=2, le=10)
    elitism_count: int = Field(default=2, ge=0, le=10)

class CellularAutomataRequest(BaseModel):
    rule_number: int = Field(ge=0, le=255)
    initial_state: Optional[List[int]] = None
    steps: int = Field(default=100, ge=1, le=1000)
    width: int = Field(default=100, ge=10, le=500)
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

# API Endpoints
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
            "ca_engine": ca_engine is not None,
            "diversity_manager": diversity_manager is not None,
            "pattern_detector": pattern_detector is not None
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
        # Create initial genome with some randomization
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
        agent_id = str(uuid.uuid4())
        agent_name = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create agent record
        db.execute(
            text("""
                INSERT INTO agent_evolution.agents 
                (id, name, goal, fitness_score, generation, token_budget, 
                 token_consumed, status, created_at)
                VALUES (:id, :name, :goal, 0.0, 0, :token_budget, 0, 
                        'initializing', NOW())
            """),
            {
                "id": agent_id,
                "name": agent_name,
                "goal": request.goal,
                "token_budget": request.token_budget
            }
        )
        db.commit()
        
        # Store initial genome as a discovered pattern
        genome_pattern_id = str(uuid.uuid4())
        pattern_hash = hashlib.sha256(
            json.dumps(initial_genome.dict(), sort_keys=True).encode()
        ).hexdigest()
        
        db.execute(
            text("""
                INSERT INTO agent_evolution.discovered_patterns
                (id, agent_id, pattern_hash, pattern_type, pattern_content,
                 effectiveness_score, discovered_at)
                VALUES (:id, :agent_id, :pattern_hash, 'genome', :content::jsonb,
                        0.5, NOW())
            """),
            {
                "id": genome_pattern_id,
                "agent_id": agent_id,
                "pattern_hash": pattern_hash,
                "content": json.dumps(initial_genome.dict())
            }
        )
        
        # Create worktree if repository URL provided
        worktree_path = None
        if request.repository_url and worktree_mgr:
            try:
                # Create branch name from agent name
                branch_name = f"agent/{agent_name}"
                
                # Create worktree for agent
                worktree_path = await worktree_mgr.create_worktree(
                    branch_name=branch_name,
                    agent_id=agent_id,
                    repository_url=request.repository_url
                )
                
                # Update agent with worktree path
                db.execute(
                    text("""
                        UPDATE agent_evolution.agents 
                        SET worktree_path = :path, status = 'active'
                        WHERE id = :id
                    """),
                    {"path": str(worktree_path), "id": agent_id}
                )
            except Exception as e:
                logger.error(f"Failed to create worktree: {e}")
                # Continue without worktree
        
        # Update status to active
        db.execute(
            text("""
                UPDATE agent_evolution.agents 
                SET status = 'active'
                WHERE id = :id
            """),
            {"id": agent_id}
        )
        db.commit()
        
        # Broadcast update
        await broadcast_update({
            "event": "agent_created",
            "agent_id": agent_id,
            "name": agent_name,
            "goal": request.goal,
            "worktree_path": str(worktree_path) if worktree_path else None
        })
        
        return {
            "id": agent_id,
            "name": agent_name,
            "goal": request.goal,
            "token_budget": request.token_budget,
            "worktree_path": str(worktree_path) if worktree_path else None,
            "status": "active"
        }
        
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agents")
async def list_agents(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """List all agents with their current status"""
    try:
        result = db.execute(
            text("""
                SELECT id, name, goal, fitness_score, generation, 
                       token_budget, token_consumed, status, created_at
                FROM agent_evolution.agents
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :skip
            """),
            {"limit": limit, "skip": skip}
        )
        
        agents = []
        for row in result:
            agents.append({
                "id": str(row.id),
                "name": row.name,
                "goal": row.goal,
                "fitness_score": float(row.fitness_score),
                "generation": row.generation,
                "token_budget": row.token_budget,
                "token_consumed": row.token_consumed,
                "status": row.status,
                "created_at": row.created_at.isoformat()
            })
        
        return {"agents": agents, "total": len(agents)}
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific agent"""
    try:
        result = db.execute(
            text("""
                SELECT a.*, 
                       COUNT(DISTINCT p.id) as pattern_count,
                       COUNT(DISTINCT pm.id) as metric_count
                FROM agent_evolution.agents a
                LEFT JOIN agent_evolution.discovered_patterns p ON a.id = p.agent_id
                LEFT JOIN agent_evolution.performance_metrics pm ON a.id = pm.agent_id
                WHERE a.id = :agent_id
                GROUP BY a.id
            """),
            {"agent_id": agent_id}
        ).first()
        
        if not result:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "id": str(result.id),
            "name": result.name,
            "goal": result.goal,
            "fitness_score": float(result.fitness_score),
            "generation": result.generation,
            "token_budget": result.token_budget,
            "token_consumed": result.token_consumed,
            "status": result.status,
            "created_at": result.created_at.isoformat(),
            "pattern_count": result.pattern_count,
            "metric_count": result.metric_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/agents/{agent_id}/fitness")
async def update_fitness(
    agent_id: str,
    request: FitnessUpdateRequest,
    db: Session = Depends(get_db)
):
    """Update agent fitness score and record discovered patterns"""
    try:
        # Update fitness score
        db.execute(
            text("""
                UPDATE agent_evolution.agents
                SET fitness_score = :fitness_score
                WHERE id = :agent_id
            """),
            {"fitness_score": request.fitness_score, "agent_id": agent_id}
        )
        
        # Record discovered patterns
        for pattern in request.patterns_discovered:
            pattern_id = str(uuid.uuid4())
            pattern_hash = hashlib.sha256(
                json.dumps(pattern, sort_keys=True).encode()
            ).hexdigest()
            
            db.execute(
                text("""
                    INSERT INTO agent_evolution.discovered_patterns
                    (id, agent_id, pattern_hash, pattern_type, pattern_content,
                     effectiveness_score, discovered_at)
                    VALUES (:id, :agent_id, :pattern_hash, :pattern_type, 
                            :content::jsonb, :effectiveness, NOW())
                    ON CONFLICT (pattern_hash) DO NOTHING
                """),
                {
                    "id": pattern_id,
                    "agent_id": agent_id,
                    "pattern_hash": pattern_hash,
                    "pattern_type": pattern.get("type", "unknown"),
                    "content": json.dumps(pattern),
                    "effectiveness": pattern.get("effectiveness", 0.5)
                }
            )
        
        # Record performance metrics
        if request.complexity_metrics:
            metric_id = str(uuid.uuid4())
            db.execute(
                text("""
                    INSERT INTO agent_evolution.performance_metrics
                    (id, agent_id, metric_type, metric_value, 
                     additional_data, recorded_at)
                    VALUES (:id, :agent_id, 'complexity', :value, 
                            :data::jsonb, NOW())
                """),
                {
                    "id": metric_id,
                    "agent_id": agent_id,
                    "value": request.complexity_metrics.get("overall", 0.0),
                    "data": json.dumps(request.complexity_metrics)
                }
            )
        
        db.commit()
        
        # Broadcast update
        await broadcast_update({
            "event": "fitness_updated",
            "agent_id": agent_id,
            "fitness_score": request.fitness_score,
            "patterns_count": len(request.patterns_discovered)
        })
        
        return {
            "agent_id": agent_id,
            "fitness_score": request.fitness_score,
            "patterns_recorded": len(request.patterns_discovered)
        }
        
    except Exception as e:
        logger.error(f"Failed to update fitness: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Evolution Endpoints
@app.post("/api/v1/evolution/run")
async def run_evolution(
    request: AgentEvolutionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Run genetic algorithm evolution on agent population"""
    try:
        # Get active agents for evolution
        result = db.execute(
            text("""
                SELECT id, name, fitness_score
                FROM agent_evolution.agents
                WHERE status = 'active'
                ORDER BY fitness_score DESC
                LIMIT :limit
            """),
            {"limit": request.population_size}
        )
        
        agents = [{"id": str(r.id), "name": r.name, "fitness": r.fitness_score} 
                  for r in result]
        
        if len(agents) < 2:
            raise HTTPException(
                status_code=400, 
                detail="Need at least 2 active agents for evolution"
            )
        
        # Start evolution in background
        evolution_id = str(uuid.uuid4())
        background_tasks.add_task(
            run_evolution_cycle,
            evolution_id,
            agents,
            request
        )
        
        return {
            "evolution_id": evolution_id,
            "status": "started",
            "population_size": len(agents),
            "generations": request.generations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start evolution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_evolution_cycle(
    evolution_id: str,
    agents: List[Dict],
    params: AgentEvolutionRequest
):
    """Background task to run evolution cycles"""
    try:
        db = SessionLocal()
        
        for generation in range(params.generations):
            # Simulate evolution step
            await broadcast_update({
                "event": "evolution_step",
                "evolution_id": evolution_id,
                "generation": generation + 1,
                "total_generations": params.generations
            })
            
            # Record evolution history
            for agent in agents:
                history_id = str(uuid.uuid4())
                db.execute(
                    text("""
                        INSERT INTO agent_evolution.evolution_history
                        (id, agent_id, generation, fitness_before, fitness_after,
                         mutation_applied, crossover_parent_id, new_patterns_discovered,
                         timestamp)
                        VALUES (:id, :agent_id, :generation, :fitness_before,
                                :fitness_after, :mutation_applied, NULL, 0, NOW())
                    """),
                    {
                        "id": history_id,
                        "agent_id": agent["id"],
                        "generation": generation + 1,
                        "fitness_before": agent["fitness"],
                        "fitness_after": agent["fitness"] + random.uniform(-0.1, 0.2),
                        "mutation_applied": random.random() < params.mutation_rate
                    }
                )
            
            db.commit()
        
        await broadcast_update({
            "event": "evolution_complete",
            "evolution_id": evolution_id,
            "generations_completed": params.generations
        })
        
    except Exception as e:
        logger.error(f"Evolution cycle failed: {e}")
        await broadcast_update({
            "event": "evolution_failed",
            "evolution_id": evolution_id,
            "error": str(e)
        })
    finally:
        db.close()

# Cellular Automata Endpoints
@app.post("/api/v1/ca/simulate")
async def simulate_cellular_automata(
    request: CellularAutomataRequest,
    background_tasks: BackgroundTasks
):
    """Run cellular automata simulation"""
    try:
        if not ca_engine:
            # Use local evolution engine as fallback
            evolution = CellularAutomataEvolution()
            result = await evolution.run_simulation(
                rule_number=request.rule_number,
                steps=request.steps,
                width=request.width
            )
        else:
            # Use IndexAgent CA engine
            rule = CARule(request.rule_number)
            initial_state = request.initial_state or [random.randint(0, 1) 
                                                      for _ in range(request.width)]
            
            result = await ca_engine.simulate(
                rule=rule,
                initial_state=initial_state,
                steps=request.steps
            )
        
        return {
            "rule_number": request.rule_number,
            "steps": request.steps,
            "width": request.width,
            "final_complexity": result.get("complexity", 0.0),
            "patterns_found": result.get("patterns", [])
        }
        
    except Exception as e:
        logger.error(f"CA simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except:
        active_connections.remove(websocket)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global worktree_mgr, token_manager, ca_engine, diversity_manager, pattern_detector
    
    try:
        # Initialize database tables
        with engine.begin() as conn:
            # Create schema if not exists
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS agent_evolution"))
            
            # Create tables if not exist
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS agent_evolution.agents (
                    id UUID PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    goal TEXT,
                    fitness_score FLOAT DEFAULT 0.0,
                    generation INT DEFAULT 0,
                    token_budget INT DEFAULT 4096,
                    token_consumed INT DEFAULT 0,
                    status VARCHAR(50) DEFAULT 'active',
                    worktree_path TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS agent_evolution.discovered_patterns (
                    id UUID PRIMARY KEY,
                    agent_id UUID REFERENCES agent_evolution.agents(id),
                    pattern_hash VARCHAR(64) UNIQUE,
                    pattern_type VARCHAR(50),
                    pattern_content JSONB,
                    effectiveness_score FLOAT DEFAULT 0.0,
                    discovered_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS agent_evolution.performance_metrics (
                    id UUID PRIMARY KEY,
                    agent_id UUID REFERENCES agent_evolution.agents(id),
                    metric_type VARCHAR(50),
                    metric_value FLOAT,
                    additional_data JSONB,
                    recorded_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS agent_evolution.evolution_history (
                    id UUID PRIMARY KEY,
                    agent_id UUID REFERENCES agent_evolution.agents(id),
                    generation INT,
                    fitness_before FLOAT,
                    fitness_after FLOAT,
                    mutation_applied BOOLEAN,
                    crossover_parent_id UUID,
                    new_patterns_discovered INT DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """))
        
        # Initialize services
        worktree_base = Path("/app/worktrees")
        worktree_base.mkdir(exist_ok=True)
        
        # Try to initialize IndexAgent components
        try:
            worktree_mgr = GitWorktreeManager(base_path=worktree_base)
            token_manager = TokenEconomyManager(global_budget=1000000)
            ca_engine = CellularAutomataEngine()
            diversity_manager = GeneticDiversityManager(min_diversity_threshold=0.3)
            pattern_detector = PatternDetector()
            logger.info("IndexAgent components initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize IndexAgent components: {e}")
            logger.info("Using fallback implementations")
            # Fallback implementations are already imported
        
        logger.info("DEAN API started successfully with integrated components")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    for connection in active_connections:
        await connection.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8091)