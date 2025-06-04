#!/usr/bin/env python3
"""
DEAN Agent Evolution Service - FastAPI Main Application

This service provides REST endpoints for the Distributed Evolutionary Agent Network (DEAN)
system, enabling agent creation, evolution, and management through HTTP APIs.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.dean.agents.fractal_agent import FractalAgent, AgentConfig, AgentState, EvolutionResult
from src.dean.agents.agent_factory import AgentFactory
from src.dean.agents.cellular_automata import CellularAutomataEngine, CAState, CARule
from src.dean.economy.token_economy import TokenEconomy
from src.dean.repository.repository_manager import RepositoryManager
from src.dean.diversity.diversity_manager import DiversityManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
agent_factory: Optional[AgentFactory] = None
ca_engine: Optional[CellularAutomataEngine] = None
token_economy: Optional[TokenEconomy] = None
repository_manager: Optional[RepositoryManager] = None
diversity_manager: Optional[DiversityManager] = None

# Active agents registry
active_agents: Dict[str, FractalAgent] = {}
evolution_tasks: Dict[str, asyncio.Task] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup
    logger.info("Starting DEAN Agent Evolution Service...")
    await initialize_services()
    logger.info("DEAN Agent Evolution Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DEAN Agent Evolution Service...")
    await cleanup_services()
    logger.info("DEAN Agent Evolution Service shut down")

async def initialize_services():
    """Initialize all core services."""
    global agent_factory, ca_engine, token_economy, repository_manager, diversity_manager
    
    try:
        # Initialize core services
        token_economy = TokenEconomy()
        repository_manager = RepositoryManager()
        diversity_manager = DiversityManager()
        
        # Initialize CA engine
        ca_engine = CellularAutomataEngine(
            token_economy=token_economy,
            repository_manager=repository_manager,
            diversity_manager=diversity_manager
        )
        
        # Initialize agent factory
        agent_factory = AgentFactory(
            token_economy=token_economy,
            repository_manager=repository_manager,
            diversity_manager=diversity_manager,
            ca_engine=ca_engine
        )
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

async def cleanup_services():
    """Cleanup all services and active agents."""
    global active_agents, evolution_tasks
    
    # Cancel all evolution tasks
    for task in evolution_tasks.values():
        if not task.done():
            task.cancel()
    
    # Wait for tasks to complete
    if evolution_tasks:
        await asyncio.gather(*evolution_tasks.values(), return_exceptions=True)
    
    # Cleanup active agents
    for agent in active_agents.values():
        try:
            await agent.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up agent {agent.agent_id}: {e}")
    
    active_agents.clear()
    evolution_tasks.clear()

# Initialize FastAPI app
app = FastAPI(
    title="DEAN Agent Evolution Service",
    description="Distributed Evolutionary Agent Network - Agent Evolution and Management API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class AgentCreateRequest(BaseModel):
    """Request model for creating a new agent."""
    agent_id: Optional[str] = None
    goal: str = Field(..., description="Primary goal/objective for the agent")
    token_budget: int = Field(default=1000, ge=100, le=100000, description="Initial token budget")
    specialized_domain: Optional[str] = Field(default=None, description="Domain of specialization")
    diversity_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for diversity in evolution")

class AgentResponse(BaseModel):
    """Response model for agent information."""
    agent_id: str
    state: str
    goal: str
    token_budget: int
    tokens_consumed: int
    efficiency: float
    generation: int
    diversity_score: float
    specialized_domain: Optional[str]

class PopulationCreateRequest(BaseModel):
    """Request model for creating a population of agents."""
    population_size: int = Field(..., ge=2, le=50, description="Number of agents in population")
    base_goal: str = Field(..., description="Base goal for all agents")
    base_token_budget: int = Field(default=1000, ge=100, le=10000, description="Base token budget")
    diversity_factor: float = Field(default=0.3, ge=0.0, le=1.0, description="Diversity factor for population")

class EvolutionRequest(BaseModel):
    """Request model for triggering evolution."""
    agent_id: str
    environment: Dict[str, Any] = Field(default_factory=dict, description="Environmental parameters")
    cycles: int = Field(default=1, ge=1, le=10, description="Number of evolution cycles")

class EvolutionResponse(BaseModel):
    """Response model for evolution results."""
    agent_id: str
    cycles_completed: int
    final_efficiency: float
    tokens_used: int
    rules_applied: List[str]
    children_created: List[str]
    patterns_discovered: List[str]

# Dependency injection
async def get_agent_factory() -> AgentFactory:
    """Dependency to get agent factory."""
    if agent_factory is None:
        raise HTTPException(status_code=503, detail="Agent factory not initialized")
    return agent_factory

async def get_ca_engine() -> CellularAutomataEngine:
    """Dependency to get CA engine."""
    if ca_engine is None:
        raise HTTPException(status_code=503, detail="CA engine not initialized")
    return ca_engine

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "DEAN Agent Evolution Service",
        "active_agents": len(active_agents),
        "running_evolutions": len([t for t in evolution_tasks.values() if not t.done()])
    }

@app.post("/agents", response_model=AgentResponse)
async def create_agent(
    request: AgentCreateRequest,
    factory: AgentFactory = Depends(get_agent_factory)
):
    """Create a new agent."""
    try:
        config = AgentConfig(
            agent_id=request.agent_id,
            goal=request.goal,
            token_budget=request.token_budget,
            specialized_domain=request.specialized_domain,
            diversity_weight=request.diversity_weight
        )
        
        agent = await factory.create_agent(config)
        active_agents[agent.agent_id] = agent
        
        return AgentResponse(
            agent_id=agent.agent_id,
            state=agent.state.value,
            goal=agent.goal,
            token_budget=agent.token_budget,
            tokens_consumed=agent.tokens_consumed,
            efficiency=agent.get_efficiency(),
            generation=agent.generation,
            diversity_score=agent.diversity_score,
            specialized_domain=agent.specialized_domain
        )
        
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@app.post("/populations", response_model=List[AgentResponse])
async def create_population(
    request: PopulationCreateRequest,
    factory: AgentFactory = Depends(get_agent_factory)
):
    """Create a population of agents."""
    try:
        base_config = AgentConfig(
            goal=request.base_goal,
            token_budget=request.base_token_budget
        )
        
        agents = await factory.create_population(
            population_size=request.population_size,
            base_config=base_config,
            diversity_factor=request.diversity_factor
        )
        
        # Register all agents
        for agent in agents:
            active_agents[agent.agent_id] = agent
        
        return [
            AgentResponse(
                agent_id=agent.agent_id,
                state=agent.state.value,
                goal=agent.goal,
                token_budget=agent.token_budget,
                tokens_consumed=agent.tokens_consumed,
                efficiency=agent.get_efficiency(),
                generation=agent.generation,
                diversity_score=agent.diversity_score,
                specialized_domain=agent.specialized_domain
            )
            for agent in agents
        ]
        
    except Exception as e:
        logger.error(f"Failed to create population: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create population: {str(e)}")

@app.get("/agents", response_model=List[AgentResponse])
async def list_agents():
    """List all active agents."""
    return [
        AgentResponse(
            agent_id=agent.agent_id,
            state=agent.state.value,
            goal=agent.goal,
            token_budget=agent.token_budget,
            tokens_consumed=agent.tokens_consumed,
            efficiency=agent.get_efficiency(),
            generation=agent.generation,
            diversity_score=agent.diversity_score,
            specialized_domain=agent.specialized_domain
        )
        for agent in active_agents.values()
    ]

@app.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get details of a specific agent."""
    if agent_id not in active_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = active_agents[agent_id]
    return AgentResponse(
        agent_id=agent.agent_id,
        state=agent.state.value,
        goal=agent.goal,
        token_budget=agent.token_budget,
        tokens_consumed=agent.tokens_consumed,
        efficiency=agent.get_efficiency(),
        generation=agent.generation,
        diversity_score=agent.diversity_score,
        specialized_domain=agent.specialized_domain
    )

@app.post("/agents/{agent_id}/evolve", response_model=EvolutionResponse)
async def evolve_agent(
    agent_id: str,
    request: EvolutionRequest,
    background_tasks: BackgroundTasks
):
    """Trigger evolution for a specific agent."""
    if agent_id not in active_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = active_agents[agent_id]
    
    # Check if agent is already evolving
    if agent_id in evolution_tasks and not evolution_tasks[agent_id].done():
        raise HTTPException(status_code=409, detail="Agent is already evolving")
    
    try:
        # Run evolution cycles
        total_tokens_used = 0
        all_rules_applied = []
        all_children_created = []
        all_patterns_discovered = []
        
        for cycle in range(request.cycles):
            result = await agent.evolve(request.environment)
            
            total_tokens_used += result.tokens_used
            all_rules_applied.extend(result.rules_applied)
            all_children_created.extend(result.children_created)
            all_patterns_discovered.extend(result.patterns_discovered)
            
            # Register any new child agents
            for child_id in result.children_created:
                if child_id in result.new_agents:
                    active_agents[child_id] = result.new_agents[child_id]
        
        return EvolutionResponse(
            agent_id=agent_id,
            cycles_completed=request.cycles,
            final_efficiency=agent.get_efficiency(),
            tokens_used=total_tokens_used,
            rules_applied=list(set(all_rules_applied)),
            children_created=list(set(all_children_created)),
            patterns_discovered=list(set(all_patterns_discovered))
        )
        
    except Exception as e:
        logger.error(f"Failed to evolve agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Evolution failed: {str(e)}")

@app.post("/population/evolve")
async def evolve_population(
    cycles: int = Field(default=1, ge=1, le=5),
    ca_engine: CellularAutomataEngine = Depends(get_ca_engine)
):
    """Trigger population-wide evolution using cellular automata rules."""
    if not active_agents:
        raise HTTPException(status_code=400, detail="No active agents to evolve")
    
    try:
        # Create CA state from active agents
        ca_state = CAState(population=list(active_agents.values()))
        
        # Run evolution cycles
        for cycle in range(cycles):
            await ca_engine.evolve_population(ca_state)
            
            # Update active agents registry with any new agents
            for agent in ca_state.population:
                active_agents[agent.agent_id] = agent
        
        return {
            "message": f"Population evolution completed ({cycles} cycles)",
            "total_agents": len(active_agents),
            "population_efficiency": ca_state.get_average_efficiency(),
            "diversity_score": ca_state.get_diversity_score()
        }
        
    except Exception as e:
        logger.error(f"Failed to evolve population: {e}")
        raise HTTPException(status_code=500, detail=f"Population evolution failed: {str(e)}")

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent and cleanup resources."""
    if agent_id not in active_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        # Cancel evolution task if running
        if agent_id in evolution_tasks and not evolution_tasks[agent_id].done():
            evolution_tasks[agent_id].cancel()
            del evolution_tasks[agent_id]
        
        # Cleanup agent
        agent = active_agents[agent_id]
        await agent.cleanup()
        del active_agents[agent_id]
        
        return {"message": f"Agent {agent_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")

@app.get("/population/stats")
async def get_population_stats():
    """Get population-wide statistics."""
    if not active_agents:
        return {
            "total_agents": 0,
            "average_efficiency": 0.0,
            "total_tokens_consumed": 0,
            "diversity_score": 0.0,
            "state_distribution": {}
        }
    
    agents = list(active_agents.values())
    total_efficiency = sum(agent.get_efficiency() for agent in agents)
    total_tokens = sum(agent.tokens_consumed for agent in agents)
    
    # Calculate state distribution
    state_counts = {}
    for agent in agents:
        state = agent.state.value
        state_counts[state] = state_counts.get(state, 0) + 1
    
    # Calculate diversity score (simplified)
    ca_state = CAState(population=agents)
    diversity_score = ca_state.get_diversity_score()
    
    return {
        "total_agents": len(agents),
        "average_efficiency": total_efficiency / len(agents),
        "total_tokens_consumed": total_tokens,
        "diversity_score": diversity_score,
        "state_distribution": state_counts
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )