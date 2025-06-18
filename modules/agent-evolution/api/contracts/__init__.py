#!/usr/bin/env python3
"""
Evolution API Contracts Module for Service Communication

This module implements the Evolution API side of the service communication contracts
specified in Service Communication Section 3.4 of the DEAN system architectural design document.
It provides standardized communication patterns from Evolution API to IndexAgent and database services.

Implements:
- Evolution API ↔ IndexAgent: Pattern detection, code analysis, worktree management
- Evolution API ↔ Database: Agent lifecycle, metrics collection, audit logging
- Service discovery and health monitoring
- Retry logic and error handling

Service Communication Patterns:
- IndexAgent Service: Pattern detection, code analysis, repository search
- Database Service: Data persistence with caching layer integration
- Cross-service metrics and monitoring
"""

from .indexagent_service import (
    IndexAgentServiceClient,
    PatternDetectionRequest,
    CodeAnalysisRequest,
    RepositorySearchRequest,
    AgentAnalysisRequest,
    WorktreeOperationRequest,
    IndexAgentResponse,
    IndexAgentCommunicationError,
    create_indexagent_service_client,
    detect_agent_patterns,
    analyze_repository_code,
    create_agent_worktree,
    check_indexagent_service_health
)

__all__ = [
    # IndexAgent Service Communication
    "IndexAgentServiceClient",
    "PatternDetectionRequest",
    "CodeAnalysisRequest", 
    "RepositorySearchRequest",
    "AgentAnalysisRequest",
    "WorktreeOperationRequest",
    "IndexAgentResponse",
    "IndexAgentCommunicationError",
    "create_indexagent_service_client",
    "detect_agent_patterns",
    "analyze_repository_code",
    "create_agent_worktree",
    "check_indexagent_service_health"
]

# Evolution API Contract Version
EVOLUTION_API_CONTRACTS_VERSION = "1.0.0"

# Service Communication Configuration for Evolution API
EVOLUTION_SERVICE_CONFIG = {
    "indexagent_api": {
        "base_url": "http://indexagent:8081/api/v1",
        "timeout": 30,
        "retry_count": 3,
        "endpoints": {
            "pattern_detection": "/patterns/detect",
            "code_analysis": "/code/analyze",
            "repository_search": "/repositories/search",
            "agent_analysis": "/agents/{agent_id}/analyze",
            "worktree_management": "/worktrees",
            "agent_metrics": "/agents/{agent_id}/metrics",
            "health": "/health"
        }
    },
    "database_api": {
        "base_url": "postgresql://postgres:password@postgres:5432/agent_evolution",
        "redis_url": "redis://agent-registry:6379",
        "connection_pool": {
            "pool_size": 20,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 3600
        }
    },
    "service_discovery": {
        "health_check_interval": 30,  # seconds
        "retry_backoff": [1, 2, 4, 8],  # exponential backoff
        "circuit_breaker": {
            "failure_threshold": 5,
            "recovery_timeout": 60
        }
    }
}

def get_evolution_service_config():
    """Get Evolution API service communication configuration"""
    return EVOLUTION_SERVICE_CONFIG

def get_evolution_api_contracts_version():
    """Get Evolution API contracts version"""
    return EVOLUTION_API_CONTRACTS_VERSION

# Service Health Monitoring
async def check_all_services_health():
    """
    Check health of all dependent services
    
    Returns:
        Dict with service health status
    """
    health_status = {
        "indexagent": False,
        "timestamp": None
    }
    
    try:
        from datetime import datetime
        
        # Check IndexAgent health
        health_status["indexagent"] = await check_indexagent_service_health()
        health_status["timestamp"] = datetime.now().isoformat()
        
        return health_status
        
    except Exception as e:
        health_status["error"] = str(e)
        return health_status

# Service Communication Utilities
def get_service_endpoints():
    """Get all service endpoints for Evolution API"""
    config = get_evolution_service_config()
    
    return {
        "indexagent": {
            "base_url": config["indexagent_api"]["base_url"],
            "endpoints": config["indexagent_api"]["endpoints"]
        },
        "database": {
            "postgres_url": config["database_api"]["base_url"],
            "redis_url": config["database_api"]["redis_url"]
        }
    }

def validate_service_configuration():
    """
    Validate service communication configuration
    
    Returns:
        Tuple of (is_valid: bool, issues: List[str])
    """
    issues = []
    config = get_evolution_service_config()
    
    # Validate IndexAgent configuration
    if not config.get("indexagent_api", {}).get("base_url"):
        issues.append("IndexAgent API base URL not configured")
    
    # Validate database configuration  
    if not config.get("database_api", {}).get("base_url"):
        issues.append("Database URL not configured")
    
    if not config.get("database_api", {}).get("redis_url"):
        issues.append("Redis URL not configured")
    
    # Validate service discovery configuration
    if not config.get("service_discovery", {}).get("health_check_interval"):
        issues.append("Health check interval not configured")
    
    return len(issues) == 0, issues