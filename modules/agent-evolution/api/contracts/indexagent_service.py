#!/usr/bin/env python3
"""
IndexAgent Service API Contracts
Implementation of Evolution API to IndexAgent communication per Service Communication Section 3.4

This module implements the service communication patterns from Evolution API to IndexAgent
as specified in the architectural design document, enabling pattern detection, agent analysis,
and code search operations.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp
from pydantic import BaseModel, Field, validator

# Implements Service Communication Section 3.4: Evolution API to IndexAgent contracts
logger = logging.getLogger(__name__)


class IndexAgentCommunicationError(Exception):
    """Exception raised for IndexAgent communication failures"""
    pass


class AnalysisRequestType(str, Enum):
    """Analysis request types for IndexAgent"""
    PATTERN_DETECTION = "pattern_detection"
    CODE_ANALYSIS = "code_analysis" 
    REPOSITORY_SEARCH = "repository_search"
    AGENT_ANALYSIS = "agent_analysis"


class PatternDetectionRequest(BaseModel):
    """Pattern detection request to IndexAgent per FR-022"""
    agent_ids: List[str] = Field(..., description="Agent IDs to analyze")
    analysis_type: str = Field(default="comprehensive", description="Analysis depth")
    pattern_types: List[str] = Field(
        default=["behavioral", "optimization", "structural", "temporal"],
        description="Pattern types to detect"
    )
    context_window: int = Field(default=100, gt=0, description="Context window size")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold")


class CodeAnalysisRequest(BaseModel):
    """Code analysis request to IndexAgent"""
    repository_url: str = Field(..., description="Repository URL to analyze")
    analysis_scope: str = Field(default="full", description="Analysis scope")
    focus_areas: List[str] = Field(
        default=["performance", "security", "maintainability"],
        description="Analysis focus areas"
    )
    agent_context: Optional[str] = Field(None, description="Agent context for analysis")


class RepositorySearchRequest(BaseModel):
    """Repository search request to IndexAgent"""
    query: str = Field(..., description="Search query")
    repository_filter: Optional[List[str]] = Field(None, description="Repository filter")
    file_types: Optional[List[str]] = Field(None, description="File type filter")
    max_results: int = Field(default=50, gt=0, le=1000, description="Maximum results")
    include_context: bool = Field(default=True, description="Include surrounding context")


class AgentAnalysisRequest(BaseModel):
    """Agent analysis request to IndexAgent per FR-006"""
    agent_id: str = Field(..., description="Agent ID to analyze")
    analysis_metrics: List[str] = Field(
        default=["efficiency", "diversity", "performance"],
        description="Metrics to analyze"
    )
    time_range: Optional[str] = Field(None, description="Time range for analysis")
    comparison_agents: Optional[List[str]] = Field(None, description="Agents for comparison")


class WorktreeOperationRequest(BaseModel):
    """Worktree operation request to IndexAgent per FR-001"""
    operation: str = Field(..., description="Worktree operation type")
    repository_url: str = Field(..., description="Repository URL")
    branch: Optional[str] = Field(default="main", description="Branch to checkout")
    agent_id: str = Field(..., description="Agent requesting worktree")
    workspace_config: Dict[str, Any] = Field(default_factory=dict, description="Workspace configuration")

    @validator('operation')
    def validate_operation(cls, v):
        """Validate worktree operation"""
        valid_operations = ['create', 'delete', 'update', 'list']
        if v not in valid_operations:
            raise ValueError(f"Operation must be one of {valid_operations}")
        return v


class IndexAgentResponse(BaseModel):
    """Standard IndexAgent service response"""
    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    service_name: str = Field(default="indexagent", description="Service name")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")


@dataclass
class IndexAgentEndpoint:
    """IndexAgent endpoint configuration"""
    name: str
    base_url: str
    timeout: int = 30
    retry_count: int = 3


class IndexAgentServiceClient:
    """
    IndexAgent Service API client implementing Service Communication Section 3.4
    
    Provides standardized communication patterns from Evolution API to IndexAgent
    for pattern detection, code analysis, and repository operations.
    """
    
    def __init__(self, indexagent_api_url: str = "http://indexagent:8081/api/v1"):
        """
        Initialize IndexAgent Service client
        
        Args:
            indexagent_api_url: Base URL for IndexAgent API service
        """
        self.indexagent_endpoint = IndexAgentEndpoint(
            name="indexagent_api",
            base_url=indexagent_api_url
        )
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.indexagent_endpoint.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def detect_patterns(self, request: PatternDetectionRequest) -> IndexAgentResponse:
        """
        Request pattern detection from IndexAgent per FR-022
        
        Implements pattern detection for specified agents using IndexAgent's
        pattern detection capabilities as specified in Service Communication Section 3.4.
        
        Args:
            request: Pattern detection request
            
        Returns:
            IndexAgentResponse with detected patterns
            
        Raises:
            IndexAgentCommunicationError: If pattern detection fails
        """
        endpoint = f"{self.indexagent_endpoint.base_url}/patterns/detect"
        start_time = datetime.now()
        
        try:
            response_data = await self._make_request(
                "POST",
                endpoint,
                request.dict()
            )
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"Pattern detection completed for {len(request.agent_ids)} agents")
            return IndexAgentResponse(
                success=True,
                message=f"Pattern detection completed for {len(request.agent_ids)} agents",
                data=response_data,
                request_id=self._generate_request_id(),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            raise IndexAgentCommunicationError(f"Pattern detection failed: {e}")
    
    async def analyze_code(self, request: CodeAnalysisRequest) -> IndexAgentResponse:
        """
        Request code analysis from IndexAgent
        
        Implements code analysis using IndexAgent's analysis capabilities
        for repository evaluation and improvement suggestions.
        
        Args:
            request: Code analysis request
            
        Returns:
            IndexAgentResponse with analysis results
            
        Raises:
            IndexAgentCommunicationError: If code analysis fails
        """
        endpoint = f"{self.indexagent_endpoint.base_url}/code/analyze"
        start_time = datetime.now()
        
        try:
            response_data = await self._make_request(
                "POST",
                endpoint,
                request.dict()
            )
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"Code analysis completed for repository: {request.repository_url}")
            return IndexAgentResponse(
                success=True,
                message=f"Code analysis completed for {request.repository_url}",
                data=response_data,
                request_id=self._generate_request_id(),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            raise IndexAgentCommunicationError(f"Code analysis failed: {e}")
    
    async def search_repositories(self, request: RepositorySearchRequest) -> IndexAgentResponse:
        """
        Search repositories using IndexAgent
        
        Implements repository search using IndexAgent's Zoekt-based search
        capabilities for code discovery and pattern identification.
        
        Args:
            request: Repository search request
            
        Returns:
            IndexAgentResponse with search results
            
        Raises:
            IndexAgentCommunicationError: If search fails
        """
        endpoint = f"{self.indexagent_endpoint.base_url}/repositories/search"
        start_time = datetime.now()
        
        try:
            response_data = await self._make_request(
                "POST",
                endpoint,
                request.dict()
            )
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"Repository search completed: '{request.query}' returned {response_data.get('result_count', 0)} results")
            return IndexAgentResponse(
                success=True,
                message=f"Repository search completed for query: {request.query}",
                data=response_data,
                request_id=self._generate_request_id(),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Repository search failed: {e}")
            raise IndexAgentCommunicationError(f"Repository search failed: {e}")
    
    async def analyze_agent(self, request: AgentAnalysisRequest) -> IndexAgentResponse:
        """
        Request agent analysis from IndexAgent per FR-006
        
        Implements agent performance analysis using IndexAgent's metrics
        collection and analysis capabilities.
        
        Args:
            request: Agent analysis request
            
        Returns:
            IndexAgentResponse with agent analysis
            
        Raises:
            IndexAgentCommunicationError: If agent analysis fails
        """
        endpoint = f"{self.indexagent_endpoint.base_url}/agents/{request.agent_id}/analyze"
        start_time = datetime.now()
        
        try:
            response_data = await self._make_request(
                "POST",
                endpoint,
                request.dict()
            )
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"Agent analysis completed for agent: {request.agent_id}")
            return IndexAgentResponse(
                success=True,
                message=f"Agent analysis completed for {request.agent_id}",
                data=response_data,
                request_id=self._generate_request_id(),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Agent analysis failed: {e}")
            raise IndexAgentCommunicationError(f"Agent analysis failed: {e}")
    
    async def manage_worktree(self, request: WorktreeOperationRequest) -> IndexAgentResponse:
        """
        Manage agent worktrees per FR-001
        
        Implements worktree management using IndexAgent's isolated git
        worktree capabilities for agent operations.
        
        Args:
            request: Worktree operation request
            
        Returns:
            IndexAgentResponse with operation results
            
        Raises:
            IndexAgentCommunicationError: If worktree operation fails
        """
        if request.operation == "create":
            endpoint = f"{self.indexagent_endpoint.base_url}/worktrees"
            method = "POST"
        elif request.operation == "delete":
            endpoint = f"{self.indexagent_endpoint.base_url}/worktrees/{request.agent_id}"
            method = "DELETE"
        elif request.operation == "list":
            endpoint = f"{self.indexagent_endpoint.base_url}/worktrees"
            method = "GET"
        else:
            endpoint = f"{self.indexagent_endpoint.base_url}/worktrees/{request.agent_id}"
            method = "PUT"
        
        start_time = datetime.now()
        
        try:
            request_data = request.dict() if method in ["POST", "PUT"] else None
            response_data = await self._make_request(method, endpoint, request_data)
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"Worktree {request.operation} completed for agent: {request.agent_id}")
            return IndexAgentResponse(
                success=True,
                message=f"Worktree {request.operation} completed for agent {request.agent_id}",
                data=response_data,
                request_id=self._generate_request_id(),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Worktree {request.operation} failed: {e}")
            raise IndexAgentCommunicationError(f"Worktree {request.operation} failed: {e}")
    
    async def get_agent_metrics(self, agent_id: str, metric_types: Optional[List[str]] = None) -> IndexAgentResponse:
        """
        Get agent metrics from IndexAgent per FR-006
        
        Args:
            agent_id: Agent identifier
            metric_types: Specific metric types to retrieve
            
        Returns:
            IndexAgentResponse with agent metrics
            
        Raises:
            IndexAgentCommunicationError: If metrics retrieval fails
        """
        endpoint = f"{self.indexagent_endpoint.base_url}/agents/{agent_id}/metrics"
        start_time = datetime.now()
        
        params = {}
        if metric_types:
            params['metric_types'] = metric_types
        
        try:
            response_data = await self._make_request("GET", endpoint, params)
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"Agent metrics retrieved for agent: {agent_id}")
            return IndexAgentResponse(
                success=True,
                message=f"Agent metrics retrieved for {agent_id}",
                data=response_data,
                request_id=self._generate_request_id(),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Agent metrics retrieval failed: {e}")
            raise IndexAgentCommunicationError(f"Agent metrics retrieval failed: {e}")
    
    async def health_check(self) -> IndexAgentResponse:
        """
        Check IndexAgent health per service monitoring requirements
        
        Returns:
            IndexAgentResponse with health status
        """
        endpoint = f"{self.indexagent_endpoint.base_url}/health"
        
        try:
            response_data = await self._make_request("GET", endpoint)
            
            return IndexAgentResponse(
                success=True,
                message="IndexAgent health check passed",
                data=response_data,
                request_id=self._generate_request_id()
            )
            
        except Exception as e:
            logger.error(f"IndexAgent health check failed: {e}")
            return IndexAgentResponse(
                success=False,
                message=f"IndexAgent health check failed: {e}",
                data={"error": str(e)},
                request_id=self._generate_request_id()
            )
    
    async def _make_request(self, method: str, url: str, 
                          data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and error handling
        
        Args:
            method: HTTP method
            url: Request URL
            data: Request data
            
        Returns:
            Response data
            
        Raises:
            IndexAgentCommunicationError: If request fails after retries
        """
        if not self.session:
            raise IndexAgentCommunicationError("Session not initialized")
        
        for attempt in range(self.indexagent_endpoint.retry_count):
            try:
                kwargs = {"headers": {"Content-Type": "application/json"}}
                
                if method in ["POST", "PUT"] and data:
                    kwargs["json"] = data
                elif method == "GET" and data:
                    kwargs["params"] = data
                
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        raise IndexAgentCommunicationError(f"Endpoint not found: {url}")
                    elif response.status == 503:
                        raise IndexAgentCommunicationError("IndexAgent service unavailable")
                    else:
                        error_text = await response.text()
                        raise IndexAgentCommunicationError(
                            f"HTTP {response.status}: {error_text}"
                        )
                        
            except aiohttp.ClientError as e:
                if attempt == self.indexagent_endpoint.retry_count - 1:
                    raise IndexAgentCommunicationError(
                        f"Request failed after {self.indexagent_endpoint.retry_count} attempts: {e}"
                    )
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                
        raise IndexAgentCommunicationError("Request failed after all retry attempts")
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid
        return str(uuid.uuid4())[:8]


# Service factory function
def create_indexagent_service_client(indexagent_api_url: Optional[str] = None) -> IndexAgentServiceClient:
    """
    Factory function to create IndexAgent Service client
    
    Args:
        indexagent_api_url: Custom IndexAgent API URL (uses default if None)
        
    Returns:
        Configured IndexAgentServiceClient
    """
    if indexagent_api_url is None:
        indexagent_api_url = "http://indexagent:8081/api/v1"
    
    return IndexAgentServiceClient(indexagent_api_url)


# Convenience functions for common operations
async def detect_agent_patterns(agent_ids: List[str], pattern_types: Optional[List[str]] = None) -> IndexAgentResponse:
    """
    Convenience function for pattern detection
    
    Implements FR-022: Novel strategy detection via IndexAgent
    """
    async with create_indexagent_service_client() as client:
        request = PatternDetectionRequest(
            agent_ids=agent_ids,
            pattern_types=pattern_types or ["behavioral", "optimization"]
        )
        return await client.detect_patterns(request)


async def analyze_repository_code(repository_url: str, focus_areas: Optional[List[str]] = None) -> IndexAgentResponse:
    """
    Convenience function for code analysis
    
    Implements repository analysis via IndexAgent capabilities
    """
    async with create_indexagent_service_client() as client:
        request = CodeAnalysisRequest(
            repository_url=repository_url,
            focus_areas=focus_areas or ["performance", "security"]
        )
        return await client.analyze_code(request)


async def create_agent_worktree(agent_id: str, repository_url: str, branch: str = "main") -> IndexAgentResponse:
    """
    Convenience function for worktree creation
    
    Implements FR-001: Agent worktree isolation via IndexAgent
    """
    async with create_indexagent_service_client() as client:
        request = WorktreeOperationRequest(
            operation="create",
            repository_url=repository_url,
            branch=branch,
            agent_id=agent_id
        )
        return await client.manage_worktree(request)


async def check_indexagent_service_health() -> bool:
    """
    Convenience function for health checking
    
    Returns:
        True if IndexAgent is healthy, False otherwise
    """
    try:
        async with create_indexagent_service_client() as client:
            response = await client.health_check()
            return response.success
    except Exception:
        return False