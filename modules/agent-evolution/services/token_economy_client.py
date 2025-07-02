"""
Client library for interacting with the Token Economy Service.

This provides a simple interface for other DEAN services to request tokens
and update consumption, with automatic enforcement of budget limits.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TokenAllocationError(Exception):
    """Raised when token allocation fails"""
    pass


class TokenBudgetExhausted(Exception):
    """Raised when an agent's token budget is exhausted"""
    pass


class TokenEconomyClient:
    """Client for interacting with the Token Economy Service"""
    
    def __init__(self, base_url: str = "http://token-economy:8091"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def allocate_tokens(
        self,
        agent_id: str,
        requested_tokens: float,
        task_complexity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Request token allocation for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            requested_tokens: Number of tokens requested
            task_complexity: Complexity factor (0.1-1.0)
            
        Returns:
            Dict with allocated_tokens, efficiency_multiplier, etc.
            
        Raises:
            TokenAllocationError: If allocation fails
            TokenBudgetExhausted: If global budget is exhausted
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._client.post(
                f"{self.base_url}/api/v1/tokens/allocate",
                json={
                    "agent_id": agent_id,
                    "requested_tokens": requested_tokens,
                    "task_complexity": task_complexity
                }
            )
            
            if response.status_code == 403:
                # Budget exhausted or insufficient allocation
                error_detail = response.json().get("detail", "Unknown error")
                if "Global token budget exhausted" in error_detail:
                    raise TokenBudgetExhausted(error_detail)
                else:
                    raise TokenAllocationError(error_detail)
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Token allocation failed: {e}")
            raise TokenAllocationError(f"Allocation failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during allocation: {e}")
            raise
    
    async def update_consumption(
        self,
        agent_id: str,
        tokens_consumed: float,
        value_generated: float = 0.0
    ) -> Dict[str, Any]:
        """
        Update token consumption for an agent.
        
        CRITICAL: This will TERMINATE the agent if budget is exhausted!
        
        Args:
            agent_id: Agent identifier
            tokens_consumed: Tokens consumed in this update
            value_generated: Value generated (for efficiency calculation)
            
        Returns:
            Dict with remaining tokens, efficiency, warnings
            
        Raises:
            TokenBudgetExhausted: If agent's budget is exhausted (agent terminated)
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._client.post(
                f"{self.base_url}/api/v1/tokens/consume",
                json={
                    "agent_id": agent_id,
                    "tokens_consumed": tokens_consumed,
                    "value_generated": value_generated
                }
            )
            
            if response.status_code == 403:
                # Agent terminated due to budget exhaustion
                error_detail = response.json().get("detail", "Unknown error")
                raise TokenBudgetExhausted(error_detail)
            
            response.raise_for_status()
            result = response.json()
            
            # Log warnings
            if result.get("warning"):
                logger.warning(f"Token warning for {agent_id}: {result['warning']}")
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Consumption update failed: {e}")
            raise
        except TokenBudgetExhausted:
            # Re-raise budget exhaustion
            raise
        except Exception as e:
            logger.error(f"Unexpected error during consumption update: {e}")
            raise
    
    async def get_budget_status(self) -> Dict[str, Any]:
        """Get global budget status"""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._client.get(f"{self.base_url}/api/v1/tokens/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get budget status: {e}")
            raise
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get token status for a specific agent"""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._client.get(
                f"{self.base_url}/api/v1/tokens/agent/{agent_id}"
            )
            
            if response.status_code == 404:
                return None  # No active allocation
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                logger.error(f"Failed to get agent status: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting agent status: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the token economy service is healthy"""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False


class TokenBudgetManager:
    """
    Context manager for token budget enforcement in agent tasks.
    
    Automatically tracks consumption and enforces limits.
    """
    
    def __init__(
        self,
        client: TokenEconomyClient,
        agent_id: str,
        requested_tokens: float,
        task_complexity: float = 0.5
    ):
        self.client = client
        self.agent_id = agent_id
        self.requested_tokens = requested_tokens
        self.task_complexity = task_complexity
        self.allocated_tokens = 0
        self.tokens_consumed = 0
        self.value_generated = 0
    
    async def __aenter__(self):
        # Request allocation
        result = await self.client.allocate_tokens(
            self.agent_id,
            self.requested_tokens,
            self.task_complexity
        )
        
        self.allocated_tokens = result["allocated_tokens"]
        logger.info(f"Allocated {self.allocated_tokens} tokens to {self.agent_id}")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Always update consumption, even on error
        if self.tokens_consumed > 0:
            try:
                await self.client.update_consumption(
                    self.agent_id,
                    self.tokens_consumed,
                    self.value_generated
                )
            except TokenBudgetExhausted:
                # Agent terminated - log but don't hide original exception
                logger.error(f"Agent {self.agent_id} terminated: budget exhausted")
                if exc_type is None:
                    # Only raise if there wasn't another exception
                    raise
    
    async def consume(self, tokens: float, value: float = 0.0):
        """
        Record token consumption during task execution.
        
        This should be called periodically during long-running tasks
        to ensure budget limits are enforced in real-time.
        """
        self.tokens_consumed += tokens
        self.value_generated += value
        
        # Check if we're approaching limit
        if self.tokens_consumed >= self.allocated_tokens * 0.9:
            # Update consumption to get current status
            result = await self.client.update_consumption(
                self.agent_id,
                tokens,
                value
            )
            
            # Reset local counters since we've reported
            self.tokens_consumed = 0
            self.value_generated = 0
            
            if result.get("warning"):
                logger.warning(f"Token budget warning: {result['warning']}")


# Example usage:
async def example_agent_task():
    """Example of how an agent would use the token economy"""
    
    async with TokenEconomyClient() as client:
        # Method 1: Direct API calls
        agent_id = "example-agent-001"
        
        # Allocate tokens
        allocation = await client.allocate_tokens(agent_id, 1000.0)
        print(f"Allocated: {allocation['allocated_tokens']} tokens")
        
        # Simulate work and consumption
        await client.update_consumption(agent_id, 100.0, value_generated=5.0)
        
        # Method 2: Using context manager for automatic tracking
        async with TokenBudgetManager(client, "example-agent-002", 500.0) as budget:
            # Simulate progressive work
            for i in range(10):
                # Do some work...
                await asyncio.sleep(0.1)
                
                # Record consumption
                await budget.consume(tokens=50.0, value=1.0)
                
                # Budget manager will automatically enforce limits


if __name__ == "__main__":
    # Run example
    asyncio.run(example_agent_task())