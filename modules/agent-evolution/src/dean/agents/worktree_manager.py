"""
Git worktree manager for isolated agent execution environments.

Manages creation, configuration, and cleanup of git worktrees
with resource constraints and isolation guarantees.
"""

import asyncio
import shutil
from typing import Dict, Optional, Set, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class WorktreeConstraints:
    """Resource and operational constraints for a worktree."""
    token_limit: int
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50
    disk_limit_mb: int = 1024
    network_access: bool = True
    timeout_minutes: int = 60
    max_files: int = 1000
    max_branches: int = 10


class GitWorktreeManager:
    """
    Manages isolated git worktrees for agent execution.
    
    Each agent gets its own worktree to prevent conflicts
    and enable parallel execution with resource limits.
    """
    
    def __init__(self, 
                 base_repo_path: Path = Path.cwd(),
                 worktree_base_path: Optional[Path] = None,
                 max_worktrees: int = 50):
        """
        Initialize worktree manager.
        
        Args:
            base_repo_path: Path to the main git repository
            worktree_base_path: Base path for creating worktrees
            max_worktrees: Maximum number of concurrent worktrees
        """
        self.base_repo_path = Path(base_repo_path)
        self.worktree_base_path = worktree_base_path or (self.base_repo_path / "agent_worktrees")
        self.max_worktrees = max_worktrees
        
        # Tracking active worktrees
        self.active_worktrees: Dict[str, Path] = {}
        self.worktree_constraints: Dict[str, WorktreeConstraints] = {}
        self.creation_times: Dict[str, datetime] = {}
        
        # Ensure base paths exist and are valid
        self._validate_setup()
        
        logger.info(f"GitWorktreeManager initialized: base={self.base_repo_path}, "
                   f"worktrees={self.worktree_base_path}")
    
    async def create_worktree(self,
                            branch_name: str,
                            agent_id: str,
                            token_limit: int,
                            constraints: Optional[WorktreeConstraints] = None) -> Path:
        """
        Create an isolated git worktree for an agent.
        
        Args:
            branch_name: Name of the branch to checkout
            agent_id: Unique identifier for the agent
            token_limit: Token budget limit for the agent
            constraints: Optional resource constraints
            
        Returns:
            Path to the created worktree
            
        Raises:
            RuntimeError: If worktree creation fails
        """
        if agent_id in self.active_worktrees:
            raise RuntimeError(f"Worktree already exists for agent {agent_id}")
        
        if len(self.active_worktrees) >= self.max_worktrees:
            await self._cleanup_expired_worktrees()
            
            if len(self.active_worktrees) >= self.max_worktrees:
                raise RuntimeError(f"Maximum worktrees ({self.max_worktrees}) exceeded")
        
        # Set up constraints
        if constraints is None:
            constraints = WorktreeConstraints(token_limit=token_limit)
        else:
            constraints.token_limit = token_limit
        
        worktree_path = self.worktree_base_path / f"agent_{agent_id}"
        
        try:
            # Ensure worktree directory doesn't exist
            if worktree_path.exists():
                shutil.rmtree(worktree_path)
            
            # Create the branch if it doesn't exist
            await self._ensure_branch_exists(branch_name)
            
            # Create the worktree
            await self._create_git_worktree(worktree_path, branch_name)
            
            # Apply resource constraints
            await self._apply_constraints(worktree_path, constraints)
            
            # Create agent configuration
            await self._create_agent_config(worktree_path, agent_id, constraints)
            
            # Track the worktree
            self.active_worktrees[agent_id] = worktree_path
            self.worktree_constraints[agent_id] = constraints
            self.creation_times[agent_id] = datetime.now()
            
            logger.info(f"Created worktree for agent {agent_id}: {worktree_path}")
            return worktree_path
            
        except Exception as e:
            # Cleanup on failure
            if worktree_path.exists():
                try:
                    await self._remove_git_worktree(worktree_path)
                except:
                    pass
            
            logger.error(f"Failed to create worktree for agent {agent_id}: {e}")
            raise RuntimeError(f"Worktree creation failed: {e}")
    
    async def cleanup_worktree(self, agent_id: str) -> bool:
        """
        Clean up a worktree for the specified agent.
        
        Args:
            agent_id: Agent whose worktree to clean up
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        if agent_id not in self.active_worktrees:
            logger.warning(f"No worktree found for agent {agent_id}")
            return False
        
        worktree_path = self.active_worktrees[agent_id]
        
        try:
            # Remove git worktree
            await self._remove_git_worktree(worktree_path)
            
            # Remove from tracking
            del self.active_worktrees[agent_id]
            del self.worktree_constraints[agent_id]
            del self.creation_times[agent_id]
            
            logger.info(f"Cleaned up worktree for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up worktree for agent {agent_id}: {e}")
            return False
    
    async def get_worktree_status(self, agent_id: str) -> Dict[str, any]:
        """Get status information for an agent's worktree."""
        if agent_id not in self.active_worktrees:
            return {'exists': False}
        
        worktree_path = self.active_worktrees[agent_id]
        constraints = self.worktree_constraints[agent_id]
        creation_time = self.creation_times[agent_id]
        
        try:
            # Get basic file system info
            stat_info = worktree_path.stat()
            
            # Count files
            file_count = sum(1 for _ in worktree_path.rglob('*') if _.is_file())
            
            # Calculate age
            age_seconds = (datetime.now() - creation_time).total_seconds()
            
            # Get git status
            git_status = await self._get_git_status(worktree_path)
            
            return {
                'exists': True,
                'path': str(worktree_path),
                'creation_time': creation_time.isoformat(),
                'age_seconds': age_seconds,
                'file_count': file_count,
                'constraints': {
                    'token_limit': constraints.token_limit,
                    'memory_limit_mb': constraints.memory_limit_mb,
                    'disk_limit_mb': constraints.disk_limit_mb,
                    'timeout_minutes': constraints.timeout_minutes
                },
                'git_status': git_status
            }
            
        except Exception as e:
            logger.error(f"Error getting worktree status for agent {agent_id}: {e}")
            return {'exists': True, 'error': str(e)}
    
    async def list_active_worktrees(self) -> List[Dict[str, any]]:
        """List all active worktrees with their status."""
        worktrees = []
        
        for agent_id in list(self.active_worktrees.keys()):
            status = await self.get_worktree_status(agent_id)
            status['agent_id'] = agent_id
            worktrees.append(status)
        
        return worktrees
    
    async def cleanup_all_worktrees(self) -> int:
        """Clean up all active worktrees. Returns count of cleaned up worktrees."""
        cleanup_count = 0
        
        for agent_id in list(self.active_worktrees.keys()):
            if await self.cleanup_worktree(agent_id):
                cleanup_count += 1
        
        logger.info(f"Cleaned up {cleanup_count} worktrees")
        return cleanup_count
    
    def _validate_setup(self) -> None:
        """Validate that the git repository and paths are set up correctly."""
        if not self.base_repo_path.exists():
            raise RuntimeError(f"Base repository path does not exist: {self.base_repo_path}")
        
        if not (self.base_repo_path / ".git").exists():
            raise RuntimeError(f"Not a git repository: {self.base_repo_path}")
        
        # Create worktree base directory
        self.worktree_base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Git worktree setup validated")
    
    async def _ensure_branch_exists(self, branch_name: str) -> None:
        """Ensure the specified branch exists."""
        try:
            # Check if branch exists
            proc = await asyncio.create_subprocess_exec(
                'git', 'show-ref', '--verify', f'refs/heads/{branch_name}',
                cwd=str(self.base_repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await proc.wait()
            
            if proc.returncode == 0:
                # Branch exists
                return
            
            # Create branch from current HEAD
            proc = await asyncio.create_subprocess_exec(
                'git', 'branch', branch_name,
                cwd=str(self.base_repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to create branch {branch_name}: {stderr.decode()}")
            
            logger.info(f"Created branch {branch_name}")
            
        except Exception as e:
            logger.error(f"Error ensuring branch {branch_name} exists: {e}")
            raise
    
    async def _create_git_worktree(self, worktree_path: Path, branch_name: str) -> None:
        """Create the actual git worktree."""
        try:
            proc = await asyncio.create_subprocess_exec(
                'git', 'worktree', 'add', str(worktree_path), branch_name,
                cwd=str(self.base_repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                raise RuntimeError(f"Git worktree creation failed: {stderr.decode()}")
            
            logger.debug(f"Git worktree created: {worktree_path}")
            
        except Exception as e:
            logger.error(f"Error creating git worktree: {e}")
            raise
    
    async def _remove_git_worktree(self, worktree_path: Path) -> None:
        """Remove a git worktree."""
        try:
            # Remove the worktree
            proc = await asyncio.create_subprocess_exec(
                'git', 'worktree', 'remove', str(worktree_path), '--force',
                cwd=str(self.base_repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await proc.wait()
            
            # If git worktree remove fails, try manual cleanup
            if worktree_path.exists():
                shutil.rmtree(worktree_path)
            
            logger.debug(f"Git worktree removed: {worktree_path}")
            
        except Exception as e:
            logger.error(f"Error removing git worktree: {e}")
            raise
    
    async def _apply_constraints(self, worktree_path: Path, constraints: WorktreeConstraints) -> None:
        """Apply resource constraints to the worktree."""
        try:
            # Create constraints configuration file
            constraints_file = worktree_path / ".dean_constraints"
            
            constraints_data = {
                'token_limit': constraints.token_limit,
                'memory_limit_mb': constraints.memory_limit_mb,
                'cpu_limit_percent': constraints.cpu_limit_percent,
                'disk_limit_mb': constraints.disk_limit_mb,
                'network_access': constraints.network_access,
                'timeout_minutes': constraints.timeout_minutes,
                'max_files': constraints.max_files,
                'max_branches': constraints.max_branches,
                'created_at': datetime.now().isoformat()
            }
            
            with open(constraints_file, 'w') as f:
                json.dump(constraints_data, f, indent=2)
            
            # Set up resource monitoring (simplified implementation)
            # In production, this could integrate with Docker, cgroups, etc.
            
            logger.debug(f"Applied constraints to worktree: {worktree_path}")
            
        except Exception as e:
            logger.error(f"Error applying constraints: {e}")
            raise
    
    async def _create_agent_config(self, worktree_path: Path, agent_id: str, constraints: WorktreeConstraints) -> None:
        """Create agent-specific configuration in the worktree."""
        try:
            config_file = worktree_path / ".dean_agent_config"
            
            config_data = {
                'agent_id': agent_id,
                'worktree_path': str(worktree_path),
                'constraints': {
                    'token_limit': constraints.token_limit,
                    'memory_limit_mb': constraints.memory_limit_mb,
                    'timeout_minutes': constraints.timeout_minutes
                },
                'environment': {
                    'DEAN_AGENT_ID': agent_id,
                    'DEAN_WORKTREE_PATH': str(worktree_path),
                    'DEAN_TOKEN_LIMIT': str(constraints.token_limit)
                },
                'created_at': datetime.now().isoformat()
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.debug(f"Created agent config for {agent_id}")
            
        except Exception as e:
            logger.error(f"Error creating agent config: {e}")
            raise
    
    async def _get_git_status(self, worktree_path: Path) -> Dict[str, any]:
        """Get git status for a worktree."""
        try:
            proc = await asyncio.create_subprocess_exec(
                'git', 'status', '--porcelain',
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                status_lines = stdout.decode().strip().split('\n') if stdout else []
                return {
                    'clean': len(status_lines) == 0 or (len(status_lines) == 1 and not status_lines[0]),
                    'modified_files': len([line for line in status_lines if line.startswith(' M')]),
                    'added_files': len([line for line in status_lines if line.startswith('A ')]),
                    'deleted_files': len([line for line in status_lines if line.startswith(' D')]),
                    'untracked_files': len([line for line in status_lines if line.startswith('??')])
                }
            else:
                return {'error': stderr.decode()}
                
        except Exception as e:
            return {'error': str(e)}
    
    async def _cleanup_expired_worktrees(self) -> None:
        """Clean up worktrees that have exceeded their timeout."""
        now = datetime.now()
        expired_agents = []
        
        for agent_id, creation_time in self.creation_times.items():
            constraints = self.worktree_constraints.get(agent_id)
            if constraints:
                timeout = timedelta(minutes=constraints.timeout_minutes)
                if now - creation_time > timeout:
                    expired_agents.append(agent_id)
        
        for agent_id in expired_agents:
            logger.info(f"Cleaning up expired worktree for agent {agent_id}")
            await self.cleanup_worktree(agent_id)