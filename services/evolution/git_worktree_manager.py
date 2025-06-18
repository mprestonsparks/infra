#!/usr/bin/env python3
"""
Git Worktree Manager
DEAN Core Component: Manages isolated git worktrees for parallel agent execution

This is a CRITICAL component that enables agent isolation and parallel evolution.
Each agent operates in its own worktree to prevent conflicts.
"""

import os
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import fcntl
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorktreeInfo:
    """Information about an active worktree"""
    agent_id: str
    worktree_path: str
    branch_name: str
    created_at: datetime
    last_accessed: datetime
    is_locked: bool
    commit_count: int


@dataclass
class WorktreeContext:
    """Context for agent worktree operations"""
    agent_id: str
    worktree_path: str
    branch_name: str
    git_dir: str
    
    def __enter__(self):
        """Enter worktree context"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit worktree context (cleanup handled separately)"""
        pass


class GitWorktreeManager:
    """Manages Git worktrees for isolated agent execution"""
    
    def __init__(self, base_repo_path: str, worktrees_dir: str = None):
        """
        Initialize worktree manager
        
        Args:
            base_repo_path: Path to the main git repository
            worktrees_dir: Directory to store worktrees (default: base_repo/.worktrees)
        """
        self.base_repo_path = Path(base_repo_path).resolve()
        
        if not self.base_repo_path.exists():
            raise ValueError(f"Base repository path does not exist: {base_repo_path}")
        
        if not (self.base_repo_path / '.git').exists():
            raise ValueError(f"Not a git repository: {base_repo_path}")
        
        # Set worktrees directory
        if worktrees_dir:
            self.worktrees_dir = Path(worktrees_dir).resolve()
        else:
            self.worktrees_dir = self.base_repo_path / '.worktrees'
        
        # Create worktrees directory
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        
        # Lock file for concurrent access
        self.lock_file = self.worktrees_dir / '.manager.lock'
        
        # Metadata file
        self.metadata_file = self.worktrees_dir / 'worktree_metadata.json'
        self._load_metadata()
        
        logger.info(f"GitWorktreeManager initialized: base={self.base_repo_path}, worktrees={self.worktrees_dir}")
    
    def _load_metadata(self):
        """Load worktree metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'worktrees': {}, 'created_at': datetime.now().isoformat()}
            self._save_metadata()
    
    def _save_metadata(self):
        """Save worktree metadata with file locking"""
        with open(self.lock_file, 'w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                with open(self.metadata_file, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    
    def _run_git_command(self, cmd: List[str], cwd: str = None) -> Tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr)"""
        if cwd is None:
            cwd = self.base_repo_path
        
        result = subprocess.run(
            ['git'] + cmd,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        
        return result.returncode, result.stdout, result.stderr
    
    def create_worktree(self, agent_id: str, base_branch: str = 'main') -> WorktreeContext:
        """
        Create a new worktree for an agent
        
        Args:
            agent_id: Unique identifier for the agent
            base_branch: Branch to base the worktree on (default: main)
            
        Returns:
            WorktreeContext for the created worktree
        """
        # Validate agent_id
        if not agent_id or not agent_id.replace('-', '').replace('_', '').isalnum():
            raise ValueError(f"Invalid agent_id: {agent_id}")
        
        # Check if worktree already exists
        if agent_id in self.metadata.get('worktrees', {}):
            existing = self.metadata['worktrees'][agent_id]
            if Path(existing['path']).exists():
                logger.warning(f"Worktree already exists for agent {agent_id}")
                return WorktreeContext(
                    agent_id=agent_id,
                    worktree_path=existing['path'],
                    branch_name=existing['branch'],
                    git_dir=str(Path(existing['path']) / '.git')
                )
        
        # Create branch name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        branch_name = f"agent/{agent_id}/{timestamp}"
        
        # Create worktree path
        worktree_path = self.worktrees_dir / agent_id
        
        # Remove existing directory if present
        if worktree_path.exists():
            shutil.rmtree(worktree_path)
        
        # Create the worktree
        logger.info(f"Creating worktree for agent {agent_id} at {worktree_path}")
        
        # Create the worktree with a new branch in one command
        returncode, stdout, stderr = self._run_git_command(
            ['worktree', 'add', '-b', branch_name, str(worktree_path), base_branch]
        )
        
        if returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {stderr}")
        
        # Update metadata
        self.metadata['worktrees'][agent_id] = {
            'path': str(worktree_path),
            'branch': branch_name,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'is_locked': False,
            'commit_count': 0
        }
        self._save_metadata()
        
        logger.info(f"Successfully created worktree for agent {agent_id}")
        
        return WorktreeContext(
            agent_id=agent_id,
            worktree_path=str(worktree_path),
            branch_name=branch_name,
            git_dir=str(worktree_path / '.git')
        )
    
    def cleanup_worktree(self, agent_id: str) -> None:
        """
        Remove a worktree and its associated branch
        
        Args:
            agent_id: Agent whose worktree to remove
        """
        if agent_id not in self.metadata.get('worktrees', {}):
            logger.warning(f"No worktree found for agent {agent_id}")
            return
        
        worktree_info = self.metadata['worktrees'][agent_id]
        worktree_path = Path(worktree_info['path'])
        branch_name = worktree_info['branch']
        
        # Remove the worktree
        if worktree_path.exists():
            logger.info(f"Removing worktree at {worktree_path}")
            returncode, stdout, stderr = self._run_git_command(
                ['worktree', 'remove', str(worktree_path), '--force']
            )
            
            if returncode != 0:
                logger.error(f"Failed to remove worktree: {stderr}")
                # Try manual removal
                shutil.rmtree(worktree_path, ignore_errors=True)
        
        # Delete the branch
        logger.info(f"Deleting branch {branch_name}")
        returncode, stdout, stderr = self._run_git_command(
            ['branch', '-D', branch_name]
        )
        
        if returncode != 0:
            logger.warning(f"Failed to delete branch: {stderr}")
        
        # Update metadata
        del self.metadata['worktrees'][agent_id]
        self._save_metadata()
        
        logger.info(f"Successfully cleaned up worktree for agent {agent_id}")
    
    def list_active_worktrees(self) -> List[WorktreeInfo]:
        """
        List all active worktrees
        
        Returns:
            List of WorktreeInfo objects
        """
        active_worktrees = []
        
        # Get actual git worktrees
        returncode, stdout, stderr = self._run_git_command(['worktree', 'list', '--porcelain'])
        
        if returncode != 0:
            logger.error(f"Failed to list worktrees: {stderr}")
            return []
        
        # Parse git worktree output
        git_worktrees = {}
        current_worktree = {}
        
        for line in stdout.strip().split('\n'):
            if not line:
                if current_worktree:
                    git_worktrees[current_worktree['worktree']] = current_worktree
                    current_worktree = {}
            elif line.startswith('worktree '):
                current_worktree['worktree'] = line.split(' ', 1)[1]
            elif line.startswith('branch '):
                current_worktree['branch'] = line.split(' ', 1)[1]
        
        if current_worktree:
            git_worktrees[current_worktree['worktree']] = current_worktree
        
        # Match with metadata
        for agent_id, info in self.metadata.get('worktrees', {}).items():
            worktree_path = info['path']
            
            # Check if worktree still exists
            if worktree_path in git_worktrees:
                # Get commit count
                returncode, stdout, stderr = self._run_git_command(
                    ['rev-list', '--count', info['branch']],
                    cwd=worktree_path
                )
                
                commit_count = int(stdout.strip()) if returncode == 0 else 0
                
                active_worktrees.append(WorktreeInfo(
                    agent_id=agent_id,
                    worktree_path=worktree_path,
                    branch_name=info['branch'],
                    created_at=datetime.fromisoformat(info['created_at']),
                    last_accessed=datetime.fromisoformat(info['last_accessed']),
                    is_locked=info.get('is_locked', False),
                    commit_count=commit_count
                ))
        
        return active_worktrees
    
    def get_worktree_status(self, agent_id: str) -> Optional[Dict[str, any]]:
        """Get detailed status of a worktree"""
        if agent_id not in self.metadata.get('worktrees', {}):
            return None
        
        info = self.metadata['worktrees'][agent_id]
        worktree_path = Path(info['path'])
        
        if not worktree_path.exists():
            return None
        
        # Get git status
        returncode, stdout, stderr = self._run_git_command(
            ['status', '--porcelain'],
            cwd=str(worktree_path)
        )
        
        has_changes = bool(stdout.strip()) if returncode == 0 else False
        
        # Get latest commit
        returncode, stdout, stderr = self._run_git_command(
            ['log', '-1', '--oneline'],
            cwd=str(worktree_path)
        )
        
        latest_commit = stdout.strip() if returncode == 0 else None
        
        return {
            'agent_id': agent_id,
            'worktree_path': str(worktree_path),
            'branch': info['branch'],
            'has_changes': has_changes,
            'latest_commit': latest_commit,
            'created_at': info['created_at'],
            'last_accessed': info['last_accessed']
        }
    
    def cleanup_all_worktrees(self):
        """Remove all worktrees (useful for cleanup)"""
        logger.warning("Cleaning up ALL worktrees")
        
        agent_ids = list(self.metadata.get('worktrees', {}).keys())
        
        for agent_id in agent_ids:
            try:
                self.cleanup_worktree(agent_id)
            except Exception as e:
                logger.error(f"Failed to cleanup worktree for {agent_id}: {e}")
    
    def prune_stale_worktrees(self, max_age_hours: int = 24):
        """Remove worktrees older than specified hours"""
        current_time = datetime.now()
        stale_agents = []
        
        for agent_id, info in self.metadata.get('worktrees', {}).items():
            last_accessed = datetime.fromisoformat(info['last_accessed'])
            age_hours = (current_time - last_accessed).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                stale_agents.append(agent_id)
        
        for agent_id in stale_agents:
            logger.info(f"Pruning stale worktree for agent {agent_id}")
            try:
                self.cleanup_worktree(agent_id)
            except Exception as e:
                logger.error(f"Failed to prune worktree for {agent_id}: {e}")
        
        return len(stale_agents)


if __name__ == "__main__":
    # Demo usage
    import tempfile
    
    # Create a test repository
    with tempfile.TemporaryDirectory() as tmpdir:
        test_repo = Path(tmpdir) / "test_repo"
        test_repo.mkdir()
        
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=test_repo, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=test_repo)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=test_repo)
        
        # Create initial commit
        (test_repo / 'README.md').write_text("# Test Repository")
        subprocess.run(['git', 'add', '.'], cwd=test_repo)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=test_repo)
        
        # Create manager
        manager = GitWorktreeManager(str(test_repo))
        
        # Create worktrees for multiple agents
        print("Creating worktrees...")
        contexts = []
        for i in range(3):
            ctx = manager.create_worktree(f"agent_{i}")
            contexts.append(ctx)
            print(f"Created worktree: {ctx.worktree_path}")
        
        # List active worktrees
        print("\nActive worktrees:")
        for wt in manager.list_active_worktrees():
            print(f"  - {wt.agent_id}: {wt.worktree_path} ({wt.branch_name})")
        
        # Cleanup
        print("\nCleaning up...")
        manager.cleanup_all_worktrees()