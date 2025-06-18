#!/usr/bin/env python3
"""
Claude Code CLI Wrapper
DEAN Core Component: Exclusive interface for code modifications via Claude Code CLI

CRITICAL: All code modifications MUST go through this wrapper.
No direct file editing or AST manipulation is allowed.
"""

import os
import subprocess
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import shlex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModificationResult:
    """Result of a Claude Code CLI modification"""
    success: bool
    prompt: str
    response: str
    files_modified: List[str]
    tokens_used: int
    execution_time_ms: int
    error: Optional[str] = None
    git_diff: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenMetrics:
    """Token usage metrics for an agent"""
    total_tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    modifications_count: int
    average_tokens_per_modification: float
    last_updated: datetime


class ClaudeCodeCLI:
    """Wrapper for Claude Code CLI - the exclusive code modification engine"""
    
    def __init__(self, api_key: str, worktree_path: str, docker_mode: bool = True):
        """
        Initialize Claude Code CLI wrapper
        
        Args:
            api_key: Anthropic API key
            worktree_path: Path to the git worktree
            docker_mode: Whether to use Docker container (True) or local CLI (False)
        """
        self.api_key = api_key
        self.worktree_path = Path(worktree_path).resolve()
        self.docker_mode = docker_mode
        
        if not self.worktree_path.exists():
            raise ValueError(f"Worktree path does not exist: {worktree_path}")
        
        # Token tracking
        self.token_metrics = TokenMetrics(
            total_tokens_used=0,
            prompt_tokens=0,
            completion_tokens=0,
            modifications_count=0,
            average_tokens_per_modification=0.0,
            last_updated=datetime.now()
        )
        
        # CLI command setup
        if docker_mode:
            self.base_cmd = [
                'docker', 'run', '--rm',
                '-v', f'{self.worktree_path}:/workspace',
                '-w', '/workspace',
                '-e', f'ANTHROPIC_API_KEY={api_key}',
                'anthropic/claude-code-cli:latest'
            ]
        else:
            # Local CLI mode
            self.base_cmd = ['claude-code']
            os.environ['ANTHROPIC_API_KEY'] = api_key
        
        # History tracking
        self.modification_history = []
        
        logger.info(f"ClaudeCodeCLI initialized: worktree={self.worktree_path}, docker={docker_mode}")
    
    def execute_modification(self, prompt: str, timeout: int = 300) -> ModificationResult:
        """
        Execute a code modification using Claude Code CLI
        
        Args:
            prompt: The modification prompt/instruction
            timeout: Maximum execution time in seconds
            
        Returns:
            ModificationResult with details of the modification
        """
        start_time = time.time()
        
        # Record initial state
        initial_files = self._get_tracked_files()
        initial_diff = self._get_git_status()
        
        # Prepare the command
        cmd = self.base_cmd + ['--prompt', prompt]
        
        logger.info(f"Executing modification: {prompt[:100]}...")
        
        try:
            # Execute the CLI command
            result = subprocess.run(
                cmd,
                cwd=self.worktree_path if not self.docker_mode else None,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            if result.returncode != 0:
                error_msg = f"CLI returned non-zero exit code: {result.returncode}\n{result.stderr}"
                logger.error(error_msg)
                
                return ModificationResult(
                    success=False,
                    prompt=prompt,
                    response=result.stdout,
                    files_modified=[],
                    tokens_used=0,
                    execution_time_ms=execution_time_ms,
                    error=error_msg
                )
            
            # Parse response and extract token usage
            response_text = result.stdout
            tokens_used = self._extract_token_usage(response_text)
            
            # Get modified files
            modified_files = self._get_modified_files(initial_files)
            
            # Get git diff
            git_diff = self._get_git_diff()
            
            # Update metrics
            self._update_token_metrics(tokens_used)
            
            # Create result
            modification_result = ModificationResult(
                success=True,
                prompt=prompt,
                response=response_text,
                files_modified=modified_files,
                tokens_used=tokens_used,
                execution_time_ms=execution_time_ms,
                git_diff=git_diff
            )
            
            # Store in history
            self.modification_history.append({
                'timestamp': datetime.now().isoformat(),
                'result': modification_result.to_dict()
            })
            
            logger.info(f"Modification successful: {len(modified_files)} files modified, {tokens_used} tokens used")
            
            return modification_result
            
        except subprocess.TimeoutExpired:
            error_msg = f"CLI execution timed out after {timeout} seconds"
            logger.error(error_msg)
            
            return ModificationResult(
                success=False,
                prompt=prompt,
                response="",
                files_modified=[],
                tokens_used=0,
                execution_time_ms=timeout * 1000,
                error=error_msg
            )
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            
            return ModificationResult(
                success=False,
                prompt=prompt,
                response="",
                files_modified=[],
                tokens_used=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error=error_msg
            )
    
    def execute_batch_modifications(self, prompts: List[str]) -> List[ModificationResult]:
        """Execute multiple modifications in sequence"""
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Executing batch modification {i+1}/{len(prompts)}")
            result = self.execute_modification(prompt)
            results.append(result)
            
            # Brief pause between modifications
            if i < len(prompts) - 1:
                time.sleep(2)
        
        return results
    
    def get_token_usage(self) -> TokenMetrics:
        """Get current token usage metrics"""
        return self.token_metrics
    
    def _extract_token_usage(self, response: str) -> int:
        """Extract token usage from CLI response"""
        # Look for token usage patterns in response
        # This would need to be adapted based on actual CLI output format
        
        # Example patterns to look for:
        # "Tokens used: 1234"
        # "Usage: { prompt_tokens: 100, completion_tokens: 200 }"
        
        import re
        
        # Try different patterns
        patterns = [
            r'[Tt]okens?\s+used:\s*(\d+)',
            r'[Tt]otal\s+tokens?:\s*(\d+)',
            r'"total_tokens":\s*(\d+)',
            r'completion_tokens":\s*(\d+).*prompt_tokens":\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                if len(match.groups()) == 1:
                    return int(match.group(1))
                elif len(match.groups()) == 2:
                    # Prompt + completion tokens
                    return int(match.group(1)) + int(match.group(2))
        
        # Default estimate based on response length
        # Rough estimate: 1 token ≈ 4 characters
        return len(response) // 4
    
    def _update_token_metrics(self, tokens_used: int):
        """Update token usage metrics"""
        self.token_metrics.total_tokens_used += tokens_used
        self.token_metrics.modifications_count += 1
        self.token_metrics.average_tokens_per_modification = (
            self.token_metrics.total_tokens_used / self.token_metrics.modifications_count
        )
        self.token_metrics.last_updated = datetime.now()
        
        # Estimate prompt/completion split (rough approximation)
        self.token_metrics.prompt_tokens += tokens_used // 3
        self.token_metrics.completion_tokens += (tokens_used * 2) // 3
    
    def _get_tracked_files(self) -> List[str]:
        """Get list of tracked files in the worktree"""
        result = subprocess.run(
            ['git', 'ls-files'],
            cwd=self.worktree_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return result.stdout.strip().split('\n') if result.stdout else []
        return []
    
    def _get_modified_files(self, initial_files: List[str]) -> List[str]:
        """Get list of modified files since initial state"""
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=self.worktree_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return []
        
        modified = []
        for line in result.stdout.strip().split('\n'):
            if line:
                # Parse git status output
                status = line[:2]
                filename = line[3:]
                if status.strip():  # Any modification
                    modified.append(filename)
        
        return modified
    
    def _get_git_status(self) -> str:
        """Get current git status"""
        result = subprocess.run(
            ['git', 'status', '--short'],
            cwd=self.worktree_path,
            capture_output=True,
            text=True
        )
        
        return result.stdout if result.returncode == 0 else ""
    
    def _get_git_diff(self) -> str:
        """Get git diff of changes"""
        result = subprocess.run(
            ['git', 'diff', '--cached', '--', '.'],
            cwd=self.worktree_path,
            capture_output=True,
            text=True
        )
        
        diff = result.stdout if result.returncode == 0 else ""
        
        # Also get unstaged changes
        result = subprocess.run(
            ['git', 'diff', '--', '.'],
            cwd=self.worktree_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout:
            diff += "\n" + result.stdout
        
        return diff
    
    def commit_changes(self, message: str) -> bool:
        """Commit current changes in the worktree"""
        # Add all changes
        result = subprocess.run(
            ['git', 'add', '-A'],
            cwd=self.worktree_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to add changes: {result.stderr}")
            return False
        
        # Commit
        result = subprocess.run(
            ['git', 'commit', '-m', message],
            cwd=self.worktree_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to commit: {result.stderr}")
            return False
        
        logger.info(f"Committed changes: {message}")
        return True
    
    def save_metrics_report(self, output_path: str):
        """Save detailed metrics report"""
        report = {
            'worktree_path': str(self.worktree_path),
            'metrics': asdict(self.token_metrics),
            'modification_count': len(self.modification_history),
            'modification_history': self.modification_history[-10:],  # Last 10
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Metrics report saved to: {output_path}")


# Mock implementation for testing without actual CLI
class MockClaudeCodeCLI(ClaudeCodeCLI):
    """Mock implementation for testing"""
    
    def __init__(self, api_key: str, worktree_path: str):
        super().__init__(api_key, worktree_path, docker_mode=False)
        self.mock_responses = []
    
    def execute_modification(self, prompt: str, timeout: int = 300) -> ModificationResult:
        """Mock execution that simulates modifications"""
        logger.info(f"MOCK: Executing modification: {prompt[:50]}...")
        
        # Simulate some token usage
        tokens = len(prompt) // 4 + 200  # Rough estimate
        
        # Create a mock file modification
        test_file = self.worktree_path / f"mock_file_{len(self.modification_history)}.py"
        test_file.write_text(f"# Modified by mock CLI\n# Prompt: {prompt}\n\ndef mock_function():\n    pass\n")
        
        return ModificationResult(
            success=True,
            prompt=prompt,
            response=f"Mock response for: {prompt}",
            files_modified=[str(test_file.relative_to(self.worktree_path))],
            tokens_used=tokens,
            execution_time_ms=1000,
            git_diff=f"+ # Modified by mock CLI\n+ # Prompt: {prompt}"
        )


if __name__ == "__main__":
    # Demo usage
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        worktree = Path(tmpdir) / "test_worktree"
        worktree.mkdir()
        
        # Initialize git
        subprocess.run(['git', 'init'], cwd=worktree, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=worktree)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=worktree)
        
        # Create initial file
        (worktree / 'test.py').write_text("def hello():\n    print('Hello')\n")
        subprocess.run(['git', 'add', '.'], cwd=worktree)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=worktree)
        
        # Create mock CLI wrapper
        cli = MockClaudeCodeCLI(api_key="mock_key", worktree_path=str(worktree))
        
        # Execute modifications
        print("Executing modifications...")
        
        result1 = cli.execute_modification("Add a function to calculate fibonacci")
        print(f"Result 1: {result1.success}, Files: {result1.files_modified}")
        
        result2 = cli.execute_modification("Optimize the fibonacci function with memoization")
        print(f"Result 2: {result2.success}, Files: {result2.files_modified}")
        
        # Get metrics
        metrics = cli.get_token_usage()
        print(f"\nToken usage: {metrics.total_tokens_used} tokens across {metrics.modifications_count} modifications")
        print(f"Average per modification: {metrics.average_tokens_per_modification:.1f}")
        
        # Save report
        cli.save_metrics_report(tmpdir + "/metrics_report.json")