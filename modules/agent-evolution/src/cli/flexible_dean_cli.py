#!/usr/bin/env python3
"""
DEAN Flexible Command Line Interface
Supports both containerized and host-based execution
"""

import os
import sys
import subprocess
import argparse
import yaml
import json
import shutil
from pathlib import Path
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode for commands"""
    AUTO = "auto"
    CONTAINER = "container"
    HOST = "host"


@dataclass
class ExecutionResult:
    """Result of command execution"""
    returncode: int
    stdout: str
    stderr: str


class EnvironmentDetector:
    """Detects the current execution environment"""
    
    def is_container_environment(self) -> bool:
        """Check if running inside a Docker container"""
        return os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
    
    def is_docker_available(self) -> bool:
        """Check if Docker is available on the system"""
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def get_compose_command(self) -> Optional[str]:
        """Detect the available Docker Compose command"""
        # Try docker-compose first
        try:
            result = subprocess.run(
                ['docker-compose', '--version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return 'docker-compose'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Try docker compose
        try:
            result = subprocess.run(
                ['docker', 'compose', 'version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return 'docker compose'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        return None
    
    def is_dean_installed(self, workspace: str = '.') -> bool:
        """Check if DEAN system is installed in the workspace"""
        required_paths = [
            Path(workspace) / 'infra' / 'docker-compose.yml',
            Path(workspace) / 'IndexAgent',
            Path(workspace) / 'airflow-hub'
        ]
        return all(path.exists() for path in required_paths)
    
    def get_python_version(self) -> Optional[str]:
        """Get the Python version"""
        try:
            result = subprocess.run(
                [sys.executable, '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None


class ConfigurationManager:
    """Manages CLI configuration"""
    
    def __init__(self):
        self.config_paths = [
            Path.home() / '.dean-cli.yml',  # Global config
            Path.cwd() / '.dean-cli.yml',   # Local config
        ]
    
    def load_config(self, config_file: Optional[Path] = None) -> Dict[str, Any]:
        """Load configuration from file"""
        config = self._get_default_config()
        
        # Load from config files
        config_files = [config_file] if config_file else self.config_paths
        for path in config_files:
            if path and path.exists():
                try:
                    with open(path, 'r') as f:
                        file_config = yaml.safe_load(f) or {}
                        config.update(file_config)
                except Exception as e:
                    logger.warning(f"Failed to load config from {path}: {e}")
        
        # Override with environment variables
        env_config = self._load_env_config()
        config = self.merge_configs(config, {}, env_config)
        
        return config
    
    def save_config(self, config: Dict[str, Any], path: Optional[Path] = None):
        """Save configuration to file"""
        save_path = path or self.config_paths[0]  # Default to global config
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def merge_configs(self, global_config: Dict[str, Any], 
                     local_config: Dict[str, Any], 
                     env_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configurations with proper precedence"""
        config = global_config.copy()
        config.update(local_config)
        
        # Environment variables take precedence
        if 'DEAN_EXECUTION_MODE' in env_vars:
            config['execution_mode'] = env_vars['DEAN_EXECUTION_MODE'].lower()
        if 'DEAN_API_URL' in env_vars:
            config['api_url'] = env_vars['DEAN_API_URL']
        if 'DEAN_WORKSPACE' in env_vars:
            config['default_workspace'] = env_vars['DEAN_WORKSPACE']
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        valid_modes = ['auto', 'container', 'host']
        if config.get('execution_mode', 'auto') not in valid_modes:
            raise ValueError(f"Invalid execution_mode: {config['execution_mode']}")
        
        return True
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'execution_mode': 'auto',
            'api_url': 'http://localhost:8090',
            'default_workspace': os.getcwd(),
            'container_settings': {
                'image': 'dean-cli:latest',
                'network': 'dean-network',
                'auto_start': True
            },
            'host_settings': {
                'python_path': sys.executable,
                'venv_path': '.venv',
                'auto_install': True
            }
        }
    
    def _load_env_config(self) -> Dict[str, str]:
        """Load configuration from environment variables"""
        return {k: v for k, v in os.environ.items() if k.startswith('DEAN_')}


class ContainerExecutor:
    """Executes commands in a Docker container"""
    
    def __init__(self, container_name: str = 'dean-cli', workspace: str = '/workspace'):
        self.container_name = container_name
        self.workspace = workspace
    
    def execute(self, command: str, env: Optional[Dict[str, str]] = None,
                cwd: Optional[str] = None, stream: bool = False,
                output_callback: Optional[callable] = None,
                auto_start: bool = True) -> subprocess.CompletedProcess:
        """Execute command in container"""
        # Build docker exec command
        docker_cmd = ['docker', 'exec']
        
        # Add environment variables
        if env:
            for key, value in env.items():
                docker_cmd.extend(['-e', f'{key}={value}'])
        
        # Add working directory
        if cwd:
            docker_cmd.extend(['-w', cwd])
        
        # Add interactive flag for streaming
        if stream:
            docker_cmd.append('-i')
        
        # Add container name and command
        docker_cmd.append(self.container_name)
        docker_cmd.extend(['bash', '-c', command])
        
        # Execute command
        try:
            if stream and output_callback:
                return self._execute_streaming(docker_cmd, output_callback)
            else:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=None if stream else 300
                )
                
                # Handle container not running
                if result.returncode == 125 and auto_start:
                    logger.info("Container not running, attempting to start...")
                    if self._start_container():
                        # Retry command
                        result = subprocess.run(
                            docker_cmd,
                            capture_output=True,
                            text=True,
                            timeout=None if stream else 300
                        )
                
                return result
                
        except subprocess.TimeoutExpired as e:
            return subprocess.CompletedProcess(
                args=docker_cmd,
                returncode=-1,
                stdout='',
                stderr=f'Command timed out: {e}'
            )
    
    def _execute_streaming(self, cmd: List[str], callback: callable) -> subprocess.CompletedProcess:
        """Execute command with streaming output"""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        stdout_lines = []
        stderr_lines = []
        
        # Stream stdout
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.rstrip()
                stdout_lines.append(line)
                callback(line)
        
        process.wait()
        
        # Capture any stderr
        stderr = process.stderr.read()
        if stderr:
            stderr_lines.append(stderr)
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout='\n'.join(stdout_lines),
            stderr='\n'.join(stderr_lines)
        )
    
    def _start_container(self) -> bool:
        """Attempt to start the container"""
        try:
            # Try to start existing container
            result = subprocess.run(
                ['docker', 'start', self.container_name],
                capture_output=True
            )
            
            if result.returncode == 0:
                logger.info(f"Started container: {self.container_name}")
                return True
            
            # If container doesn't exist, try to create it
            # This would need the full docker run command with proper setup
            logger.warning("Container doesn't exist. Please run deployment script first.")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            return False


class HostExecutor:
    """Executes commands directly on the host"""
    
    def __init__(self, python_path: str = sys.executable, venv_path: Optional[str] = None):
        self.python_path = python_path
        self.venv_path = venv_path
    
    def execute(self, command: str, env: Optional[Dict[str, str]] = None,
                cwd: Optional[str] = None, stream: bool = False,
                output_callback: Optional[callable] = None,
                auto_install: bool = True) -> subprocess.CompletedProcess:
        """Execute command on host"""
        # Parse command to extract dean subcommand
        parts = command.split()
        if parts[0] != 'dean':
            raise ValueError(f"Expected 'dean' command, got: {parts[0]}")
        
        # Build Python module execution command
        cmd = [self.python_path, '-m', 'infra.modules.agent_evolution.src.cli.dean_cli']
        cmd.extend(parts[1:])  # Add subcommand and arguments
        
        # Prepare environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)
        
        # Add virtual environment if specified
        if self.venv_path and os.path.exists(self.venv_path):
            venv_bin = os.path.join(self.venv_path, 'bin')
            exec_env['VIRTUAL_ENV'] = self.venv_path
            exec_env['PATH'] = f"{venv_bin}:{exec_env.get('PATH', '')}"
        
        # Add PYTHONPATH for module imports
        python_paths = [
            os.getcwd(),
            os.path.join(os.getcwd(), 'IndexAgent'),
            os.path.join(os.getcwd(), 'airflow-hub'),
            os.path.join(os.getcwd(), 'infra')
        ]
        exec_env['PYTHONPATH'] = ':'.join(python_paths)
        
        # Execute command
        try:
            if stream and output_callback:
                return self._execute_streaming(cmd, exec_env, cwd, output_callback)
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=exec_env,
                    cwd=cwd,
                    timeout=None if stream else 300
                )
                
                # Handle missing dependencies
                if result.returncode == 1 and 'ModuleNotFoundError' in result.stderr and auto_install:
                    logger.info("Missing dependencies detected, attempting to install...")
                    if self._install_dependencies():
                        # Retry command
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            env=exec_env,
                            cwd=cwd,
                            timeout=None if stream else 300
                        )
                
                return result
                
        except subprocess.TimeoutExpired as e:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout='',
                stderr=f'Command timed out: {e}'
            )
    
    def _execute_streaming(self, cmd: List[str], env: Dict[str, str], 
                          cwd: Optional[str], callback: callable) -> subprocess.CompletedProcess:
        """Execute command with streaming output"""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=cwd,
            bufsize=1
        )
        
        stdout_lines = []
        
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.rstrip()
                stdout_lines.append(line)
                callback(line)
        
        process.wait()
        stderr = process.stderr.read()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout='\n'.join(stdout_lines),
            stderr=stderr
        )
    
    def _install_dependencies(self) -> bool:
        """Attempt to install missing dependencies"""
        try:
            # Install DEAN CLI dependencies
            requirements = [
                'click',
                'requests',
                'psycopg2-binary',
                'tabulate',
                'pyyaml'
            ]
            
            cmd = [self.python_path, '-m', 'pip', 'install'] + requirements
            
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                logger.info("Dependencies installed successfully")
                return True
            else:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False


class CommandExecutor:
    """Main command executor that delegates to appropriate backend"""
    
    def __init__(self, mode: ExecutionMode = ExecutionMode.AUTO,
                 config: Optional[Dict[str, Any]] = None):
        self.mode = mode
        self.config = config or {}
        self.detector = EnvironmentDetector()
        self._executor_cache = {}
    
    def execute(self, command: str, env: Optional[Dict[str, str]] = None,
                cwd: Optional[str] = None, **kwargs) -> subprocess.CompletedProcess:
        """Execute command using appropriate executor"""
        executor = self._get_executor()
        return executor.execute(command, env=env, cwd=cwd, **kwargs)
    
    def _get_executor(self) -> Union[ContainerExecutor, HostExecutor]:
        """Get the appropriate executor based on mode"""
        if self.mode == ExecutionMode.CONTAINER:
            return self._get_container_executor()
        elif self.mode == ExecutionMode.HOST:
            return self._get_host_executor()
        else:  # AUTO mode
            return self._auto_select_executor()
    
    def _auto_select_executor(self) -> Union[ContainerExecutor, HostExecutor]:
        """Automatically select the best executor"""
        # If in container, use host executor (direct execution)
        if self.detector.is_container_environment():
            return self._get_host_executor()
        
        # If Docker is available, prefer container
        if self.detector.is_docker_available():
            return self._get_container_executor()
        
        # Fall back to host execution
        return self._get_host_executor()
    
    def _get_container_executor(self) -> ContainerExecutor:
        """Get or create container executor"""
        if 'container' not in self._executor_cache:
            settings = self.config.get('container_settings', {})
            self._executor_cache['container'] = ContainerExecutor(
                container_name=settings.get('container_name', 'dean-cli'),
                workspace=settings.get('workspace', '/workspace')
            )
        return self._executor_cache['container']
    
    def _get_host_executor(self) -> HostExecutor:
        """Get or create host executor"""
        if 'host' not in self._executor_cache:
            settings = self.config.get('host_settings', {})
            self._executor_cache['host'] = HostExecutor(
                python_path=settings.get('python_path', sys.executable),
                venv_path=settings.get('venv_path')
            )
        return self._executor_cache['host']


class FlexibleCLI:
    """Main flexible CLI interface"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.load_config()
        self.detector = EnvironmentDetector()
        self.executor = CommandExecutor(
            mode=ExecutionMode(self.config.get('execution_mode', 'auto')),
            config=self.config
        )
    
    def parse_args(self, args: List[str]) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description='DEAN Flexible CLI - Unified interface for DEAN commands',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Global options
        parser.add_argument('--mode', choices=['auto', 'container', 'host'],
                          help='Execution mode (default: auto)')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug output')
        parser.add_argument('--retry', action='store_true',
                          help='Retry failed commands once')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Status command
        subparsers.add_parser('status', help='Show system status')
        
        # Evolution commands
        evo_parser = subparsers.add_parser('evolution', help='Evolution management')
        evo_sub = evo_parser.add_subparsers(dest='subcommand')
        
        evo_start = evo_sub.add_parser('start', help='Start evolution')
        evo_start.add_argument('--generations', type=int, default=10)
        evo_start.add_argument('--agents', type=int, default=5)
        evo_start.add_argument('--strategies', type=str)
        
        evo_sub.add_parser('stop', help='Stop evolution')
        evo_sub.add_parser('list', help='List evolution runs')
        
        # Logs command
        logs_parser = subparsers.add_parser('logs', help='View logs')
        logs_parser.add_argument('-f', '--follow', action='store_true')
        logs_parser.add_argument('--component', choices=['evolution', 'agent', 'pattern'])
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export data')
        export_parser.add_argument('type', choices=['patterns', 'metrics', 'lineage'])
        export_parser.add_argument('--format', choices=['json', 'csv'], default='json')
        
        # Pattern command
        pattern_parser = subparsers.add_parser('pattern', help='Pattern management')
        pattern_sub = pattern_parser.add_subparsers(dest='subcommand')
        pattern_sub.add_parser('list', help='List patterns')
        
        # Agent command
        subparsers.add_parser('agent', help='Agent information')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_sub = config_parser.add_subparsers(dest='subcommand')
        
        config_set = config_sub.add_parser('set', help='Set configuration value')
        config_set.add_argument('key', help='Configuration key')
        config_set.add_argument('value', help='Configuration value')
        
        config_sub.add_parser('show', help='Show configuration')
        
        # Environment info
        subparsers.add_parser('env-info', help='Show environment information')
        
        return parser.parse_args(args)
    
    def execute(self, args: List[str]) -> int:
        """Execute command with given arguments"""
        parsed_args = self.parse_args(args)
        
        # Handle mode override
        if parsed_args.mode:
            self.executor.mode = ExecutionMode(parsed_args.mode)
        
        # Enable debug if requested
        if parsed_args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Handle special commands
        if parsed_args.command == 'env-info':
            return self._show_environment_info()
        
        if parsed_args.command == 'config':
            return self._handle_config_command(parsed_args)
        
        # Build dean command
        dean_cmd = self._build_dean_command(parsed_args)
        if not dean_cmd:
            logger.error("No command specified")
            return 1
        
        # Execute command
        try:
            result = self.executor.execute(dean_cmd)
            
            # Handle retry
            if result.returncode != 0 and parsed_args.retry:
                logger.info("Command failed, retrying...")
                result = self.executor.execute(dean_cmd)
            
            # Output results
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            
            return result.returncode
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return 1
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("DEAN Flexible CLI - Interactive Mode")
        print("Type 'help' for commands, 'exit' to quit")
        print()
        
        while True:
            try:
                command = input("dean> ").strip()
                if command.lower() in ['exit', 'quit']:
                    break
                elif command.lower() == 'help':
                    self._print_help()
                elif command:
                    args = command.split()
                    self.execute(args)
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                logger.error(f"Error: {e}")
    
    def _build_dean_command(self, args: argparse.Namespace) -> Optional[str]:
        """Build dean command from parsed arguments"""
        if not args.command:
            return None
        
        cmd_parts = ['dean', args.command]
        
        # Add subcommand if present
        if hasattr(args, 'subcommand') and args.subcommand:
            cmd_parts.append(args.subcommand)
        
        # Add options based on command
        if args.command == 'evolution' and args.subcommand == 'start':
            cmd_parts.extend(['--generations', str(args.generations)])
            cmd_parts.extend(['--agents', str(args.agents)])
            if args.strategies:
                cmd_parts.extend(['--strategies', args.strategies])
        
        elif args.command == 'logs':
            if args.follow:
                cmd_parts.append('-f')
            if args.component:
                cmd_parts.extend(['--component', args.component])
        
        elif args.command == 'export':
            cmd_parts.append(args.type)
            cmd_parts.extend(['--format', args.format])
        
        return ' '.join(cmd_parts)
    
    def _show_environment_info(self) -> int:
        """Display environment information"""
        print("=== DEAN Execution Environment ===")
        print(f"Current Mode: {self.executor.mode.value}")
        print(f"In Container: {self.detector.is_container_environment()}")
        print(f"Docker Available: {self.detector.is_docker_available()}")
        print(f"Compose Command: {self.detector.get_compose_command() or 'Not found'}")
        print(f"Python Version: {self.detector.get_python_version()}")
        print(f"DEAN Installed: {self.detector.is_dean_installed()}")
        print(f"Working Directory: {os.getcwd()}")
        print()
        print("=== Configuration ===")
        for key, value in self.config.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        return 0
    
    def _handle_config_command(self, args: argparse.Namespace) -> int:
        """Handle configuration commands"""
        if args.subcommand == 'show':
            print(yaml.dump(self.config, default_flow_style=False))
            return 0
        
        elif args.subcommand == 'set':
            # Update configuration
            keys = args.key.split('.')
            current = self.config
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = args.value
            
            # Save configuration
            self.config_manager.save_config(self.config)
            print(f"Set {args.key} = {args.value}")
            return 0
        
        return 1
    
    def _print_help(self):
        """Print help information"""
        print("""
Available commands:
  status              - Show system status
  evolution start     - Start evolution run
  evolution stop      - Stop evolution run
  evolution list      - List evolution runs
  logs [-f]           - View logs (use -f to follow)
  export <type>       - Export data (patterns, metrics, lineage)
  pattern list        - List discovered patterns
  agent               - Show agent information
  config show         - Show current configuration
  config set <k> <v>  - Set configuration value
  env-info            - Show environment information
  help                - Show this help
  exit                - Exit interactive mode
""")


def main():
    """Main entry point"""
    cli = FlexibleCLI()
    
    # If no arguments, run in interactive mode
    if len(sys.argv) == 1:
        cli.interactive_mode()
    else:
        sys.exit(cli.execute(sys.argv[1:]))


if __name__ == "__main__":
    main()