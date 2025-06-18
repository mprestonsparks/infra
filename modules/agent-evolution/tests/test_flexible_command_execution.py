#!/usr/bin/env python3
"""
Test suite for DEAN Flexible Command Execution
Tests both containerized and host-based execution modes
"""

import pytest
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any, Optional

# Import the flexible CLI we're about to build
from infra.modules.agent_evolution.src.cli.flexible_dean_cli import (
    FlexibleCLI,
    ExecutionMode,
    CommandExecutor,
    ContainerExecutor,
    HostExecutor,
    EnvironmentDetector,
    ConfigurationManager
)


class TestEnvironmentDetector:
    """Test suite for environment detection"""
    
    @pytest.fixture
    def detector(self):
        """Create an environment detector instance"""
        return EnvironmentDetector()
    
    def test_detect_container_environment(self, detector):
        """Test detection when running inside a container"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True  # /.dockerenv exists
            assert detector.is_container_environment() is True
            mock_exists.assert_called_with('/.dockerenv')
    
    def test_detect_host_environment(self, detector):
        """Test detection when running on host"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False  # /.dockerenv doesn't exist
            assert detector.is_container_environment() is False
    
    def test_detect_docker_availability(self, detector):
        """Test Docker availability detection"""
        with patch('subprocess.run') as mock_run:
            # Docker is available
            mock_run.return_value = subprocess.CompletedProcess(
                args=['docker', '--version'],
                returncode=0,
                stdout='Docker version 20.10.0'
            )
            assert detector.is_docker_available() is True
            
            # Docker not available
            mock_run.side_effect = FileNotFoundError()
            assert detector.is_docker_available() is False
    
    def test_detect_compose_command(self, detector):
        """Test Docker Compose command detection"""
        with patch('subprocess.run') as mock_run:
            # docker-compose available
            mock_run.return_value = subprocess.CompletedProcess(
                args=['docker-compose', '--version'],
                returncode=0
            )
            assert detector.get_compose_command() == 'docker-compose'
            
            # Only docker compose available
            mock_run.side_effect = [
                FileNotFoundError(),  # docker-compose fails
                subprocess.CompletedProcess(args=['docker', 'compose', 'version'], returncode=0)
            ]
            assert detector.get_compose_command() == 'docker compose'
            
            # Neither available
            mock_run.side_effect = FileNotFoundError()
            assert detector.get_compose_command() is None
    
    def test_detect_dean_installation(self, detector):
        """Test DEAN system installation detection"""
        with patch('os.path.exists') as mock_exists:
            # Check for key DEAN directories
            mock_exists.side_effect = lambda path: path in [
                '/infra/docker-compose.yml',
                '/IndexAgent',
                '/airflow-hub'
            ]
            
            assert detector.is_dean_installed('/') is True
            assert detector.is_dean_installed('/nonexistent') is False


class TestConfigurationManager:
    """Test suite for configuration management"""
    
    @pytest.fixture
    def config_manager(self):
        """Create a configuration manager instance"""
        return ConfigurationManager()
    
    def test_load_configuration(self, config_manager):
        """Test loading configuration from files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test config
            config_file = Path(tmpdir) / '.dean-cli.yml'
            config_file.write_text("""
execution_mode: auto
default_workspace: /workspace
container_settings:
  image: dean-cli:latest
  network: dean-network
host_settings:
  python_path: /usr/bin/python3.11
  venv_path: .venv
""")
            
            config = config_manager.load_config(config_file)
            
            assert config['execution_mode'] == 'auto'
            assert config['default_workspace'] == '/workspace'
            assert config['container_settings']['image'] == 'dean-cli:latest'
            assert config['host_settings']['python_path'] == '/usr/bin/python3.11'
    
    def test_merge_configurations(self, config_manager):
        """Test merging multiple configuration sources"""
        global_config = {
            'execution_mode': 'auto',
            'default_workspace': '/workspace'
        }
        
        local_config = {
            'execution_mode': 'host',
            'custom_setting': 'value'
        }
        
        env_vars = {
            'DEAN_EXECUTION_MODE': 'container',
            'DEAN_API_URL': 'http://localhost:8090'
        }
        
        merged = config_manager.merge_configs(global_config, local_config, env_vars)
        
        # Env vars should take precedence
        assert merged['execution_mode'] == 'container'
        assert merged['api_url'] == 'http://localhost:8090'
        assert merged['custom_setting'] == 'value'
        assert merged['default_workspace'] == '/workspace'
    
    def test_validate_configuration(self, config_manager):
        """Test configuration validation"""
        valid_config = {
            'execution_mode': 'auto',
            'api_url': 'http://localhost:8090',
            'container_settings': {
                'image': 'dean-cli:latest'
            }
        }
        
        assert config_manager.validate_config(valid_config) is True
        
        invalid_config = {
            'execution_mode': 'invalid_mode'
        }
        
        with pytest.raises(ValueError):
            config_manager.validate_config(invalid_config)


class TestContainerExecutor:
    """Test suite for container-based command execution"""
    
    @pytest.fixture
    def executor(self):
        """Create a container executor instance"""
        return ContainerExecutor(
            container_name='dean-cli',
            workspace='/workspace'
        )
    
    def test_execute_simple_command(self, executor):
        """Test executing a simple command in container"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='Command output',
                stderr=''
            )
            
            result = executor.execute('dean status')
            
            assert result.returncode == 0
            assert result.stdout == 'Command output'
            
            # Verify Docker exec was called correctly
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == 'docker'
            assert args[1] == 'exec'
            assert 'dean-cli' in args
            assert 'dean status' in ' '.join(args)
    
    def test_execute_with_environment_variables(self, executor):
        """Test executing command with environment variables"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0
            )
            
            executor.execute('dean evolution start', env={'DEAN_TOKEN_BUDGET': '1000000'})
            
            # Verify environment variable was passed
            args = mock_run.call_args[0][0]
            assert '-e' in args
            env_index = args.index('-e')
            assert args[env_index + 1] == 'DEAN_TOKEN_BUDGET=1000000'
    
    def test_execute_with_working_directory(self, executor):
        """Test executing command with specific working directory"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0
            )
            
            executor.execute('ls', cwd='/workspace/IndexAgent')
            
            # Verify working directory was set
            args = mock_run.call_args[0][0]
            assert '-w' in args
            wd_index = args.index('-w')
            assert args[wd_index + 1] == '/workspace/IndexAgent'
    
    def test_handle_container_not_running(self, executor):
        """Test handling when container is not running"""
        with patch('subprocess.run') as mock_run:
            # First call fails (container not running)
            mock_run.side_effect = [
                subprocess.CompletedProcess(
                    args=[], returncode=125,
                    stderr='Error: No such container'
                ),
                # Start container succeeds
                subprocess.CompletedProcess(args=[], returncode=0),
                # Retry command succeeds
                subprocess.CompletedProcess(
                    args=[], returncode=0,
                    stdout='Success'
                )
            ]
            
            result = executor.execute('dean status', auto_start=True)
            
            assert result.returncode == 0
            assert mock_run.call_count == 3  # Check, start, retry
    
    def test_stream_output(self, executor):
        """Test streaming command output"""
        output_lines = []
        
        def capture_output(line):
            output_lines.append(line)
        
        with patch('subprocess.Popen') as mock_popen:
            process = MagicMock()
            process.stdout.readline.side_effect = [
                b'Line 1\n',
                b'Line 2\n',
                b''  # EOF
            ]
            process.poll.side_effect = [None, None, 0]
            process.returncode = 0
            mock_popen.return_value = process
            
            result = executor.execute('dean logs -f', stream=True, 
                                    output_callback=capture_output)
            
            assert len(output_lines) == 2
            assert output_lines[0] == 'Line 1'
            assert output_lines[1] == 'Line 2'


class TestHostExecutor:
    """Test suite for host-based command execution"""
    
    @pytest.fixture
    def executor(self):
        """Create a host executor instance"""
        return HostExecutor(
            python_path='/usr/bin/python3.11',
            venv_path='/workspace/.venv'
        )
    
    def test_execute_python_command(self, executor):
        """Test executing Python-based DEAN commands"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='Status output'
            )
            
            result = executor.execute('dean status')
            
            # Verify Python was called with correct module
            args = mock_run.call_args[0][0]
            assert args[0] == '/usr/bin/python3.11'
            assert '-m' in args
            assert 'dean_cli' in args
            assert 'status' in args
    
    def test_activate_virtual_environment(self, executor):
        """Test virtual environment activation"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0
            )
            
            executor.execute('dean evolution start')
            
            # Verify venv activation in environment
            env = mock_run.call_args[1]['env']
            assert env['VIRTUAL_ENV'] == '/workspace/.venv'
            assert '/workspace/.venv/bin' in env['PATH']
    
    def test_handle_missing_dependencies(self, executor):
        """Test handling missing Python dependencies"""
        with patch('subprocess.run') as mock_run:
            # First attempt fails with import error
            mock_run.side_effect = [
                subprocess.CompletedProcess(
                    args=[], returncode=1,
                    stderr='ModuleNotFoundError: No module named "dean_cli"'
                ),
                # Install dependencies succeeds
                subprocess.CompletedProcess(args=[], returncode=0),
                # Retry succeeds
                subprocess.CompletedProcess(
                    args=[], returncode=0,
                    stdout='Success'
                )
            ]
            
            result = executor.execute('dean status', auto_install=True)
            
            assert result.returncode == 0
            assert mock_run.call_count == 3
            
            # Verify pip install was called
            install_args = mock_run.call_args_list[1][0][0]
            assert 'pip' in install_args
            assert 'install' in install_args
    
    def test_execute_with_pythonpath(self, executor):
        """Test setting PYTHONPATH for module imports"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0
            )
            
            executor.execute('dean pattern list')
            
            env = mock_run.call_args[1]['env']
            assert 'PYTHONPATH' in env
            assert '/workspace' in env['PYTHONPATH']
            assert '/workspace/IndexAgent' in env['PYTHONPATH']


class TestCommandExecutor:
    """Test suite for the main command executor"""
    
    @pytest.fixture
    def executor(self):
        """Create a command executor instance"""
        return CommandExecutor(mode=ExecutionMode.AUTO)
    
    def test_auto_mode_selection(self, executor):
        """Test automatic execution mode selection"""
        with patch.object(EnvironmentDetector, 'is_container_environment') as mock_detect:
            # In container environment
            mock_detect.return_value = True
            selected_executor = executor._get_executor()
            assert isinstance(selected_executor, ContainerExecutor)
            
            # On host with Docker available
            mock_detect.return_value = False
            with patch.object(EnvironmentDetector, 'is_docker_available') as mock_docker:
                mock_docker.return_value = True
                selected_executor = executor._get_executor()
                assert isinstance(selected_executor, ContainerExecutor)
                
                # On host without Docker
                mock_docker.return_value = False
                selected_executor = executor._get_executor()
                assert isinstance(selected_executor, HostExecutor)
    
    def test_forced_mode_selection(self, executor):
        """Test forced execution mode"""
        # Force container mode
        executor.mode = ExecutionMode.CONTAINER
        selected_executor = executor._get_executor()
        assert isinstance(selected_executor, ContainerExecutor)
        
        # Force host mode
        executor.mode = ExecutionMode.HOST
        selected_executor = executor._get_executor()
        assert isinstance(selected_executor, HostExecutor)
    
    def test_command_routing(self, executor):
        """Test routing commands to appropriate executor"""
        with patch.object(ContainerExecutor, 'execute') as mock_container:
            with patch.object(HostExecutor, 'execute') as mock_host:
                mock_container.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout='Container result'
                )
                mock_host.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout='Host result'
                )
                
                # Test container routing
                executor.mode = ExecutionMode.CONTAINER
                result = executor.execute('dean status')
                assert result.stdout == 'Container result'
                mock_container.assert_called_once()
                mock_host.assert_not_called()
                
                # Reset mocks
                mock_container.reset_mock()
                mock_host.reset_mock()
                
                # Test host routing
                executor.mode = ExecutionMode.HOST
                result = executor.execute('dean status')
                assert result.stdout == 'Host result'
                mock_host.assert_called_once()
                mock_container.assert_not_called()


class TestFlexibleCLI:
    """Test suite for the main flexible CLI interface"""
    
    @pytest.fixture
    def cli(self):
        """Create a flexible CLI instance"""
        return FlexibleCLI()
    
    def test_initialization(self, cli):
        """Test CLI initialization with configuration"""
        assert cli.config is not None
        assert cli.executor is not None
        assert cli.detector is not None
    
    def test_command_parsing(self, cli):
        """Test parsing command line arguments"""
        # Test status command
        args = cli.parse_args(['status'])
        assert args.command == 'status'
        
        # Test evolution command with options
        args = cli.parse_args(['evolution', 'start', '--generations', '10'])
        assert args.command == 'evolution'
        assert args.subcommand == 'start'
        assert args.generations == 10
        
        # Test mode override
        args = cli.parse_args(['--mode', 'host', 'status'])
        assert args.mode == 'host'
    
    def test_execute_command(self, cli):
        """Test executing commands through the CLI"""
        with patch.object(CommandExecutor, 'execute') as mock_execute:
            mock_execute.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='Success'
            )
            
            # Execute status command
            result = cli.execute(['status'])
            assert result == 0
            mock_execute.assert_called_once_with('dean status', env=None, cwd=None)
    
    def test_handle_complex_commands(self, cli):
        """Test handling complex commands with options"""
        with patch.object(CommandExecutor, 'execute') as mock_execute:
            mock_execute.return_value = subprocess.CompletedProcess(
                args=[], returncode=0
            )
            
            # Evolution command with multiple options
            cli.execute(['evolution', 'start', '--generations', '20', 
                        '--agents', '10', '--strategies', 'optimization,refactoring'])
            
            expected_cmd = 'dean evolution start --generations 20 --agents 10 --strategies optimization,refactoring'
            mock_execute.assert_called_once_with(expected_cmd, env=None, cwd=None)
    
    def test_environment_info_command(self, cli):
        """Test environment information display"""
        with patch('builtins.print') as mock_print:
            cli.execute(['env-info'])
            
            # Verify environment information was printed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any('Execution Environment' in call for call in print_calls)
            assert any('Docker' in call for call in print_calls)
    
    def test_interactive_mode(self, cli):
        """Test interactive command mode"""
        with patch('builtins.input') as mock_input:
            with patch.object(CommandExecutor, 'execute') as mock_execute:
                mock_execute.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout='Result'
                )
                
                # Simulate interactive commands
                mock_input.side_effect = ['status', 'evolution list', 'exit']
                
                cli.interactive_mode()
                
                assert mock_execute.call_count == 2
                mock_execute.assert_any_call('dean status', env=None, cwd=None)
                mock_execute.assert_any_call('dean evolution list', env=None, cwd=None)
    
    def test_configuration_commands(self, cli):
        """Test configuration management commands"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / '.dean-cli.yml'
            
            # Test saving configuration
            with patch.object(ConfigurationManager, 'save_config') as mock_save:
                cli.execute(['config', 'set', 'execution_mode', 'host'])
                mock_save.assert_called_once()
            
            # Test showing configuration
            with patch('builtins.print') as mock_print:
                cli.execute(['config', 'show'])
                assert mock_print.called
    
    def test_error_handling(self, cli):
        """Test error handling and recovery"""
        with patch.object(CommandExecutor, 'execute') as mock_execute:
            # Simulate command failure
            mock_execute.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stderr='Error occurred'
            )
            
            # Test that error is handled gracefully
            result = cli.execute(['status'])
            assert result == 1
            
            # Test retry mechanism
            mock_execute.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=1),
                subprocess.CompletedProcess(args=[], returncode=0)
            ]
            
            result = cli.execute(['status', '--retry'])
            assert result == 0
            assert mock_execute.call_count == 2


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios"""
    
    def test_seamless_mode_switching(self):
        """Test seamless switching between execution modes"""
        cli = FlexibleCLI()
        
        with patch.object(CommandExecutor, 'execute') as mock_execute:
            mock_execute.return_value = subprocess.CompletedProcess(
                args=[], returncode=0
            )
            
            # Start in auto mode (detects container)
            with patch.object(EnvironmentDetector, 'is_container_environment', return_value=True):
                cli.execute(['status'])
                assert isinstance(cli.executor._get_executor(), ContainerExecutor)
            
            # Switch to host mode
            cli.execute(['--mode', 'host', 'status'])
            assert isinstance(cli.executor._get_executor(), HostExecutor)
            
            # Back to auto mode (detects host now)
            with patch.object(EnvironmentDetector, 'is_container_environment', return_value=False):
                cli.execute(['status'])
                # Should use host executor when no Docker available
                with patch.object(EnvironmentDetector, 'is_docker_available', return_value=False):
                    assert isinstance(cli.executor._get_executor(), HostExecutor)
    
    def test_development_workflow(self):
        """Test typical development workflow"""
        cli = FlexibleCLI()
        
        with patch.object(CommandExecutor, 'execute') as mock_execute:
            mock_execute.return_value = subprocess.CompletedProcess(
                args=[], returncode=0
            )
            
            # Developer workflow
            commands = [
                ['status'],  # Check system status
                ['evolution', 'start', '--generations', '5', '--agents', '3'],  # Start small test
                ['logs', '-f', '--component', 'evolution'],  # Watch logs
                ['evolution', 'stop'],  # Stop when done
                ['export', 'patterns', '--format', 'json']  # Export results
            ]
            
            for cmd in commands:
                result = cli.execute(cmd)
                assert result == 0
            
            assert mock_execute.call_count == len(commands)
    
    def test_ci_cd_integration(self):
        """Test CI/CD pipeline integration"""
        # Test running in CI environment
        with patch.dict(os.environ, {'CI': 'true', 'DEAN_MODE': 'container'}):
            cli = FlexibleCLI()
            
            with patch.object(CommandExecutor, 'execute') as mock_execute:
                mock_execute.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout='{"status": "healthy"}'
                )
                
                # CI health check
                result = cli.execute(['status', '--json'])
                assert result == 0
                
                # Verify container mode was forced
                assert cli.executor.mode == ExecutionMode.CONTAINER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])