#!/usr/bin/env python3
"""
Test suite for DEAN Advanced Deployment Capabilities
Tests multi-environment deployments, rolling updates, and rollback procedures
"""

import pytest
import os
import yaml
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any, Optional
import subprocess

# Import the advanced deployment system we're about to build
from infra.modules.agent_evolution.src.deployment.advanced_deployment import (
    DeploymentManager,
    Environment,
    DeploymentConfig,
    DeploymentStrategy,
    RollbackManager,
    ConfigurationTemplateEngine,
    HealthChecker,
    DeploymentOrchestrator,
    BlueGreenDeployment,
    CanaryDeployment,
    RollingUpdateDeployment
)


class TestEnvironment:
    """Test suite for environment management"""
    
    def test_environment_creation(self):
        """Test creating different environment configurations"""
        dev_env = Environment(
            name='development',
            api_url='http://localhost:8090',
            airflow_url='http://localhost:8080',
            indexagent_url='http://localhost:8081',
            config_overrides={
                'token_budget': 100000,
                'debug_mode': True
            }
        )
        
        assert dev_env.name == 'development'
        assert dev_env.api_url == 'http://localhost:8090'
        assert dev_env.config_overrides['debug_mode'] is True
    
    def test_environment_validation(self):
        """Test environment configuration validation"""
        # Valid environment
        env = Environment(
            name='production',
            api_url='https://dean-api.prod.company.com',
            airflow_url='https://airflow.prod.company.com',
            indexagent_url='https://indexagent.prod.company.com'
        )
        assert env.validate() is True
        
        # Invalid environment (missing required URL)
        with pytest.raises(ValueError):
            invalid_env = Environment(
                name='invalid',
                api_url='',
                airflow_url='http://localhost:8080',
                indexagent_url='http://localhost:8081'
            )
            invalid_env.validate()
    
    def test_environment_comparison(self):
        """Test comparing environment configurations"""
        env1 = Environment(
            name='staging',
            api_url='http://staging:8090',
            config_overrides={'version': '1.0.0'}
        )
        
        env2 = Environment(
            name='staging',
            api_url='http://staging:8090',
            config_overrides={'version': '1.0.1'}
        )
        
        differences = env1.diff(env2)
        assert 'config_overrides.version' in differences
        assert differences['config_overrides.version'] == ('1.0.0', '1.0.1')


class TestDeploymentConfig:
    """Test suite for deployment configuration"""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample deployment configuration"""
        return {
            'environments': {
                'development': {
                    'api_url': 'http://localhost:8090',
                    'airflow_url': 'http://localhost:8080',
                    'indexagent_url': 'http://localhost:8081',
                    'config_overrides': {
                        'token_budget': 100000,
                        'debug_mode': True
                    }
                },
                'staging': {
                    'api_url': 'http://staging.dean.internal:8090',
                    'airflow_url': 'http://staging.dean.internal:8080',
                    'indexagent_url': 'http://staging.dean.internal:8081',
                    'config_overrides': {
                        'token_budget': 500000,
                        'debug_mode': False
                    }
                },
                'production': {
                    'api_url': 'https://api.dean.company.com',
                    'airflow_url': 'https://airflow.dean.company.com',
                    'indexagent_url': 'https://indexagent.dean.company.com',
                    'config_overrides': {
                        'token_budget': 1000000,
                        'debug_mode': False,
                        'high_availability': True
                    }
                }
            },
            'deployment_strategies': {
                'development': 'direct',
                'staging': 'rolling_update',
                'production': 'blue_green'
            },
            'health_checks': {
                'timeout': 300,
                'interval': 10,
                'required_services': ['evolution_api', 'airflow', 'indexagent']
            }
        }
    
    def test_load_deployment_config(self, sample_config):
        """Test loading deployment configuration from file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            config_path = f.name
        
        try:
            config = DeploymentConfig.from_file(config_path)
            
            assert len(config.environments) == 3
            assert 'development' in config.environments
            assert config.environments['production'].config_overrides['high_availability'] is True
            assert config.deployment_strategies['production'] == DeploymentStrategy.BLUE_GREEN
            
        finally:
            os.unlink(config_path)
    
    def test_deployment_config_validation(self, sample_config):
        """Test deployment configuration validation"""
        config = DeploymentConfig(sample_config)
        
        # Valid config should pass
        assert config.validate() is True
        
        # Invalid strategy should fail
        sample_config['deployment_strategies']['production'] = 'invalid_strategy'
        with pytest.raises(ValueError):
            invalid_config = DeploymentConfig(sample_config)
            invalid_config.validate()
    
    def test_environment_specific_config(self, sample_config):
        """Test getting environment-specific configuration"""
        config = DeploymentConfig(sample_config)
        
        dev_env = config.get_environment('development')
        assert dev_env.config_overrides['debug_mode'] is True
        assert dev_env.config_overrides['token_budget'] == 100000
        
        prod_env = config.get_environment('production')
        assert prod_env.config_overrides['debug_mode'] is False
        assert prod_env.config_overrides['high_availability'] is True


class TestConfigurationTemplateEngine:
    """Test suite for configuration templating"""
    
    @pytest.fixture
    def template_engine(self):
        """Create a template engine instance"""
        return ConfigurationTemplateEngine()
    
    def test_simple_template_rendering(self, template_engine):
        """Test rendering simple templates"""
        template = """
        api_url: {{ api_url }}
        token_budget: {{ token_budget }}
        environment: {{ environment }}
        """
        
        variables = {
            'api_url': 'http://localhost:8090',
            'token_budget': 100000,
            'environment': 'development'
        }
        
        result = template_engine.render(template, variables)
        
        assert 'http://localhost:8090' in result
        assert '100000' in result
        assert 'development' in result
    
    def test_conditional_template_rendering(self, template_engine):
        """Test conditional logic in templates"""
        template = """
        debug_mode: {{ debug_mode }}
        {% if environment == 'production' %}
        high_availability: true
        replicas: 3
        {% else %}
        high_availability: false
        replicas: 1
        {% endif %}
        """
        
        # Development environment
        dev_result = template_engine.render(template, {
            'debug_mode': True,
            'environment': 'development'
        })
        assert 'high_availability: false' in dev_result
        assert 'replicas: 1' in dev_result
        
        # Production environment
        prod_result = template_engine.render(template, {
            'debug_mode': False,
            'environment': 'production'
        })
        assert 'high_availability: true' in prod_result
        assert 'replicas: 3' in prod_result
    
    def test_template_with_secrets(self, template_engine):
        """Test handling secrets in templates"""
        template = """
        database_url: {{ vault('database/url') }}
        api_key: {{ vault('api/keys/claude') }}
        """
        
        with patch.object(template_engine, 'get_secret') as mock_get_secret:
            mock_get_secret.side_effect = lambda key: {
                'database/url': 'postgresql://user:pass@db:5432/dean',
                'api/keys/claude': 'sk-ant-123456'
            }.get(key)
            
            result = template_engine.render(template, {})
            
            assert 'postgresql://user:pass@db:5432/dean' in result
            assert 'sk-ant-123456' in result
    
    def test_template_inheritance(self, template_engine):
        """Test template inheritance for configuration reuse"""
        base_template = """
        # Base configuration
        log_level: info
        metrics_enabled: true
        {% block custom_config %}{% endblock %}
        """
        
        env_template = """
        {% extends 'base.yaml' %}
        {% block custom_config %}
        environment: {{ environment }}
        api_url: {{ api_url }}
        {% endblock %}
        """
        
        with patch.object(template_engine, 'load_template') as mock_load:
            mock_load.return_value = base_template
            
            result = template_engine.render(env_template, {
                'environment': 'staging',
                'api_url': 'http://staging:8090'
            }, enable_inheritance=True)
            
            assert 'log_level: info' in result
            assert 'metrics_enabled: true' in result
            assert 'environment: staging' in result


class TestHealthChecker:
    """Test suite for health checking functionality"""
    
    @pytest.fixture
    def health_checker(self):
        """Create a health checker instance"""
        return HealthChecker(
            timeout=30,
            interval=5,
            max_retries=6
        )
    
    @pytest.mark.asyncio
    async def test_service_health_check(self, health_checker):
        """Test checking individual service health"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Healthy service
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {'status': 'healthy'}
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            result = await health_checker.check_service('http://localhost:8090/health')
            assert result.is_healthy is True
            assert result.response_time > 0
    
    @pytest.mark.asyncio
    async def test_environment_health_check(self, health_checker):
        """Test checking entire environment health"""
        environment = Environment(
            name='staging',
            api_url='http://staging:8090',
            airflow_url='http://staging:8080',
            indexagent_url='http://staging:8081'
        )
        
        with patch.object(health_checker, 'check_service') as mock_check:
            mock_check.return_value = MagicMock(is_healthy=True, response_time=0.1)
            
            result = await health_checker.check_environment(environment)
            
            assert result.is_healthy is True
            assert len(result.service_results) == 3
            assert all(r.is_healthy for r in result.service_results.values())
    
    @pytest.mark.asyncio
    async def test_health_check_with_retries(self, health_checker):
        """Test health check retry mechanism"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # First 2 attempts fail, third succeeds
            mock_get.side_effect = [
                Exception("Connection error"),
                Exception("Timeout"),
                MagicMock(status=200, __aenter__=lambda self: self, 
                         __aexit__=lambda *args: None)
            ]
            
            result = await health_checker.check_service(
                'http://localhost:8090/health',
                retry=True
            )
            
            assert mock_get.call_count == 3
            assert result.is_healthy is True
    
    def test_health_check_criteria(self, health_checker):
        """Test custom health check criteria"""
        # Define custom criteria
        criteria = {
            'min_response_time': 0.5,
            'required_status': 200,
            'required_body_contains': 'healthy'
        }
        
        # Test passing criteria
        result = MagicMock(
            status_code=200,
            response_time=0.3,
            body='{"status": "healthy"}'
        )
        assert health_checker.evaluate_criteria(result, criteria) is True
        
        # Test failing criteria (slow response)
        result.response_time = 0.6
        assert health_checker.evaluate_criteria(result, criteria) is False


class TestRollbackManager:
    """Test suite for rollback functionality"""
    
    @pytest.fixture
    def rollback_manager(self):
        """Create a rollback manager instance"""
        return RollbackManager(backup_dir='/tmp/dean-rollbacks')
    
    def test_create_backup(self, rollback_manager):
        """Test creating deployment backup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rollback_manager.backup_dir = tmpdir
            
            deployment_state = {
                'version': '1.2.3',
                'timestamp': datetime.utcnow().isoformat(),
                'environment': 'production',
                'config': {
                    'token_budget': 1000000,
                    'services': ['api', 'airflow', 'indexagent']
                }
            }
            
            backup_id = rollback_manager.create_backup('production', deployment_state)
            
            assert backup_id is not None
            backup_path = Path(tmpdir) / 'production' / backup_id
            assert backup_path.exists()
            
            # Verify backup content
            with open(backup_path / 'state.json', 'r') as f:
                saved_state = json.load(f)
                assert saved_state['version'] == '1.2.3'
    
    def test_list_backups(self, rollback_manager):
        """Test listing available backups"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rollback_manager.backup_dir = tmpdir
            
            # Create multiple backups
            for i in range(3):
                rollback_manager.create_backup('production', {
                    'version': f'1.2.{i}',
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            backups = rollback_manager.list_backups('production')
            assert len(backups) == 3
            
            # Backups should be sorted by timestamp (newest first)
            versions = [b['state']['version'] for b in backups]
            assert versions == ['1.2.2', '1.2.1', '1.2.0']
    
    def test_rollback_deployment(self, rollback_manager):
        """Test rolling back to a previous deployment"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rollback_manager.backup_dir = tmpdir
            
            # Create a backup
            original_state = {
                'version': '1.2.3',
                'config': {'token_budget': 1000000}
            }
            backup_id = rollback_manager.create_backup('production', original_state)
            
            # Mock deployment process
            with patch.object(rollback_manager, 'apply_deployment') as mock_apply:
                mock_apply.return_value = True
                
                result = rollback_manager.rollback('production', backup_id)
                
                assert result is True
                mock_apply.assert_called_once_with('production', original_state)
    
    def test_automatic_rollback_on_failure(self, rollback_manager):
        """Test automatic rollback when deployment fails"""
        with patch.object(rollback_manager, 'create_backup') as mock_backup:
            with patch.object(rollback_manager, 'rollback') as mock_rollback:
                mock_backup.return_value = 'backup-123'
                
                # Simulate deployment failure
                def failing_deployment():
                    raise Exception("Deployment failed")
                
                with pytest.raises(Exception):
                    with rollback_manager.auto_rollback('production', {'version': '1.2.4'}):
                        failing_deployment()
                
                # Verify rollback was triggered
                mock_rollback.assert_called_once_with('production', 'backup-123')


class TestDeploymentStrategies:
    """Test suite for different deployment strategies"""
    
    def test_rolling_update_deployment(self):
        """Test rolling update deployment strategy"""
        deployment = RollingUpdateDeployment(
            batch_size=2,
            wait_between_batches=10
        )
        
        services = ['api-1', 'api-2', 'api-3', 'api-4', 'api-5']
        
        # Mock service update
        updated_services = []
        
        def mock_update(service):
            updated_services.append(service)
            return True
        
        with patch.object(deployment, 'update_service', side_effect=mock_update):
            with patch('time.sleep'):  # Skip actual waiting
                result = deployment.deploy(services)
        
        assert result is True
        assert len(updated_services) == 5
        # Verify services were updated in batches
        assert updated_services[:2] == ['api-1', 'api-2']
        assert updated_services[2:4] == ['api-3', 'api-4']
        assert updated_services[4:] == ['api-5']
    
    def test_blue_green_deployment(self):
        """Test blue-green deployment strategy"""
        deployment = BlueGreenDeployment()
        
        with patch.object(deployment, 'create_green_environment') as mock_create:
            with patch.object(deployment, 'test_green_environment') as mock_test:
                with patch.object(deployment, 'switch_traffic') as mock_switch:
                    with patch.object(deployment, 'cleanup_blue_environment') as mock_cleanup:
                        mock_create.return_value = 'green-env-123'
                        mock_test.return_value = True
                        mock_switch.return_value = True
                        
                        result = deployment.deploy('production', {
                            'version': '1.3.0',
                            'config': {'replicas': 3}
                        })
                        
                        assert result is True
                        mock_create.assert_called_once()
                        mock_test.assert_called_once_with('green-env-123')
                        mock_switch.assert_called_once()
                        mock_cleanup.assert_called_once()
    
    def test_canary_deployment(self):
        """Test canary deployment strategy"""
        deployment = CanaryDeployment(
            initial_percentage=10,
            increment=20,
            analysis_duration=60
        )
        
        with patch.object(deployment, 'deploy_canary') as mock_deploy:
            with patch.object(deployment, 'analyze_metrics') as mock_analyze:
                with patch.object(deployment, 'adjust_traffic') as mock_adjust:
                    mock_deploy.return_value = 'canary-123'
                    mock_analyze.return_value = {'error_rate': 0.01, 'latency': 100}
                    mock_adjust.return_value = True
                    
                    with patch('time.sleep'):  # Skip waiting
                        result = deployment.deploy('production', {
                            'version': '1.3.0'
                        })
                    
                    assert result is True
                    # Verify progressive rollout
                    traffic_adjustments = [call[0][1] for call in mock_adjust.call_args_list]
                    assert traffic_adjustments == [10, 30, 50, 70, 90, 100]
    
    def test_deployment_strategy_selection(self):
        """Test automatic strategy selection based on environment"""
        config = DeploymentConfig({
            'deployment_strategies': {
                'development': 'direct',
                'staging': 'rolling_update',
                'production': 'blue_green'
            }
        })
        
        manager = DeploymentManager(config)
        
        # Development uses direct deployment
        dev_strategy = manager.get_strategy('development')
        assert isinstance(dev_strategy, DirectDeployment)
        
        # Staging uses rolling update
        staging_strategy = manager.get_strategy('staging')
        assert isinstance(staging_strategy, RollingUpdateDeployment)
        
        # Production uses blue-green
        prod_strategy = manager.get_strategy('production')
        assert isinstance(prod_strategy, BlueGreenDeployment)


class TestDeploymentOrchestrator:
    """Test suite for the main deployment orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a deployment orchestrator instance"""
        config = DeploymentConfig({
            'environments': {
                'staging': {
                    'api_url': 'http://staging:8090',
                    'airflow_url': 'http://staging:8080',
                    'indexagent_url': 'http://staging:8081'
                }
            },
            'deployment_strategies': {
                'staging': 'rolling_update'
            }
        })
        return DeploymentOrchestrator(config)
    
    @pytest.mark.asyncio
    async def test_full_deployment_flow(self, orchestrator):
        """Test complete deployment flow"""
        with patch.object(orchestrator, 'pre_deployment_checks') as mock_pre:
            with patch.object(orchestrator, 'create_backup') as mock_backup:
                with patch.object(orchestrator, 'execute_deployment') as mock_deploy:
                    with patch.object(orchestrator, 'post_deployment_validation') as mock_post:
                        mock_pre.return_value = True
                        mock_backup.return_value = 'backup-123'
                        mock_deploy.return_value = True
                        mock_post.return_value = True
                        
                        result = await orchestrator.deploy('staging', '1.3.0', dry_run=False)
                        
                        assert result.success is True
                        assert result.version == '1.3.0'
                        assert result.backup_id == 'backup-123'
                        
                        # Verify all steps were executed
                        mock_pre.assert_called_once()
                        mock_backup.assert_called_once()
                        mock_deploy.assert_called_once()
                        mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deployment_with_pre_check_failure(self, orchestrator):
        """Test deployment abortion on pre-check failure"""
        with patch.object(orchestrator, 'pre_deployment_checks') as mock_pre:
            mock_pre.return_value = False
            
            result = await orchestrator.deploy('staging', '1.3.0')
            
            assert result.success is False
            assert 'pre-deployment checks failed' in result.error
    
    @pytest.mark.asyncio
    async def test_deployment_with_rollback(self, orchestrator):
        """Test deployment with automatic rollback on failure"""
        with patch.object(orchestrator, 'pre_deployment_checks') as mock_pre:
            with patch.object(orchestrator, 'create_backup') as mock_backup:
                with patch.object(orchestrator, 'execute_deployment') as mock_deploy:
                    with patch.object(orchestrator, 'rollback') as mock_rollback:
                        mock_pre.return_value = True
                        mock_backup.return_value = 'backup-123'
                        mock_deploy.side_effect = Exception("Deployment failed")
                        mock_rollback.return_value = True
                        
                        result = await orchestrator.deploy('staging', '1.3.0')
                        
                        assert result.success is False
                        assert result.rolled_back is True
                        mock_rollback.assert_called_once_with('staging', 'backup-123')
    
    @pytest.mark.asyncio
    async def test_dry_run_deployment(self, orchestrator):
        """Test dry-run deployment mode"""
        with patch.object(orchestrator, 'pre_deployment_checks') as mock_pre:
            with patch.object(orchestrator, 'simulate_deployment') as mock_simulate:
                mock_pre.return_value = True
                mock_simulate.return_value = {
                    'changes': [
                        'Update API version to 1.3.0',
                        'Scale replicas from 2 to 3',
                        'Update token budget to 1500000'
                    ],
                    'estimated_duration': 300
                }
                
                result = await orchestrator.deploy('staging', '1.3.0', dry_run=True)
                
                assert result.success is True
                assert result.dry_run is True
                assert len(result.planned_changes) == 3
                assert result.estimated_duration == 300


class TestMultiEnvironmentDeployment:
    """Test suite for multi-environment deployment workflows"""
    
    @pytest.mark.asyncio
    async def test_sequential_environment_deployment(self):
        """Test deploying to multiple environments sequentially"""
        manager = DeploymentManager(DeploymentConfig({
            'environments': {
                'dev': {'api_url': 'http://dev:8090'},
                'staging': {'api_url': 'http://staging:8090'},
                'production': {'api_url': 'http://prod:8090'}
            }
        }))
        
        deployment_sequence = ['dev', 'staging', 'production']
        results = []
        
        with patch.object(manager, 'deploy_to_environment') as mock_deploy:
            mock_deploy.return_value = MagicMock(success=True)
            
            for env in deployment_sequence:
                result = await manager.deploy_to_environment(env, '1.3.0')
                results.append(result)
                
                # Verify previous environments were successful before proceeding
                if env != 'dev':
                    assert all(r.success for r in results[:-1])
        
        assert len(results) == 3
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_promotion_based_deployment(self):
        """Test promotion-based deployment workflow"""
        manager = DeploymentManager(DeploymentConfig({
            'promotion_gates': {
                'dev_to_staging': {
                    'manual_approval': False,
                    'automated_tests': True,
                    'min_soak_time': 3600  # 1 hour
                },
                'staging_to_production': {
                    'manual_approval': True,
                    'automated_tests': True,
                    'min_soak_time': 86400  # 24 hours
                }
            }
        }))
        
        # Deploy to dev
        with patch.object(manager, 'deploy_to_environment') as mock_deploy:
            with patch.object(manager, 'run_promotion_checks') as mock_checks:
                mock_deploy.return_value = MagicMock(success=True)
                mock_checks.return_value = True
                
                # Dev deployment
                dev_result = await manager.deploy_with_promotion('dev', '1.3.0')
                assert dev_result.success is True
                
                # Auto-promote to staging (no manual approval needed)
                staging_result = await manager.promote('dev', 'staging')
                assert staging_result.success is True
                
                # Attempt production promotion (requires approval)
                mock_checks.return_value = False  # No approval yet
                prod_result = await manager.promote('staging', 'production')
                assert prod_result.success is False
                assert 'manual approval required' in prod_result.error


class TestConfigurationManagement:
    """Test suite for configuration management across environments"""
    
    def test_configuration_diff_generation(self):
        """Test generating configuration differences between environments"""
        config_manager = ConfigurationManager()
        
        dev_config = {
            'api': {
                'replicas': 1,
                'memory': '512Mi',
                'token_budget': 100000
            },
            'features': {
                'debug_mode': True,
                'metrics_enabled': True
            }
        }
        
        prod_config = {
            'api': {
                'replicas': 3,
                'memory': '2Gi',
                'token_budget': 1000000
            },
            'features': {
                'debug_mode': False,
                'metrics_enabled': True,
                'high_availability': True
            }
        }
        
        diff = config_manager.diff_configs(dev_config, prod_config)
        
        assert diff['api.replicas'] == (1, 3)
        assert diff['api.memory'] == ('512Mi', '2Gi')
        assert diff['features.debug_mode'] == (True, False)
        assert 'features.high_availability' in diff
        assert diff['features.high_availability'] == (None, True)
    
    def test_configuration_validation_rules(self):
        """Test configuration validation against environment rules"""
        validator = ConfigurationValidator({
            'production': {
                'required_fields': ['api.replicas', 'api.memory', 'features.high_availability'],
                'constraints': {
                    'api.replicas': {'min': 2},
                    'api.memory': {'pattern': r'\d+Gi'},
                    'features.debug_mode': {'value': False}
                }
            }
        })
        
        # Valid production config
        valid_config = {
            'api': {'replicas': 3, 'memory': '2Gi'},
            'features': {'debug_mode': False, 'high_availability': True}
        }
        assert validator.validate('production', valid_config) is True
        
        # Invalid production config (debug mode enabled)
        invalid_config = {
            'api': {'replicas': 3, 'memory': '2Gi'},
            'features': {'debug_mode': True, 'high_availability': True}
        }
        with pytest.raises(ValueError):
            validator.validate('production', invalid_config)
    
    def test_secret_management_in_deployments(self):
        """Test handling secrets during deployment"""
        secret_manager = SecretManager(vault_url='http://vault:8200')
        
        with patch.object(secret_manager, 'get_secret') as mock_get:
            with patch.object(secret_manager, 'create_secret') as mock_create:
                mock_get.side_effect = lambda path: {
                    'api/claude_key': 'sk-ant-old-key',
                    'db/password': 'old-password'
                }.get(path)
                
                # Rotate secrets during deployment
                new_secrets = {
                    'api/claude_key': 'sk-ant-new-key',
                    'db/password': 'new-password'
                }
                
                for path, value in new_secrets.items():
                    secret_manager.rotate_secret(path, value)
                
                assert mock_create.call_count == 2
                mock_create.assert_any_call('api/claude_key', 'sk-ant-new-key')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])