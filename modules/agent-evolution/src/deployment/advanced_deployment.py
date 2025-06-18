#!/usr/bin/env python3
"""
DEAN Advanced Deployment System
Provides multi-environment deployments, rolling updates, and rollback capabilities
"""

import os
import json
import yaml
import shutil
import asyncio
import time
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import aiohttp
from jinja2 import Environment as JinjaEnv, Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Available deployment strategies"""
    DIRECT = "direct"
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"


@dataclass
class Environment:
    """Represents a deployment environment"""
    name: str
    api_url: str
    airflow_url: str = ""
    indexagent_url: str = ""
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate environment configuration"""
        if not self.api_url:
            raise ValueError(f"Environment {self.name} missing required api_url")
        return True
    
    def diff(self, other: 'Environment') -> Dict[str, Tuple[Any, Any]]:
        """Compare with another environment"""
        differences = {}
        
        # Compare URLs
        if self.api_url != other.api_url:
            differences['api_url'] = (self.api_url, other.api_url)
        if self.airflow_url != other.airflow_url:
            differences['airflow_url'] = (self.airflow_url, other.airflow_url)
        if self.indexagent_url != other.indexagent_url:
            differences['indexagent_url'] = (self.indexagent_url, other.indexagent_url)
        
        # Compare config overrides
        for key in set(self.config_overrides.keys()) | set(other.config_overrides.keys()):
            self_val = self.config_overrides.get(key)
            other_val = other.config_overrides.get(key)
            if self_val != other_val:
                differences[f'config_overrides.{key}'] = (self_val, other_val)
        
        return differences


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    is_healthy: bool
    response_time: float
    status_code: Optional[int] = None
    body: Optional[str] = None
    error: Optional[str] = None


@dataclass
class EnvironmentHealthResult:
    """Result of environment-wide health check"""
    is_healthy: bool
    service_results: Dict[str, HealthCheckResult]
    
    @property
    def unhealthy_services(self) -> List[str]:
        """Get list of unhealthy services"""
        return [s for s, r in self.service_results.items() if not r.is_healthy]


@dataclass
class DeploymentResult:
    """Result of a deployment operation"""
    success: bool
    environment: str
    version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    backup_id: Optional[str] = None
    error: Optional[str] = None
    rolled_back: bool = False
    dry_run: bool = False
    planned_changes: List[str] = field(default_factory=list)
    estimated_duration: Optional[int] = None


class ConfigurationTemplateEngine:
    """Handles configuration templating with Jinja2"""
    
    def __init__(self):
        self.env = JinjaEnv(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.env.globals['vault'] = self.get_secret
    
    def render(self, template_str: str, variables: Dict[str, Any], 
               enable_inheritance: bool = False) -> str:
        """Render a template with given variables"""
        if enable_inheritance:
            # For inheritance, we'd need to set up a proper loader
            # For now, simple rendering
            pass
        
        template = self.env.from_string(template_str)
        return template.render(**variables)
    
    def get_secret(self, key: str) -> str:
        """Retrieve secret from vault (mock implementation)"""
        # In production, this would call Vault API
        return f"secret-{key}"
    
    def load_template(self, name: str) -> str:
        """Load template by name (mock implementation)"""
        # In production, this would load from template directory
        return ""


class HealthChecker:
    """Performs health checks on services"""
    
    def __init__(self, timeout: int = 30, interval: int = 5, max_retries: int = 6):
        self.timeout = timeout
        self.interval = interval
        self.max_retries = max_retries
    
    async def check_service(self, url: str, retry: bool = True) -> HealthCheckResult:
        """Check health of a single service"""
        start_time = time.time()
        attempts = 0
        
        while attempts < self.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=self.timeout) as response:
                        response_time = time.time() - start_time
                        body = await response.text()
                        
                        return HealthCheckResult(
                            is_healthy=response.status == 200,
                            response_time=response_time,
                            status_code=response.status,
                            body=body
                        )
                        
            except Exception as e:
                attempts += 1
                if not retry or attempts >= self.max_retries:
                    return HealthCheckResult(
                        is_healthy=False,
                        response_time=time.time() - start_time,
                        error=str(e)
                    )
                await asyncio.sleep(self.interval)
        
        return HealthCheckResult(
            is_healthy=False,
            response_time=time.time() - start_time,
            error="Max retries exceeded"
        )
    
    async def check_environment(self, environment: Environment) -> EnvironmentHealthResult:
        """Check health of all services in an environment"""
        services = {
            'evolution_api': f"{environment.api_url}/health",
            'airflow': f"{environment.airflow_url}/health",
            'indexagent': f"{environment.indexagent_url}/health"
        }
        
        # Check all services concurrently
        tasks = {
            name: self.check_service(url)
            for name, url in services.items()
            if url and url != "/health"  # Skip if URL not configured
        }
        
        results = {}
        for name, task in tasks.items():
            results[name] = await task
        
        all_healthy = all(r.is_healthy for r in results.values())
        
        return EnvironmentHealthResult(
            is_healthy=all_healthy,
            service_results=results
        )
    
    def evaluate_criteria(self, result: Any, criteria: Dict[str, Any]) -> bool:
        """Evaluate custom health check criteria"""
        if criteria.get('required_status') and result.status_code != criteria['required_status']:
            return False
        
        if criteria.get('min_response_time') and result.response_time > criteria['min_response_time']:
            return False
        
        if criteria.get('required_body_contains') and criteria['required_body_contains'] not in result.body:
            return False
        
        return True


class RollbackManager:
    """Manages deployment backups and rollbacks"""
    
    def __init__(self, backup_dir: str = '/tmp/dean-rollbacks'):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, environment: str, state: Dict[str, Any]) -> str:
        """Create a backup of current deployment state"""
        backup_id = f"{environment}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        backup_path = self.backup_dir / environment / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Save state
        with open(backup_path / 'state.json', 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Created backup {backup_id} for {environment}")
        return backup_id
    
    def list_backups(self, environment: str) -> List[Dict[str, Any]]:
        """List available backups for an environment"""
        env_backup_dir = self.backup_dir / environment
        if not env_backup_dir.exists():
            return []
        
        backups = []
        for backup_dir in sorted(env_backup_dir.iterdir(), reverse=True):
            if backup_dir.is_dir():
                state_file = backup_dir / 'state.json'
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    backups.append({
                        'backup_id': backup_dir.name,
                        'state': state
                    })
        
        return backups
    
    def rollback(self, environment: str, backup_id: str) -> bool:
        """Rollback to a specific backup"""
        backup_path = self.backup_dir / environment / backup_id / 'state.json'
        
        if not backup_path.exists():
            logger.error(f"Backup {backup_id} not found for {environment}")
            return False
        
        with open(backup_path, 'r') as f:
            state = json.load(f)
        
        return self.apply_deployment(environment, state)
    
    def apply_deployment(self, environment: str, state: Dict[str, Any]) -> bool:
        """Apply a deployment state (mock implementation)"""
        logger.info(f"Applying deployment state to {environment}")
        # In production, this would apply the actual deployment
        return True
    
    def auto_rollback(self, environment: str, current_state: Dict[str, Any]):
        """Context manager for automatic rollback on failure"""
        class AutoRollback:
            def __init__(self, manager, env, state):
                self.manager = manager
                self.environment = env
                self.state = state
                self.backup_id = None
            
            def __enter__(self):
                self.backup_id = self.manager.create_backup(self.environment, self.state)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    logger.error(f"Deployment failed, rolling back to {self.backup_id}")
                    self.manager.rollback(self.environment, self.backup_id)
        
        return AutoRollback(self, environment, current_state)


class BaseDeploymentStrategy:
    """Base class for deployment strategies"""
    
    def deploy(self, environment: str, config: Dict[str, Any]) -> bool:
        """Deploy using this strategy"""
        raise NotImplementedError


class DirectDeployment(BaseDeploymentStrategy):
    """Direct deployment (no special strategy)"""
    
    def deploy(self, environment: str, config: Dict[str, Any]) -> bool:
        """Direct deployment"""
        logger.info(f"Performing direct deployment to {environment}")
        # In production, this would perform the actual deployment
        return True


class RollingUpdateDeployment(BaseDeploymentStrategy):
    """Rolling update deployment strategy"""
    
    def __init__(self, batch_size: int = 2, wait_between_batches: int = 10):
        self.batch_size = batch_size
        self.wait_between_batches = wait_between_batches
    
    def deploy(self, services: List[str]) -> bool:
        """Deploy services in rolling batches"""
        logger.info(f"Starting rolling update with batch size {self.batch_size}")
        
        for i in range(0, len(services), self.batch_size):
            batch = services[i:i + self.batch_size]
            logger.info(f"Updating batch: {batch}")
            
            for service in batch:
                if not self.update_service(service):
                    return False
            
            if i + self.batch_size < len(services):
                logger.info(f"Waiting {self.wait_between_batches}s before next batch")
                time.sleep(self.wait_between_batches)
        
        return True
    
    def update_service(self, service: str) -> bool:
        """Update a single service"""
        logger.info(f"Updating service: {service}")
        # In production, this would update the actual service
        return True


class BlueGreenDeployment(BaseDeploymentStrategy):
    """Blue-green deployment strategy"""
    
    def deploy(self, environment: str, config: Dict[str, Any]) -> bool:
        """Perform blue-green deployment"""
        logger.info(f"Starting blue-green deployment to {environment}")
        
        # Create green environment
        green_env = self.create_green_environment(environment, config)
        if not green_env:
            return False
        
        # Test green environment
        if not self.test_green_environment(green_env):
            self.cleanup_green_environment(green_env)
            return False
        
        # Switch traffic
        if not self.switch_traffic(environment, green_env):
            self.cleanup_green_environment(green_env)
            return False
        
        # Clean up old blue environment
        self.cleanup_blue_environment(environment)
        
        return True
    
    def create_green_environment(self, environment: str, config: Dict[str, Any]) -> str:
        """Create the green environment"""
        logger.info("Creating green environment")
        return f"green-{environment}-{int(time.time())}"
    
    def test_green_environment(self, green_env: str) -> bool:
        """Test the green environment"""
        logger.info(f"Testing green environment: {green_env}")
        return True
    
    def switch_traffic(self, environment: str, green_env: str) -> bool:
        """Switch traffic from blue to green"""
        logger.info(f"Switching traffic to {green_env}")
        return True
    
    def cleanup_blue_environment(self, environment: str):
        """Clean up the old blue environment"""
        logger.info(f"Cleaning up blue environment for {environment}")
    
    def cleanup_green_environment(self, green_env: str):
        """Clean up failed green environment"""
        logger.info(f"Cleaning up failed green environment: {green_env}")


class CanaryDeployment(BaseDeploymentStrategy):
    """Canary deployment strategy"""
    
    def __init__(self, initial_percentage: int = 10, increment: int = 20, 
                 analysis_duration: int = 60):
        self.initial_percentage = initial_percentage
        self.increment = increment
        self.analysis_duration = analysis_duration
    
    def deploy(self, environment: str, config: Dict[str, Any]) -> bool:
        """Perform canary deployment"""
        logger.info(f"Starting canary deployment to {environment}")
        
        # Deploy canary
        canary_id = self.deploy_canary(environment, config)
        if not canary_id:
            return False
        
        # Progressive rollout
        current_percentage = self.initial_percentage
        
        while current_percentage <= 100:
            logger.info(f"Routing {current_percentage}% traffic to canary")
            
            if not self.adjust_traffic(canary_id, current_percentage):
                self.rollback_canary(canary_id)
                return False
            
            if current_percentage < 100:
                # Analyze metrics
                time.sleep(self.analysis_duration)
                metrics = self.analyze_metrics(canary_id)
                
                if not self.metrics_acceptable(metrics):
                    logger.error("Canary metrics not acceptable, rolling back")
                    self.rollback_canary(canary_id)
                    return False
            
            current_percentage = min(100, current_percentage + self.increment)
        
        # Finalize deployment
        self.finalize_canary(canary_id)
        return True
    
    def deploy_canary(self, environment: str, config: Dict[str, Any]) -> str:
        """Deploy the canary version"""
        logger.info("Deploying canary")
        return f"canary-{int(time.time())}"
    
    def adjust_traffic(self, canary_id: str, percentage: int) -> bool:
        """Adjust traffic percentage to canary"""
        return True
    
    def analyze_metrics(self, canary_id: str) -> Dict[str, float]:
        """Analyze canary metrics"""
        return {'error_rate': 0.01, 'latency': 100}
    
    def metrics_acceptable(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics are acceptable"""
        return metrics.get('error_rate', 1.0) < 0.05
    
    def rollback_canary(self, canary_id: str):
        """Rollback canary deployment"""
        logger.info(f"Rolling back canary: {canary_id}")
    
    def finalize_canary(self, canary_id: str):
        """Finalize canary deployment"""
        logger.info(f"Finalizing canary: {canary_id}")


class DeploymentConfig:
    """Manages deployment configuration"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.environments: Dict[str, Environment] = {}
        self.deployment_strategies: Dict[str, DeploymentStrategy] = {}
        
        self._parse_config()
    
    @classmethod
    def from_file(cls, path: str) -> 'DeploymentConfig':
        """Load configuration from file"""
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        return cls(config)
    
    def _parse_config(self):
        """Parse configuration into objects"""
        # Parse environments
        for name, env_config in self.config.get('environments', {}).items():
            self.environments[name] = Environment(
                name=name,
                api_url=env_config.get('api_url', ''),
                airflow_url=env_config.get('airflow_url', ''),
                indexagent_url=env_config.get('indexagent_url', ''),
                config_overrides=env_config.get('config_overrides', {})
            )
        
        # Parse deployment strategies
        for env, strategy in self.config.get('deployment_strategies', {}).items():
            try:
                self.deployment_strategies[env] = DeploymentStrategy(strategy)
            except ValueError:
                raise ValueError(f"Invalid deployment strategy '{strategy}' for {env}")
    
    def validate(self) -> bool:
        """Validate configuration"""
        for env in self.environments.values():
            env.validate()
        return True
    
    def get_environment(self, name: str) -> Environment:
        """Get environment configuration"""
        if name not in self.environments:
            raise ValueError(f"Environment {name} not found")
        return self.environments[name]


class DeploymentManager:
    """Manages deployments across environments"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.strategies = {
            DeploymentStrategy.DIRECT: DirectDeployment(),
            DeploymentStrategy.ROLLING_UPDATE: RollingUpdateDeployment(),
            DeploymentStrategy.BLUE_GREEN: BlueGreenDeployment(),
            DeploymentStrategy.CANARY: CanaryDeployment()
        }
    
    def get_strategy(self, environment: str) -> BaseDeploymentStrategy:
        """Get deployment strategy for environment"""
        strategy_type = self.config.deployment_strategies.get(
            environment,
            DeploymentStrategy.DIRECT
        )
        return self.strategies[strategy_type]
    
    async def deploy_to_environment(self, environment: str, version: str) -> DeploymentResult:
        """Deploy to a specific environment"""
        # Mock implementation
        return DeploymentResult(
            success=True,
            environment=environment,
            version=version
        )
    
    async def deploy_with_promotion(self, environment: str, version: str) -> DeploymentResult:
        """Deploy with promotion workflow"""
        return await self.deploy_to_environment(environment, version)
    
    async def promote(self, from_env: str, to_env: str) -> DeploymentResult:
        """Promote deployment from one environment to another"""
        # Check promotion gates
        if not await self.run_promotion_checks(from_env, to_env):
            return DeploymentResult(
                success=False,
                environment=to_env,
                version='',
                error='manual approval required'
            )
        
        # Get version from source environment
        # In production, this would fetch actual deployed version
        version = '1.0.0'
        
        return await self.deploy_to_environment(to_env, version)
    
    async def run_promotion_checks(self, from_env: str, to_env: str) -> bool:
        """Run promotion gate checks"""
        # Mock implementation
        return from_env == 'dev' and to_env == 'staging'


class ConfigurationManager:
    """Manages configuration across environments"""
    
    def diff_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """Generate diff between configurations"""
        differences = {}
        
        def _diff_recursive(d1: Dict, d2: Dict, prefix: str = ''):
            all_keys = set(d1.keys()) | set(d2.keys())
            
            for key in all_keys:
                current_path = f"{prefix}.{key}" if prefix else key
                val1 = d1.get(key)
                val2 = d2.get(key)
                
                if isinstance(val1, dict) and isinstance(val2, dict):
                    _diff_recursive(val1, val2, current_path)
                elif val1 != val2:
                    differences[current_path] = (val1, val2)
        
        _diff_recursive(config1, config2)
        return differences


class ConfigurationValidator:
    """Validates configurations against environment rules"""
    
    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules
    
    def validate(self, environment: str, config: Dict[str, Any]) -> bool:
        """Validate configuration for environment"""
        if environment not in self.rules:
            return True
        
        env_rules = self.rules[environment]
        
        # Check required fields
        for field in env_rules.get('required_fields', []):
            if not self._has_field(config, field):
                raise ValueError(f"Required field {field} missing")
        
        # Check constraints
        for field, constraint in env_rules.get('constraints', {}).items():
            value = self._get_field(config, field)
            
            if 'min' in constraint and value < constraint['min']:
                raise ValueError(f"{field} value {value} below minimum {constraint['min']}")
            
            if 'value' in constraint and value != constraint['value']:
                raise ValueError(f"{field} must be {constraint['value']}, got {value}")
            
            if 'pattern' in constraint:
                import re
                if not re.match(constraint['pattern'], str(value)):
                    raise ValueError(f"{field} value {value} doesn't match pattern")
        
        return True
    
    def _has_field(self, config: Dict, field_path: str) -> bool:
        """Check if field exists in config"""
        parts = field_path.split('.')
        current = config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        
        return True
    
    def _get_field(self, config: Dict, field_path: str) -> Any:
        """Get field value from config"""
        parts = field_path.split('.')
        current = config
        
        for part in parts:
            current = current[part]
        
        return current


class SecretManager:
    """Manages secrets during deployment"""
    
    def __init__(self, vault_url: str):
        self.vault_url = vault_url
    
    def get_secret(self, path: str) -> str:
        """Get secret from vault"""
        # Mock implementation
        return f"secret-{path}"
    
    def create_secret(self, path: str, value: str):
        """Create or update secret"""
        logger.info(f"Creating/updating secret at {path}")
    
    def rotate_secret(self, path: str, new_value: str):
        """Rotate a secret"""
        old_value = self.get_secret(path)
        self.create_secret(path, new_value)
        logger.info(f"Rotated secret at {path}")


class DeploymentOrchestrator:
    """Orchestrates the deployment process"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.rollback_manager = RollbackManager()
        self.health_checker = HealthChecker()
        self.deployment_manager = DeploymentManager(config)
    
    async def deploy(self, environment: str, version: str, 
                    dry_run: bool = False) -> DeploymentResult:
        """Orchestrate deployment to environment"""
        logger.info(f"Starting deployment of {version} to {environment}")
        
        # Pre-deployment checks
        if not await self.pre_deployment_checks(environment):
            return DeploymentResult(
                success=False,
                environment=environment,
                version=version,
                error="pre-deployment checks failed"
            )
        
        if dry_run:
            # Dry run mode
            changes = await self.simulate_deployment(environment, version)
            return DeploymentResult(
                success=True,
                environment=environment,
                version=version,
                dry_run=True,
                planned_changes=changes['changes'],
                estimated_duration=changes['estimated_duration']
            )
        
        # Create backup
        current_state = await self.get_current_state(environment)
        backup_id = self.create_backup(environment, current_state)
        
        try:
            # Execute deployment
            success = await self.execute_deployment(environment, version)
            
            if not success:
                raise Exception("Deployment execution failed")
            
            # Post-deployment validation
            if not await self.post_deployment_validation(environment):
                raise Exception("Post-deployment validation failed")
            
            return DeploymentResult(
                success=True,
                environment=environment,
                version=version,
                backup_id=backup_id
            )
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            
            # Attempt rollback
            rolled_back = await self.rollback(environment, backup_id)
            
            return DeploymentResult(
                success=False,
                environment=environment,
                version=version,
                backup_id=backup_id,
                error=str(e),
                rolled_back=rolled_back
            )
    
    async def pre_deployment_checks(self, environment: str) -> bool:
        """Run pre-deployment checks"""
        logger.info("Running pre-deployment checks")
        
        # Check environment health
        env_config = self.config.get_environment(environment)
        health_result = await self.health_checker.check_environment(env_config)
        
        if not health_result.is_healthy:
            logger.error(f"Environment unhealthy: {health_result.unhealthy_services}")
            return False
        
        return True
    
    async def get_current_state(self, environment: str) -> Dict[str, Any]:
        """Get current deployment state"""
        # Mock implementation
        return {
            'version': '1.2.0',
            'timestamp': datetime.utcnow().isoformat(),
            'config': {}
        }
    
    def create_backup(self, environment: str, state: Dict[str, Any]) -> str:
        """Create deployment backup"""
        return self.rollback_manager.create_backup(environment, state)
    
    async def execute_deployment(self, environment: str, version: str) -> bool:
        """Execute the actual deployment"""
        strategy = self.deployment_manager.get_strategy(environment)
        env_config = self.config.get_environment(environment)
        
        return strategy.deploy(environment, {'version': version})
    
    async def post_deployment_validation(self, environment: str) -> bool:
        """Validate deployment after execution"""
        logger.info("Running post-deployment validation")
        
        # Check health again
        env_config = self.config.get_environment(environment)
        health_result = await self.health_checker.check_environment(env_config)
        
        return health_result.is_healthy
    
    async def rollback(self, environment: str, backup_id: str) -> bool:
        """Rollback deployment"""
        return self.rollback_manager.rollback(environment, backup_id)
    
    async def simulate_deployment(self, environment: str, version: str) -> Dict[str, Any]:
        """Simulate deployment and return planned changes"""
        # Mock implementation
        return {
            'changes': [
                f'Update API version to {version}',
                'Scale replicas from 2 to 3',
                'Update token budget to 1500000'
            ],
            'estimated_duration': 300
        }


def main():
    """Main entry point for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DEAN Advanced Deployment')
    parser.add_argument('command', choices=['deploy', 'rollback', 'status'])
    parser.add_argument('environment', help='Target environment')
    parser.add_argument('--version', help='Version to deploy')
    parser.add_argument('--config', default='deployment-config.yaml', help='Config file')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DeploymentConfig.from_file(args.config)
    orchestrator = DeploymentOrchestrator(config)
    
    # Execute command
    if args.command == 'deploy':
        if not args.version:
            print("Version required for deployment")
            return 1
        
        result = asyncio.run(orchestrator.deploy(
            args.environment,
            args.version,
            dry_run=args.dry_run
        ))
        
        if result.success:
            print(f"Deployment successful: {result.version} to {result.environment}")
            if result.dry_run:
                print("Planned changes:")
                for change in result.planned_changes:
                    print(f"  - {change}")
        else:
            print(f"Deployment failed: {result.error}")
            return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())