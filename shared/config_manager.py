#!/usr/bin/env python3
"""
DEAN System Configuration Manager
Centralized configuration management for all services
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
import json

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration across DEAN system services"""
    
    # Default configuration paths
    DEFAULT_PATHS = {
        'production': '.env.production',
        'development': '.env.development',
        'test': '.env.test'
    }
    
    # Required environment variables by service
    REQUIRED_VARS = {
        'github_integration': [
            'GITHUB_TOKEN',
            'GITHUB_API_URL',
            'ENABLE_GITHUB_INTEGRATION'
        ],
        'claude_integration': [
            'CLAUDE_API_KEY',
            'CLAUDE_CODE_CLI_PATH'
        ],
        'database': [
            'DATABASE_URL'
        ],
        'services': [
            'EVOLUTION_API_PORT',
            'INDEXAGENT_API_PORT',
            'REDIS_URL'
        ],
        'security': [
            'DRY_RUN_MODE',
            'ALLOWED_REPOSITORIES',
            'RESTRICTED_PATHS'
        ]
    }
    
    def __init__(self, environment: str = 'production'):
        """Initialize configuration manager"""
        self.environment = environment
        self.config = {}
        self.loaded_files = []
        self._load_configuration()
    
    def _find_env_file(self, filename: str) -> Optional[Path]:
        """Find environment file in standard locations"""
        # Search paths in order of priority
        search_paths = [
            Path.cwd() / filename,
            Path.cwd().parent / filename,
            Path.cwd().parent.parent / filename,
            Path.home() / '.dean' / filename,
            Path('/etc/dean') / filename
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def _load_configuration(self):
        """Load configuration from environment files"""
        env_filename = self.DEFAULT_PATHS.get(self.environment, '.env')
        
        # Find and load main environment file
        env_path = self._find_env_file(env_filename)
        if env_path:
            load_dotenv(env_path)
            self.loaded_files.append(str(env_path))
            logger.info(f"Loaded environment from {env_path}")
        else:
            logger.warning(f"Environment file {env_filename} not found")
        
        # Load any additional .env files
        for additional in ['.env.local', '.env.secrets']:
            path = self._find_env_file(additional)
            if path:
                load_dotenv(path, override=True)
                self.loaded_files.append(str(path))
                logger.info(f"Loaded additional config from {path}")
        
        # Capture all environment variables
        self.config = dict(os.environ)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value"""
        value = self.get(key, str(default))
        return str(value).lower() in ('true', 'yes', '1', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        try:
            return int(self.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def get_list(self, key: str, separator: str = ',') -> List[str]:
        """Get list configuration value"""
        value = self.get(key, '')
        if not value:
            return []
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def validate_service(self, service_name: str) -> Dict[str, Any]:
        """Validate configuration for a specific service"""
        results = {
            'service': service_name,
            'valid': True,
            'missing': [],
            'warnings': []
        }
        
        if service_name not in self.REQUIRED_VARS:
            results['warnings'].append(f"No validation rules for service {service_name}")
            return results
        
        required = self.REQUIRED_VARS[service_name]
        for var in required:
            value = self.get(var)
            if not value or value == f'your_{var.lower()}_here':
                results['missing'].append(var)
                results['valid'] = False
        
        return results
    
    def validate_all(self) -> Dict[str, Any]:
        """Validate all service configurations"""
        results = {
            'environment': self.environment,
            'loaded_files': self.loaded_files,
            'services': {},
            'overall_valid': True
        }
        
        for service in self.REQUIRED_VARS:
            service_result = self.validate_service(service)
            results['services'][service] = service_result
            if not service_result['valid']:
                results['overall_valid'] = False
        
        return results
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get all configuration for a specific service"""
        config = {}
        
        # Get service-specific prefixes
        prefixes = {
            'github': ['GITHUB_', 'GIT_'],
            'claude': ['CLAUDE_', 'ANTHROPIC_'],
            'database': ['DATABASE_', 'DB_', 'POSTGRES_'],
            'redis': ['REDIS_'],
            'evolution_api': ['EVOLUTION_', 'DEAN_']
        }
        
        service_prefixes = prefixes.get(service_name, [service_name.upper() + '_'])
        
        for key, value in self.config.items():
            for prefix in service_prefixes:
                if key.startswith(prefix):
                    config[key] = value
                    break
        
        return config
    
    def export_config(self, output_path: str, service: Optional[str] = None):
        """Export configuration to file"""
        config_to_export = self.get_service_config(service) if service else self.config
        
        # Filter sensitive values
        safe_config = {}
        sensitive_patterns = ['KEY', 'TOKEN', 'PASSWORD', 'SECRET']
        
        for key, value in config_to_export.items():
            is_sensitive = any(pattern in key.upper() for pattern in sensitive_patterns)
            if is_sensitive:
                safe_config[key] = f"{value[:10]}..." if len(value) > 10 else "***"
            else:
                safe_config[key] = value
        
        with open(output_path, 'w') as f:
            json.dump({
                'environment': self.environment,
                'exported_at': str(Path.cwd()),
                'config': safe_config
            }, f, indent=2)
    
    def get_connection_string(self, service: str) -> Optional[str]:
        """Get connection string for a service"""
        if service == 'postgres':
            return self.get('DATABASE_URL')
        elif service == 'redis':
            return self.get('REDIS_URL')
        elif service == 'vault':
            return self.get('VAULT_URL')
        return None
    
    def get_api_credentials(self) -> Dict[str, Dict[str, Any]]:
        """Get API credentials status"""
        return {
            'github': {
                'configured': bool(self.get('GITHUB_TOKEN')),
                'token_prefix': self.get('GITHUB_TOKEN', '')[:10] + '...' if self.get('GITHUB_TOKEN') else None,
                'integration_enabled': self.get_bool('ENABLE_GITHUB_INTEGRATION')
            },
            'claude': {
                'configured': bool(self.get('CLAUDE_API_KEY')),
                'key_prefix': self.get('CLAUDE_API_KEY', '')[:20] + '...' if self.get('CLAUDE_API_KEY') else None,
                'cli_path': self.get('CLAUDE_CODE_CLI_PATH')
            }
        }

# Singleton instance
_config_manager = None

def get_config_manager(environment: Optional[str] = None) -> ConfigManager:
    """Get the singleton ConfigManager instance"""
    global _config_manager
    if _config_manager is None:
        env = environment or os.getenv('DEAN_ENVIRONMENT', 'production')
        _config_manager = ConfigManager(env)
    return _config_manager

# Convenience functions
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return get_config_manager().get(key, default)

def get_bool_config(key: str, default: bool = False) -> bool:
    """Get boolean configuration value"""
    return get_config_manager().get_bool(key, default)

def validate_config() -> Dict[str, Any]:
    """Validate all configuration"""
    return get_config_manager().validate_all()