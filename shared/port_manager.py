#!/usr/bin/env python3
"""
DEAN System Port Manager
Centralized port allocation and conflict prevention for all services
"""

import socket
import logging
from typing import Dict, Optional, List
from pathlib import Path
import json
import fcntl
import os

logger = logging.getLogger(__name__)

class PortManager:
    """Manages port allocation across DEAN system services"""
    
    # Default port assignments per DEAN specifications
    DEFAULT_PORTS = {
        'airflow': 8080,
        'indexagent': 8081,
        'evolution_api': 8090,
        'market_analysis': 8000,
        'zoekt': 6070,
        'postgres': 5432,
        'redis': 6379,
        'vault': 8200,
        'prometheus': 9090,
        'grafana': 3000
    }
    
    # Port ranges for dynamic allocation
    DYNAMIC_RANGES = {
        'test_services': (9000, 9099),
        'agent_workspaces': (9100, 9199),
        'monitoring': (9200, 9299),
        'development': (9300, 9399)
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize port manager with optional config file"""
        self.config_path = config_path or Path.home() / '.dean' / 'port_config.json'
        self.allocated_ports = {}
        self._lock_file = None
        self._load_config()
    
    def _load_config(self):
        """Load port configuration from file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.allocated_ports = saved_config.get('allocated_ports', {})
                    logger.info(f"Loaded port configuration from {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not load port config: {e}")
            self.allocated_ports = {}
    
    def _save_config(self):
        """Save port configuration to file"""
        try:
            config_dir = Path(self.config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump({
                    'allocated_ports': self.allocated_ports,
                    'version': '1.0'
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save port config: {e}")
    
    def is_port_available(self, port: int, host: str = '0.0.0.0') -> bool:
        """Check if a port is available for binding"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                return True
        except OSError:
            return False
    
    def find_available_port(self, start_port: int, end_port: int, host: str = '0.0.0.0') -> Optional[int]:
        """Find an available port in the given range"""
        for port in range(start_port, end_port + 1):
            if self.is_port_available(port, host):
                return port
        return None
    
    def allocate_port(self, service_name: str, preferred_port: Optional[int] = None, 
                     category: str = 'default') -> Optional[int]:
        """Allocate a port for a service"""
        # Check if service already has an allocated port
        if service_name in self.allocated_ports:
            allocated = self.allocated_ports[service_name]
            if self.is_port_available(allocated):
                logger.info(f"Reusing allocated port {allocated} for {service_name}")
                return allocated
            else:
                logger.warning(f"Previously allocated port {allocated} for {service_name} is in use")
        
        # Try preferred port first
        if preferred_port and self.is_port_available(preferred_port):
            self.allocated_ports[service_name] = preferred_port
            self._save_config()
            logger.info(f"Allocated preferred port {preferred_port} for {service_name}")
            return preferred_port
        
        # Try default port for known services
        if service_name in self.DEFAULT_PORTS:
            default_port = self.DEFAULT_PORTS[service_name]
            if self.is_port_available(default_port):
                self.allocated_ports[service_name] = default_port
                self._save_config()
                logger.info(f"Allocated default port {default_port} for {service_name}")
                return default_port
        
        # Find port in category range
        if category in self.DYNAMIC_RANGES:
            start, end = self.DYNAMIC_RANGES[category]
            port = self.find_available_port(start, end)
            if port:
                self.allocated_ports[service_name] = port
                self._save_config()
                logger.info(f"Allocated dynamic port {port} for {service_name} in category {category}")
                return port
        
        # Last resort: find any available port
        port = self.find_available_port(9000, 9999)
        if port:
            self.allocated_ports[service_name] = port
            self._save_config()
            logger.info(f"Allocated fallback port {port} for {service_name}")
            return port
        
        logger.error(f"Could not allocate port for {service_name}")
        return None
    
    def release_port(self, service_name: str):
        """Release a port allocation"""
        if service_name in self.allocated_ports:
            port = self.allocated_ports[service_name]
            del self.allocated_ports[service_name]
            self._save_config()
            logger.info(f"Released port {port} for {service_name}")
    
    def get_service_port(self, service_name: str) -> Optional[int]:
        """Get the allocated port for a service"""
        return self.allocated_ports.get(service_name) or self.DEFAULT_PORTS.get(service_name)
    
    def list_allocations(self) -> Dict[str, int]:
        """List all current port allocations"""
        return {**self.DEFAULT_PORTS, **self.allocated_ports}
    
    def check_conflicts(self) -> List[Dict[str, any]]:
        """Check for port conflicts across all services"""
        conflicts = []
        all_ports = self.list_allocations()
        
        for service, port in all_ports.items():
            if not self.is_port_available(port):
                # Check what's using the port
                try:
                    import subprocess
                    result = subprocess.run(
                        ['lsof', '-i', f':{port}'], 
                        capture_output=True, 
                        text=True
                    )
                    if result.returncode == 0:
                        conflicts.append({
                            'service': service,
                            'port': port,
                            'conflict': result.stdout.strip()
                        })
                except:
                    conflicts.append({
                        'service': service,
                        'port': port,
                        'conflict': 'Port in use (details unavailable)'
                    })
        
        return conflicts
    
    def suggest_alternative(self, service_name: str, current_port: int) -> Optional[int]:
        """Suggest an alternative port for a service"""
        # Try ports near the current one
        for offset in range(1, 10):
            alt_port = current_port + offset
            if self.is_port_available(alt_port):
                return alt_port
        
        # Try category-based allocation
        category = 'default'
        if 'test' in service_name.lower():
            category = 'test_services'
        elif 'agent' in service_name.lower():
            category = 'agent_workspaces'
        
        return self.allocate_port(f"{service_name}_alt", category=category)

# Singleton instance
_port_manager = None

def get_port_manager(config_path: Optional[str] = None) -> PortManager:
    """Get the singleton PortManager instance"""
    global _port_manager
    if _port_manager is None:
        _port_manager = PortManager(config_path)
    return _port_manager

# Convenience functions
def allocate_port(service_name: str, preferred_port: Optional[int] = None) -> Optional[int]:
    """Allocate a port for a service"""
    return get_port_manager().allocate_port(service_name, preferred_port)

def get_service_port(service_name: str) -> Optional[int]:
    """Get the port for a service"""
    return get_port_manager().get_service_port(service_name)

def check_port_available(port: int) -> bool:
    """Check if a port is available"""
    return get_port_manager().is_port_available(port)