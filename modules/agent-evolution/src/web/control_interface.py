#!/usr/bin/env python3
"""
DEAN Interactive Control Interface
Provides web-based controls for managing evolution runs
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import aiohttp
from aiohttp import web
import websockets
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Manages WebSocket connections and real-time updates"""
    
    def __init__(self):
        self.clients: Dict[str, web.WebSocketResponse] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def register_client(self, websocket: web.WebSocketResponse) -> str:
        """Register a new WebSocket client"""
        client_id = str(uuid.uuid4())
        async with self._lock:
            self.clients[client_id] = websocket
        logger.info(f"Registered WebSocket client: {client_id}")
        return client_id
    
    async def unregister_client(self, client_id: str):
        """Remove a disconnected client"""
        async with self._lock:
            if client_id in self.clients:
                del self.clients[client_id]
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]
        logger.info(f"Unregistered WebSocket client: {client_id}")
    
    async def subscribe_client(self, client_id: str, channels: List[str]):
        """Subscribe a client to specific channels"""
        async with self._lock:
            self.subscriptions[client_id].update(channels)
        logger.info(f"Client {client_id} subscribed to: {channels}")
    
    def get_client_subscriptions(self, client_id: str) -> Set[str]:
        """Get channels a client is subscribed to"""
        return self.subscriptions.get(client_id, set())
    
    def has_client(self, client_id: str) -> bool:
        """Check if a client is registered"""
        return client_id in self.clients
    
    async def broadcast_update(self, channel: str, data: Dict[str, Any]):
        """Broadcast an update to all clients subscribed to a channel"""
        message = json.dumps(data)
        
        # Find all clients subscribed to this channel
        clients_to_notify = []
        async with self._lock:
            for client_id, channels in self.subscriptions.items():
                if channel in channels and client_id in self.clients:
                    clients_to_notify.append((client_id, self.clients[client_id]))
        
        # Send updates outside the lock
        for client_id, websocket in clients_to_notify:
            try:
                await websocket.send_str(message)
            except Exception as e:
                logger.error(f"Failed to send to client {client_id}: {e}")
                await self.unregister_client(client_id)


class EvolutionController:
    """Manages evolution runs through the Agent Evolution API"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def start_evolution(self, generations: int, agents_per_generation: int, 
                            initial_strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        """Start a new evolution run"""
        await self._ensure_session()
        
        payload = {
            'generations': generations,
            'agent_count': agents_per_generation,
            'initial_strategies': initial_strategies or ['optimization', 'refactoring']
        }
        
        async with self.session.post(f"{self.api_url}/evolution/start", json=payload) as resp:
            return await resp.json()
    
    async def pause_evolution(self, trial_id: str) -> Dict[str, Any]:
        """Pause an evolution run"""
        await self._ensure_session()
        
        async with self.session.post(f"{self.api_url}/evolution/{trial_id}/pause") as resp:
            return await resp.json()
    
    async def resume_evolution(self, trial_id: str) -> Dict[str, Any]:
        """Resume a paused evolution run"""
        await self._ensure_session()
        
        async with self.session.post(f"{self.api_url}/evolution/{trial_id}/resume") as resp:
            return await resp.json()
    
    async def stop_evolution(self, trial_id: str) -> Dict[str, Any]:
        """Stop an evolution run"""
        await self._ensure_session()
        
        async with self.session.post(f"{self.api_url}/evolution/{trial_id}/stop") as resp:
            return await resp.json()
    
    async def get_evolution_status(self, trial_id: str) -> Dict[str, Any]:
        """Get current status of an evolution run"""
        await self._ensure_session()
        
        async with self.session.get(f"{self.api_url}/evolution/{trial_id}/status") as resp:
            return await resp.json()
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()


class PatternApprovalManager:
    """Manages pattern discovery approval workflow"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def get_pending_patterns(self) -> Dict[str, Any]:
        """Get patterns awaiting approval"""
        await self._ensure_session()
        
        async with self.session.get(f"{self.api_url}/patterns/pending") as resp:
            return await resp.json()
    
    async def approve_pattern(self, pattern_id: str, auto_propagate: bool = True, 
                            notes: str = "") -> Dict[str, Any]:
        """Approve a discovered pattern"""
        await self._ensure_session()
        
        payload = {
            'action': 'approve',
            'auto_propagate': auto_propagate,
            'notes': notes
        }
        
        async with self.session.post(f"{self.api_url}/patterns/{pattern_id}/review", 
                                     json=payload) as resp:
            return await resp.json()
    
    async def reject_pattern(self, pattern_id: str, reason: str, 
                           notes: str = "") -> Dict[str, Any]:
        """Reject a discovered pattern"""
        await self._ensure_session()
        
        payload = {
            'action': 'reject',
            'reason': reason,
            'notes': notes
        }
        
        async with self.session.post(f"{self.api_url}/patterns/{pattern_id}/review", 
                                     json=payload) as resp:
            return await resp.json()
    
    async def set_auto_approval_rules(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Configure automatic pattern approval rules"""
        await self._ensure_session()
        
        async with self.session.put(f"{self.api_url}/patterns/auto-approval", 
                                   json=rules) as resp:
            return await resp.json()
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()


class SystemParameterManager:
    """Manages system parameter adjustments"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def get_current_parameters(self) -> Dict[str, Any]:
        """Get current system parameters"""
        await self._ensure_session()
        
        async with self.session.get(f"{self.api_url}/parameters") as resp:
            return await resp.json()
    
    async def update_token_budget(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update token budget parameters"""
        await self._ensure_session()
        
        async with self.session.patch(f"{self.api_url}/parameters/token_budget", 
                                     json=updates) as resp:
            return await resp.json()
    
    async def update_diversity_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update diversity management settings"""
        await self._ensure_session()
        
        async with self.session.patch(f"{self.api_url}/parameters/diversity", 
                                     json=updates) as resp:
            return await resp.json()
    
    async def toggle_ca_rule(self, rule_id: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable a cellular automata rule"""
        await self._ensure_session()
        
        payload = {'enabled': enabled}
        async with self.session.patch(f"{self.api_url}/parameters/ca_rules/{rule_id}", 
                                     json=payload) as resp:
            return await resp.json()
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()


class RESTAPIHandler:
    """Handles REST API endpoints for the control interface"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.start_time = datetime.utcnow()
        self.evolution_controller = EvolutionController(api_url)
        self.pattern_manager = PatternApprovalManager(api_url)
        self.parameter_manager = SystemParameterManager(api_url)
    
    def create_app(self) -> web.Application:
        """Create the aiohttp application"""
        app = web.Application()
        
        # Add routes
        app.router.add_get('/health', self.health_check)
        
        # Evolution control
        app.router.add_post('/api/evolution/start', self.start_evolution)
        app.router.add_post('/api/evolution/{trial_id}/pause', self.pause_evolution)
        app.router.add_post('/api/evolution/{trial_id}/resume', self.resume_evolution)
        app.router.add_post('/api/evolution/{trial_id}/stop', self.stop_evolution)
        app.router.add_get('/api/evolution/{trial_id}/status', self.get_evolution_status)
        
        # Pattern approval
        app.router.add_get('/api/patterns/pending', self.get_pending_patterns)
        app.router.add_post('/api/patterns/{pattern_id}/approve', self.approve_pattern)
        app.router.add_post('/api/patterns/{pattern_id}/reject', self.reject_pattern)
        app.router.add_put('/api/patterns/auto-approval', self.set_auto_approval)
        
        # Parameter management
        app.router.add_get('/api/parameters', self.get_parameters)
        app.router.add_patch('/api/parameters/token_budget', self.update_token_budget)
        app.router.add_patch('/api/parameters/diversity', self.update_diversity)
        app.router.add_patch('/api/parameters/ca_rules/{rule_id}', self.toggle_ca_rule)
        
        # Proxy endpoint for direct API access
        app.router.add_get('/api/proxy/{path:.*}', self.proxy_request)
        
        return app
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        return web.json_response({
            'status': 'healthy',
            'uptime': uptime,
            'version': '1.0.0'
        })
    
    async def start_evolution(self, request: web.Request) -> web.Response:
        """Start a new evolution run"""
        data = await request.json()
        result = await self.evolution_controller.start_evolution(
            generations=data.get('generations', 10),
            agents_per_generation=data.get('agents', 5),
            initial_strategies=data.get('strategies')
        )
        return web.json_response(result, status=201)
    
    async def pause_evolution(self, request: web.Request) -> web.Response:
        """Pause an evolution run"""
        trial_id = request.match_info['trial_id']
        result = await self.evolution_controller.pause_evolution(trial_id)
        return web.json_response(result)
    
    async def resume_evolution(self, request: web.Request) -> web.Response:
        """Resume an evolution run"""
        trial_id = request.match_info['trial_id']
        result = await self.evolution_controller.resume_evolution(trial_id)
        return web.json_response(result)
    
    async def stop_evolution(self, request: web.Request) -> web.Response:
        """Stop an evolution run"""
        trial_id = request.match_info['trial_id']
        result = await self.evolution_controller.stop_evolution(trial_id)
        return web.json_response(result)
    
    async def get_evolution_status(self, request: web.Request) -> web.Response:
        """Get evolution run status"""
        trial_id = request.match_info['trial_id']
        result = await self.evolution_controller.get_evolution_status(trial_id)
        return web.json_response(result)
    
    async def get_pending_patterns(self, request: web.Request) -> web.Response:
        """Get patterns pending approval"""
        result = await self.pattern_manager.get_pending_patterns()
        return web.json_response(result)
    
    async def approve_pattern(self, request: web.Request) -> web.Response:
        """Approve a pattern"""
        pattern_id = request.match_info['pattern_id']
        data = await request.json()
        result = await self.pattern_manager.approve_pattern(
            pattern_id,
            auto_propagate=data.get('auto_propagate', True),
            notes=data.get('notes', '')
        )
        return web.json_response(result)
    
    async def reject_pattern(self, request: web.Request) -> web.Response:
        """Reject a pattern"""
        pattern_id = request.match_info['pattern_id']
        data = await request.json()
        result = await self.pattern_manager.reject_pattern(
            pattern_id,
            reason=data.get('reason', 'manual_rejection'),
            notes=data.get('notes', '')
        )
        return web.json_response(result)
    
    async def set_auto_approval(self, request: web.Request) -> web.Response:
        """Set auto-approval rules"""
        rules = await request.json()
        result = await self.pattern_manager.set_auto_approval_rules(rules)
        return web.json_response(result)
    
    async def get_parameters(self, request: web.Request) -> web.Response:
        """Get current system parameters"""
        result = await self.parameter_manager.get_current_parameters()
        return web.json_response(result)
    
    async def update_token_budget(self, request: web.Request) -> web.Response:
        """Update token budget"""
        updates = await request.json()
        result = await self.parameter_manager.update_token_budget(updates)
        return web.json_response(result)
    
    async def update_diversity(self, request: web.Request) -> web.Response:
        """Update diversity settings"""
        updates = await request.json()
        result = await self.parameter_manager.update_diversity_settings(updates)
        return web.json_response(result)
    
    async def toggle_ca_rule(self, request: web.Request) -> web.Response:
        """Toggle CA rule"""
        rule_id = request.match_info['rule_id']
        data = await request.json()
        result = await self.parameter_manager.toggle_ca_rule(
            rule_id,
            enabled=data.get('enabled', True)
        )
        return web.json_response(result)
    
    async def proxy_request(self, request: web.Request) -> web.Response:
        """Proxy requests to the Agent Evolution API"""
        path = request.match_info['path']
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/{path}") as resp:
                data = await resp.json()
                return web.json_response(data, status=resp.status)


class ControlInterface:
    """Main control interface coordinating all components"""
    
    def __init__(self, api_url: str = "http://localhost:8090", ws_port: int = 8091):
        self.api_url = api_url
        self.ws_port = ws_port
        self.is_running = False
        
        # Components
        self.websocket_handler = WebSocketHandler()
        self.evolution_controller = EvolutionController(api_url)
        self.pattern_manager = PatternApprovalManager(api_url)
        self.parameter_manager = SystemParameterManager(api_url)
        self.api_handler = RESTAPIHandler(api_url)
        
        # Web app
        self.app = None
        self.runner = None
        self.site = None
        
        # Background tasks
        self.update_task = None
    
    async def start(self):
        """Start the control interface server"""
        if self.is_running:
            return
        
        # Create web application
        self.app = self.api_handler.create_app()
        
        # Add WebSocket endpoint
        self.app.router.add_get('/ws', self.websocket_endpoint)
        
        # Add health check endpoint
        async def health_check(request):
            return web.json_response({
                'status': 'healthy',
                'service': 'agent-evolution-control',
                'version': '2.0.0',
                'websocket_port': self.ws_port,
                'is_running': self.is_running
            })
        
        self.app.router.add_get('/health', health_check)
        
        # Add static file serving for web UI
        static_path = Path(__file__).parent / 'static'
        if static_path.exists():
            self.app.router.add_static('/', static_path, name='static')
        
        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '0.0.0.0', self.ws_port)
        await self.site.start()
        
        # Start background update task
        self.update_task = asyncio.create_task(self._background_updates())
        
        self.is_running = True
        logger.info(f"Control interface started on port {self.ws_port}")
    
    async def stop(self):
        """Stop the control interface server"""
        if not self.is_running:
            return
        
        # Cancel background task
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Close components
        await self.evolution_controller.close()
        await self.pattern_manager.close()
        await self.parameter_manager.close()
        
        # Stop server
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        self.is_running = False
        logger.info("Control interface stopped")
    
    async def websocket_endpoint(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Register client
        client_id = await self.websocket_handler.register_client(ws)
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data['type'] == 'subscribe':
                        channels = data.get('channels', [])
                        await self.websocket_handler.subscribe_client(client_id, channels)
                        
                        # Send confirmation
                        await ws.send_str(json.dumps({
                            'type': 'subscription_confirmed',
                            'channels': channels
                        }))
                    
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            await self.websocket_handler.unregister_client(client_id)
            
        return ws
    
    async def _background_updates(self):
        """Send periodic updates to WebSocket clients"""
        while True:
            try:
                # Get current evolution status
                # In production, this would poll the actual API
                await asyncio.sleep(5)
                
                # Example update broadcast
                await self.websocket_handler.broadcast_update('evolution', {
                    'type': 'status_update',
                    'timestamp': datetime.utcnow().isoformat(),
                    'data': {
                        'active_trials': 1,
                        'system_health': 'good'
                    }
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background update error: {e}")


async def main():
    """Main entry point for standalone operation"""
    interface = ControlInterface()
    
    try:
        await interface.start()
        logger.info("Control interface is running. Press Ctrl+C to stop.")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await interface.stop()


if __name__ == "__main__":
    asyncio.run(main())