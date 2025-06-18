#!/usr/bin/env python3
"""
Test suite for DEAN Interactive Control Interface
Tests all functionality before implementation (TDD approach)
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import websockets
import aiohttp

# Import the interface we're about to build
from infra.modules.agent_evolution.src.web.control_interface import (
    ControlInterface,
    EvolutionController,
    PatternApprovalManager,
    SystemParameterManager,
    WebSocketHandler,
    RESTAPIHandler
)


class TestControlInterface:
    """Test suite for the main control interface"""
    
    @pytest.fixture
    def interface(self):
        """Create a control interface instance for testing"""
        return ControlInterface(
            api_url="http://localhost:8090",
            ws_port=8091
        )
    
    def test_initialization(self, interface):
        """Test interface initializes with correct configuration"""
        assert interface.api_url == "http://localhost:8090"
        assert interface.ws_port == 8091
        assert interface.is_running is False
        
    @pytest.mark.asyncio
    async def test_start_stop_server(self, interface):
        """Test starting and stopping the web server"""
        # Start server
        await interface.start()
        assert interface.is_running is True
        
        # Verify server is accessible
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8091/health') as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data['status'] == 'healthy'
        
        # Stop server
        await interface.stop()
        assert interface.is_running is False
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, interface):
        """Test WebSocket connection for real-time updates"""
        await interface.start()
        
        try:
            # Connect via WebSocket
            async with websockets.connect('ws://localhost:8091/ws') as websocket:
                # Send subscription message
                await websocket.send(json.dumps({
                    'type': 'subscribe',
                    'channels': ['evolution', 'agents', 'patterns']
                }))
                
                # Receive confirmation
                response = await websocket.recv()
                data = json.loads(response)
                assert data['type'] == 'subscription_confirmed'
                assert set(data['channels']) == {'evolution', 'agents', 'patterns'}
        finally:
            await interface.stop()


class TestEvolutionController:
    """Test suite for evolution run management"""
    
    @pytest.fixture
    def controller(self):
        """Create an evolution controller instance"""
        return EvolutionController(api_url="http://localhost:8090")
    
    @pytest.mark.asyncio
    async def test_start_evolution_run(self, controller):
        """Test starting a new evolution run"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = MagicMock()
            mock_resp.status = 201
            mock_resp.json = asyncio.coroutine(lambda: {
                'trial_id': 'trial_001',
                'status': 'started',
                'generation': 0,
                'agent_count': 5
            })
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            result = await controller.start_evolution(
                generations=10,
                agents_per_generation=5,
                initial_strategies=['optimization', 'refactoring']
            )
            
            assert result['trial_id'] == 'trial_001'
            assert result['status'] == 'started'
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pause_resume_evolution(self, controller):
        """Test pausing and resuming evolution runs"""
        trial_id = 'trial_001'
        
        # Test pause
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {'status': 'paused'})
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            result = await controller.pause_evolution(trial_id)
            assert result['status'] == 'paused'
        
        # Test resume
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {'status': 'running'})
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            result = await controller.resume_evolution(trial_id)
            assert result['status'] == 'running'
    
    @pytest.mark.asyncio
    async def test_stop_evolution(self, controller):
        """Test stopping an evolution run"""
        trial_id = 'trial_001'
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {
                'status': 'stopped',
                'final_generation': 7,
                'reason': 'user_requested'
            })
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            result = await controller.stop_evolution(trial_id)
            assert result['status'] == 'stopped'
            assert result['reason'] == 'user_requested'
    
    @pytest.mark.asyncio
    async def test_get_evolution_status(self, controller):
        """Test retrieving evolution run status"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {
                'trial_id': 'trial_001',
                'status': 'running',
                'current_generation': 5,
                'total_generations': 10,
                'active_agents': 8,
                'patterns_discovered': 3,
                'token_usage': {
                    'used': 45000,
                    'allocated': 100000,
                    'efficiency': 1.2
                }
            })
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            status = await controller.get_evolution_status('trial_001')
            assert status['current_generation'] == 5
            assert status['patterns_discovered'] == 3
            assert status['token_usage']['efficiency'] == 1.2


class TestPatternApprovalManager:
    """Test suite for pattern discovery approval"""
    
    @pytest.fixture
    def manager(self):
        """Create a pattern approval manager instance"""
        return PatternApprovalManager(api_url="http://localhost:8090")
    
    @pytest.mark.asyncio
    async def test_get_pending_patterns(self, manager):
        """Test retrieving patterns pending approval"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {
                'patterns': [
                    {
                        'pattern_id': 'pat_001',
                        'type': 'optimization',
                        'description': 'Cache frequently accessed values',
                        'discovery_agent': 'agent_003',
                        'generation': 5,
                        'confidence': 0.85,
                        'impact_estimate': 0.15
                    },
                    {
                        'pattern_id': 'pat_002',
                        'type': 'refactoring',
                        'description': 'Extract common logic to utility function',
                        'discovery_agent': 'agent_007',
                        'generation': 6,
                        'confidence': 0.92,
                        'impact_estimate': 0.08
                    }
                ]
            })
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            patterns = await manager.get_pending_patterns()
            assert len(patterns['patterns']) == 2
            assert patterns['patterns'][0]['pattern_id'] == 'pat_001'
            assert patterns['patterns'][1]['confidence'] == 0.92
    
    @pytest.mark.asyncio
    async def test_approve_pattern(self, manager):
        """Test approving a discovered pattern"""
        pattern_id = 'pat_001'
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {
                'pattern_id': pattern_id,
                'status': 'approved',
                'approved_by': 'operator',
                'approved_at': '2025-01-15T10:30:00Z',
                'auto_propagate': True
            })
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            result = await manager.approve_pattern(
                pattern_id,
                auto_propagate=True,
                notes="Looks good, high confidence"
            )
            
            assert result['status'] == 'approved'
            assert result['auto_propagate'] is True
    
    @pytest.mark.asyncio
    async def test_reject_pattern(self, manager):
        """Test rejecting a discovered pattern"""
        pattern_id = 'pat_002'
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {
                'pattern_id': pattern_id,
                'status': 'rejected',
                'rejected_by': 'operator',
                'rejected_at': '2025-01-15T10:35:00Z',
                'reason': 'potential_side_effects'
            })
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            result = await manager.reject_pattern(
                pattern_id,
                reason='potential_side_effects',
                notes="Could break existing functionality"
            )
            
            assert result['status'] == 'rejected'
            assert result['reason'] == 'potential_side_effects'
    
    @pytest.mark.asyncio
    async def test_set_auto_approval_rules(self, manager):
        """Test setting automatic approval rules"""
        rules = {
            'min_confidence': 0.9,
            'max_impact': 0.1,
            'allowed_types': ['optimization', 'documentation'],
            'require_test_pass': True
        }
        
        with patch('aiohttp.ClientSession.put') as mock_put:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {
                'rules': rules,
                'updated_at': '2025-01-15T10:40:00Z'
            })
            mock_put.return_value.__aenter__.return_value = mock_resp
            
            result = await manager.set_auto_approval_rules(rules)
            assert result['rules']['min_confidence'] == 0.9
            assert 'documentation' in result['rules']['allowed_types']


class TestSystemParameterManager:
    """Test suite for system parameter adjustment"""
    
    @pytest.fixture
    def manager(self):
        """Create a system parameter manager instance"""
        return SystemParameterManager(api_url="http://localhost:8090")
    
    @pytest.mark.asyncio
    async def test_get_current_parameters(self, manager):
        """Test retrieving current system parameters"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {
                'parameters': {
                    'token_budget': {
                        'total': 1000000,
                        'per_generation': 100000,
                        'per_agent': 10000
                    },
                    'diversity': {
                        'min_threshold': 0.3,
                        'intervention_rate': 0.1
                    },
                    'evolution': {
                        'mutation_rate': 0.05,
                        'crossover_rate': 0.7,
                        'selection_pressure': 1.5
                    },
                    'ca_rules': {
                        'enabled': ['Rule110', 'Rule30', 'Rule90'],
                        'thresholds': {
                            'Rule110': {'efficiency': 0.4},
                            'Rule30': {'stall_count': 3}
                        }
                    }
                }
            })
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            params = await manager.get_current_parameters()
            assert params['parameters']['token_budget']['total'] == 1000000
            assert params['parameters']['diversity']['min_threshold'] == 0.3
            assert 'Rule110' in params['parameters']['ca_rules']['enabled']
    
    @pytest.mark.asyncio
    async def test_update_token_budget(self, manager):
        """Test updating token budget parameters"""
        updates = {
            'total': 2000000,
            'per_agent': 15000
        }
        
        with patch('aiohttp.ClientSession.patch') as mock_patch:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {
                'updated': True,
                'parameters': {
                    'token_budget': {
                        'total': 2000000,
                        'per_generation': 100000,
                        'per_agent': 15000
                    }
                },
                'effective_at': 'next_generation'
            })
            mock_patch.return_value.__aenter__.return_value = mock_resp
            
            result = await manager.update_token_budget(updates)
            assert result['parameters']['token_budget']['total'] == 2000000
            assert result['parameters']['token_budget']['per_agent'] == 15000
            assert result['effective_at'] == 'next_generation'
    
    @pytest.mark.asyncio
    async def test_update_diversity_settings(self, manager):
        """Test updating diversity management settings"""
        updates = {
            'min_threshold': 0.25,
            'intervention_rate': 0.15,
            'mutation_strength': 0.8
        }
        
        with patch('aiohttp.ClientSession.patch') as mock_patch:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {
                'updated': True,
                'parameters': {
                    'diversity': {
                        'min_threshold': 0.25,
                        'intervention_rate': 0.15,
                        'mutation_strength': 0.8
                    }
                }
            })
            mock_patch.return_value.__aenter__.return_value = mock_resp
            
            result = await manager.update_diversity_settings(updates)
            assert result['parameters']['diversity']['min_threshold'] == 0.25
            assert result['parameters']['diversity']['mutation_strength'] == 0.8
    
    @pytest.mark.asyncio
    async def test_toggle_ca_rules(self, manager):
        """Test enabling/disabling cellular automata rules"""
        with patch('aiohttp.ClientSession.patch') as mock_patch:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {
                'updated': True,
                'parameters': {
                    'ca_rules': {
                        'enabled': ['Rule110', 'Rule30', 'Rule184'],
                        'disabled': ['Rule90', 'Rule1']
                    }
                }
            })
            mock_patch.return_value.__aenter__.return_value = mock_resp
            
            result = await manager.toggle_ca_rule('Rule184', enabled=True)
            assert 'Rule184' in result['parameters']['ca_rules']['enabled']
            assert 'Rule90' in result['parameters']['ca_rules']['disabled']


class TestWebSocketHandler:
    """Test suite for WebSocket real-time updates"""
    
    @pytest.fixture
    def handler(self):
        """Create a WebSocket handler instance"""
        return WebSocketHandler()
    
    @pytest.mark.asyncio
    async def test_client_subscription(self, handler):
        """Test client subscribing to channels"""
        websocket = MagicMock()
        client_id = await handler.register_client(websocket)
        
        # Subscribe to channels
        await handler.subscribe_client(client_id, ['evolution', 'patterns'])
        
        subscriptions = handler.get_client_subscriptions(client_id)
        assert 'evolution' in subscriptions
        assert 'patterns' in subscriptions
        assert 'agents' not in subscriptions
    
    @pytest.mark.asyncio
    async def test_broadcast_update(self, handler):
        """Test broadcasting updates to subscribed clients"""
        # Register multiple clients
        ws1 = MagicMock()
        ws1.send = asyncio.coroutine(lambda x: None)
        ws2 = MagicMock()
        ws2.send = asyncio.coroutine(lambda x: None)
        ws3 = MagicMock()
        ws3.send = asyncio.coroutine(lambda x: None)
        
        client1 = await handler.register_client(ws1)
        client2 = await handler.register_client(ws2)
        client3 = await handler.register_client(ws3)
        
        # Subscribe to different channels
        await handler.subscribe_client(client1, ['evolution', 'patterns'])
        await handler.subscribe_client(client2, ['evolution'])
        await handler.subscribe_client(client3, ['agents'])
        
        # Broadcast evolution update
        evolution_update = {
            'type': 'evolution_update',
            'data': {
                'generation': 5,
                'active_agents': 8
            }
        }
        
        await handler.broadcast_update('evolution', evolution_update)
        
        # Verify correct clients received update
        ws1.send.assert_called_once()
        ws2.send.assert_called_once()
        ws3.send.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_client_disconnection(self, handler):
        """Test handling client disconnection"""
        websocket = MagicMock()
        client_id = await handler.register_client(websocket)
        await handler.subscribe_client(client_id, ['evolution'])
        
        # Verify client is registered
        assert handler.has_client(client_id)
        
        # Disconnect client
        await handler.unregister_client(client_id)
        
        # Verify client is removed
        assert not handler.has_client(client_id)
        assert len(handler.get_client_subscriptions(client_id)) == 0


class TestRESTAPIHandler:
    """Test suite for REST API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create a test application"""
        api_handler = RESTAPIHandler(api_url="http://localhost:8090")
        return api_handler.create_app()
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, app):
        """Test health check endpoint"""
        from aiohttp.test_utils import TestClient, TestServer
        
        async with TestClient(TestServer(app)) as client:
            resp = await client.get('/health')
            assert resp.status == 200
            data = await resp.json()
            assert data['status'] == 'healthy'
            assert 'uptime' in data
            assert 'version' in data
    
    @pytest.mark.asyncio
    async def test_evolution_control_endpoints(self, app):
        """Test evolution control REST endpoints"""
        from aiohttp.test_utils import TestClient, TestServer
        
        async with TestClient(TestServer(app)) as client:
            # Test starting evolution
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_resp = MagicMock()
                mock_resp.status = 201
                mock_resp.json = asyncio.coroutine(lambda: {'trial_id': 'trial_001'})
                mock_post.return_value.__aenter__.return_value = mock_resp
                
                resp = await client.post('/api/evolution/start', json={
                    'generations': 10,
                    'agents': 5
                })
                assert resp.status == 201
                data = await resp.json()
                assert data['trial_id'] == 'trial_001'
    
    @pytest.mark.asyncio
    async def test_pattern_approval_endpoints(self, app):
        """Test pattern approval REST endpoints"""
        from aiohttp.test_utils import TestClient, TestServer
        
        async with TestClient(TestServer(app)) as client:
            # Test getting pending patterns
            with patch('aiohttp.ClientSession.get') as mock_get:
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.json = asyncio.coroutine(lambda: {'patterns': []})
                mock_get.return_value.__aenter__.return_value = mock_resp
                
                resp = await client.get('/api/patterns/pending')
                assert resp.status == 200
                data = await resp.json()
                assert 'patterns' in data
    
    @pytest.mark.asyncio
    async def test_parameter_management_endpoints(self, app):
        """Test parameter management REST endpoints"""
        from aiohttp.test_utils import TestClient, TestServer
        
        async with TestClient(TestServer(app)) as client:
            # Test getting parameters
            with patch('aiohttp.ClientSession.get') as mock_get:
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.json = asyncio.coroutine(lambda: {'parameters': {}})
                mock_get.return_value.__aenter__.return_value = mock_resp
                
                resp = await client.get('/api/parameters')
                assert resp.status == 200
                data = await resp.json()
                assert 'parameters' in data


class TestIntegration:
    """Integration tests for the complete control interface"""
    
    @pytest.mark.asyncio
    async def test_full_evolution_control_flow(self):
        """Test complete evolution control flow"""
        interface = ControlInterface(api_url="http://localhost:8090")
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock all API responses
            mock_post = MagicMock()
            mock_get = MagicMock()
            mock_session.return_value.post = mock_post
            mock_session.return_value.get = mock_get
            
            # Start interface
            await interface.start()
            
            try:
                # Start evolution
                controller = interface.evolution_controller
                result = await controller.start_evolution(
                    generations=10,
                    agents_per_generation=5
                )
                
                # Monitor status
                status = await controller.get_evolution_status(result['trial_id'])
                
                # Handle pattern approval
                pattern_mgr = interface.pattern_manager
                patterns = await pattern_mgr.get_pending_patterns()
                
                if patterns['patterns']:
                    await pattern_mgr.approve_pattern(
                        patterns['patterns'][0]['pattern_id']
                    )
                
                # Adjust parameters
                param_mgr = interface.parameter_manager
                await param_mgr.update_token_budget({'per_agent': 12000})
                
                # Stop evolution
                await controller.stop_evolution(result['trial_id'])
                
            finally:
                await interface.stop()
    
    @pytest.mark.asyncio
    async def test_websocket_real_time_monitoring(self):
        """Test real-time monitoring via WebSocket"""
        interface = ControlInterface(api_url="http://localhost:8090")
        await interface.start()
        
        try:
            # Connect WebSocket client
            async with websockets.connect('ws://localhost:8091/ws') as websocket:
                # Subscribe to all channels
                await websocket.send(json.dumps({
                    'type': 'subscribe',
                    'channels': ['evolution', 'agents', 'patterns', 'metrics']
                }))
                
                # Simulate evolution events
                await interface.websocket_handler.broadcast_update('evolution', {
                    'type': 'generation_complete',
                    'data': {
                        'generation': 5,
                        'agents_evaluated': 8,
                        'patterns_found': 2
                    }
                })
                
                # Receive update
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                data = json.loads(message)
                assert data['type'] == 'generation_complete'
                assert data['data']['generation'] == 5
                
        finally:
            await interface.stop()


# Performance and reliability tests
class TestPerformance:
    """Performance tests to ensure < 5% impact"""
    
    @pytest.mark.asyncio
    async def test_interface_overhead(self):
        """Test that interface adds minimal overhead"""
        import time
        
        # Baseline: Direct API call
        start = time.time()
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = asyncio.coroutine(lambda: {'status': 'ok'})
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8090/health') as resp:
                    await resp.json()
        
        baseline_time = time.time() - start
        
        # With interface
        interface = ControlInterface(api_url="http://localhost:8090")
        await interface.start()
        
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8091/api/proxy/health') as resp:
                    await resp.json()
        finally:
            await interface.stop()
        
        interface_time = time.time() - start
        
        # Calculate overhead
        overhead = (interface_time - baseline_time) / baseline_time
        assert overhead < 0.05, f"Interface overhead {overhead:.2%} exceeds 5% limit"
    
    @pytest.mark.asyncio
    async def test_websocket_scalability(self):
        """Test WebSocket handling with many concurrent clients"""
        handler = WebSocketHandler()
        
        # Register 100 clients
        clients = []
        for i in range(100):
            ws = MagicMock()
            ws.send = asyncio.coroutine(lambda x: None)
            client_id = await handler.register_client(ws)
            await handler.subscribe_client(client_id, ['evolution'])
            clients.append((client_id, ws))
        
        # Broadcast update
        start = time.time()
        await handler.broadcast_update('evolution', {
            'type': 'test',
            'data': {'value': 42}
        })
        broadcast_time = time.time() - start
        
        # Should complete quickly even with many clients
        assert broadcast_time < 0.1, f"Broadcast took {broadcast_time:.3f}s for 100 clients"
        
        # Verify all clients received update
        for _, ws in clients:
            ws.send.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])