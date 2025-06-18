#!/usr/bin/env python3
"""
DEAN CLI - Extended Commands
Provides operational commands for the DEAN system
"""

import click
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from tabulate import tabulate
import os

# Color utilities
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'


@click.group()
@click.option('--api-url', default='http://localhost:8090', 
              envvar='DEAN_API_URL', help='Evolution API URL')
@click.pass_context
def cli(ctx, api_url):
    """DEAN System CLI - Extended operational commands"""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url


@cli.command()
@click.pass_context
def status(ctx):
    """Show comprehensive system health information"""
    api_url = ctx.obj['api_url']
    
    click.echo(f"{Colors.BOLD}DEAN System Status{Colors.NC}")
    click.echo("=" * 50)
    
    # Check services
    services = [
        ("Evolution API", f"{api_url}/health"),
        ("IndexAgent", "http://localhost:8081/health"),
        ("Airflow", "http://localhost:8080/health"),
        ("Prometheus", "http://localhost:9090/-/healthy"),
    ]
    
    click.echo(f"\n{Colors.CYAN}Service Status:{Colors.NC}")
    service_status = []
    
    for name, url in services:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                status_icon = f"{Colors.GREEN}✓{Colors.NC}"
                status_text = "Healthy"
            else:
                status_icon = f"{Colors.YELLOW}!{Colors.NC}"
                status_text = f"Status {resp.status_code}"
        except:
            status_icon = f"{Colors.RED}✗{Colors.NC}"
            status_text = "Unreachable"
        
        service_status.append([status_icon, name, status_text])
    
    click.echo(tabulate(service_status, headers=["", "Service", "Status"], 
                       tablefmt="simple"))
    
    # Get evolution status
    try:
        resp = requests.get(f"{api_url}/evolution/status", timeout=5)
        if resp.status_code == 200:
            evo_data = resp.json()
            
            click.echo(f"\n{Colors.CYAN}Evolution Status:{Colors.NC}")
            click.echo(f"  Current Generation: {Colors.BOLD}{evo_data.get('generation', 0)}{Colors.NC}")
            click.echo(f"  Active Agents: {evo_data.get('active_agents', 0)}")
            click.echo(f"  Total Agents: {evo_data.get('total_agents', 0)}")
            
            # Recent discoveries
            patterns = evo_data.get('recent_patterns', [])
            if patterns:
                click.echo(f"\n{Colors.CYAN}Recent Pattern Discoveries:{Colors.NC}")
                for p in patterns[:5]:
                    click.echo(f"  • {p['type']}: {p['description']} (gen {p['generation']})")
            
            # Token budget
            budget = evo_data.get('token_budget', {})
            if budget:
                used = budget.get('used', 0)
                total = budget.get('total', 1)
                remaining = total - used
                usage_pct = (used / total * 100) if total > 0 else 0
                
                click.echo(f"\n{Colors.CYAN}Token Budget:{Colors.NC}")
                click.echo(f"  Used: {used:,} / {total:,} ({usage_pct:.1f}%)")
                
                # Progress bar
                bar_width = 30
                filled = int(bar_width * usage_pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                if usage_pct < 50:
                    bar_color = Colors.GREEN
                elif usage_pct < 80:
                    bar_color = Colors.YELLOW
                else:
                    bar_color = Colors.RED
                
                click.echo(f"  [{bar_color}{bar}{Colors.NC}] {remaining:,} remaining")
    
    except Exception as e:
        click.echo(f"{Colors.RED}Failed to get evolution status: {e}{Colors.NC}")


@cli.command()
@click.option('--level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Log level filter')
@click.option('--component', type=click.Choice(['all', 'evolution', 'agent', 'pattern', 'economic']), 
              default='all', help='Component to filter')
@click.option('--since', default='1h', help='Time range (e.g., 1h, 30m, 1d)')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--grep', '-g', help='Pattern to search for')
@click.pass_context
def logs(ctx, level, component, since, follow, grep):
    """View and filter system logs"""
    # Parse time range
    time_map = {'m': 'minutes', 'h': 'hours', 'd': 'days'}
    
    amount = int(since[:-1])
    unit = since[-1]
    
    if unit not in time_map:
        click.echo(f"{Colors.RED}Invalid time format. Use format like '1h', '30m', '1d'{Colors.NC}")
        return
    
    # In production, this would query actual log aggregation
    # For now, show mock implementation
    click.echo(f"{Colors.CYAN}Showing {component} logs ({level}+) from last {since}{Colors.NC}")
    
    if grep:
        click.echo(f"{Colors.YELLOW}Filtering for: {grep}{Colors.NC}")
    
    click.echo("-" * 80)
    
    # Mock log entries
    log_entries = [
        {"timestamp": datetime.now() - timedelta(minutes=45), "level": "INFO", 
         "component": "evolution", "message": "Started generation 15"},
        {"timestamp": datetime.now() - timedelta(minutes=30), "level": "INFO", 
         "component": "agent", "message": "Agent agent_015_003 achieved 0.85 fitness"},
        {"timestamp": datetime.now() - timedelta(minutes=20), "level": "WARNING", 
         "component": "economic", "message": "Token budget at 75% utilization"},
        {"timestamp": datetime.now() - timedelta(minutes=10), "level": "INFO", 
         "component": "pattern", "message": "Discovered optimization pattern opt_cache_001"},
        {"timestamp": datetime.now() - timedelta(minutes=5), "level": "ERROR", 
         "component": "agent", "message": "Agent agent_015_007 failed action execution"},
    ]
    
    # Filter and display
    for entry in log_entries:
        if component != 'all' and entry['component'] != component:
            continue
        
        if grep and grep.lower() not in entry['message'].lower():
            continue
        
        # Color by level
        level_colors = {
            'DEBUG': Colors.BLUE,
            'INFO': Colors.GREEN,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.RED
        }
        
        level_color = level_colors.get(entry['level'], Colors.NC)
        
        click.echo(f"{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} "
                  f"[{level_color}{entry['level']:>7}{Colors.NC}] "
                  f"[{entry['component']:>10}] {entry['message']}")
    
    if follow:
        click.echo(f"\n{Colors.CYAN}Following logs... (Ctrl+C to stop){Colors.NC}")
        try:
            while True:
                time.sleep(1)
                # In production, would tail actual logs
        except KeyboardInterrupt:
            click.echo("\nStopped following logs")


@cli.command()
@click.argument('data-type', type=click.Choice(['patterns', 'metrics', 'lineage', 'history']))
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']), 
              default='json', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file')
@click.option('--generation', '-g', type=int, help='Filter by generation')
@click.option('--compress', is_flag=True, help='Compress output')
@click.pass_context
def export(ctx, data_type, output_format, output, generation, compress):
    """Export data for analysis"""
    api_url = ctx.obj['api_url']
    
    click.echo(f"Exporting {data_type} data...")
    
    # Build export URL
    export_url = f"{api_url}/export/{data_type}"
    params = {'format': output_format}
    
    if generation:
        params['generation'] = generation
    
    try:
        resp = requests.get(export_url, params=params, timeout=30)
        
        if resp.status_code == 200:
            data = resp.content
            
            # Determine output file
            if not output:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                ext = 'csv' if output_format == 'csv' else 'json'
                output = f"dean_{data_type}_{timestamp}.{ext}"
                
                if compress:
                    output += '.gz'
            
            # Write data
            if compress:
                import gzip
                with gzip.open(output, 'wb') as f:
                    f.write(data)
            else:
                with open(output, 'wb') as f:
                    f.write(data)
            
            file_size = Path(output).stat().st_size
            click.echo(f"{Colors.GREEN}✓ Exported to {output} ({file_size:,} bytes){Colors.NC}")
            
            # Show preview for JSON
            if output_format == 'json' and not compress:
                try:
                    preview_data = json.loads(data)
                    if isinstance(preview_data, dict):
                        click.echo(f"\nExport Summary:")
                        for key, value in list(preview_data.items())[:5]:
                            if isinstance(value, (list, dict)):
                                click.echo(f"  {key}: {len(value)} items")
                            else:
                                click.echo(f"  {key}: {value}")
                except:
                    pass
        else:
            click.echo(f"{Colors.RED}Export failed: {resp.status_code}{Colors.NC}")
            
    except Exception as e:
        click.echo(f"{Colors.RED}Export error: {e}{Colors.NC}")


@cli.command()
@click.option('--generations', '-g', type=int, default=10, help='Number of generations')
@click.option('--agents', '-a', type=int, default=5, help='Agents per generation')
@click.option('--strategies', '-s', multiple=True, help='Initial strategies')
@click.pass_context
def evolution(ctx, generations, agents, strategies):
    """Start a new evolution trial"""
    api_url = ctx.obj['api_url']
    
    click.echo(f"{Colors.BOLD}Starting DEAN Evolution Trial{Colors.NC}")
    click.echo(f"Generations: {generations}")
    click.echo(f"Agents per generation: {agents}")
    
    if strategies:
        click.echo(f"Initial strategies: {', '.join(strategies)}")
    
    # Confirm
    if not click.confirm("\nProceed with evolution trial?"):
        click.echo("Cancelled")
        return
    
    # Start evolution
    payload = {
        'generations': generations,
        'agent_count': agents,
        'initial_strategies': list(strategies) if strategies else None
    }
    
    try:
        resp = requests.post(f"{api_url}/evolution/start", json=payload, timeout=10)
        
        if resp.status_code == 200:
            result = resp.json()
            trial_id = result.get('trial_id')
            
            click.echo(f"\n{Colors.GREEN}✓ Evolution trial started{Colors.NC}")
            click.echo(f"Trial ID: {trial_id}")
            click.echo(f"\nMonitor progress at:")
            click.echo(f"  • Grafana: http://localhost:3000/d/dean-evolution")
            click.echo(f"  • Airflow: http://localhost:8080")
            click.echo(f"\nView status: dean status")
            click.echo(f"View logs: dean logs -f")
        else:
            click.echo(f"{Colors.RED}Failed to start evolution: {resp.status_code}{Colors.NC}")
            
    except Exception as e:
        click.echo(f"{Colors.RED}Error starting evolution: {e}{Colors.NC}")


@cli.command()
@click.argument('pattern-id')
@click.pass_context
def pattern(ctx, pattern_id):
    """Show details about a discovered pattern"""
    api_url = ctx.obj['api_url']
    
    try:
        resp = requests.get(f"{api_url}/patterns/{pattern_id}", timeout=5)
        
        if resp.status_code == 200:
            pattern = resp.json()
            
            click.echo(f"\n{Colors.BOLD}Pattern: {pattern_id}{Colors.NC}")
            click.echo("=" * 50)
            
            # Basic info
            click.echo(f"Type: {pattern['pattern_type']}")
            click.echo(f"Action: {pattern['action_type']}")
            click.echo(f"Description: {pattern['description']}")
            click.echo(f"Discovery Agent: {pattern['discovery_agent']}")
            click.echo(f"Generation: {pattern['discovery_generation']}")
            
            # Success metrics
            click.echo(f"\n{Colors.CYAN}Performance:{Colors.NC}")
            click.echo(f"  Average Success Delta: {pattern['avg_success_delta']:.3f}")
            click.echo(f"  Reuse Count: {pattern['reuse_count']}")
            click.echo(f"  Last Used: {pattern['last_used']}")
            
            # Pattern data
            if 'pattern_data' in pattern:
                click.echo(f"\n{Colors.CYAN}Pattern Details:{Colors.NC}")
                for key, value in pattern['pattern_data'].items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    click.echo(f"  {key}: {value}")
            
            # Usage history
            if 'recent_usage' in pattern:
                click.echo(f"\n{Colors.CYAN}Recent Usage:{Colors.NC}")
                usage_table = []
                for usage in pattern['recent_usage'][:5]:
                    usage_table.append([
                        usage['agent_id'],
                        usage['generation'],
                        f"{usage['success_score']:.2f}",
                        usage['used_at']
                    ])
                
                if usage_table:
                    click.echo(tabulate(usage_table, 
                                      headers=["Agent", "Gen", "Success", "Time"],
                                      tablefmt="simple"))
        else:
            click.echo(f"{Colors.RED}Pattern not found{Colors.NC}")
            
    except Exception as e:
        click.echo(f"{Colors.RED}Error fetching pattern: {e}{Colors.NC}")


@cli.command()
@click.argument('agent-id')
@click.pass_context  
def agent(ctx, agent_id):
    """Show details about a specific agent"""
    api_url = ctx.obj['api_url']
    
    try:
        resp = requests.get(f"{api_url}/agents/{agent_id}", timeout=5)
        
        if resp.status_code == 200:
            agent_data = resp.json()
            
            click.echo(f"\n{Colors.BOLD}Agent: {agent_id}{Colors.NC}")
            click.echo("=" * 50)
            
            # Basic info
            click.echo(f"Generation: {agent_data['generation']}")
            click.echo(f"Fitness Score: {agent_data['fitness_score']:.3f}")
            click.echo(f"Token Efficiency: {agent_data['token_efficiency']:.3f}")
            click.echo(f"Current Budget: {agent_data['current_budget']:,} tokens")
            
            # Strategies
            click.echo(f"\n{Colors.CYAN}Active Strategies:{Colors.NC}")
            for strategy in agent_data['active_strategies']:
                click.echo(f"  • {strategy}")
            
            # Lineage
            if agent_data.get('lineage'):
                click.echo(f"\n{Colors.CYAN}Lineage:{Colors.NC}")
                for i, parent in enumerate(agent_data['lineage'][-5:]):
                    click.echo(f"  {'└' if i == len(agent_data['lineage'][-5:])-1 else '├'}─ {parent}")
            
            # Recent improvements
            if agent_data.get('recent_improvements'):
                click.echo(f"\n{Colors.CYAN}Recent Performance:{Colors.NC}")
                improvements = agent_data['recent_improvements']
                for i, imp in enumerate(improvements):
                    if imp > 0:
                        color = Colors.GREEN
                        symbol = "↑"
                    elif imp < 0:
                        color = Colors.RED
                        symbol = "↓"
                    else:
                        color = Colors.YELLOW
                        symbol = "→"
                    
                    click.echo(f"  Gen {agent_data['generation']-len(improvements)+i+1}: "
                              f"{color}{symbol} {imp:+.3f}{Colors.NC}")
        else:
            click.echo(f"{Colors.RED}Agent not found{Colors.NC}")
            
    except Exception as e:
        click.echo(f"{Colors.RED}Error fetching agent: {e}{Colors.NC}")


if __name__ == '__main__':
    cli()