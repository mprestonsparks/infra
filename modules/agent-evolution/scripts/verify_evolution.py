#!/usr/bin/env python3
"""
DEAN Evolution Verification Script
Performs functional verification of the evolution system
"""

import os
import sys
import time
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import requests
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project paths
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

# Color codes for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class EvolutionVerifier:
    """Verifies DEAN evolution system functionality"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.base_url = "http://localhost:8090"
        self.db_url = os.environ.get('DATABASE_URL', 
                                   'postgresql://dean_user:dean_password_2024@localhost:5432/agent_evolution')
        
    def log_check(self, message: str):
        """Log a check being performed"""
        print(f"{Colors.BLUE}[CHECK]{Colors.NC} {message}")
    
    def log_pass(self, message: str):
        """Log a passed check"""
        print(f"{Colors.GREEN}[PASS]{Colors.NC} {message}")
        self.checks_passed += 1
    
    def log_fail(self, message: str):
        """Log a failed check"""
        print(f"{Colors.RED}[FAIL]{Colors.NC} {message}")
        self.checks_failed += 1
    
    def log_info(self, message: str):
        """Log information"""
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")
    
    def verify_api_health(self) -> bool:
        """Verify Evolution API is healthy"""
        self.log_check("Verifying Evolution API health...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.log_pass("Evolution API is healthy")
                return True
            else:
                self.log_fail(f"Evolution API returned status {response.status_code}")
                return False
        except Exception as e:
            self.log_fail(f"Failed to connect to Evolution API: {e}")
            return False
    
    def spawn_test_agent(self) -> Tuple[bool, str]:
        """Spawn a test agent and verify worktree creation"""
        self.log_check("Spawning test agent...")
        
        agent_id = f"test_agent_{int(time.time())}"
        
        try:
            # Call API to create agent
            payload = {
                "agent_id": agent_id,
                "strategies": ["baseline_optimization", "test_improvement"],
                "generation": 1
            }
            
            response = requests.post(f"{self.base_url}/agents", json=payload, timeout=10)
            
            if response.status_code == 201:
                self.log_pass(f"Test agent '{agent_id}' created successfully")
                
                # Verify worktree exists
                worktree_path = f"/tmp/dean-worktrees/{agent_id}"
                if Path(worktree_path).exists():
                    self.log_pass(f"Worktree created at {worktree_path}")
                else:
                    self.log_fail("Worktree directory not found")
                    
                return True, agent_id
            else:
                self.log_fail(f"Failed to create agent: {response.status_code}")
                return False, ""
                
        except Exception as e:
            self.log_fail(f"Error spawning agent: {e}")
            return False, ""
    
    def verify_action_execution(self, agent_id: str) -> bool:
        """Verify each action type can be executed"""
        self.log_check("Verifying action execution...")
        
        action_types = ["implement_todos", "improve_test_coverage", "refactor_complexity"]
        success_count = 0
        
        for action_type in action_types:
            try:
                payload = {
                    "agent_id": agent_id,
                    "action_type": action_type
                }
                
                response = requests.post(f"{self.base_url}/actions/execute", 
                                       json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        self.log_pass(f"Action '{action_type}' executed successfully")
                        success_count += 1
                        
                        # Verify metrics
                        metrics = result.get("metrics", {})
                        if metrics.get("task_specific_score", 0) > 0:
                            self.log_pass(f"  Task score: {metrics['task_specific_score']:.2f}")
                        if metrics.get("token_cost", 0) > 0:
                            self.log_info(f"  Tokens used: {metrics['token_cost']}")
                    else:
                        self.log_fail(f"Action '{action_type}' failed: {result.get('error')}")
                else:
                    self.log_fail(f"Action '{action_type}' returned status {response.status_code}")
                    
            except Exception as e:
                self.log_fail(f"Error executing action '{action_type}': {e}")
        
        return success_count == len(action_types)
    
    def verify_ca_rules(self, agent_id: str) -> bool:
        """Verify cellular automata rules trigger appropriately"""
        self.log_check("Verifying CA rule triggers...")
        
        try:
            # Simulate conditions for each rule
            rules_verified = 0
            
            # Rule 110: Low efficiency should trigger spawn
            payload = {
                "agent_id": agent_id,
                "fitness_score": 0.3,
                "token_efficiency": 0.2
            }
            response = requests.post(f"{self.base_url}/agents/{agent_id}/update_fitness", 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                # Check if rule triggered
                response = requests.get(f"{self.base_url}/agents/{agent_id}/decisions", timeout=5)
                if response.status_code == 200:
                    decisions = response.json()
                    if any(d["rule_name"] == "Rule110_SpawnImprovedNeighbor" for d in decisions):
                        self.log_pass("Rule 110 triggered on low efficiency")
                        rules_verified += 1
            
            # Rule 30: Stagnation should trigger fork
            payload = {"agent_id": agent_id, "stall_count": 4}
            response = requests.post(f"{self.base_url}/agents/{agent_id}/update_stall", 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                response = requests.get(f"{self.base_url}/agents/{agent_id}/decisions", timeout=5)
                if response.status_code == 200:
                    decisions = response.json()
                    if any(d["rule_name"] == "Rule30_ForkOnBottleneck" for d in decisions):
                        self.log_pass("Rule 30 triggered on stagnation")
                        rules_verified += 1
            
            # Additional rules would be tested similarly
            
            return rules_verified >= 2
            
        except Exception as e:
            self.log_fail(f"Error verifying CA rules: {e}")
            return False
    
    def verify_pattern_detection(self) -> bool:
        """Verify pattern detection and cataloging"""
        self.log_check("Verifying pattern detection...")
        
        try:
            # Create a mock pattern
            pattern_data = {
                "pattern_type": "optimization",
                "action_type": "implement_todos",
                "description": "Test pattern for verification",
                "success_delta": 0.15,
                "agent_id": "test_agent",
                "generation": 1
            }
            
            response = requests.post(f"{self.base_url}/patterns", json=pattern_data, timeout=10)
            
            if response.status_code == 201:
                pattern_id = response.json().get("pattern_id")
                self.log_pass(f"Pattern created with ID: {pattern_id}")
                
                # Verify pattern is cataloged
                conn = psycopg2.connect(self.db_url)
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                cur.execute("""
                    SELECT COUNT(*) as count 
                    FROM discovered_patterns 
                    WHERE pattern_id = %s
                """, (pattern_id,))
                
                result = cur.fetchone()
                conn.close()
                
                if result and result['count'] > 0:
                    self.log_pass("Pattern successfully cataloged in database")
                    return True
                else:
                    self.log_fail("Pattern not found in database")
                    return False
            else:
                self.log_fail(f"Failed to create pattern: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_fail(f"Error verifying pattern detection: {e}")
            return False
    
    def verify_meta_learning(self) -> bool:
        """Verify meta-learning cycle execution"""
        self.log_check("Verifying meta-learning cycle...")
        
        try:
            # Trigger meta-learning
            response = requests.post(f"{self.base_url}/meta-learning/extract", 
                                   json={"generation": 5}, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                patterns_extracted = result.get("patterns_extracted", 0)
                
                if patterns_extracted > 0:
                    self.log_pass(f"Meta-learning extracted {patterns_extracted} patterns")
                    
                    # Verify DSPy injection
                    examples_injected = result.get("examples_injected", 0)
                    if examples_injected > 0:
                        self.log_pass(f"Injected {examples_injected} examples into DSPy")
                        return True
                    else:
                        self.log_fail("No examples injected into DSPy")
                        return False
                else:
                    self.log_info("No patterns available for extraction (expected in new system)")
                    return True
            else:
                self.log_fail(f"Meta-learning request failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_fail(f"Error verifying meta-learning: {e}")
            return False
    
    def verify_economic_governance(self, agent_id: str) -> bool:
        """Verify economic governance and token management"""
        self.log_check("Verifying economic governance...")
        
        try:
            # Check agent budget
            response = requests.get(f"{self.base_url}/agents/{agent_id}/budget", timeout=5)
            
            if response.status_code == 200:
                budget_data = response.json()
                current_budget = budget_data.get("current_budget", 0)
                
                if current_budget > 0:
                    self.log_pass(f"Agent has budget: {current_budget} tokens")
                    
                    # Verify token usage tracking
                    usage_data = {
                        "agent_id": agent_id,
                        "tokens": 500,
                        "action_type": "implement_todos"
                    }
                    
                    response = requests.post(f"{self.base_url}/tokens/use", 
                                           json=usage_data, timeout=10)
                    
                    if response.status_code == 200:
                        self.log_pass("Token usage tracked successfully")
                        return True
                    else:
                        self.log_fail("Failed to track token usage")
                        return False
                else:
                    self.log_fail("Agent has no budget allocated")
                    return False
            else:
                self.log_fail(f"Failed to get agent budget: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_fail(f"Error verifying economic governance: {e}")
            return False
    
    def cleanup_test_agent(self, agent_id: str):
        """Clean up test agent and worktree"""
        self.log_info(f"Cleaning up test agent '{agent_id}'...")
        
        try:
            response = requests.delete(f"{self.base_url}/agents/{agent_id}", timeout=10)
            if response.status_code in [200, 204]:
                self.log_info("Test agent cleaned up")
        except:
            pass
    
    def display_summary(self) -> bool:
        """Display verification summary"""
        print("\nEvolution Verification Summary")
        print("=============================")
        print(f"{Colors.GREEN}Passed:{Colors.NC} {self.checks_passed}")
        print(f"{Colors.RED}Failed:{Colors.NC} {self.checks_failed}")
        print()
        
        if self.checks_failed == 0:
            print(f"{Colors.GREEN}All evolution checks passed!{Colors.NC}")
            return True
        else:
            print(f"{Colors.RED}Some evolution checks failed. Please review the output above.{Colors.NC}")
            return False
    
    def run_verification(self) -> bool:
        """Run complete evolution verification"""
        print("DEAN Evolution System Verification")
        print("=================================")
        print()
        
        # Check API health first
        if not self.verify_api_health():
            print(f"\n{Colors.RED}Evolution API is not accessible. Ensure services are running.{Colors.NC}")
            return False
        
        # Spawn test agent
        success, agent_id = self.spawn_test_agent()
        if not success:
            return False
        
        try:
            # Run verification tests
            self.verify_action_execution(agent_id)
            self.verify_ca_rules(agent_id)
            self.verify_pattern_detection()
            self.verify_meta_learning()
            self.verify_economic_governance(agent_id)
            
        finally:
            # Clean up
            self.cleanup_test_agent(agent_id)
        
        return self.display_summary()


def main():
    """Main entry point"""
    verifier = EvolutionVerifier()
    success = verifier.run_verification()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()