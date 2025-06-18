#!/bin/bash
# DEAN Infrastructure Verification Script
# Validates proper system deployment and operation

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFRA_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Compose command detection
if command -v docker-compose &> /dev/null; then
    compose_cmd="docker-compose"
elif docker compose version &> /dev/null; then
    compose_cmd="docker compose"
else
    echo -e "${RED}[ERROR]${NC} Docker Compose not found"
    exit 1
fi

# Verification results
CHECKS_PASSED=0
CHECKS_FAILED=0

# Log functions
log_check() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check Docker containers
check_containers() {
    log_check "Verifying Docker containers..."
    
    cd "$INFRA_DIR"
    
    local required_containers=(
        "dean-postgres"
        "dean-redis"
        "dean-vault"
        "dean-evolution-api"
        "airflow-webserver"
        "airflow-scheduler"
        "indexagent"
    )
    
    for container in "${required_containers[@]}"; do
        if $compose_cmd ps | grep -q "$container.*Up.*healthy\|running"; then
            log_pass "$container is running"
        else
            log_fail "$container is not healthy"
        fi
    done
}

# Check PostgreSQL
check_postgresql() {
    log_check "Verifying PostgreSQL..."
    
    # Check connection
    if $compose_cmd exec -T dean-postgres pg_isready -U dean_user &> /dev/null; then
        log_pass "PostgreSQL connection successful"
    else
        log_fail "PostgreSQL connection failed"
        return
    fi
    
    # Check databases
    local required_dbs=("airflow" "indexagent" "agent_evolution" "market_analysis")
    for db in "${required_dbs[@]}"; do
        if $compose_cmd exec -T dean-postgres psql -U dean_user -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$db'" | grep -q 1; then
            log_pass "Database '$db' exists"
        else
            log_fail "Database '$db' missing"
        fi
    done
    
    # Check agent_evolution schema
    if $compose_cmd exec -T dean-postgres psql -U dean_user -d agent_evolution -tc "\dt" 2>/dev/null | grep -q "discovered_patterns"; then
        log_pass "agent_evolution schema verified"
    else
        log_fail "agent_evolution schema incomplete"
    fi
}

# Check Redis
check_redis() {
    log_check "Verifying Redis..."
    
    # Test connection
    if $compose_cmd exec -T dean-redis redis-cli -p 6380 ping | grep -q PONG; then
        log_pass "Redis connection successful"
    else
        log_fail "Redis connection failed"
        return
    fi
    
    # Test persistence
    $compose_cmd exec -T dean-redis redis-cli -p 6380 SET test_key "test_value" &> /dev/null
    if $compose_cmd exec -T dean-redis redis-cli -p 6380 GET test_key | grep -q "test_value"; then
        log_pass "Redis persistence verified"
        $compose_cmd exec -T dean-redis redis-cli -p 6380 DEL test_key &> /dev/null
    else
        log_fail "Redis persistence failed"
    fi
}

# Check Airflow
check_airflow() {
    log_check "Verifying Airflow..."
    
    # Check webserver health
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health | grep -q "200"; then
        log_pass "Airflow webserver is healthy"
    else
        log_fail "Airflow webserver health check failed"
    fi
    
    # Check DAG registration
    local dean_dag=$($compose_cmd exec -T airflow-webserver airflow dags list 2>/dev/null | grep -c "dean_evolution_cycle" || true)
    if [ "$dean_dag" -gt 0 ]; then
        log_pass "DEAN evolution DAG registered"
    else
        log_fail "DEAN evolution DAG not found"
    fi
    
    # Check scheduler
    if $compose_cmd ps | grep -q "airflow-scheduler.*Up"; then
        log_pass "Airflow scheduler running"
    else
        log_fail "Airflow scheduler not running"
    fi
}

# Check Prometheus
check_prometheus() {
    log_check "Verifying Prometheus..."
    
    # Check Prometheus API
    if curl -s http://localhost:9090/-/healthy | grep -q "Prometheus is Healthy"; then
        log_pass "Prometheus is healthy"
    else
        log_fail "Prometheus health check failed"
        return
    fi
    
    # Check key metrics
    local metrics=(
        "dean_agent_success_rate"
        "dean_diversity_index"
        "dean_patterns_discovered_total"
        "dean_token_budget_allocated"
    )
    
    for metric in "${metrics[@]}"; do
        if curl -s "http://localhost:9090/api/v1/query?query=$metric" | grep -q "\"status\":\"success\""; then
            log_pass "Metric '$metric' available"
        else
            log_fail "Metric '$metric' not found"
        fi
    done
}

# Check Evolution API
check_evolution_api() {
    log_check "Verifying Evolution API..."
    
    # Check health endpoint
    if curl -s http://localhost:8090/health | grep -q "ok\|healthy"; then
        log_pass "Evolution API is healthy"
    else
        log_fail "Evolution API health check failed"
        return
    fi
    
    # Check metrics endpoint
    if curl -s http://localhost:8090/metrics | grep -q "dean_"; then
        log_pass "Evolution API metrics exposed"
    else
        log_fail "Evolution API metrics not available"
    fi
}

# Check IndexAgent
check_indexagent() {
    log_check "Verifying IndexAgent..."
    
    # Check health endpoint
    if curl -s http://localhost:8081/health | grep -q "ok\|healthy"; then
        log_pass "IndexAgent API is healthy"
    else
        log_fail "IndexAgent API health check failed"
    fi
}

# Check Vault
check_vault() {
    log_check "Verifying Vault..."
    
    # Check health
    if curl -s http://localhost:8200/v1/sys/health | grep -q "initialized"; then
        log_pass "Vault is initialized"
    else
        log_fail "Vault not initialized"
    fi
}

# Check file permissions
check_permissions() {
    log_check "Verifying file permissions..."
    
    # Check log directories
    local log_dirs=(
        "$INFRA_DIR/logs"
        "$INFRA_DIR/modules/agent-evolution/logs"
    )
    
    for dir in "${log_dirs[@]}"; do
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            log_pass "Log directory '$dir' is writable"
        else
            log_fail "Log directory '$dir' is not writable"
        fi
    done
}

# Check network connectivity
check_networking() {
    log_check "Verifying network connectivity..."
    
    # Check if services can communicate
    if $compose_cmd exec -T dean-evolution-api curl -s http://indexagent:8081/health &> /dev/null; then
        log_pass "Inter-service communication working"
    else
        log_fail "Inter-service communication failed"
    fi
}

# Display summary
display_summary() {
    echo
    echo "Verification Summary"
    echo "==================="
    echo -e "${GREEN}Passed:${NC} $CHECKS_PASSED"
    echo -e "${RED}Failed:${NC} $CHECKS_FAILED"
    echo
    
    if [ $CHECKS_FAILED -eq 0 ]; then
        echo -e "${GREEN}All checks passed! DEAN system is ready.${NC}"
        return 0
    else
        echo -e "${RED}Some checks failed. Please review the output above.${NC}"
        return 1
    fi
}

# Main verification flow
main() {
    echo "DEAN Infrastructure Verification"
    echo "==============================="
    echo
    
    check_containers
    check_postgresql
    check_redis
    check_airflow
    check_prometheus
    check_evolution_api
    check_indexagent
    check_vault
    check_permissions
    check_networking
    
    display_summary
}

# Run verification
main "$@"