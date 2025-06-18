#!/bin/bash
# DEAN System Deployment Script
# Automates deployment of the complete DEAN infrastructure
# Compliant with DEAN specifications for three-repository architecture

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFRA_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
WORKSPACE_ROOT="$(dirname "$INFRA_DIR")"

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_deps=()
    
    # Check Docker version
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    else
        docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
        required_version="20.10"
        if [ "$(printf '%s\n' "$required_version" "$docker_version" | sort -V | head -n1)" != "$required_version" ]; then
            log_warning "Docker version $docker_version found, but $required_version or higher is recommended"
        else
            log_success "Docker version $docker_version ✓"
        fi
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        compose_cmd="docker-compose"
    elif docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    else
        missing_deps+=("docker-compose")
    fi
    
    if [ -z "${missing_deps[@]:-}" ]; then
        log_success "Docker Compose ✓"
    fi
    
    # Check Python 3.11
    if ! command -v python3.11 &> /dev/null && ! command -v python3 &> /dev/null; then
        missing_deps+=("python3.11")
    else
        python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [[ "$python_version" != "3.11" && "$python_version" != "3.12" ]]; then
            log_warning "Python $python_version found, but 3.11+ is recommended"
        else
            log_success "Python $python_version ✓"
        fi
    fi
    
    # Check PostgreSQL client
    if ! command -v psql &> /dev/null; then
        missing_deps+=("postgresql-client")
    else
        log_success "PostgreSQL client ✓"
    fi
    
    # Check Redis client
    if ! command -v redis-cli &> /dev/null; then
        missing_deps+=("redis-tools")
    else
        log_success "Redis client ✓"
    fi
    
    # Report missing dependencies
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        
        # Provide OS-specific installation instructions
        if [[ "$OSTYPE" == "darwin"* ]]; then
            log_info "On macOS, install missing dependencies with:"
            echo "    brew install ${missing_deps[*]}"
        elif [[ -f /etc/debian_version ]]; then
            log_info "On Ubuntu/Debian, install missing dependencies with:"
            echo "    sudo apt-get update && sudo apt-get install -y ${missing_deps[*]}"
        elif [[ -f /etc/redhat-release ]]; then
            log_info "On RHEL/CentOS, install missing dependencies with:"
            echo "    sudo yum install -y ${missing_deps[*]}"
        else
            log_info "Please install the missing dependencies using your system's package manager"
        fi
        
        return 1
    fi
    
    log_success "All prerequisites satisfied"
    return 0
}

# Configure environment
configure_environment() {
    log_info "Configuring environment..."
    
    cd "$INFRA_DIR"
    
    # Check for .env file
    if [ -f .env ]; then
        log_info "Found existing .env file"
        # Source it to check for required variables
        set -a
        source .env
        set +a
    else
        if [ -f .env.example ]; then
            log_info "Creating .env from .env.example"
            cp .env.example .env
        else
            log_info "Creating new .env file"
            touch .env
        fi
    fi
    
    # Check and prompt for required variables
    local env_updated=false
    
    # CLAUDE_API_KEY
    if [ -z "${CLAUDE_API_KEY:-}" ]; then
        read -p "Enter CLAUDE_API_KEY: " CLAUDE_API_KEY
        echo "CLAUDE_API_KEY=$CLAUDE_API_KEY" >> .env
        env_updated=true
    else
        log_success "CLAUDE_API_KEY configured ✓"
    fi
    
    # GITHUB_TOKEN
    if [ -z "${GITHUB_TOKEN:-}" ]; then
        read -p "Enter GITHUB_TOKEN (optional, press Enter to skip): " GITHUB_TOKEN
        if [ -n "$GITHUB_TOKEN" ]; then
            echo "GITHUB_TOKEN=$GITHUB_TOKEN" >> .env
            env_updated=true
        fi
    else
        log_success "GITHUB_TOKEN configured ✓"
    fi
    
    # ANTHROPIC_API_KEY (alias for CLAUDE_API_KEY)
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "ANTHROPIC_API_KEY=${CLAUDE_API_KEY}" >> .env
        env_updated=true
    fi
    
    # Generate secure passwords if not set
    if [ -z "${DEAN_DB_PASSWORD:-}" ]; then
        DEAN_DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        echo "DEAN_DB_PASSWORD=$DEAN_DB_PASSWORD" >> .env
        env_updated=true
        log_info "Generated secure database password"
    fi
    
    if [ -z "${REDIS_PASSWORD:-}" ]; then
        REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        echo "REDIS_PASSWORD=$REDIS_PASSWORD" >> .env
        env_updated=true
        log_info "Generated secure Redis password"
    fi
    
    if [ -z "${VAULT_TOKEN:-}" ]; then
        VAULT_TOKEN=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        echo "VAULT_TOKEN=$VAULT_TOKEN" >> .env
        env_updated=true
        log_info "Generated secure Vault token"
    fi
    
    # Set default ports per specification
    if [ -z "${EVOLUTION_API_PORT:-}" ]; then
        echo "EVOLUTION_API_PORT=8090" >> .env
        env_updated=true
    fi
    
    if [ -z "${AIRFLOW_PORT:-}" ]; then
        echo "AIRFLOW_PORT=8080" >> .env
        env_updated=true
    fi
    
    if [ -z "${INDEXAGENT_PORT:-}" ]; then
        echo "INDEXAGENT_PORT=8081" >> .env
        env_updated=true
    fi
    
    # Platform-specific settings
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [ -z "${DOCKER_DEFAULT_PLATFORM:-}" ]; then
            echo "DOCKER_DEFAULT_PLATFORM=linux/amd64" >> .env
            env_updated=true
        fi
    fi
    
    if [ "$env_updated" = true ]; then
        log_success "Environment configuration updated"
    else
        log_success "Environment already configured"
    fi
}

# Initialize database
initialize_database() {
    log_info "Initializing database..."
    
    cd "$INFRA_DIR"
    
    # Start only PostgreSQL first
    log_info "Starting PostgreSQL service..."
    $compose_cmd up -d dean-postgres
    
    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if $compose_cmd exec -T dean-postgres pg_isready -U dean_user -d dean_system &> /dev/null; then
            log_success "PostgreSQL is ready"
            break
        fi
        attempt=$((attempt + 1))
        log_info "Waiting for PostgreSQL... ($attempt/$max_attempts)"
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "PostgreSQL failed to start"
        return 1
    fi
    
    # Execute schema creation scripts
    log_info "Creating agent_evolution schema..."
    
    # Check if schema SQL file exists
    local schema_file="$INFRA_DIR/modules/agent-evolution/scripts/complete_schema.sql"
    if [ -f "$schema_file" ]; then
        log_info "Executing schema creation script..."
        $compose_cmd exec -T dean-postgres psql -U dean_user -d agent_evolution < "$schema_file"
        log_success "Database schema created"
    else
        log_warning "Schema file not found at $schema_file, will be created by services"
    fi
    
    # Create additional required databases
    log_info "Ensuring all databases exist..."
    for db in airflow indexagent market_analysis agent_evolution; do
        $compose_cmd exec -T dean-postgres psql -U dean_user -d postgres -c "CREATE DATABASE $db;" 2>/dev/null || true
    done
    
    log_success "Database initialization complete"
}

# Start services
start_services() {
    log_info "Starting DEAN services..."
    
    cd "$INFRA_DIR"
    
    # Build images if needed
    log_info "Building Docker images..."
    $compose_cmd build
    
    # Start all services
    log_info "Starting all services..."
    $compose_cmd up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    
    local services=("dean-postgres" "dean-redis" "dean-vault" "dean-evolution-api" "airflow-webserver" "indexagent")
    local all_healthy=false
    local max_wait=180  # 3 minutes
    local waited=0
    
    while [ $waited -lt $max_wait ]; do
        all_healthy=true
        
        for service in "${services[@]}"; do
            if ! $compose_cmd ps "$service" 2>/dev/null | grep -q "healthy\|running"; then
                all_healthy=false
                break
            fi
        done
        
        if [ "$all_healthy" = true ]; then
            log_success "All services are healthy"
            break
        fi
        
        sleep 5
        waited=$((waited + 5))
        log_info "Waiting for services to be healthy... ($waited/$max_wait seconds)"
    done
    
    if [ "$all_healthy" = false ]; then
        log_error "Some services failed to become healthy"
        log_info "Service status:"
        $compose_cmd ps
        return 1
    fi
    
    # Initialize Airflow
    log_info "Initializing Airflow..."
    $compose_cmd exec -T airflow-webserver airflow db init || true
    
    # Create Airflow admin user
    $compose_cmd exec -T airflow-webserver airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@dean.ai \
        --password admin 2>/dev/null || true
    
    log_success "All services started successfully"
}

# Display service URLs
display_urls() {
    log_info "DEAN System deployed successfully!"
    echo
    echo "Service URLs:"
    echo "  - Evolution API:    http://localhost:8090"
    echo "  - Airflow UI:       http://localhost:8080 (admin/admin)"
    echo "  - IndexAgent API:   http://localhost:8081"
    echo "  - Prometheus:       http://localhost:9090"
    echo "  - Vault UI:         http://localhost:8200"
    echo
    echo "Grafana Dashboards:"
    echo "  - Main Dashboard:   http://localhost:3000/d/dean-evolution"
    echo "  - Agent Details:    http://localhost:3000/d/dean-agents"
    echo "  - Economic Analysis: http://localhost:3000/d/dean-economics"
    echo
    echo "To verify deployment, run:"
    echo "  $SCRIPT_DIR/verify_infrastructure.sh"
    echo
    echo "To start an evolution trial, run:"
    echo "  docker exec -it dean-cli dean evolution start --generations 10 --agents 5"
}

# Main deployment flow
main() {
    echo "DEAN System Deployment"
    echo "====================="
    echo
    
    # Check if running from correct directory
    if [ ! -f "$INFRA_DIR/docker-compose.yml" ]; then
        log_error "docker-compose.yml not found in $INFRA_DIR"
        log_error "Please run this script from the DEAN workspace root"
        exit 1
    fi
    
    # Execute deployment steps
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi
    
    if ! configure_environment; then
        log_error "Environment configuration failed"
        exit 1
    fi
    
    if ! initialize_database; then
        log_error "Database initialization failed"
        exit 1
    fi
    
    if ! start_services; then
        log_error "Service startup failed"
        exit 1
    fi
    
    display_urls
}

# Run main function
main "$@"