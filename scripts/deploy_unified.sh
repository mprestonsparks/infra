#!/bin/bash
# Unified deployment script for infra + DEAN integration
# Handles both integrated and standalone deployments

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$INFRA_DIR")"
DEAN_DIR="$WORKSPACE_ROOT/DEAN"

# Default values
DEPLOYMENT_MODE="integrated"
ENVIRONMENT="development"
SKIP_VALIDATION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --mode <integrated|standalone|hybrid>  Deployment mode (default: integrated)"
            echo "  --env <development|production>         Environment (default: development)"
            echo "  --skip-validation                      Skip validation checks"
            echo "  --help                                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Header
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Unified DEAN + Infra Deployment      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo
log_info "Deployment Mode: $DEPLOYMENT_MODE"
log_info "Environment: $ENVIRONMENT"
log_info "Workspace Root: $WORKSPACE_ROOT"
echo

# Validation
validate_deployment() {
    log_info "Running deployment validation..."
    
    if [ -f "$INFRA_DIR/scripts/validate_deployment.ps1" ]; then
        pwsh "$INFRA_DIR/scripts/validate_deployment.ps1" -Environment "$ENVIRONMENT" || {
            log_error "Validation failed. Run with -AutoFix to attempt fixes."
            return 1
        }
    else
        log_warning "Validation script not found, skipping validation"
    fi
    
    log_success "Validation passed"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        return 1
    fi
    
    # Check Docker Compose
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        log_error "Docker Compose is not installed"
        return 1
    fi
    
    log_success "Prerequisites satisfied"
}

# Configure environment
configure_environment() {
    log_info "Configuring environment..."
    
    cd "$INFRA_DIR"
    
    # Create .env if it doesn't exist
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            log_info "Created .env from template"
        else
            touch .env
            log_info "Created empty .env file"
        fi
    fi
    
    # Ensure required variables are set
    if ! grep -q "DOCKER_DEFAULT_PLATFORM" .env; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "DOCKER_DEFAULT_PLATFORM=linux/amd64" >> .env
            log_info "Set DOCKER_DEFAULT_PLATFORM for macOS"
        fi
    fi
    
    # For DEAN standalone/hybrid mode
    if [[ "$DEPLOYMENT_MODE" != "integrated" ]] && [ -d "$DEAN_DIR" ]; then
        cd "$DEAN_DIR"
        if [ ! -f .env ] && [ -f .env.template ]; then
            cp .env.template .env
            log_info "Created DEAN .env from template"
        fi
        cd "$INFRA_DIR"
    fi
    
    log_success "Environment configured"
}

# Deploy integrated mode
deploy_integrated() {
    log_info "Deploying in integrated mode..."
    
    cd "$INFRA_DIR"
    
    # Check which compose files to use
    COMPOSE_FILES="-f docker-compose.yml"
    
    if [ -f "docker-compose.dean.yml" ]; then
        COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.dean.yml"
    fi
    
    if [[ "$ENVIRONMENT" == "production" ]] && [ -f "docker-compose.dean.prod.yml" ]; then
        COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.dean.prod.yml"
    fi
    
    # Build images
    log_info "Building Docker images..."
    $COMPOSE_CMD $COMPOSE_FILES build
    
    # Start core services first
    log_info "Starting core services..."
    $COMPOSE_CMD $COMPOSE_FILES up -d postgres redis vault
    
    # Wait for core services
    log_info "Waiting for core services to be ready..."
    sleep 10
    
    # Initialize databases
    log_info "Initializing databases..."
    if [ -f "scripts/init_database.sh" ]; then
        bash scripts/init_database.sh || log_warning "Database initialization script failed"
    fi
    
    # Start all services
    log_info "Starting all services..."
    $COMPOSE_CMD $COMPOSE_FILES up -d
    
    log_success "Integrated deployment complete"
}

# Deploy standalone mode
deploy_standalone() {
    log_info "Deploying DEAN in standalone mode..."
    
    if [ ! -d "$DEAN_DIR" ]; then
        log_error "DEAN directory not found at $DEAN_DIR"
        return 1
    fi
    
    cd "$DEAN_DIR"
    
    # Run DEAN setup
    if [ -f "scripts/setup_environment.ps1" ]; then
        log_info "Running DEAN environment setup..."
        pwsh scripts/setup_environment.ps1 -Environment "$ENVIRONMENT"
    fi
    
    # Deploy DEAN
    if [ -f "deploy_windows.ps1" ]; then
        log_info "Running DEAN deployment..."
        pwsh deploy_windows.ps1
    else
        # Fallback to docker compose
        if [[ "$ENVIRONMENT" == "production" ]]; then
            $COMPOSE_CMD -f docker-compose.prod.yml up -d
        else
            $COMPOSE_CMD up -d
        fi
    fi
    
    log_success "Standalone DEAN deployment complete"
}

# Deploy hybrid mode
deploy_hybrid() {
    log_info "Deploying in hybrid mode..."
    
    # Start infra services
    cd "$INFRA_DIR"
    log_info "Starting infrastructure services..."
    $COMPOSE_CMD up -d postgres redis vault prometheus grafana
    
    # Wait for services
    sleep 10
    
    # Deploy DEAN with its own nginx
    deploy_standalone
    
    log_success "Hybrid deployment complete"
}

# Show deployment info
show_deployment_info() {
    echo
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   Deployment Complete!                 ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
    echo
    
    case "$DEPLOYMENT_MODE" in
        integrated)
            echo "Service URLs:"
            echo "  • Airflow:        http://localhost:8080 (admin/admin)"
            echo "  • IndexAgent:     http://localhost:8081"
            echo "  • DEAN Evolution: http://localhost:8090"
            echo "  • DEAN API:       http://localhost:8091"
            echo "  • DEAN Dashboard: http://localhost:8092"
            echo "  • Vault:          http://localhost:8200"
            echo "  • Prometheus:     http://localhost:9090"
            echo "  • Grafana:        http://localhost:3000 (admin/admin)"
            ;;
        standalone)
            echo "DEAN Service URLs:"
            echo "  • HTTP:           http://localhost"
            echo "  • HTTPS:          https://localhost"
            echo "  • API:            http://localhost:8082"
            echo "  • Documentation:  http://localhost:8082/docs"
            ;;
        hybrid)
            echo "Infrastructure URLs:"
            echo "  • PostgreSQL:     localhost:5432"
            echo "  • Redis:          localhost:6379"
            echo "  • Vault:          http://localhost:8200"
            echo "  • Prometheus:     http://localhost:9090"
            echo "  • Grafana:        http://localhost:3000"
            echo
            echo "DEAN URLs:"
            echo "  • HTTP:           http://localhost"
            echo "  • HTTPS:          https://localhost"
            echo "  • API:            http://localhost:8082"
            ;;
    esac
    
    echo
    echo "Health Checks:"
    case "$DEPLOYMENT_MODE" in
        integrated)
            echo "  curl http://localhost:8091/health"
            echo "  curl http://localhost:8090/health"
            echo "  curl http://localhost:8081/health"
            ;;
        *)
            echo "  curl http://localhost:8082/health"
            echo "  curl http://localhost/health"
            ;;
    esac
    
    echo
    echo "Logs:"
    echo "  docker compose logs -f"
    echo
}

# Main execution
main() {
    # Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi
    
    # Run validation unless skipped
    if [ "$SKIP_VALIDATION" = false ]; then
        if ! validate_deployment; then
            log_error "Deployment validation failed"
            exit 1
        fi
    fi
    
    # Configure environment
    if ! configure_environment; then
        log_error "Environment configuration failed"
        exit 1
    fi
    
    # Deploy based on mode
    case "$DEPLOYMENT_MODE" in
        integrated)
            deploy_integrated
            ;;
        standalone)
            deploy_standalone
            ;;
        hybrid)
            deploy_hybrid
            ;;
        *)
            log_error "Invalid deployment mode: $DEPLOYMENT_MODE"
            exit 1
            ;;
    esac
    
    # Show deployment information
    show_deployment_info
}

# Run main function
main