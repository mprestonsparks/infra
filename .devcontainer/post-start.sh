#!/bin/bash

# Infrastructure Post-Start Script
# This script runs every time the container starts

set -e

echo "🔄 Starting Infrastructure services..."

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Set up environment variables
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
export VAULT_ADDR=http://host.docker.internal:8200

# =============================================================================
# SERVICE HEALTH CHECKS
# =============================================================================

echo "🏥 Performing infrastructure health checks..."

# Function to check if a service is reachable
check_service() {
    local service_name=$1
    local host=$2
    local port=$3
    
    if nc -z "$host" "$port" 2>/dev/null; then
        echo "✅ $service_name ($host:$port) is reachable"
        return 0
    else
        echo "⚠️  $service_name ($host:$port) is not reachable"
        return 1
    fi
}

# Check core infrastructure services
check_service "Docker Daemon" "localhost" "2375" || echo "Docker daemon check skipped (socket-based)"
check_service "PostgreSQL" "host.docker.internal" "5432"
check_service "Vault" "host.docker.internal" "8200"
check_service "Redis" "host.docker.internal" "6379"

# =============================================================================
# DOCKER ENVIRONMENT VERIFICATION
# =============================================================================

echo "🐳 Verifying Docker environment..."

# Check Docker version
echo "📋 Docker version:"
docker --version

# Check Docker Compose version
echo "📋 Docker Compose version:"
docker-compose --version

# Check running containers
echo "📊 Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "No containers running"

# Check Docker networks
echo "🌐 Docker networks:"
docker network ls | grep -E "(multi-repo|bridge)" || echo "No custom networks found"

# Check Docker volumes
echo "💾 Docker volumes:"
docker volume ls | grep -E "(postgres|vault|redis)" || echo "No persistent volumes found"

# =============================================================================
# TERRAFORM VERIFICATION
# =============================================================================

echo "🏗️ Verifying Terraform environment..."

# Check Terraform version
echo "📋 Terraform version:"
terraform --version

# Initialize Terraform if not already done
if [ -d "terraform" ]; then
    cd terraform
    if [ ! -d ".terraform" ]; then
        echo "🔧 Initializing Terraform..."
        terraform init
    else
        echo "✅ Terraform already initialized"
    fi
    
    # Validate Terraform configuration
    echo "🔍 Validating Terraform configuration..."
    terraform validate && echo "✅ Terraform configuration is valid" || echo "❌ Terraform configuration has errors"
    
    cd ..
else
    echo "⚠️  Terraform directory not found"
fi

# =============================================================================
# KUBERNETES VERIFICATION
# =============================================================================

echo "☸️ Verifying Kubernetes tools..."

# Check kubectl version
echo "📋 kubectl version:"
kubectl version --client

# Check Helm version
echo "📋 Helm version:"
helm version

# Check if there's a Kubernetes cluster available
echo "🔍 Checking Kubernetes cluster connectivity..."
if kubectl cluster-info >/dev/null 2>&1; then
    echo "✅ Kubernetes cluster is accessible"
    kubectl get nodes 2>/dev/null || echo "No nodes found or insufficient permissions"
else
    echo "⚠️  No Kubernetes cluster available (this is normal for local development)"
fi

# =============================================================================
# ANSIBLE VERIFICATION
# =============================================================================

echo "📋 Verifying Ansible environment..."

# Check Ansible version
echo "📋 Ansible version:"
ansible --version

# Check Ansible inventory
if [ -f "ansible/inventory/hosts.yml" ]; then
    echo "🔍 Validating Ansible inventory..."
    ansible-inventory -i ansible/inventory/hosts.yml --list >/dev/null && echo "✅ Ansible inventory is valid" || echo "❌ Ansible inventory has errors"
else
    echo "⚠️  Ansible inventory not found"
fi

# =============================================================================
# VAULT INTEGRATION
# =============================================================================

echo "🔐 Setting up Vault integration..."

export VAULT_ADDR=http://host.docker.internal:8200
export VAULT_TOKEN=dev-token

# Wait for Vault to be ready
echo "⏳ Waiting for Vault to be ready..."
for i in {1..30}; do
    if curl -s $VAULT_ADDR/v1/sys/health >/dev/null 2>&1; then
        echo "✅ Vault is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Vault health check timeout"
        break
    fi
    sleep 2
done

# Test Vault connectivity
if vault status >/dev/null 2>&1; then
    echo "✅ Vault CLI connectivity successful"
    
    # List available secret engines
    echo "📋 Available Vault secret engines:"
    vault secrets list 2>/dev/null || echo "Unable to list secret engines"
else
    echo "⚠️  Vault CLI connectivity failed"
fi

# =============================================================================
# INFRASTRUCTURE MONITORING
# =============================================================================

echo "📊 Setting up infrastructure monitoring..."

# Create monitoring script if it doesn't exist
if [ ! -f "scripts/monitoring/system-status.sh" ]; then
    mkdir -p scripts/monitoring
    cat > scripts/monitoring/system-status.sh << 'EOF'
#!/bin/bash

echo "🖥️ System Status Report"
echo "======================="

echo ""
echo "💾 Disk Usage:"
df -h | head -5

echo ""
echo "🧠 Memory Usage:"
free -h

echo ""
echo "⚡ CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2 $3 $4 $5}'

echo ""
echo "🐳 Docker Stats:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>/dev/null || echo "No containers running"

echo ""
echo "🌐 Network Connections:"
netstat -tuln | grep -E "(5432|8080|8081|8000|8200|6379)" || echo "No relevant services listening"
EOF
    chmod +x scripts/monitoring/system-status.sh
fi

# Run system status check
if [ -f "scripts/monitoring/system-status.sh" ]; then
    echo "🔍 Running system status check..."
    ./scripts/monitoring/system-status.sh
fi

# =============================================================================
# DEVELOPMENT ENVIRONMENT STATUS
# =============================================================================

echo ""
echo "📊 Infrastructure Status Summary:"
echo "=================================="

# Tool versions
echo "🛠️ Tool Versions:"
echo "   - Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"
echo "   - Docker Compose: $(docker-compose --version | cut -d' ' -f3 | tr -d ',')"
echo "   - Terraform: $(terraform --version | head -1 | cut -d' ' -f2)"
echo "   - kubectl: $(kubectl version --client --short | cut -d' ' -f3)"
echo "   - Helm: $(helm version --short | cut -d' ' -f2)"
echo "   - Ansible: $(ansible --version | head -1 | cut -d' ' -f2)"

# Service connectivity
echo ""
echo "🔧 Service Connectivity:"
echo "   - PostgreSQL: $(check_service "PostgreSQL" "host.docker.internal" "5432" >/dev/null 2>&1 && echo "✅ Connected" || echo "❌ Disconnected")"
echo "   - Vault: $(check_service "Vault" "host.docker.internal" "8200" >/dev/null 2>&1 && echo "✅ Connected" || echo "❌ Disconnected")"
echo "   - Redis: $(check_service "Redis" "host.docker.internal" "6379" >/dev/null 2>&1 && echo "✅ Connected" || echo "❌ Disconnected")"

# Infrastructure status
echo ""
echo "🏗️ Infrastructure:"
echo "   - Docker daemon: $(docker info >/dev/null 2>&1 && echo "✅ Running" || echo "❌ Not running")"
echo "   - Terraform: $([ -d "terraform/.terraform" ] && echo "✅ Initialized" || echo "⚠️  Not initialized")"
echo "   - Kubernetes: $(kubectl cluster-info >/dev/null 2>&1 && echo "✅ Available" || echo "⚠️  Not available")"

# Available commands
echo ""
echo "💡 Quick commands:"
echo "   - 'make help' - Show all available commands"
echo "   - 'make deploy' - Deploy the entire infrastructure stack"
echo "   - 'make health' - Run comprehensive health checks"
echo "   - 'make status' - Show service status"
echo "   - 'terraform plan' - Plan infrastructure changes"
echo "   - 'ansible-playbook ansible/playbooks/deploy.yml' - Deploy with Ansible"

# Configuration files
echo ""
echo "📁 Configuration Files:"
echo "   - Docker Compose: docker/docker-compose.yml"
echo "   - Terraform: terraform/main.tf"
echo "   - Kubernetes: kubernetes/manifests/"
echo "   - Ansible: ansible/playbooks/deploy.yml"

echo ""
echo "🎉 Infrastructure environment is ready for orchestration!"