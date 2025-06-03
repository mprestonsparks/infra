#!/bin/bash

# Infrastructure Post-Create Script
# This script runs after the container is created

set -e

echo "🚀 Setting up Infrastructure development environment..."

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Set up environment variables
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
export VAULT_ADDR=http://host.docker.internal:8200

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================

echo "📁 Creating infrastructure directory structure..."

# Create main infrastructure directories
mkdir -p docker
mkdir -p docker/compose
mkdir -p docker/images
mkdir -p terraform
mkdir -p terraform/modules
mkdir -p terraform/environments
mkdir -p kubernetes
mkdir -p kubernetes/manifests
mkdir -p kubernetes/helm-charts
mkdir -p ansible
mkdir -p ansible/playbooks
mkdir -p ansible/roles
mkdir -p scripts
mkdir -p scripts/deployment
mkdir -p scripts/monitoring
mkdir -p scripts/backup
mkdir -p docs
mkdir -p configs
mkdir -p templates

echo "✅ Directory structure created"

# =============================================================================
# DOCKER CONFIGURATION
# =============================================================================

echo "🐳 Setting up Docker configurations..."

# Create main docker-compose file for the entire stack
if [ ! -f "docker/docker-compose.yml" ]; then
    cat > docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: airflow
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_MULTIPLE_DATABASES: indexagent,market_analysis
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ../scripts/init-multiple-databases.sh:/docker-entrypoint-initdb.d/init-multiple-databases.sh:ro
    ports:
      - "5432:5432"
    networks:
      - multi-repo-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U airflow"]
      interval: 10s
      timeout: 5s
      retries: 5

  # HashiCorp Vault
  vault:
    image: hashicorp/vault:1.15
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: dev-token
      VAULT_DEV_LISTEN_ADDRESS: 0.0.0.0:8200
      VAULT_ADDR: http://0.0.0.0:8200
    ports:
      - "8200:8200"
    networks:
      - multi-repo-network
    cap_add:
      - IPC_LOCK
    volumes:
      - vault_data:/vault/data
    command: vault server -dev -dev-root-token-id=dev-token -dev-listen-address=0.0.0.0:8200

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - multi-repo-network
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  # Airflow Webserver
  airflow-webserver:
    build:
      context: ../../airflow-hub
      dockerfile: .devcontainer/Dockerfile
    environment:
      AIRFLOW_HOME: /opt/airflow
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
    ports:
      - "8080:8080"
    networks:
      - multi-repo-network
    depends_on:
      - postgres
      - vault
    volumes:
      - ../../airflow-hub:/opt/airflow
    command: airflow webserver

  # IndexAgent API
  indexagent:
    build:
      context: ../../IndexAgent
      dockerfile: .devcontainer/Dockerfile
    environment:
      PYTHONPATH: /app
      DATABASE_URL: postgresql+psycopg2://airflow:airflow@postgres:5432/indexagent
    ports:
      - "8081:8081"
    networks:
      - multi-repo-network
    depends_on:
      - postgres
      - vault
    volumes:
      - ../../IndexAgent:/app

  # Market Analysis API
  market-analysis:
    build:
      context: ../../market-analysis
      dockerfile: .devcontainer/Dockerfile
    environment:
      PYTHONPATH: /app
      DATABASE_URL: postgresql+psycopg2://airflow:airflow@postgres:5432/market_analysis
    ports:
      - "8000:8000"
    networks:
      - multi-repo-network
    depends_on:
      - postgres
      - vault
    volumes:
      - ../../market-analysis:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000

volumes:
  postgres_data:
    driver: local
  vault_data:
    driver: local
  redis_data:
    driver: local

networks:
  multi-repo-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
EOF
fi

# Create individual service compose files
if [ ! -f "docker/compose/postgres.yml" ]; then
    mkdir -p docker/compose
    cat > docker/compose/postgres.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: airflow
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_MULTIPLE_DATABASES: indexagent,market_analysis
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ../../scripts/init-multiple-databases.sh:/docker-entrypoint-initdb.d/init-multiple-databases.sh:ro
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U airflow"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
    driver: local
EOF
fi

echo "✅ Docker configurations created"

# =============================================================================
# TERRAFORM CONFIGURATION
# =============================================================================

echo "🏗️ Setting up Terraform configurations..."

# Create main Terraform configuration
if [ ! -f "terraform/main.tf" ]; then
    cat > terraform/main.tf << 'EOF'
# Multi-Repository Infrastructure Configuration

terraform {
  required_version = ">= 1.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
    vault = {
      source  = "hashicorp/vault"
      version = "~> 3.0"
    }
  }
}

# Docker provider configuration
provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Vault provider configuration
provider "vault" {
  address = var.vault_address
  token   = var.vault_token
}

# Variables
variable "vault_address" {
  description = "Vault server address"
  type        = string
  default     = "http://localhost:8200"
}

variable "vault_token" {
  description = "Vault authentication token"
  type        = string
  default     = "dev-token"
  sensitive   = true
}

# Docker network
resource "docker_network" "multi_repo_network" {
  name = "multi-repo-network"
  ipam_config {
    subnet = "172.20.0.0/16"
  }
}

# PostgreSQL container
resource "docker_container" "postgres" {
  name  = "postgres"
  image = "postgres:15-alpine"
  
  env = [
    "POSTGRES_DB=airflow",
    "POSTGRES_USER=airflow",
    "POSTGRES_PASSWORD=airflow",
    "POSTGRES_MULTIPLE_DATABASES=indexagent,market_analysis"
  ]
  
  ports {
    internal = 5432
    external = 5432
  }
  
  networks_advanced {
    name = docker_network.multi_repo_network.name
  }
  
  volumes {
    volume_name    = docker_volume.postgres_data.name
    container_path = "/var/lib/postgresql/data"
  }
  
  healthcheck {
    test         = ["CMD-SHELL", "pg_isready -U airflow"]
    interval     = "10s"
    timeout      = "5s"
    retries      = 5
    start_period = "30s"
  }
}

# Docker volumes
resource "docker_volume" "postgres_data" {
  name = "postgres_data"
}

resource "docker_volume" "vault_data" {
  name = "vault_data"
}

resource "docker_volume" "redis_data" {
  name = "redis_data"
}

# Vault secrets
resource "vault_mount" "kv" {
  path        = "secret"
  type        = "kv-v2"
  description = "KV Version 2 secret engine mount"
}

resource "vault_kv_secret_v2" "database" {
  mount = vault_mount.kv.path
  name  = "database"
  
  data_json = jsonencode({
    username = "airflow"
    password = "airflow"
    host     = "postgres"
    port     = "5432"
  })
}

# Outputs
output "postgres_connection_string" {
  description = "PostgreSQL connection string"
  value       = "postgresql://airflow:airflow@localhost:5432/airflow"
  sensitive   = true
}

output "vault_address" {
  description = "Vault server address"
  value       = var.vault_address
}
EOF
fi

# Create Terraform variables file
if [ ! -f "terraform/terraform.tfvars.example" ]; then
    cat > terraform/terraform.tfvars.example << 'EOF'
# Terraform Variables Example
# Copy this file to terraform.tfvars and customize

vault_address = "http://localhost:8200"
vault_token   = "dev-token"
EOF
fi

echo "✅ Terraform configurations created"

# =============================================================================
# KUBERNETES CONFIGURATION
# =============================================================================

echo "☸️ Setting up Kubernetes configurations..."

# Create namespace
if [ ! -f "kubernetes/manifests/namespace.yaml" ]; then
    mkdir -p kubernetes/manifests
    cat > kubernetes/manifests/namespace.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: multi-repo-dev
  labels:
    name: multi-repo-dev
    environment: development
EOF
fi

# Create PostgreSQL deployment
if [ ! -f "kubernetes/manifests/postgres.yaml" ]; then
    cat > kubernetes/manifests/postgres.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: multi-repo-dev
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: "airflow"
        - name: POSTGRES_USER
          value: "airflow"
        - name: POSTGRES_PASSWORD
          value: "airflow"
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: multi-repo-dev
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: multi-repo-dev
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF
fi

echo "✅ Kubernetes configurations created"

# =============================================================================
# ANSIBLE CONFIGURATION
# =============================================================================

echo "📋 Setting up Ansible configurations..."

# Create Ansible inventory
if [ ! -f "ansible/inventory/hosts.yml" ]; then
    mkdir -p ansible/inventory
    cat > ansible/inventory/hosts.yml << 'EOF'
all:
  children:
    development:
      hosts:
        localhost:
          ansible_connection: local
          ansible_python_interpreter: "{{ ansible_playbook_python }}"
    services:
      children:
        databases:
          hosts:
            postgres:
              ansible_host: localhost
              ansible_port: 5432
        apis:
          hosts:
            indexagent:
              ansible_host: localhost
              ansible_port: 8081
            market_analysis:
              ansible_host: localhost
              ansible_port: 8000
        workflows:
          hosts:
            airflow:
              ansible_host: localhost
              ansible_port: 8080
EOF
fi

# Create main playbook
if [ ! -f "ansible/playbooks/deploy.yml" ]; then
    mkdir -p ansible/playbooks
    cat > ansible/playbooks/deploy.yml << 'EOF'
---
- name: Deploy Multi-Repository Stack
  hosts: localhost
  gather_facts: yes
  vars:
    project_root: "{{ playbook_dir }}/../.."
    
  tasks:
    - name: Ensure Docker is running
      service:
        name: docker
        state: started
      become: yes
      
    - name: Create Docker network
      docker_network:
        name: multi-repo-network
        ipam_config:
          - subnet: "172.20.0.0/16"
            
    - name: Deploy PostgreSQL
      docker_container:
        name: postgres
        image: postgres:15-alpine
        state: started
        restart_policy: unless-stopped
        env:
          POSTGRES_DB: airflow
          POSTGRES_USER: airflow
          POSTGRES_PASSWORD: airflow
        ports:
          - "5432:5432"
        networks:
          - name: multi-repo-network
        volumes:
          - postgres_data:/var/lib/postgresql/data
          
    - name: Deploy Vault
      docker_container:
        name: vault
        image: hashicorp/vault:1.15
        state: started
        restart_policy: unless-stopped
        env:
          VAULT_DEV_ROOT_TOKEN_ID: dev-token
          VAULT_DEV_LISTEN_ADDRESS: 0.0.0.0:8200
        ports:
          - "8200:8200"
        networks:
          - name: multi-repo-network
        capabilities:
          - IPC_LOCK
        command: vault server -dev -dev-root-token-id=dev-token -dev-listen-address=0.0.0.0:8200
EOF
fi

echo "✅ Ansible configurations created"

# =============================================================================
# SCRIPTS AND UTILITIES
# =============================================================================

echo "📜 Creating utility scripts..."

# Create deployment script
if [ ! -f "scripts/deploy.sh" ]; then
    cat > scripts/deploy.sh << 'EOF'
#!/bin/bash

# Multi-Repository Deployment Script

set -e

echo "🚀 Starting multi-repository deployment..."

# Function to check if a service is running
check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo "✅ $service_name is ready"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start within timeout"
    return 1
}

# Deploy infrastructure services
echo "🏗️ Deploying infrastructure services..."
cd docker
docker-compose up -d postgres vault redis

# Wait for services to be ready
check_service "PostgreSQL" 5432
check_service "Vault" 8200
check_service "Redis" 6379

# Deploy application services
echo "📱 Deploying application services..."
docker-compose up -d airflow-webserver indexagent market-analysis

# Wait for application services
check_service "Airflow" 8080
check_service "IndexAgent" 8081
check_service "Market Analysis" 8000

echo "🎉 Deployment complete!"
echo ""
echo "🌐 Service URLs:"
echo "   - Airflow UI: http://localhost:8080"
echo "   - IndexAgent API: http://localhost:8081"
echo "   - Market Analysis API: http://localhost:8000"
echo "   - Vault UI: http://localhost:8200"
EOF
    chmod +x scripts/deploy.sh
fi

# Create monitoring script
if [ ! -f "scripts/monitoring/health-check.sh" ]; then
    mkdir -p scripts/monitoring
    cat > scripts/monitoring/health-check.sh << 'EOF'
#!/bin/bash

# Health Check Script for Multi-Repository Services

echo "🏥 Performing health checks..."

# Function to check service health
check_health() {
    local service_name=$1
    local url=$2
    
    if curl -s "$url" >/dev/null 2>&1; then
        echo "✅ $service_name: Healthy"
        return 0
    else
        echo "❌ $service_name: Unhealthy"
        return 1
    fi
}

# Check all services
check_health "PostgreSQL" "localhost:5432"
check_health "Vault" "http://localhost:8200/v1/sys/health"
check_health "Redis" "localhost:6379"
check_health "Airflow" "http://localhost:8080/health"
check_health "IndexAgent" "http://localhost:8081/health"
check_health "Market Analysis" "http://localhost:8000/health"

echo ""
echo "📊 Docker container status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
EOF
    chmod +x scripts/monitoring/health-check.sh
fi

echo "✅ Utility scripts created"

# =============================================================================
# MAKEFILE
# =============================================================================

echo "🛠️ Creating Makefile..."

if [ ! -f "Makefile" ]; then
    cat > Makefile << 'EOF'
.PHONY: help deploy destroy status health terraform-init terraform-plan terraform-apply ansible-deploy

help:
	@echo "Available commands:"
	@echo "  deploy         - Deploy the entire stack"
	@echo "  destroy        - Destroy the entire stack"
	@echo "  status         - Show service status"
	@echo "  health         - Run health checks"
	@echo "  terraform-init - Initialize Terraform"
	@echo "  terraform-plan - Plan Terraform changes"
	@echo "  terraform-apply - Apply Terraform changes"
	@echo "  ansible-deploy - Deploy using Ansible"

deploy:
	@echo "🚀 Deploying multi-repository stack..."
	./scripts/deploy.sh

destroy:
	@echo "🔥 Destroying multi-repository stack..."
	cd docker && docker-compose down -v
	docker network rm multi-repo-network || true

status:
	@echo "📊 Service status:"
	cd docker && docker-compose ps

health:
	@echo "🏥 Running health checks..."
	./scripts/monitoring/health-check.sh

terraform-init:
	@echo "🏗️ Initializing Terraform..."
	cd terraform && terraform init

terraform-plan:
	@echo "📋 Planning Terraform changes..."
	cd terraform && terraform plan

terraform-apply:
	@echo "🚀 Applying Terraform changes..."
	cd terraform && terraform apply

ansible-deploy:
	@echo "📋 Deploying with Ansible..."
	cd ansible && ansible-playbook -i inventory/hosts.yml playbooks/deploy.yml
EOF
fi

echo "✅ Makefile created"

# =============================================================================
# COMPLETION
# =============================================================================

echo ""
echo "🎉 Infrastructure setup complete!"
echo ""
echo "📋 Available tools and configurations:"
echo "   - Docker Compose configurations in docker/"
echo "   - Terraform infrastructure in terraform/"
echo "   - Kubernetes manifests in kubernetes/"
echo "   - Ansible playbooks in ansible/"
echo "   - Utility scripts in scripts/"
echo ""
echo "🔧 Quick commands:"
echo "   - 'make help' to see all available commands"
echo "   - 'make deploy' to deploy the entire stack"
echo "   - 'make health' to run health checks"
echo "   - 'make status' to check service status"
echo ""
echo "🚀 Ready for infrastructure orchestration!"