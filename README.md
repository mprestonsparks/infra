# Infrastructure Orchestration Repository

A comprehensive infrastructure orchestration layer that provides unified Docker-in-Docker capabilities, service deployment automation, and development environment management for the multi-repository workspace consisting of IndexAgent, airflow-hub, market-analysis, and supporting services.

## Overview

This repository serves as the infrastructure backbone for the entire development ecosystem, providing:

- **Unified Service Orchestration**: Single command deployment of all services
- **Docker-in-Docker Support**: Containerized infrastructure management
- **Environment Configuration**: Centralized configuration management
- **Service Discovery**: Automated service registration and discovery
- **Development Tooling**: Infrastructure automation and deployment scripts

## Repository Structure

```
infra/
├── docker-compose.yml                    # Main orchestration file
├── Makefile                             # Infrastructure automation commands
├── README.md                            # This documentation
├── docker-parameterization-deliverable.txt  # Docker configuration guide
├── scripts/                             # Infrastructure automation scripts
│   ├── deploy.sh                        # Service deployment automation
│   ├── backup.sh                        # Data backup procedures
│   ├── restore.sh                       # Data restoration procedures
│   ├── health-check.sh                  # Service health monitoring
│   ├── log-aggregation.sh               # Centralized logging setup
│   └── security/                        # Security configuration scripts
│       ├── setup-vault.sh               # Vault initialization
│       ├── generate-certs.sh            # SSL certificate generation
│       └── rotate-secrets.sh            # Secret rotation automation
├── config/                              # Configuration templates
│   ├── nginx/                           # Reverse proxy configuration
│   ├── prometheus/                      # Monitoring configuration
│   ├── grafana/                         # Dashboard configuration
│   └── vault/                           # Vault policies and configuration
├── terraform/                           # Infrastructure as Code
│   ├── main.tf                          # Main Terraform configuration
│   ├── variables.tf                     # Variable definitions
│   ├── outputs.tf                       # Output definitions
│   └── modules/                         # Reusable Terraform modules
├── kubernetes/                          # Kubernetes manifests
│   ├── namespace.yaml                   # Namespace definitions
│   ├── deployments/                     # Application deployments
│   ├── services/                        # Service definitions
│   ├── ingress/                         # Ingress configurations
│   └── monitoring/                      # Monitoring stack
└── ansible/                             # Configuration management
    ├── playbooks/                       # Ansible playbooks
    ├── roles/                           # Reusable roles
    └── inventory/                       # Environment inventories
```

## Development Environment Setup

### Option 1: Multi-Repository Dev Container (Recommended)

For integrated infrastructure development across all repositories:

1. **Prerequisites:**
   - [VSCode](https://code.visualstudio.com/) with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/) with Docker-in-Docker support

2. **Setup:**
   ```bash
   # Ensure all repositories are cloned as siblings
   ~/Documents/gitRepos/
     ├── airflow-hub/
     ├── IndexAgent/
     ├── market-analysis/
     └── infra/
   
   # Open the parent directory in VSCode
   code ~/Documents/gitRepos
   
   # Select "Reopen in Container" and choose the workspace configuration
   ```

3. **Benefits:**
   - Docker-in-Docker capabilities for infrastructure testing
   - Terraform, Kubernetes, and Ansible tooling pre-installed
   - Shared volume access across all repositories
   - Infrastructure automation and deployment tools
   - Service orchestration and monitoring capabilities

### Option 2: Standalone Infrastructure Development

For infrastructure-only development:

1. **Local Setup:**
   ```bash
   git clone https://github.com/mprestonsparks/infra.git
   cd infra
   ```

2. **Prerequisites:**
   ```bash
   # Install required tools
   # Docker and Docker Compose
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   
   # Terraform
   wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
   unzip terraform_1.6.0_linux_amd64.zip
   sudo mv terraform /usr/local/bin/
   
   # Kubernetes CLI
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
   
   # Ansible
   pip install ansible
   ```

## Service Architecture and Communication

### Port Allocation Strategy

The infrastructure manages the following port allocations to prevent conflicts:

| Service | Port | Protocol | Description | Access URL |
|---------|------|----------|-------------|------------|
| **Core Services** |
| PostgreSQL | 5432 | TCP | Shared database | Internal only |
| Vault | 8200 | HTTP/HTTPS | Secrets management | http://localhost:8200 |
| Redis | 6379 | TCP | Caching and queues | Internal only |
| **Application Services** |
| Airflow UI | 8080 | HTTP | Workflow management | http://localhost:8080 |
| IndexAgent API | 8081 | HTTP | Code indexing service | http://localhost:8081 |
| Market Analysis API | 8000 | HTTP | Financial data API | http://localhost:8000 |
| Zoekt UI | 6070 | HTTP | Code search interface | http://localhost:6070 |
| Sourcebot UI | 3000 | HTTP | Source code assistant | http://localhost:3000 |
| **Infrastructure Services** |
| Nginx Proxy | 80/443 | HTTP/HTTPS | Reverse proxy | http://localhost |
| Prometheus | 9090 | HTTP | Metrics collection | http://localhost:9090 |
| Grafana | 3001 | HTTP | Monitoring dashboards | http://localhost:3001 |
| Jaeger | 16686 | HTTP | Distributed tracing | http://localhost:16686 |

### Service Discovery and Communication

```yaml
# Example service discovery configuration
version: '3.8'
services:
  nginx-proxy:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - airflow-webserver
      - indexagent
      - market-analysis
    networks:
      - infrastructure

  # Service mesh configuration
  consul:
    image: consul:latest
    ports:
      - "8500:8500"
    environment:
      - CONSUL_BIND_INTERFACE=eth0
    networks:
      - infrastructure
```

## Docker-in-Docker Usage and Best Practices

### Container Architecture

The infrastructure repository leverages Docker-in-Docker (DinD) for:

- **Isolated Testing**: Test infrastructure changes without affecting host
- **CI/CD Pipelines**: Build and deploy containers within containers
- **Development Environments**: Consistent development across platforms
- **Service Orchestration**: Manage complex multi-container applications

### DinD Configuration

```dockerfile
# Example DinD setup in workspace
FROM docker:dind

# Install additional tools
RUN apk add --no-cache \
    python3 \
    py3-pip \
    terraform \
    kubectl \
    ansible

# Configure Docker daemon
COPY config/docker/daemon.json /etc/docker/daemon.json

# Setup Docker Compose
RUN pip3 install docker-compose

# Configure service discovery
COPY config/docker/docker-compose.override.yml /docker-compose.override.yml
```

### Best Practices

**Security:**
```bash
# Use rootless Docker when possible
dockerd-rootless-setuptool.sh install

# Implement proper network segmentation
docker network create --driver bridge infrastructure
docker network create --driver bridge application
docker network create --driver bridge monitoring
```

**Performance:**
```bash
# Optimize Docker daemon configuration
cat > /etc/docker/daemon.json << EOF
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-address-pools": [
    {
      "base": "172.17.0.0/16",
      "size": 24
    }
  ]
}
EOF
```

**Resource Management:**
```bash
# Set resource limits
docker run --memory="2g" --cpus="1.5" --name service-name image-name

# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## Service Deployment Procedures

### Unified Deployment

```bash
# Deploy all services
make deploy

# Deploy specific environment
make deploy ENV=production

# Deploy with custom configuration
make deploy CONFIG=custom-config.yml
```

### Individual Service Deployment

```bash
# Deploy core infrastructure
make deploy-core

# Deploy application services
make deploy-apps

# Deploy monitoring stack
make deploy-monitoring
```

### Rolling Updates

```bash
# Perform rolling update
./scripts/deploy.sh --rolling-update --service=indexagent

# Blue-green deployment
./scripts/deploy.sh --blue-green --service=market-analysis

# Canary deployment
./scripts/deploy.sh --canary --service=airflow-hub --percentage=10
```

## Environment Configuration Management

### Configuration Hierarchy

```
config/
├── base/                    # Base configuration for all environments
│   ├── docker-compose.yml
│   ├── .env.base
│   └── services/
├── development/             # Development-specific overrides
│   ├── docker-compose.override.yml
│   ├── .env.development
│   └── services/
├── staging/                 # Staging environment configuration
│   ├── docker-compose.override.yml
│   ├── .env.staging
│   └── services/
└── production/              # Production configuration
    ├── docker-compose.override.yml
    ├── .env.production
    └── services/
```

### Environment Variables Management

```bash
# Development environment
export ENVIRONMENT=development
export LOG_LEVEL=debug
export DATABASE_URL=postgresql://airflow:airflow@postgres:5432/airflow
export VAULT_ADDR=http://vault:8200
export VAULT_TOKEN=dev-token

# Production environment
export ENVIRONMENT=production
export LOG_LEVEL=info
export DATABASE_URL=postgresql://user:pass@prod-db:5432/airflow
export VAULT_ADDR=https://vault.company.com:8200
export VAULT_TOKEN_FILE=/run/secrets/vault-token
```

### Secrets Management

```bash
# Initialize Vault
./scripts/security/setup-vault.sh

# Store secrets
vault kv put secret/database \
  username=airflow \
  password=secure_password \
  host=postgres \
  port=5432

vault kv put secret/api-keys \
  binance_api_key=your_api_key \
  binance_secret_key=your_secret_key

# Rotate secrets
./scripts/security/rotate-secrets.sh --service=database
```

## Terraform Usage Examples

### Infrastructure Provisioning

```hcl
# main.tf - Core infrastructure
terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 2.0"
    }
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Network configuration
resource "docker_network" "infrastructure" {
  name = "infrastructure"
  driver = "bridge"
  ipam_config {
    subnet = "172.20.0.0/16"
  }
}

# PostgreSQL database
resource "docker_container" "postgres" {
  name  = "postgres"
  image = "postgres:15"
  
  env = [
    "POSTGRES_USER=airflow",
    "POSTGRES_PASSWORD=airflow",
    "POSTGRES_DB=airflow"
  ]
  
  ports {
    internal = 5432
    external = 5432
  }
  
  networks_advanced {
    name = docker_network.infrastructure.name
  }
  
  volumes {
    host_path      = "/data/postgres"
    container_path = "/var/lib/postgresql/data"
  }
}
```

### Service Modules

```hcl
# modules/service/main.tf
variable "service_name" {
  description = "Name of the service"
  type        = string
}

variable "image" {
  description = "Docker image for the service"
  type        = string
}

variable "ports" {
  description = "Port mappings"
  type = list(object({
    internal = number
    external = number
  }))
}

resource "docker_container" "service" {
  name  = var.service_name
  image = var.image
  
  dynamic "ports" {
    for_each = var.ports
    content {
      internal = ports.value.internal
      external = ports.value.external
    }
  }
  
  networks_advanced {
    name = "infrastructure"
  }
}
```

### Deployment Commands

```bash
# Initialize Terraform
terraform init

# Plan infrastructure changes
terraform plan -var-file="environments/development.tfvars"

# Apply changes
terraform apply -var-file="environments/development.tfvars"

# Destroy infrastructure
terraform destroy -var-file="environments/development.tfvars"
```

## Kubernetes Integration

### Namespace Configuration

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: multi-repo-workspace
  labels:
    name: multi-repo-workspace
    environment: development
```

### Service Deployments

```yaml
# kubernetes/deployments/indexagent.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: indexagent
  namespace: multi-repo-workspace
spec:
  replicas: 2
  selector:
    matchLabels:
      app: indexagent
  template:
    metadata:
      labels:
        app: indexagent
    spec:
      containers:
      - name: indexagent
        image: indexagent:latest
        ports:
        - containerPort: 8081
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Service Discovery

```yaml
# kubernetes/services/indexagent.yaml
apiVersion: v1
kind: Service
metadata:
  name: indexagent-service
  namespace: multi-repo-workspace
spec:
  selector:
    app: indexagent
  ports:
  - protocol: TCP
    port: 8081
    targetPort: 8081
  type: ClusterIP
```

### Ingress Configuration

```yaml
# kubernetes/ingress/main.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: multi-repo-ingress
  namespace: multi-repo-workspace
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: localhost
    http:
      paths:
      - path: /airflow
        pathType: Prefix
        backend:
          service:
            name: airflow-service
            port:
              number: 8080
      - path: /indexagent
        pathType: Prefix
        backend:
          service:
            name: indexagent-service
            port:
              number: 8081
      - path: /market
        pathType: Prefix
        backend:
          service:
            name: market-analysis-service
            port:
              number: 8000
```

## Ansible Configuration Management

### Playbook Structure

```yaml
# ansible/playbooks/deploy-services.yml
---
- name: Deploy Multi-Repository Services
  hosts: localhost
  connection: local
  vars:
    environment: "{{ env | default('development') }}"
    
  tasks:
    - name: Create Docker networks
      docker_network:
        name: "{{ item }}"
        driver: bridge
      loop:
        - infrastructure
        - application
        - monitoring
    
    - name: Deploy PostgreSQL
      docker_container:
        name: postgres
        image: postgres:15
        state: started
        restart_policy: unless-stopped
        env:
          POSTGRES_USER: airflow
          POSTGRES_PASSWORD: airflow
          POSTGRES_DB: airflow
        ports:
          - "5432:5432"
        networks:
          - name: infrastructure
        volumes:
          - postgres_data:/var/lib/postgresql/data
    
    - name: Deploy Vault
      docker_container:
        name: vault
        image: vault:latest
        state: started
        restart_policy: unless-stopped
        env:
          VAULT_DEV_ROOT_TOKEN_ID: dev-token
          VAULT_DEV_LISTEN_ADDRESS: 0.0.0.0:8200
        ports:
          - "8200:8200"
        networks:
          - name: infrastructure
        cap_add:
          - IPC_LOCK
```

### Role-Based Configuration

```yaml
# ansible/roles/database/tasks/main.yml
---
- name: Create database directories
  file:
    path: "{{ item }}"
    state: directory
    mode: '0755'
  loop:
    - /data/postgres
    - /data/backups

- name: Deploy PostgreSQL container
  docker_container:
    name: postgres
    image: "postgres:{{ postgres_version }}"
    state: started
    restart_policy: unless-stopped
    env: "{{ postgres_env }}"
    ports: "{{ postgres_ports }}"
    volumes: "{{ postgres_volumes }}"
    networks: "{{ postgres_networks }}"

- name: Initialize databases
  postgresql_db:
    name: "{{ item }}"
    login_host: localhost
    login_user: "{{ postgres_user }}"
    login_password: "{{ postgres_password }}"
  loop: "{{ databases }}"
```

### Inventory Management

```ini
# ansible/inventory/development
[local]
localhost ansible_connection=local

[local:vars]
environment=development
postgres_version=15
postgres_user=airflow
postgres_password=airflow
databases=['airflow', 'indexagent', 'market_analysis']
```

## Troubleshooting Guide

### Service Health Monitoring

```bash
# Check all service status
./scripts/health-check.sh

# Check specific service
docker ps --filter "name=postgres" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# View service logs
docker logs postgres --tail 50 --follow

# Check resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
```

### Common Infrastructure Issues

**Docker Daemon Issues:**
```bash
# Restart Docker daemon
sudo systemctl restart docker

# Check Docker daemon status
sudo systemctl status docker

# View Docker daemon logs
sudo journalctl -u docker.service --since "1 hour ago"
```

**Network Connectivity:**
```bash
# Test network connectivity between containers
docker exec postgres ping -c 3 vault
docker exec indexagent curl -f http://postgres:5432

# Check network configuration
docker network ls
docker network inspect infrastructure
```

**Volume Mount Issues:**
```bash
# Check volume mounts
docker volume ls
docker volume inspect postgres_data

# Fix permission issues
sudo chown -R 999:999 /data/postgres
sudo chmod -R 755 /data
```

**Port Conflicts:**
```bash
# Check port usage
netstat -tulpn | grep :8080
lsof -i :8080

# Kill processes using conflicting ports
sudo kill -9 $(lsof -t -i:8080)
```

### Performance Troubleshooting

**Container Performance:**
```bash
# Monitor container performance
docker stats --no-stream

# Check container resource limits
docker inspect postgres | grep -A 10 "Resources"

# Optimize container performance
docker update --memory=2g --cpus=1.5 postgres
```

**Disk Space Issues:**
```bash
# Check Docker disk usage
docker system df

# Clean up unused resources
docker system prune -a --volumes

# Remove unused images
docker image prune -a
```

**Memory Issues:**
```bash
# Check system memory
free -h

# Monitor container memory usage
docker stats --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Adjust container memory limits
docker update --memory=1g container_name
```

### Log Analysis

```bash
# Centralized log collection
./scripts/log-aggregation.sh

# Search logs across all services
grep -r "ERROR" /logs/

# Monitor real-time logs
tail -f /logs/*/application.log

# Log rotation and cleanup
./scripts/log-rotation.sh
```

## Best Practices

### Security Best Practices

**Container Security:**
```bash
# Run containers as non-root user
docker run --user 1000:1000 image_name

# Use read-only root filesystem
docker run --read-only --tmpfs /tmp image_name

# Limit container capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE image_name
```

**Network Security:**
```bash
# Create isolated networks
docker network create --driver bridge --internal secure_network

# Use encrypted communication
docker run -e TLS_ENABLED=true -v /certs:/certs image_name
```

**Secrets Management:**
```bash
# Use Docker secrets
echo "secret_value" | docker secret create db_password -

# Mount secrets as files
docker run --secret db_password image_name
```

### Performance Best Practices

**Resource Optimization:**
```bash
# Set appropriate resource limits
docker run --memory=512m --cpus=0.5 image_name

# Use multi-stage builds
FROM node:16 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:16-alpine
COPY --from=builder /app/node_modules ./node_modules
```

**Caching Strategies:**
```bash
# Use build cache effectively
docker build --cache-from=image_name:latest .

# Implement layer caching
COPY package*.json ./
RUN npm install
COPY . .
```

### Monitoring and Observability

**Metrics Collection:**
```yaml
# Prometheus configuration
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'docker-containers'
    static_configs:
      - targets: ['localhost:9323']
  
  - job_name: 'application-metrics'
    static_configs:
      - targets: ['indexagent:8081', 'market-analysis:8000']
```

**Distributed Tracing:**
```yaml
# Jaeger configuration
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411
```

## Contributing

### Infrastructure Development Guidelines

1. **Code Standards:**
   - Use Infrastructure as Code (IaC) principles
   - Version control all configuration files
   - Implement proper testing for infrastructure changes
   - Document all configuration options

2. **Security Requirements:**
   - Follow security best practices for container deployment
   - Implement proper secrets management
   - Use network segmentation and access controls
   - Regular security audits and updates

3. **Testing Requirements:**
   - Test infrastructure changes in isolated environments
   - Implement automated testing for deployment scripts
   - Validate service connectivity and health checks
   - Performance testing for resource allocation

4. **Documentation:**
   - Update documentation for configuration changes
   - Include troubleshooting guides for new services
   - Document deployment procedures and rollback plans
   - Maintain architecture diagrams and service maps

### Pull Request Process

1. Create feature branch for infrastructure changes
2. Test changes in development environment
3. Update documentation and configuration examples
4. Submit pull request with detailed change description
5. Address review feedback and security concerns
6. Merge after approval and successful testing

## License

This project is the private property of M. Preston Sparks. All rights reserved.