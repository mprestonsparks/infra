include .env

.PHONY: check-ports up down status logs clean validate deploy-integrated deploy-standalone deploy-hybrid

# Basic commands
check-ports:
	@bash scripts/check-ports.sh

up: check-ports
	DOCKER_DEFAULT_PLATFORM=${DOCKER_DEFAULT_PLATFORM} \
	docker compose \
	-f docker-compose.yml \
	-f ../airflow-hub/docker-compose.yml \
	-f ../IndexAgent/config/docker-compose.yml \
	-f ../market-analysis/docker-compose.yml \
	up -d

down:
	docker compose down

status:
	docker compose ps

logs:
	docker compose logs -f --tail=100

clean:
	docker compose down -v
	docker system prune -f

# Validation
validate:
	@pwsh scripts/validate_deployment.ps1 -Environment production

validate-fix:
	@pwsh scripts/validate_deployment.ps1 -Environment production -AutoFix

# DEAN Integration Commands
deploy-integrated:
	@bash scripts/deploy_unified.sh --mode integrated --env production

deploy-standalone:
	@bash scripts/deploy_unified.sh --mode standalone --env production

deploy-hybrid:
	@bash scripts/deploy_unified.sh --mode hybrid --env production

# DEAN-specific commands
dean-up:
	docker compose -f docker-compose.dean.yml up -d

dean-down:
	docker compose -f docker-compose.dean.yml down

dean-logs:
	docker compose -f docker-compose.dean.yml logs -f --tail=100

dean-status:
	docker compose -f docker-compose.dean.yml ps

# Database management
db-backup:
	@bash scripts/backup_dean_data.sh

db-init:
	@bash scripts/init_database.sh

# Health checks
health-check:
	@echo "Checking service health..."
	@curl -sf http://localhost:8090/health || echo "DEAN Evolution API: DOWN"
	@curl -sf http://localhost:8091/health || echo "DEAN API: DOWN"
	@curl -sf http://localhost:8081/health || echo "IndexAgent: DOWN"
	@curl -sf http://localhost:8080/health || echo "Airflow: DOWN"

# Help
help:
	@echo "Multi-Repository Infrastructure Management"
	@echo ""
	@echo "Basic Commands:"
	@echo "  make up              - Start all infrastructure services"
	@echo "  make down            - Stop all services"
	@echo "  make status          - Show service status"
	@echo "  make logs            - Follow service logs"
	@echo "  make clean           - Remove all containers and volumes"
	@echo ""
	@echo "Validation:"
	@echo "  make validate        - Run deployment validation"
	@echo "  make validate-fix    - Run validation with auto-fix"
	@echo ""
	@echo "DEAN Deployment:"
	@echo "  make deploy-integrated - Deploy DEAN with shared infrastructure"
	@echo "  make deploy-standalone - Deploy DEAN independently"
	@echo "  make deploy-hybrid     - Deploy DEAN with infra services"
	@echo ""
	@echo "DEAN Management:"
	@echo "  make dean-up         - Start DEAN services only"
	@echo "  make dean-down       - Stop DEAN services only"
	@echo "  make dean-logs       - View DEAN logs"
	@echo "  make dean-status     - Show DEAN service status"
	@echo ""
	@echo "Database:"
	@echo "  make db-backup       - Backup all databases"
	@echo "  make db-init         - Initialize databases"
	@echo ""
	@echo "Health:"
	@echo "  make health-check    - Check all service health endpoints"