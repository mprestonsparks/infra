include .env

.PHONY: check-ports up

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