#!/bin/bash
#
# DEAN System Backup Script
# Performs daily backups of PostgreSQL database and discovered patterns
#

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups/dean}"
POSTGRES_CONTAINER="dean-postgres"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${BACKUP_DIR}/backup_${TIMESTAMP}.log"

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Error handling
handle_error() {
    log "ERROR: Backup failed at line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Start backup process
log "Starting DEAN system backup..."

# 1. Backup PostgreSQL database
log "Backing up PostgreSQL database..."

# Create database backup
DB_BACKUP_FILE="${BACKUP_DIR}/dean_db_${TIMESTAMP}.sql.gz"
docker exec "${POSTGRES_CONTAINER}" pg_dumpall -U postgres | gzip > "${DB_BACKUP_FILE}"

if [ -f "${DB_BACKUP_FILE}" ]; then
    SIZE=$(du -h "${DB_BACKUP_FILE}" | cut -f1)
    log "Database backup completed: ${DB_BACKUP_FILE} (${SIZE})"
else
    log "ERROR: Database backup failed"
    exit 1
fi

# 2. Backup discovered patterns and agent data
log "Backing up patterns and agent data..."

# Export patterns as JSON
PATTERNS_BACKUP_FILE="${BACKUP_DIR}/dean_patterns_${TIMESTAMP}.json.gz"
docker exec "${POSTGRES_CONTAINER}" psql -U postgres -d agent_evolution -t -A -c \
    "SELECT json_agg(row_to_json(t)) FROM (
        SELECT dp.*, a.name as agent_name 
        FROM agent_evolution.discovered_patterns dp 
        JOIN agent_evolution.agents a ON dp.agent_id = a.id
    ) t;" | gzip > "${PATTERNS_BACKUP_FILE}"

if [ -f "${PATTERNS_BACKUP_FILE}" ]; then
    log "Patterns backup completed: ${PATTERNS_BACKUP_FILE}"
else
    log "WARNING: Patterns backup may be empty"
fi

# 3. Backup evolution history
EVOLUTION_BACKUP_FILE="${BACKUP_DIR}/dean_evolution_${TIMESTAMP}.json.gz"
docker exec "${POSTGRES_CONTAINER}" psql -U postgres -d agent_evolution -t -A -c \
    "SELECT json_agg(row_to_json(t)) FROM (
        SELECT eh.*, a.name as agent_name 
        FROM agent_evolution.evolution_history eh 
        JOIN agent_evolution.agents a ON eh.agent_id = a.id
        WHERE eh.timestamp > NOW() - INTERVAL '7 days'
    ) t;" | gzip > "${EVOLUTION_BACKUP_FILE}"

log "Evolution history backup completed: ${EVOLUTION_BACKUP_FILE}"

# 4. Backup Redis data (if needed)
log "Backing up Redis data..."
REDIS_BACKUP_FILE="${BACKUP_DIR}/dean_redis_${TIMESTAMP}.rdb"
docker exec dean-redis redis-cli BGSAVE
sleep 5  # Wait for background save
docker cp dean-redis:/data/dump.rdb "${REDIS_BACKUP_FILE}"
log "Redis backup completed: ${REDIS_BACKUP_FILE}"

# 5. Create backup manifest
MANIFEST_FILE="${BACKUP_DIR}/dean_backup_manifest_${TIMESTAMP}.json"
cat > "${MANIFEST_FILE}" <<EOF
{
    "timestamp": "${TIMESTAMP}",
    "backup_date": "$(date -Iseconds)",
    "files": {
        "database": "$(basename ${DB_BACKUP_FILE})",
        "patterns": "$(basename ${PATTERNS_BACKUP_FILE})",
        "evolution": "$(basename ${EVOLUTION_BACKUP_FILE})",
        "redis": "$(basename ${REDIS_BACKUP_FILE})"
    },
    "sizes": {
        "database": "$(du -h ${DB_BACKUP_FILE} | cut -f1)",
        "patterns": "$(du -h ${PATTERNS_BACKUP_FILE} | cut -f1)",
        "evolution": "$(du -h ${EVOLUTION_BACKUP_FILE} | cut -f1)",
        "redis": "$(du -h ${REDIS_BACKUP_FILE} | cut -f1)"
    }
}
EOF

log "Backup manifest created: ${MANIFEST_FILE}"

# 6. Clean up old backups
log "Cleaning up backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -name "dean_*" -type f -mtime +${RETENTION_DAYS} -delete
DELETED_COUNT=$(find "${BACKUP_DIR}" -name "dean_*" -type f -mtime +${RETENTION_DAYS} | wc -l)
log "Deleted ${DELETED_COUNT} old backup files"

# 7. Verify backup integrity
log "Verifying backup integrity..."
gzip -t "${DB_BACKUP_FILE}" 2>/dev/null && log "Database backup verified" || log "ERROR: Database backup corrupted"
gzip -t "${PATTERNS_BACKUP_FILE}" 2>/dev/null && log "Patterns backup verified" || log "WARNING: Patterns backup may be corrupted"

# Calculate total backup size
TOTAL_SIZE=$(du -sh "${BACKUP_DIR}" | cut -f1)
log "Total backup directory size: ${TOTAL_SIZE}"

# Send notification (placeholder for actual notification system)
# notify_slack "DEAN backup completed successfully"

log "DEAN system backup completed successfully!"

exit 0