#!/bin/bash
# DEAN System Database Initialization Script
# Ensures the database is properly set up with all required schemas and permissions

set -e

echo "=== DEAN System Database Initialization ==="

# Database connection parameters
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-agent_evolution}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"

# Export for psql
export PGPASSWORD=$DB_PASSWORD

echo "Checking PostgreSQL connection..."
until psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c '\q' 2>/dev/null; do
  echo "Waiting for PostgreSQL to be ready..."
  sleep 2
done

echo "PostgreSQL is ready!"

# Create database if it doesn't exist
echo "Creating database $DB_NAME if it doesn't exist..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_NAME"

# Create Airflow database if it doesn't exist
echo "Creating Airflow database if it doesn't exist..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -tc "SELECT 1 FROM pg_database WHERE datname = 'airflow'" | grep -q 1 || \
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE airflow"

# Run the initialization SQL
echo "Running database initialization script..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /app/database/init_agent_evolution.sql

# Create Airflow user if needed
echo "Creating Airflow database user..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d airflow <<EOF
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'airflow') THEN
        CREATE USER airflow WITH PASSWORD 'airflow';
    END IF;
END
\$\$;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
GRANT ALL PRIVILEGES ON SCHEMA public TO airflow;
EOF

# Verify setup
echo "Verifying database setup..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<EOF
-- Check schema exists
SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'agent_evolution';

-- Check tables exist
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'agent_evolution' 
ORDER BY table_name;

-- Check user permissions
SELECT grantee, privilege_type 
FROM information_schema.role_table_grants 
WHERE table_schema = 'agent_evolution' 
AND grantee = 'dean_api'
LIMIT 5;
EOF

echo "=== Database initialization complete! ==="