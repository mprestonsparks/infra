#!/usr/bin/env bash
set -e
[ -f .env ] && export $(grep -v '^#' .env | xargs)
for PORT in "$AIRFLOW_PORT" "$VAULT_PORT" "$INDEXAGENT_API_PORT" "$ZOEKT_UI_PORT" "$SOURCEBOT_UI_PORT" "$MARKET_API_PORT"; do
  if lsof -i ":${PORT}" -sTCP:LISTEN -t >/dev/null; then
    echo "Port $PORT in use. Free it or override in infra/.env."
    exit 1
  fi
done
echo "All ports free."