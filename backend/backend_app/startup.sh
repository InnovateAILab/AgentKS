#!/bin/sh
set -e

# startup.sh - run alembic migrations (with simple retry) then launch supervisord

ALCHEMY_CFG=/app/alembic.ini
RETRIES=${ALEMBIC_RETRIES:-10}
SLEEP_SECONDS=${ALEMBIC_SLEEP_SECONDS:-2}

echo "Starting startup.sh: running alembic migrations (retries=${RETRIES})"
attempt=0
while [ "$attempt" -lt "$RETRIES" ]; do
  attempt=$((attempt + 1))
  if alembic -c "$ALCHEMY_CFG" upgrade head; then
    echo "Alembic migration succeeded"
    break
  fi
  echo "Alembic failed (attempt ${attempt}/${RETRIES}), sleeping ${SLEEP_SECONDS}s and retrying..."
  sleep "$SLEEP_SECONDS"
done

if [ "$attempt" -ge "$RETRIES" ]; then
  echo "Alembic failed after ${RETRIES} attempts. Exiting."
  exit 1
fi

echo "Starting supervisord"
exec supervisord -c /app/supervisord.conf
