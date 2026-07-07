#!/usr/bin/env bash
set -euo pipefail

# Factory reset: destroys ALL application state.
#
# The wiki (pages, audit log, users, orgs, jobs, graph, embeddings) lives
# entirely in PostgreSQL now — there is no data/wiki/ directory anymore.
# Raw uploaded source files live in the `data` volume (or S3). This script
# tears both down by removing the Docker volumes, so the next
# `docker compose up` starts from a clean slate: migrations re-run and the
# default org + admin user are re-seeded from AUTH_USERNAME / AUTH_PASSWORD.

read -r -p "This will PERMANENTLY delete ALL wiki data, users, and uploaded files. Continue? [y/N] " reply
case "$reply" in
  [yY] | [yY][eE][sS]) ;;
  *) echo "Aborted."; exit 1 ;;
esac

echo "Factory reset: stopping containers and removing data volumes..."
docker compose down -v

echo "Done. Run 'docker compose up -d' to start fresh (migrations + default admin re-seed on boot)."
