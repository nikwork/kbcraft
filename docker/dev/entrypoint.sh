#!/bin/bash
# kbcraft devcontainer entrypoint
# 1. Installs kbcraft in editable mode from the mounted source tree
# 2. Generates SSH host keys if missing (first start only)
# 3. Starts sshd in the foreground
set -e

# ── 1. Editable install ───────────────────────────────────────────────────────
# The project root is mounted at /app — install in editable mode so every
# source change on the host is immediately visible inside the container.
if [ -f "/app/pyproject.toml" ]; then
    echo "[entrypoint] Installing kbcraft in editable mode …"
    cd /app
    poetry install --with dev --no-interaction --no-ansi 2>&1 | tail -5
else
    echo "[entrypoint] WARNING: /app/pyproject.toml not found — is the project mounted at /app?"
fi

# ── 2. SSH host keys ──────────────────────────────────────────────────────────
ssh-keygen -A -q   # no-op if keys already exist in /etc/ssh/

# ── 3. Start SSH daemon ───────────────────────────────────────────────────────
echo "[entrypoint] sshd ready — connect with:  ssh root@localhost -p 2222"
exec /usr/sbin/sshd -D -e
