#!/bin/bash

# Ports must match frontend/vite.config.ts's `server.proxy['/api']` target.
HOST="127.0.0.1"
PORT="8010"
FE_PORT="5173"

export PATH="$(pwd)/.venv/bin:$PATH"

if [ ! -d frontend/node_modules ]; then
  echo "Installing frontend dependencies..."
  npm --prefix frontend install
fi

echo "Starting backend on http://$HOST:$PORT"
uv run uvicorn src.api.main:app --host "$HOST" --port "$PORT" --reload --reload-dir src &
BACKEND_PID=$!

echo "Starting frontend on http://127.0.0.1:$FE_PORT"
npm --prefix frontend run dev -- --port "$FE_PORT" &
FRONTEND_PID=$!

cleanup() {
  echo "Shutting down..."
  for PID in $BACKEND_PID $FRONTEND_PID; do
    kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "Done."
}

trap cleanup INT TERM

wait
