#!/usr/bin/env bash
set -euo pipefail

case "${1:-api}" in
  api)
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
    ;;
  ui)
    exec streamlit run app/ui.py --server.address 0.0.0.0 --server.port 8501
    ;;
  *)
    exec "$@"
    ;;
esac
