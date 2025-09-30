#!/usr/bin/env bash
set -euo pipefail
streamlit run "$(dirname "$0")/../app/dashboard.py"

