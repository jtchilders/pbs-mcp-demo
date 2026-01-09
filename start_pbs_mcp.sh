#!/bin/bash
# Startup script for PBS MCP server
# This script can be used with Claude Code's mcpServers configuration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the local virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Source the PBS environment setup (defaults to aurora)
source "$SCRIPT_DIR/external/pbs-python-api/setup_env.sh" "${PBS_SYSTEM:-aurora}"

# Launch the PBS MCP server
exec python -m pbs_mcp
