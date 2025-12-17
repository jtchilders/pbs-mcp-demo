# PBS MCP Server

Bridge PBS schedulers with Model Context Protocol (MCP) clients like Claude Desktop by wrapping the existing `pbs_api.py` module in a FastMCP server.

## Project Layout

```
pbs-mcp-demo/
├── src/pbs_mcp/        # Package exposing server, tools, resources, prompts
├── tests/              # pytest-based unit and integration scaffolding
├── examples/           # Small runnable snippets demonstrating future workflows
├── config/             # Sample MCP client configuration files
├── pyproject.toml      # Modern Python package configuration
├── setup.py            # Legacy entry point for editable installs
├── requirements.txt    # Runtime dependencies (PBS API added locally)
└── PLAN.md             # Detailed development roadmap
```

## Getting Started

1. Create/activate a Python 3.8+ environment and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Install the PBS MCP server package itself so `python -m pbs_mcp.server` can resolve modules:
   ```bash
   pip install -e .
   ```
3. Install your existing PBS API implementation so `from pbs_api import PBSClient` works. For local development you can install it editable, e.g.:
   ```bash
   pip install -e ../pbs-python-api
   ```
4. Export the environment variables expected by the server (if `PBS_ACCOUNT` is omitted it defaults to `datascience`, and `PBS_ROLE` defaults to `user` for privilege-aware tools):
   ```bash
   export PBS_SERVER="aurora-pbs-0001.host"
   export PBS_ACCOUNT="datascience"
   export PBS_ROLE="user"  # or operator/administrator
   ```
5. Run the server via stdio transport:
   ```bash
   python -m pbs_mcp.server
   ```

## Development

- Install dev extras with `pip install -e .[dev]`.
- Run tests using `pytest`.
- Use the sample Claude Desktop config (`config/claude_desktop_config.json`) to wire the server into an MCP client.

## Resources

The MCP server now exposes the Phase 4 resources described in `PLAN.md`, including:

- `pbs://job/{job_id}` – detailed job breakdown
- `pbs://queue/{queue}` – queue configuration snapshot
- `pbs://system/status` – server, scheduler, queue, and job summary
- `pbs://user/jobs` – current user's jobs
- `pbs://system/privileges` – current privilege role and required levels

## Prompts

The Phase 5 prompt templates provide reusable guidance for common workflows:

- `submit_job_guide(job_type=\"python\", queue=\"debug\")` – tailored submission checklist with script template snippets.
- `diagnose_job_failure(job_id)` – conversation scaffold for investigating failing jobs using MCP tools/resources.
- `optimize_resources(workload_description)` – structured plan for right-sizing resource requests.

The current code contains placeholders so Phase 2+ implementation can safely iterate while the structure, packaging, and testing scaffolding are ready.
