# PBS MCP Server

Bridge PBS schedulers with Model Context Protocol (MCP) clients like Claude Desktop by wrapping the existing `pbs-python-api` module in a FastMCP server.

## Getting Started

1. Initialize the submodule and install requirements:
   ```bash
   git submodule update --init --recursive
   pip install -r requirements.txt
   ```
2. Install the PBS MCP server package itself so `python -m pbs_mcp.server` can resolve modules:
   ```bash
   pip install -e .
   ```
3. Export the environment variables expected by the server (if `PBS_ACCOUNT` is omitted it defaults to `datascience`, and `PBS_ROLE` defaults to `user` for privilege-aware tools):
   ```bash
   export PBS_SERVER="aurora-pbs-0001.host"
   export PBS_ACCOUNT="datascience"
   export PBS_ROLE="user"  # or operator/administrator
   ```
4. Run the server via stdio transport:
   ```bash
   python -m pbs_mcp.server
   ```


## Running the OpenAI Bridge Demo

The fastest way to see this server in action is to run the OpenAI bridge script. This script:
1. Launches the PBS MCP server locally.
2. Connects it to OpenAI's API (using your key).
3. Prompts the model to submit (`examples/hello_world_script.sh`) and monitor a job on your behalf.

### Prerequisites
- `pip install openai`
- `export OPENAI_API_KEY=sk-...`

### Run the Demo
```bash
python examples/openai_mcp_bridge.py --prompt "Submit a job to PBS using this submit script: $PWD/examples/hello_world_script.sh. Using 'home' for the filesystems input."
```

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

```


## Development

- Install dev extras with `pip install -e .[dev]`.
- Run tests using `pytest`.
- Use the sample Claude Desktop config (`config/claude_desktop_config.json`) to wire the server into an MCP client.

## Resources

The MCP server now exposes the following resources:

- `pbs://job/{job_id}` – detailed job breakdown
- `pbs://queue/{queue}` – queue configuration snapshot
- `pbs://system/status` – server, scheduler, queue, and job summary
- `pbs://user/jobs` – current user's jobs
- `pbs://system/privileges` – current privilege role and required levels

## Prompts

The prompt templates provide reusable guidance for common workflows:

- `submit_job_guide(job_type=\"python\", queue=\"debug\")` – tailored submission checklist with script template snippets.
- `diagnose_job_failure(job_id)` – conversation scaffold for investigating failing jobs using MCP tools/resources.
- `optimize_resources(workload_description)` – structured plan for right-sizing resource requests.

The current code contains placeholders so implementation can safely iterate while the structure, packaging, and testing scaffolding are ready.
