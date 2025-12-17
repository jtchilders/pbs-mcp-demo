# PBS MCP Server Development Plan

## Project Overview

Build a Model Context Protocol (MCP) server that exposes PBS (Portable Batch System) scheduler functionality to AI assistants like Claude. This will enable AI tools to interact with HPC job scheduling systems through a standardized interface, wrapping your existing `pbs_api.py` Python API.

## Architecture Overview

```
┌─────────────────┐
│  MCP Client     │
│  (Claude/AI)    │
└────────┬────────┘
         │ MCP Protocol (JSON-RPC 2.0)
         │
┌────────▼────────┐
│  MCP Server     │
│  (FastMCP)      │
└────────┬────────┘
         │ Python API
         │
┌────────▼────────┐
│  pbs_api.py     │
│  (PBSClient)    │
└────────┬────────┘
         │ PBS IFL
         │
┌────────▼────────┐
│  PBS Scheduler  │
│  (Aurora/etc)   │
└─────────────────┘
```

## Technology Stack

- **MCP Framework**: FastMCP (official Python MCP SDK)
- **PBS Interface**: Your existing `pbs_api.py` with `PBSClient` class
- **Python Version**: 3.8+ (match your current environment)
- **Transport**: stdio (for local execution) and HTTP/SSE (for remote deployment)
- **Dependencies**: `mcp`, your PBS Python API, standard library

## Phase 1: Project Setup and Structure

### 1.1 Directory Structure

```
pbs-mcp-server/
├── README.md
├── pyproject.toml          # Project metadata and dependencies
├── setup.py                # Alternative setup for pip install
├── requirements.txt        # Dependencies list
├── .gitignore
├── src/
│   └── pbs_mcp/
│       ├── __init__.py
│       ├── server.py       # Main MCP server implementation
│       ├── tools.py        # PBS tool definitions
│       ├── resources.py    # PBS resource exposures
│       ├── prompts.py      # PBS prompt templates
│       └── utils.py        # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_tools.py
│   ├── test_resources.py
│   └── test_integration.py
├── examples/
│   ├── simple_submit.py
│   └── job_monitoring.py
└── config/
    └── claude_desktop_config.json  # Example Claude Desktop config
```

### 1.2 Dependencies to Install

```toml
[project]
name = "pbs-mcp-server"
version = "0.1.0"
dependencies = [
    "mcp>=1.0.0",
    # Your PBS API is local, so add path or package reference
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black",
    "mypy",
]
```

### 1.3 Key Configuration Considerations

- **PBS Server Connection**: Read from environment variables (PBS_SERVER, PBS_ACCOUNT)
- **Authentication**: Leverage system credentials (Kerberos, SSH keys)
- **Error Handling**: Comprehensive error messages for PBS operations
- **Logging**: Structured logging for debugging MCP interactions

## Phase 2: Core MCP Server Implementation

### 2.1 Main Server Setup (`src/pbs_mcp/server.py`)

**Key Implementation Points:**

```python
from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager
import os

# Import your PBS API
from pbs_api import PBSClient, PBSException

@asynccontextmanager
async def lifespan(server: FastMCP):
    """Manage PBS connection lifecycle"""
    # Initialize PBS client connection
    pbs_server = os.environ.get('PBS_SERVER')
    context = {
        'pbs_server': pbs_server,
        'pbs_account': os.environ.get('PBS_ACCOUNT', 'datascience')
    }
    yield context
    # Cleanup if needed

mcp = FastMCP(
    "PBS Scheduler",
    dependencies=["pbs-python-api"],  # Adjust based on your API packaging
    lifespan=lifespan
)
```

**Design Considerations:**

- Use context managers for PBS connections within each tool
- Don't maintain persistent PBS connections (they're lightweight)
- Handle PBS connection failures gracefully
- Provide clear error messages about missing environment variables

### 2.2 Privilege Management

**Important Note:** Many PBS operations require Operator or Administrator privileges. Your MCP server should:

1. Check privileges before operations that require them
2. Return clear error messages when privileges are insufficient
3. Document which tools require elevated privileges
4. Consider implementing a `check_privileges` resource

## Phase 3: Tool Definitions (`src/pbs_mcp/tools.py`)

### 3.1 Job Submission Tools

**Tool: `submit_job`**

```python
@mcp.tool()
def submit_job(
    script_path: str,
    queue: str = "debug",
    job_name: str = "",
    walltime: str = "00:10:00",
    node_count: int = 1,
    account: str = ""
) -> dict:
    """
    Submit a PBS job to the scheduler.
    
    Args:
        script_path: Path to the job script file
        queue: PBS queue name (default: debug)
        job_name: Name for the job
        walltime: Requested walltime (HH:MM:SS format)
        node_count: Number of nodes to request
        account: PBS account/project name (uses PBS_ACCOUNT env if not provided)
        
    Returns:
        dict with job_id, submission_status, and any errors
    """
    # Implementation using PBSClient context manager
    # Handle errors and return structured results
```

**Key Implementation Details:**
- Validate script_path exists before submission
- Build proper PBS attributes dictionary from parameters
- Use PBSClient context manager for connection
- Return job_id on success, detailed error on failure
- Consider adding support for additional Resource_List parameters

### 3.2 Job Query Tools

**Tool: `get_job_status`**

```python
@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Query detailed status of a specific PBS job.
    
    Args:
        job_id: PBS job identifier (e.g., '12345.aurora-pbs-01')
        
    Returns:
        dict containing job attributes: state, queue, resources, etc.
    """
```

**Tool: `list_jobs`**

```python
@mcp.tool()
def list_jobs(
    state_filter: str = "all",
    user_filter: str = "",
    queue_filter: str = ""
) -> list[dict]:
    """
    List PBS jobs with optional filtering.
    
    Args:
        state_filter: Filter by job state (Q/R/H/all)
        user_filter: Filter by username
        queue_filter: Filter by queue name
        
    Returns:
        List of job dictionaries with key attributes
    """
```

**Tool: `get_job_summary`**

```python
@mcp.tool()
def get_job_summary() -> dict:
    """
    Get summary statistics of jobs by state.
    
    Returns:
        dict with counts: {queued: N, running: N, held: N, ...}
    """
```

### 3.3 Job Control Tools

**Tool: `delete_job`**

```python
@mcp.tool()
def delete_job(job_id: str, force: bool = False) -> dict:
    """
    Delete/cancel a PBS job.
    
    Args:
        job_id: PBS job identifier
        force: Force deletion (requires privileges)
        
    Returns:
        dict with success status and message
    """
```

**Tool: `hold_job`** and **`release_job`**

```python
@mcp.tool()
def hold_job(job_id: str) -> dict:
    """Place a user hold on a PBS job."""

@mcp.tool()
def release_job(job_id: str) -> dict:
    """Release a user hold from a PBS job."""
```

### 3.4 System Query Tools

**Tool: `list_queues`**

```python
@mcp.tool()
def list_queues() -> list[dict]:
    """
    List available PBS queues with statistics.
    
    Returns:
        List of queue info: name, state, total_jobs, enabled, started
    """
```

**Tool: `list_nodes`**

```python
@mcp.tool()
def list_nodes(state_filter: str = "all") -> list[dict]:
    """
    List PBS compute nodes and their status.
    
    Args:
        state_filter: Filter by node state (free/busy/down/all)
        
    Returns:
        List of node info: name, state, jobs, resources
    """
```

### 3.5 Advanced Tools (Optional for v2)

- `modify_job`: Alter job attributes
- `submit_batch_jobs`: Submit multiple jobs at once
- `manage_reservation`: Create/modify/delete reservations
- `get_server_info`: Query PBS server configuration

## Phase 4: Resource Definitions (`src/pbs_mcp/resources.py`)

Resources provide read-only data that can be loaded into the LLM's context.

### 4.1 Resource: Job Details

```python
@mcp.resource("pbs://job/{job_id}")
def get_job_resource(job_id: str) -> str:
    """
    Get comprehensive job information as formatted text.
    
    URI: pbs://job/12345.aurora-pbs-01
    
    Returns formatted job details including:
    - Job state and queue
    - Resource requests and usage
    - Submit/start/end times
    - Output/error file locations
    """
```

### 4.2 Resource: Queue Information

```python
@mcp.resource("pbs://queue/{queue_name}")
def get_queue_resource(queue_name: str) -> str:
    """
    Get detailed queue configuration and statistics.
    
    URI: pbs://queue/debug
    """
```

### 4.3 Resource: System Status

```python
@mcp.resource("pbs://system/status")
def get_system_status() -> str:
    """
    Get overall PBS system status including:
    - Total nodes and utilization
    - Job statistics by state
    - Queue summaries
    - Server information
    """
```

### 4.4 Resource: User's Jobs

```python
@mcp.resource("pbs://user/jobs")
def get_user_jobs() -> str:
    """
    Get all jobs belonging to the current user.
    
    Returns formatted list of user's jobs with key details.
    """
```

## Phase 5: Prompt Templates (`src/pbs_mcp/prompts.py`)

Prompts are reusable templates that help guide LLM interactions with PBS.

### 5.1 Job Submission Prompts

```python
@mcp.prompt()
def submit_job_guide(
    job_type: str = "python",
    queue: str = "debug"
) -> str:
    """
    Generate guidance for submitting a PBS job.
    
    Args:
        job_type: Type of job (python/mpi/gpu)
        queue: Target queue
        
    Returns:
        Detailed instructions for job submission including
        script template, resource recommendations, and best practices.
    """
```

### 5.2 Troubleshooting Prompts

```python
@mcp.prompt()
def diagnose_job_failure(job_id: str) -> list[Message]:
    """
    Generate diagnostic prompt for a failed job.
    
    Creates a conversation flow to help debug job failures:
    1. Fetch job details
    2. Analyze error messages
    3. Suggest solutions
    """
```

### 5.3 Optimization Prompts

```python
@mcp.prompt()
def optimize_resources(workload_description: str) -> str:
    """
    Generate guidance for optimal PBS resource requests.
    
    Args:
        workload_description: Description of computational workload
        
    Returns:
        Recommendations for nodes, walltime, queue selection
    """
```

## Phase 6: Error Handling and Utilities

### 6.1 Error Handling Strategy (`src/pbs_mcp/utils.py`)

```python
class PBSMCPError(Exception):
    """Base exception for PBS MCP server errors."""
    
class PBSConnectionError(PBSMCPError):
    """PBS server connection failed."""
    
class PBSPrivilegeError(PBSMCPError):
    """Insufficient privileges for operation."""
    
def handle_pbs_exception(func):
    """Decorator to convert PBS exceptions to MCP-friendly errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PBSException as e:
            # Convert to structured error response
            return {
                "success": False,
                "error": str(e),
                "error_type": "PBS_ERROR"
            }
    return wrapper
```

### 6.2 Connection Helper

```python
def get_pbs_client(server: str = None, account: str = None):
    """
    Create PBS client with proper configuration.
    
    Reads from environment variables if not provided:
    - PBS_SERVER
    - PBS_ACCOUNT
    
    Validates configuration before returning client.
    """
```

### 6.3 Data Formatting Helpers

```python
def format_job_info(job_dict: dict) -> str:
    """Format job information for human-readable display."""
    
def format_duration(seconds: int) -> str:
    """Convert seconds to HH:MM:SS format."""
    
def parse_job_id(job_id: str) -> tuple[str, str]:
    """Parse job ID into sequence number and server name."""
```

## Phase 7: Testing Strategy

### 7.1 Unit Tests (`tests/test_tools.py`)

Test each tool in isolation:

```python
import pytest
from pbs_mcp.tools import submit_job, get_job_status

@pytest.mark.asyncio
async def test_submit_job_validation():
    """Test job submission parameter validation."""
    # Test missing script file
    # Test invalid walltime format
    # Test invalid resource requests
    
@pytest.mark.asyncio
async def test_get_job_status_invalid_id():
    """Test handling of invalid job IDs."""
```

### 7.2 Integration Tests (`tests/test_integration.py`)

Test against a real or mock PBS server:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_job_lifecycle():
    """Test submit -> query -> delete workflow."""
    # Submit a simple test job
    # Query its status
    # Delete the job
    # Verify deletion
```

### 7.3 Mock PBS Server for Testing

Consider creating a mock PBS server for CI/CD:

```python
class MockPBSClient:
    """Mock PBS client for testing without real PBS server."""
    
    def __init__(self):
        self.jobs = {}
        self.job_counter = 1000
        
    def submit(self, script_path, queue, attrs):
        # Simulate job submission
        job_id = f"{self.job_counter}.mock-pbs"
        self.job_counter += 1
        return job_id
```

## Phase 8: Documentation

### 8.1 README.md Structure

```markdown
# PBS MCP Server

## What is this?
Connect AI assistants like Claude to PBS job schedulers on HPC systems.

## Prerequisites
- PBS Pro environment
- Python 3.8+
- pbs_api.py library

## Installation
[Step-by-step installation]

## Configuration
[Environment variables and setup]

## Usage with Claude Desktop
[Configuration file examples]

## Usage with Other MCP Clients
[Generic client setup]

## Available Tools
[Complete tool reference]

## Troubleshooting
[Common issues and solutions]
```

### 8.2 Tool Documentation

Each tool should have:
- Clear description of purpose
- Parameter types and defaults
- Return value structure
- Example usage
- Required privileges (if any)
- Common error scenarios

### 8.3 Examples Directory

Create example scripts showing:
- Basic job submission
- Job monitoring workflow
- Batch operations
- Queue and node queries
- Integration with Claude Desktop
- Integration with other MCP clients

## Phase 9: Deployment Considerations

### 9.1 Local Deployment (stdio)

For Claude Desktop and similar clients:

```json
{
  "mcpServers": {
    "pbs": {
      "command": "python",
      "args": ["/path/to/pbs_mcp/server.py"],
      "env": {
        "PBS_SERVER": "aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
        "PBS_ACCOUNT": "datascience"
      }
    }
  }
}
```

### 9.2 Remote Deployment (HTTP/SSE)

For remote access via HTTP:

```python
if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000
    )
```

**Security Considerations:**
- Implement authentication (OAuth, API keys)
- Use HTTPS in production
- Validate all inputs
- Rate limiting
- Audit logging

### 9.3 Container Deployment

Create a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.10-slim

# Install PBS client libraries
# Copy application code
# Set environment variables
# Expose port
# Run server
```

## Phase 10: Advanced Features (Future Enhancements)

### 10.1 Job Script Generation

Tool to generate PBS job scripts from high-level descriptions:

```python
@mcp.tool()
def generate_job_script(
    application: str,
    input_files: list[str],
    node_count: int,
    gpu_required: bool = False
) -> dict:
    """
    Generate a PBS job script based on application requirements.
    
    Returns:
        Script content and recommended submission parameters
    """
```

### 10.2 Job Dependency Management

Tools for managing job dependencies and workflows:

```python
@mcp.tool()
def submit_dependent_jobs(
    jobs: list[dict],
    dependency_graph: dict
) -> dict:
    """
    Submit multiple jobs with dependencies.
    
    Args:
        jobs: List of job specifications
        dependency_graph: Dependencies between jobs
        
    Returns:
        Job IDs and dependency relationships
    """
```

### 10.3 Historical Analysis

Resources for job history and performance analysis:

```python
@mcp.resource("pbs://history/user/{username}")
def get_user_history(username: str, days: int = 30) -> str:
    """
    Get historical job information for analysis.
    
    Useful for identifying patterns, optimizing resource requests,
    and tracking utilization over time.
    """
```

### 10.4 Automated Job Monitoring

Prompts for setting up automated monitoring:

```python
@mcp.prompt()
def setup_job_monitoring(job_pattern: str) -> str:
    """
    Generate instructions for monitoring job progress.
    
    Creates a monitoring strategy including:
    - Key metrics to track
    - Warning signs to watch for
    - Automated actions to take
    """
```

## Phase 11: Security and Best Practices

### 11.1 Security Checklist

- [ ] Never log sensitive information (credentials, full job scripts)
- [ ] Validate all file paths to prevent directory traversal
- [ ] Implement rate limiting for tool calls
- [ ] Use secure environment variable handling
- [ ] Implement proper error messages without exposing internals
- [ ] Add audit logging for administrative operations
- [ ] Document privilege requirements clearly
- [ ] Implement timeout mechanisms for long-running operations

### 11.2 PBS Best Practices

- [ ] Use context managers for PBS connections
- [ ] Don't maintain long-lived connections
- [ ] Batch similar operations when possible
- [ ] Cache queue/node information appropriately
- [ ] Provide clear user feedback on operation progress
- [ ] Handle PBS server unavailability gracefully
- [ ] Respect PBS server rate limits

### 11.3 MCP Best Practices

- [ ] Keep tool functions focused and single-purpose
- [ ] Use clear, descriptive tool names and descriptions
- [ ] Provide detailed parameter documentation
- [ ] Return structured, consistent response formats
- [ ] Include examples in tool descriptions
- [ ] Use resources for large read-only data
- [ ] Use prompts for complex workflows

## Phase 12: Development Workflow

### 12.1 Development Setup

1. Clone repository and set up virtual environment
2. Install development dependencies
3. Configure local PBS environment variables
4. Run tests to verify setup
5. Start server in development mode

### 12.2 Iterative Development Process

1. **Start Simple**: Implement basic tools first (submit, status, list)
2. **Test Early**: Write tests alongside implementation
3. **Use MCP Inspector**: Test tools interactively during development
4. **Iterate**: Add more sophisticated tools based on usage
5. **Document**: Keep documentation current with code

### 12.3 Testing During Development

```bash
# Run tests
pytest tests/

# Test with MCP Inspector
mcp dev server.py

# Test with Claude Desktop
# Update config and restart Claude Desktop
```

## Phase 13: Monitoring and Maintenance

### 13.1 Logging Strategy

Implement structured logging:

```python
import logging

logger = logging.getLogger("pbs_mcp")

# Log levels:
# - DEBUG: Detailed PBS API calls and responses
# - INFO: Tool invocations and outcomes
# - WARNING: Recoverable errors, deprecations
# - ERROR: Tool failures, connection issues
# - CRITICAL: Server failures
```

### 13.2 Metrics to Track

- Tool invocation counts
- Success/failure rates
- Response times
- PBS connection statistics
- Error types and frequencies
- User activity patterns

### 13.3 Maintenance Tasks

- Regular updates to match PBS API changes
- Dependency updates
- Security patches
- Performance optimization based on usage patterns
- Documentation updates

## Success Criteria

### Minimum Viable Product (MVP)

- [ ] Submit PBS jobs with basic parameters
- [ ] Query job status by ID
- [ ] List current jobs
- [ ] Delete/cancel jobs
- [ ] List available queues
- [ ] Works with Claude Desktop via stdio transport
- [ ] Basic error handling and user feedback
- [ ] README with setup instructions

### Version 1.0 Features

- [ ] All MVP features
- [ ] Job hold/release operations
- [ ] Node status queries
- [ ] Queue information resources
- [ ] System status resource
- [ ] Job submission prompt templates
- [ ] Comprehensive test coverage
- [ ] Production-ready error handling
- [ ] Complete documentation

### Future Enhancements

- [ ] Job script generation
- [ ] Dependency management
- [ ] Historical analysis
- [ ] Remote HTTP deployment
- [ ] Advanced prompts for optimization
- [ ] Integration with multiple PBS servers
- [ ] Reservation management

## AI Coding Assistant Guidance

When implementing this with an AI coding assistant:

1. **Start with Phase 1**: Set up project structure first
2. **Implement Incrementally**: Complete one phase before moving to the next
3. **Test Frequently**: Verify each tool works before adding more
4. **Use Type Hints**: Help AI understand data structures
5. **Provide Context**: Share relevant portions of `pbs_api.py` when implementing tools
6. **Review Carefully**: AI may not understand PBS-specific constraints
7. **Iterate on Errors**: Use error messages to guide improvements
8. **Document as You Go**: Add docstrings immediately after implementation

## Example Implementation Prompt

When starting implementation, you might provide this to your AI assistant:

```
Implement the PBS MCP Server Phase 2 (Core Server Setup).

Context:
- Using FastMCP from the official MCP Python SDK
- Wrapping the PBSClient from pbs_api.py (attached)
- PBS connection requires PBS_SERVER and PBS_ACCOUNT environment variables
- Need lifecycle management for any shared resources

Requirements:
1. Create src/pbs_mcp/server.py
2. Import FastMCP and PBSClient
3. Implement lifespan context manager
4. Create main FastMCP instance
5. Add basic configuration from environment
6. Include error handling for missing environment variables

Here's the PBSClient code: [paste relevant sections]
```

## Conclusion

This plan provides a comprehensive roadmap for building a production-ready PBS MCP server. The phased approach allows for incremental development and testing, while the detailed guidance ensures that AI-assisted coding can proceed efficiently.

Key success factors:
- **Incremental Development**: Build and test in small pieces
- **Comprehensive Testing**: Validate against real PBS systems early
- **Clear Documentation**: Help users understand capabilities and limitations
- **Error Handling**: Provide helpful feedback for common issues
- **Security Awareness**: Respect PBS privileges and protect credentials

The result will be a powerful bridge between AI assistants and HPC job scheduling, enabling natural language interaction with PBS systems while maintaining security and reliability.