"""Prompt templates for guiding PBS workflows."""
from __future__ import annotations

from typing import Dict, List

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = object  # type: ignore

JOB_TYPE_HINTS: Dict[str, Dict[str, str]] = {
    "python": {
        "script_template": "#!/bin/bash\n#PBS -l select=1:ncpus=32\n#PBS -l walltime=00:30:00\n#PBS -q {queue}\n#PBS -N {job_name}\n\nsource /path/to/env\npython your_script.py\n",
        "resources": "1 node, 32 CPUs, short walltime",
    },
    "mpi": {
        "script_template": "#!/bin/bash\n#PBS -l select={nodes}:ncpus=64:mpiprocs=64\n#PBS -l walltime=01:00:00\n#PBS -q {queue}\n#PBS -N {job_name}\n\nmodule load mpi\nmpirun -np $PBS_NP ./app\n",
        "resources": "Multi-node with mpiprocs, align select/place directives",
    },
    "gpu": {
        "script_template": "#!/bin/bash\n#PBS -l select=1:ncpus=32:ngpus=4\n#PBS -l walltime=02:00:00\n#PBS -q {queue}\n#PBS -N {job_name}\n\nmodule load cuda\npython train.py\n",
        "resources": "Request ngpus plus matching gpus per node policies",
    },
}


def _job_type_hint(job_type: str) -> Dict[str, str]:
    return JOB_TYPE_HINTS.get(job_type.lower(), JOB_TYPE_HINTS["python"])


def register_prompts(server: FastMCP) -> None:
    """Register reusable prompt templates."""

    @server.prompt()
    def submit_job_guide(job_type: str = "python", queue: str = "debug") -> str:
        """Provide PBS job submission guidance tailored to a workload type."""

        hints = _job_type_hint(job_type)
        return (
            f"PBS job submission guide ({job_type} workloads)\n\n"
            "1. Inspect target queue resources:\n"
            f"   - Load resource: pbs://queue/{queue}\n"
            "   - Use tool: list_queues to compare throughput and limits.\n"
            "2. Author your job script:\n"
            f"{hints['script_template']}\n"
            "3. Validate resource requests:\n"
            f"   - Recommended resources: {hints['resources']}\n"
            "   - Adjust walltime/select/place according to queue policy.\n"
            "4. Submit via submit_job tool:\n"
            "   - Required params: script_path, queue\n"
            "   - Optional: job_name, walltime, node_count, account\n"
            "5. Monitor progress:\n"
            "   - get_job_status(job_id)\n"
            "   - list_jobs(state_filter=\"R\") for running jobs\n"
            "6. Troubleshoot using resources:\n"
            "   - pbs://job/{job_id}\n"
            "   - pbs://system/status\n"
        )

    @server.prompt()
    def diagnose_job_failure(job_id: str) -> List[Dict[str, str]]:
        """Create a conversation flow for diagnosing failed jobs."""

        if not job_id:
            raise ValueError("job_id is required for diagnostics")

        steps = [
            (
                "system",
                "You are a PBS scheduling expert helping diagnose failed jobs. "
                "Leverage available MCP tools/resources to gather context.",
            ),
            (
                "user",
                f"Investigate PBS job {job_id}. Determine why it failed and suggest fixes.",
            ),
            (
                "assistant",
                "1. Fetch job snapshot from resource pbs://job/{job_id}.\n"
                "2. Inspect stderr via Output/Error paths reported.\n"
                "3. Use get_job_status(job_id) for live attributes if job still exists.\n"
                "4. Provide likely failure reasons and remediation steps.",
            ),
        ]
        return [{"role": role, "content": content.format(job_id=job_id)} for role, content in steps]

    @server.prompt()
    def optimize_resources(workload_description: str) -> str:
        """Recommend optimal PBS resource requests based on workload details."""

        if not workload_description:
            raise ValueError("workload_description is required")

        return (
            "PBS Resource Optimization Plan\n"
            f"Workload: {workload_description}\n\n"
            "1. Identify resource drivers:\n"
            "   - CPU vs GPU, memory footprint, I/O intensity.\n"
            "2. Gather live system info:\n"
            "   - list_nodes(state_filter='free') for available inventory.\n"
            "   - pbs://system/status for queue backlogs.\n"
            "3. Draft resource list:\n"
            "   - select={nodes}:ncpus={cpus}:mem={mem}\n"
            "   - walltime={duration}\n"
            "   - queue=tuned to workload priority.\n"
            "4. Validate account/privilege:\n"
            "   - Ensure PBS_ROLE matches operations via pbs://system/privileges.\n"
            "5. Recommend submission parameters:\n"
            "   - Use submit_job with calculated node_count and walltime.\n"
            "   - Capture rationale (e.g., scaling tests, previous runs).\n"
            "6. Provide follow-up steps:\n"
            "   - Monitor with get_job_summary for queue impact.\n"
            "   - Adjust select/place/walltime iteratively.\n"
        )
