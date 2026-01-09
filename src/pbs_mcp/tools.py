"""Tool declarations for the PBS MCP server."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = object  # type: ignore

from .utils import (
    PBSException,
    PBSMCPError,
    PBSPrivilegeError,
    ensure_privilege,
    load_pbs_context,
    pbs_session,
)


def _error_response(error: Exception) -> Dict[str, Any]:
    return {
        "success": False,
        "error": str(error),
        "error_type": error.__class__.__name__,
    }


def _success_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = {"success": True}
    data.update(payload)
    return data


def _filtered_state(target: str, actual: Optional[str]) -> bool:
    if target.lower() == "all":
        return True
    return (actual or "").upper() == target.upper()


def _filtered_value(target: str, actual: Optional[str]) -> bool:
    if not target:
        return True
    return (actual or "").lower() == target.lower()


def _node_state_matches(state_filter: str, node_state: Optional[str]) -> bool:
    if state_filter.lower() == "all":
        return True
    states = (node_state or "").replace("+", " ").replace(",", " ").lower().split()
    return state_filter.lower() in states


def register_tools(server: FastMCP) -> None:
    """Register PBS-related tools."""

    @server.tool()
    def submit_job(
        script_path: str,
        queue: str = "debug",
        job_name: str = "",
        walltime: str = "00:10:00",
        node_count: int = 1,
        account: str = "",
        filesystems: str = "",
        stdout: str = "tmp_stdout.txt",
        stderr: str = "tmp_stderr.txt"
    ) -> Dict[str, Any]:
        """Submit a PBS job to the scheduler."""

        context = load_pbs_context()
        script = Path(script_path).expanduser()

        if not script.exists() or not script.is_file():
            return _error_response(FileNotFoundError(f"Script not found: {script}"))

        if node_count <= 0:
            return _error_response(ValueError("node_count must be a positive integer"))

        attrs: Dict[str, Any] = {
            "job_name": job_name or script.stem,
            "Account_Name": account or context.account,
            "Resource_List": {
                "walltime": walltime,
                "select": node_count,
            },
            "Output_Path": stdout,
            "Error_Path": stderr
        }

        if filesystems:
            attrs["Resource_List"]["filesystems"] = filesystems

        try:
            with pbs_session(context) as client:
                job_id = client.submit(str(script), queue or None, attrs)
        except (PBSException, PBSMCPError, RuntimeError, OSError) as error:
            return _error_response(error)

        return _success_response(
            {
                "job_id": job_id,
                "queue": queue,
                "walltime": walltime,
                "node_count": node_count,
                "account": attrs["Account_Name"],
                "filesystems": filesystems,
            }
        )

    @server.tool()
    def get_job_status(job_id: str, history: bool = False) -> Dict[str, Any]:
        """Query detailed status of a specific PBS job.

        Args:
            job_id: The ID of the job to check.
            history: If True, search job history (finished jobs). Default is False (active jobs only).
        """

        if not job_id:
            return _error_response(ValueError("job_id must be provided"))

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                # Only extend="x" if history is requested
                jobs = client.stat_jobs(job_id, extend="x" if history else None)
        except (PBSException, PBSMCPError, RuntimeError) as error:
            return _error_response(error)

        if not jobs:
            return _error_response(LookupError(f"Job {job_id} was not found"))

        job = jobs[0]
        return _success_response(
            {
                "job_id": job.name,
                "attributes": job.attrs,
            }
        )

    @server.tool()
    def list_jobs(
        job_id: str = "",
        state_filter: str = "all",
        user_filter: str = "",
        queue_filter: str = "",
    ) -> Dict[str, Any]:
        """List PBS jobs with optional filtering.

        Args:
            job_id: Optional specific job ID to list.
            state_filter: Filter by job state (e.g. "R", "Q", "H", "F") or "all".
            user_filter: Filter by job owner.
            queue_filter: Filter by queue name.
        """

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                # If job_id is provided, only stat that job
                if job_id:
                    jobs = client.stat_jobs(job_id)
                else:
                    jobs = client.stat_jobs()
        except (PBSException, PBSMCPError, RuntimeError) as error:
            return _error_response(error)

        filtered: List[Dict[str, Any]] = []
        for job in jobs:
            attrs = job.attrs
            if not _filtered_state(state_filter, attrs.get("job_state")):
                continue
            if not _filtered_value(user_filter, attrs.get("owner")):
                continue
            if not _filtered_value(queue_filter, attrs.get("queue")):
                continue

            filtered.append(
                {
                    "job_id": job.name,
                    "name": attrs.get("Job_Name"),
                    "state": attrs.get("job_state"),
                    "queue": attrs.get("queue"),
                    "owner": attrs.get("owner"),
                }
            )

        return _success_response(
            {
                "jobs": filtered,
                "total": len(filtered),
                "filters": {
                    "state": state_filter,
                    "user": user_filter,
                    "queue": queue_filter,
                },
            }
        )

    @server.tool()
    def get_job_summary() -> Dict[str, Any]:
        """Get summary statistics of jobs by state."""

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                jobs = client.stat_jobs()
        except (PBSException, PBSMCPError, RuntimeError) as error:
            return _error_response(error)

        summary: Dict[str, int] = {}
        for job in jobs:
            state = (job.attrs.get("job_state") or "UNKNOWN").upper()
            summary[state] = summary.get(state, 0) + 1

        return _success_response({"summary": summary})

    @server.tool()
    def delete_job(job_id: str, force: bool = False) -> Dict[str, Any]:
        """Delete/cancel a PBS job."""

        if not job_id:
            return _error_response(ValueError("job_id must be provided"))

        context = load_pbs_context()

        try:
            if force:
                ensure_privilege(context, "administrator")
            with pbs_session(context) as client:
                client.delete_job(job_id, force=force)
        except (PBSPrivilegeError, PBSException, PBSMCPError, RuntimeError) as error:
            return _error_response(error)

        return _success_response({"job_id": job_id, "force": force})

    @server.tool()
    def hold_job(job_id: str) -> Dict[str, Any]:
        """Place a user hold on a PBS job."""

        if not job_id:
            return _error_response(ValueError("job_id must be provided"))

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                client.hold_job(job_id)
        except (PBSPrivilegeError, PBSException, PBSMCPError, RuntimeError) as error:
            return _error_response(error)

        return _success_response({"job_id": job_id, "action": "held"})

    @server.tool()
    def release_job(job_id: str) -> Dict[str, Any]:
        """Release a user hold from a PBS job."""

        if not job_id:
            return _error_response(ValueError("job_id must be provided"))

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                client.release_job(job_id)
        except (PBSPrivilegeError, PBSException, PBSMCPError, RuntimeError) as error:
            return _error_response(error)

        return _success_response({"job_id": job_id, "action": "released"})

    @server.tool()
    def list_queues() -> Dict[str, Any]:
        """List available PBS queues with statistics."""

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                queues = client.stat_queues()
        except (PBSException, PBSMCPError, RuntimeError) as error:
            return _error_response(error)

        data = [
            {
                "name": queue.name,
                "state": queue.attrs.get("state_count"),
                "enabled": queue.attrs.get("enabled"),
                "started": queue.attrs.get("started"),
                "total_jobs": queue.attrs.get("total_jobs"),
            }
            for queue in queues
        ]

        return _success_response({"queues": data})

    @server.tool()
    def list_nodes(state_filter: str = "all") -> Dict[str, Any]:
        """List PBS compute nodes and their status."""

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                nodes = client.stat_nodes()
        except (PBSException, PBSMCPError, RuntimeError) as error:
            return _error_response(error)

        data = []
        for node in nodes:
            state = node.attrs.get("state")
            if not _node_state_matches(state_filter, state):
                continue
            data.append(
                {
                    "name": node.name,
                    "state": state,
                    "np": node.attrs.get("np"),
                    "jobs": node.attrs.get("jobs"),
                    "resources": {
                        "ncpus": node.attrs.get("resources_available.ncpus"),
                        "mem": node.attrs.get("resources_available.mem"),
                    },
                }
            )

        return _success_response({"nodes": data, "filter": state_filter})

    @server.tool()
    def generate_aurora_pytorch_script(
        script_name: str,
        num_nodes: int = 1,
        walltime: str = "01:00:00",
        queue: str = "debug",
        account: str = "",
        python_script: str = "train.py",
        conda_env: str = "",
        filesystems: str = "home:flare",
        ranks_per_node: int = 12,
        output_dir: str = ".",
    ) -> Dict[str, Any]:
        """Generate a PBS submit script for distributed PyTorch training on Aurora.

        Creates a ready-to-submit PBS script configured for Intel GPUs with:
        - Proper module loading (frameworks)
        - GPU affinity setup via ZE_AFFINITY_MASK
        - oneCCL configuration for distributed training
        - mpiexec launch with correct binding

        Args:
            script_name: Name for the generated PBS script (e.g., "train_job.sh")
            num_nodes: Number of Aurora nodes (each has 6 Intel GPUs with 2 tiles = 12 devices)
            walltime: Job walltime in HH:MM:SS format
            queue: PBS queue (debug, prod, prod-large)
            account: Project allocation name (uses PBS_ACCOUNT env var if empty)
            python_script: Path to your PyTorch training script
            conda_env: Optional conda environment to activate
            filesystems: Filesystems to mount (e.g., "home:flare", "home:eagle")
            ranks_per_node: MPI ranks per node (default 12 = 1 per GPU tile)
            output_dir: Directory to write the generated script
        """
        context = load_pbs_context()
        resolved_account = account or context.account

        if num_nodes <= 0:
            return _error_response(ValueError("num_nodes must be positive"))
        if ranks_per_node <= 0 or ranks_per_node > 12:
            return _error_response(ValueError("ranks_per_node must be between 1 and 12"))

        total_ranks = num_nodes * ranks_per_node

        # Conda activation if specified
        conda_block = ""
        if conda_env:
            conda_block = f"""
# Activate conda environment
conda activate {conda_env}
"""

        script_content = f'''#!/bin/bash
#PBS -A {resolved_account}
#PBS -l select={num_nodes}
#PBS -l walltime={walltime}
#PBS -l filesystems={filesystems}
#PBS -q {queue}
#PBS -N pytorch_distributed
#PBS -j oe
#PBS -k doe

# Change to submission directory
cd ${{PBS_O_WORKDIR}}

# Load PyTorch environment
module load frameworks
{conda_block}
# Intel GPU and CCL configuration
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

# CCL settings for optimal distributed performance
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi

# Horovod thread affinity (for 12 ranks per node)
export HOROVOD_THREAD_AFFINITY="4,8,12,16,20,24,56,60,64,68,72,76"

# GPU affinity helper script
cat > /tmp/gpu_affinity_${{PBS_JOBID}}.sh << 'AFFINITY_EOF'
#!/bin/bash
num_tiles_per_gpu=$1
shift
gpu_id=$(( PALS_LOCAL_RANKID / num_tiles_per_gpu ))
export ZE_AFFINITY_MASK=$gpu_id
exec "$@"
AFFINITY_EOF
chmod +x /tmp/gpu_affinity_${{PBS_JOBID}}.sh

echo "Running distributed PyTorch on {num_nodes} nodes with {total_ranks} total ranks"
echo "Start time: $(date)"

# Launch distributed training
# 2 tiles per GPU, so divide ranks_per_node by 2 to get tiles_per_gpu for affinity
mpiexec -n {total_ranks} -ppn {ranks_per_node} \\
    --pmi=pmix \\
    /tmp/gpu_affinity_${{PBS_JOBID}}.sh 2 \\
    python {python_script}

echo "End time: $(date)"
rm -f /tmp/gpu_affinity_${{PBS_JOBID}}.sh
'''

        # Write the script
        output_path = Path(output_dir).expanduser() / script_name
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(script_content)
            output_path.chmod(0o755)
        except OSError as error:
            return _error_response(error)

        return _success_response({
            "script_path": str(output_path.absolute()),
            "script_name": script_name,
            "num_nodes": num_nodes,
            "total_ranks": total_ranks,
            "ranks_per_node": ranks_per_node,
            "queue": queue,
            "walltime": walltime,
            "python_script": python_script,
            "message": f"Generated PBS script at {output_path.absolute()}. Submit with: qsub {script_name}",
        })

    @server.tool()
    def generate_aurora_mpi_script(
        script_name: str,
        num_nodes: int = 1,
        walltime: str = "01:00:00",
        queue: str = "debug",
        account: str = "",
        executable: str = "./a.out",
        filesystems: str = "home:flare",
        ranks_per_node: int = 12,
        use_gpu: bool = True,
        output_dir: str = ".",
    ) -> Dict[str, Any]:
        """Generate a PBS submit script for MPI applications on Aurora.

        Creates a ready-to-submit PBS script for general MPI workloads.

        Args:
            script_name: Name for the generated PBS script
            num_nodes: Number of Aurora nodes
            walltime: Job walltime in HH:MM:SS format
            queue: PBS queue (debug, prod, prod-large)
            account: Project allocation name
            executable: Path to your MPI executable
            filesystems: Filesystems to mount
            ranks_per_node: MPI ranks per node
            use_gpu: Whether to set up GPU affinity
            output_dir: Directory to write the generated script
        """
        context = load_pbs_context()
        resolved_account = account or context.account

        if num_nodes <= 0:
            return _error_response(ValueError("num_nodes must be positive"))

        total_ranks = num_nodes * ranks_per_node

        gpu_setup = ""
        affinity_wrapper = ""
        cleanup = ""
        if use_gpu:
            gpu_setup = """
# Intel GPU configuration
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

# GPU affinity helper
cat > /tmp/gpu_affinity_${PBS_JOBID}.sh << 'AFFINITY_EOF'
#!/bin/bash
gpu_id=$(( PALS_LOCAL_RANKID / 2 ))
export ZE_AFFINITY_MASK=$gpu_id
exec "$@"
AFFINITY_EOF
chmod +x /tmp/gpu_affinity_${PBS_JOBID}.sh
"""
            affinity_wrapper = "/tmp/gpu_affinity_${PBS_JOBID}.sh "
            cleanup = "rm -f /tmp/gpu_affinity_${PBS_JOBID}.sh\n"

        script_content = f'''#!/bin/bash
#PBS -A {resolved_account}
#PBS -l select={num_nodes}
#PBS -l walltime={walltime}
#PBS -l filesystems={filesystems}
#PBS -q {queue}
#PBS -N mpi_job
#PBS -j oe
#PBS -k doe

cd ${{PBS_O_WORKDIR}}
{gpu_setup}
echo "Running MPI job on {num_nodes} nodes with {total_ranks} ranks"
echo "Start time: $(date)"

mpiexec -n {total_ranks} -ppn {ranks_per_node} --pmi=pmix {affinity_wrapper}{executable}

echo "End time: $(date)"
{cleanup}'''

        output_path = Path(output_dir).expanduser() / script_name
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(script_content)
            output_path.chmod(0o755)
        except OSError as error:
            return _error_response(error)

        return _success_response({
            "script_path": str(output_path.absolute()),
            "script_name": script_name,
            "num_nodes": num_nodes,
            "total_ranks": total_ranks,
            "ranks_per_node": ranks_per_node,
            "queue": queue,
            "walltime": walltime,
            "executable": executable,
            "use_gpu": use_gpu,
            "message": f"Generated PBS script at {output_path.absolute()}. Submit with: qsub {script_name}",
        })
