"""Tool declarations for the PBS MCP server."""
from __future__ import annotations

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
    def get_job_status(job_id: str) -> Dict[str, Any]:
        """Query detailed status of a specific PBS job."""

        if not job_id:
            return _error_response(ValueError("job_id must be provided"))

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                jobs = client.stat_jobs(job_id)
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
        state_filter: str = "all",
        user_filter: str = "",
        queue_filter: str = "",
    ) -> Dict[str, Any]:
        """List PBS jobs with optional filtering."""

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
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
