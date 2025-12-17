"""Resource definitions for the PBS MCP server."""
from __future__ import annotations

import getpass
import os
from typing import Any, List

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = object  # type: ignore

from .utils import (
    PBSException,
    PBSMCPError,
    PRIVILEGE_LEVELS,
    load_pbs_context,
    pbs_session,
)


def _format_error(message: str) -> str:
    return f"PBS resource error: {message}"


def _format_job(job) -> str:
    attrs = job.attrs
    lines = [
        f"Job ID: {job.name}",
        f"Name: {attrs.get('Job_Name', 'unknown')}",
        f"Owner: {attrs.get('owner', 'unknown')}",
        f"State: {attrs.get('job_state', 'unknown')}",
        f"Queue: {attrs.get('queue', 'unknown')}",
        f"Account: {attrs.get('Account_Name', '-')}",
        f"Walltime: {attrs.get('Resource_List.walltime', '-')}",
        f"Nodes: {attrs.get('Resource_List.select', attrs.get('Resource_List.nodect', '-'))}",
        f"Submit Time: {attrs.get('ctime', '-')}",
        f"Start Time: {attrs.get('start_time', '-')}",
        f"Output: {attrs.get('Output_Path', '-')}",
        f"Error: {attrs.get('Error_Path', '-')}",
    ]
    return "\n".join(lines)


def _format_queue(queue) -> str:
    attrs = queue.attrs
    return "\n".join(
        [
            f"Queue: {queue.name}",
            f"Enabled: {attrs.get('enabled', 'unknown')}",
            f"Started: {attrs.get('started', 'unknown')}",
            f"State Counts: {attrs.get('state_count', '-')}",
            f"Total Jobs: {attrs.get('total_jobs', '-')}",
            f"Max User Run: {attrs.get('max_user_run', '-')}",
            f"Max Running: {attrs.get('max_running', '-')}",
            f"Resources Default: {attrs.get('resources_default.walltime', '-')}",
        ]
    )


def _format_job_summary(jobs: List[Any]) -> str:
    summary = {}
    for job in jobs:
        state = (job.attrs.get("job_state") or "UNKNOWN").upper()
        summary[state] = summary.get(state, 0) + 1
    parts = [f"{state}: {count}" for state, count in sorted(summary.items())]
    return ", ".join(parts) if parts else "No jobs reported"


def register_resources(server: FastMCP) -> None:
    """Register read-only resources exposed by the server."""

    @server.resource("pbs://job/{job_id}")
    def get_job_resource(job_id: str) -> str:
        """Get comprehensive job information."""

        if not job_id:
            return _format_error("job_id must be provided")

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                jobs = client.stat_jobs(job_id)
        except (PBSException, PBSMCPError, RuntimeError) as error:
            return _format_error(str(error))

        if not jobs:
            return _format_error(f"Job {job_id} was not found on server {context.server}")

        return _format_job(jobs[0])

    @server.resource("pbs://queue/{queue_name}")
    def get_queue_resource(queue_name: str) -> str:
        """Get detailed queue configuration and statistics."""

        if not queue_name:
            return _format_error("queue_name must be provided")

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                queues = client.stat_queues(queue_name)
        except (PBSException, PBSMCPError, RuntimeError) as error:
            return _format_error(str(error))

        if not queues:
            return _format_error(f"Queue {queue_name} was not found on server {context.server}")

        return _format_queue(queues[0])

    @server.resource("pbs://system/status")
    def get_system_status() -> str:
        """Summarize PBS system status including job and queue information."""

        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                server_info = client.stat_server()
                scheduler_info = client.stat_sched()
                queues = client.stat_queues()
                jobs = client.stat_jobs()
        except (PBSException, PBSMCPError, RuntimeError) as error:
            return _format_error(str(error))

        lines = [f"PBS System Status for {context.server}"]
        if server_info:
            attrs = server_info[0].attrs
            lines.append(f"Server State: {attrs.get('state', 'unknown')}")
            lines.append(f"Total Nodes: {attrs.get('resources_available.ncpus', '-')}")
        if scheduler_info:
            attrs = scheduler_info[0].attrs
            lines.append(f"Scheduler Cycle: {attrs.get('sched_cycle_length', '-')}")

        lines.append(f"Job Summary: {_format_job_summary(jobs)}")
        lines.append(f"Queues: {len(queues)} available")
        for queue in queues[:5]:
            lines.append(f"- {queue.name}: {queue.attrs.get('state_count', '-')}")

        return "\n".join(lines)

    @server.resource("pbs://user/jobs")
    def get_user_jobs() -> str:
        """List all jobs belonging to the current user."""

        username = os.getenv("PBS_USER") or getpass.getuser()
        context = load_pbs_context()

        try:
            with pbs_session(context) as client:
                jobs = client.stat_jobs()
        except (PBSException, PBSMCPError, RuntimeError) as error:
            return _format_error(str(error))

        user_jobs = []
        for job in jobs:
            owner = job.attrs.get("owner", "")
            owner_name = owner.split("@", 1)[0] if owner else ""
            if owner_name == username:
                user_jobs.append(job)

        if not user_jobs:
            return f"No jobs found for user {username} on server {context.server}"

        job_blocks = "\n\n".join(_format_job(job) for job in user_jobs)
        return f"Jobs for {username}:\n\n{job_blocks}"

    @server.resource("pbs://system/privileges")
    def get_privilege_requirements() -> str:
        """Report the configured PBS privilege role and the hierarchy expected by MCP tools."""

        context = load_pbs_context()
        return (
            "PBS privilege overview\n"
            f"- Current role: {context.role}\n"
            f"- Hierarchy: {', '.join(PRIVILEGE_LEVELS)}\n"
            "- Operator-level tools require at least 'operator'\n"
            "- Administrative tools require 'administrator'\n"
            "Update the PBS_ROLE environment variable to match your scheduler account."
        )
