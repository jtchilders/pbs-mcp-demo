#!/usr/bin/env python3
"""Bridge the PBS MCP server with OpenAI's function-calling API.

This script launches the local PBS MCP server via stdio, wraps its tools as
OpenAI-compatible functions, and lets you drive a conversation where OpenAI
decides when to call those tools. It is intended for experimentation on HPC
login nodes where you want an OpenAI model (e.g., GPT-4o) to submit PBS jobs
through the MCP layer.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool

READ_FILE_FUNCTION = "read_local_file"
MAX_FILE_BYTES = 16_384
COMPLETED_STATES = {"F", "C", "E"}


@dataclass
class JobTracker:
    job_id: Optional[str] = None
    state: Optional[str] = None
    deleted: bool = False
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    stdout_reported: bool = False
    stderr_reported: bool = False
    polls: int = 0

    def record_submit(self, structured: Optional[Dict[str, Any]]) -> None:
        if isinstance(structured, dict):
            candidate = structured.get("job_id") or structured.get("jobId")
            if candidate:
                self.job_id = candidate

    def record_status(self, structured: Optional[Dict[str, Any]]) -> None:
        if not isinstance(structured, dict):
            return
        attrs = structured.get("attributes") or {}
        state = (attrs.get("job_state") or "").upper()
        if state:
            self.state = state
        output_path = attrs.get("Output_Path") or attrs.get("output_path")
        error_path = attrs.get("Error_Path") or attrs.get("error_path")
        if output_path:
            self.stdout_path = self._normalize_path(output_path)
        if error_path:
            self.stderr_path = self._normalize_path(error_path)

    def record_resource(self, structured: Optional[Dict[str, Any]]) -> None:
        if not isinstance(structured, dict):
            return
        # Resources return text blobs; nothing to parse here yet.

    def record_deletion(self, structured: Optional[Dict[str, Any]]) -> None:
        if isinstance(structured, dict) and structured.get("success"):
            self.deleted = True

    def record_local_read(self, path: str) -> None:
        norm = self._normalize_path(path)
        if self.stdout_path and norm == self.stdout_path:
            self.stdout_reported = True
        if self.stderr_path and norm == self.stderr_path:
            self.stderr_reported = True

    @property
    def job_completed(self) -> bool:
        if self.deleted:
            return True
        return (self.state or "").upper() in COMPLETED_STATES

    @staticmethod
    def _normalize_path(path: str) -> str:
        return path.split(":", 1)[1] if ":" in path else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an OpenAI conversation that can call PBS MCP tools."
    )
    parser.add_argument(
        "--prompt",
        default="Submit a hello world PBS job on the debug queue and report the job id.",
        help="Initial user prompt passed to OpenAI.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model identifier supporting tool calls.",
    )
    parser.add_argument(
        "--server-command",
        default="python",
        help="Executable used to launch the MCP server.",
    )
    parser.add_argument(
        "--server-args",
        nargs=argparse.REMAINDER,
        default=["-m", "pbs_mcp"],
        help="Arguments passed to the MCP server command.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between follow-up status requests once a job is submitted.",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=10,
        help="Maximum number of follow-up polling rounds before exiting.",
    )
    parser.add_argument(
        "--demo",
        choices=["simple", "lifecycle", "admin", "resources"],
        default="simple",
        help="Select a pre-canned demo scenario.",
    )
    return parser.parse_args()


def tool_to_openai_schema(tool: Tool) -> Dict[str, Any]:
    """Convert an MCP tool definition into an OpenAI function schema."""

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object"},
        },
    }


def local_read_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": READ_FILE_FUNCTION,
            "description": (
                "Read up to 16KB from a text file accessible on the login node. "
                "Use this after retrieving stdout/stderr paths from PBS job attributes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    }


def list_resources_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "list_mcp_resources",
            "description": "List all available resources on the MCP server.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def read_resource_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_mcp_resource",
            "description": "Read the content of a specific MCP resource.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "The URI of the resource to read (e.g., pbs://system/status).",
                    }
                },
                "required": ["uri"],
            },
        },
    }

def read_local_file(arguments: str) -> Dict[str, Any]:
    """Read a text file so the LLM can inspect stdout/stderr contents."""

    try:
        parsed_args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"read_local_file arguments are invalid JSON: {arguments}") from exc

    path_value = parsed_args.get("path")
    if not path_value:
        raise ValueError("read_local_file requires a 'path' parameter.")

    normalized_path = path_value.split(":", 1)[1] if ":" in path_value else path_value
    path = Path(normalized_path).expanduser()
    try:
        data = path.read_text(errors="replace")
    except OSError as exc:
        return {"success": False, "path": str(path), "error": str(exc)}

    snippet = data[:MAX_FILE_BYTES]
    return {"success": True, "path": str(path), "bytes": len(snippet), "content": snippet}


async def call_mcp_tool(
    session: ClientSession, name: str, arguments: str
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Invoke an MCP tool and return a JSON-serializable payload."""

    try:
        parsed_args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool arguments for {name} are not valid JSON: {arguments}") from exc

    result = await session.call_tool(name, parsed_args)
    return result.model_dump(), result.structuredContent


def _extract_structured_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    structured = payload.get("structuredContent")
    if isinstance(structured, dict):
        return structured

    content_blocks = payload.get("content") or []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "json" and isinstance(block.get("json"), dict):
            return block["json"]
        text = block.get("text")
        if isinstance(text, str):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None

    return None


async def list_mcp_resources(session: ClientSession) -> Dict[str, Any]:
    """List available resources."""
    try:
        resources = await session.list_resources()
        return {
            "resources": [
                {"name": r.name, "uri": str(r.uri), "description": r.description}
                for r in resources.resources
            ]
        }
    except Exception as exc:
        return {"error": str(exc)}


async def read_mcp_resource(session: ClientSession, arguments: str) -> Dict[str, Any]:
    """Read a specific resource."""
    try:
        parsed_args = json.loads(arguments or "{}")
        uri = parsed_args.get("uri")
        if not uri:
            return {"error": "uri parameter is required"}
        
        result = await session.read_resource(uri)
        # Combine contents directly
        contents = []
        for content in result.contents:
            contents.append({"uri": str(content.uri), "text": content.text})
        return {"contents": contents}
    except Exception as exc:
        return {"error": str(exc)}

async def run_conversation_loop(
    client: OpenAI,
    session: ClientSession,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tracker: JobTracker,
    model: str,
) -> str:
    """Iteratively call OpenAI until there are no outstanding tool calls."""

    while True:
        print(f"\n[DEBUG] Sending to LLM (History: {len(messages)} messages). Last message:")
        if messages:
            last_msg = messages[-1]
            print(f"Role: {last_msg.get('role')}")
            print(f"Content: {last_msg.get('content')}")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        print("\n[DEBUG] LLM Response:")
        if message.content:
            print(f"Content: {message.content}")
        if message.tool_calls:
            for tc in message.tool_calls:
                print(f"Tool Call: {tc.function.name} args={tc.function.arguments}")

        assistant_entry: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_entry["tool_calls"] = message.tool_calls
        messages.append(assistant_entry)

        if not message.tool_calls:
            return message.content or ""

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments or "{}"


            if tool_name == READ_FILE_FUNCTION:
                payload = read_local_file(arguments)
                if payload.get("success"):
                    tracker.record_local_read(payload.get("path", ""))
            elif tool_name == "list_mcp_resources":
                payload = await list_mcp_resources(session)
            elif tool_name == "read_mcp_resource":
                payload = await read_mcp_resource(session, arguments)
            else:
                payload, structured = await call_mcp_tool(session, tool_name, arguments)
                structured_data = structured or _extract_structured_from_payload(payload)
                if structured_data and "result" in structured_data:
                    structured_data = structured_data["result"]

                if tool_name == "submit_job":
                    tracker.record_submit(structured_data)
                elif tool_name == "get_job_status":
                    tracker.record_status(structured_data)
                elif tool_name == "delete_job":
                    tracker.record_deletion(structured_data)
                elif tool_name.startswith("get_job_resource") or tool_name.startswith("read_resource"):
                    tracker.record_resource(structured_data)


            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(payload, indent=2),
                }
            )


async def poll_until_complete(
    client: OpenAI,
    session: ClientSession,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tracker: JobTracker,
    poll_interval: int,
    max_polls: int,
    model: str,
) -> None:
    if not tracker.job_id:
        return

    while tracker.polls < max_polls:
        tracker.polls += 1
        await asyncio.sleep(poll_interval)

        followup_prompt = (
            f"Please check whether PBS job {tracker.job_id} has finished. Use get_job_status "
            "or other PBS tools as needed. When it finishes, read and report the stdout and stderr "
            "files referenced in the job attributes using read_local_file."
        )
        messages.append({"role": "user", "content": followup_prompt})

        final_message = await run_conversation_loop(client, session, messages, tools, tracker, model)
        print(final_message)

        if tracker.job_completed and tracker.stdout_reported and tracker.stderr_reported:
            return
        
        if tracker.deleted:
            print("Job was deleted explicitly.")
            return

    print(
        f"Polling limit reached without collecting both stdout/stderr. "
        f"Job state: {tracker.state or 'unknown'}."
    )


async def main_async() -> None:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")

    server = StdioServerParameters(
        command=args.server_command,
        args=args.server_args,
        env=os.environ.copy(),
    )

    # Demo prompts
    prompts = {
        "simple": args.prompt,
        "lifecycle": (
            "Submit a PBS job using the following script: ./examples/hello_world_script.sh. "
            "REQUIRED: You MUST specify 'home' for the 'filesystems' parameter. "
            "Then, immediately hold the job, check that its state is 'H' (Held), "
            " wait five seconds "
            "then release the job, and check that its state is 'Q' or 'R'. "
            "Finally, delete the job."
        ),
        "admin": (
            "List all nodes and report how many are 'free'. "
            "Then list all queues and report which ones are enabled."
        ),
        "resources": (
            "List all available MCP resources. "
            "Then read the 'pbs://system/status' resource to get a system overview. "
            "Finally, read 'pbs://user/jobs' to see my current jobs."
        ),
    }

    selected_prompt = prompts.get(args.demo, args.prompt)
    if args.demo != "simple":
        print(f"--- Running Demo: {args.demo} ---")
        print(f"Prompt: {selected_prompt}\n")

    tracker = JobTracker()

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            registered_tools = await session.list_tools()
            openai_tools = [tool_to_openai_schema(tool) for tool in registered_tools.tools]
            openai_tools.append(local_read_tool_schema())
            openai_tools.append(list_resources_tool_schema())
            openai_tools.append(read_resource_tool_schema())

            if not openai_tools:
                raise RuntimeError("The MCP server did not expose any tools.")

            client = OpenAI()
            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are an HPC assistant. Use the provided PBS MCP tools to fulfill requests. "
                        "Call tools whenever you need live scheduler data, and report final results clearly."
                    ),
                },
                {"role": "user", "content": selected_prompt},
            ]

            final_message = await run_conversation_loop(
                client=client,
                session=session,
                messages=messages,
                tools=openai_tools,
                tracker=tracker,
                model=args.model,
            )
            print(final_message)

            if not tracker.job_id or tracker.deleted:
                return

            await poll_until_complete(
                client=client,
                session=session,
                messages=messages,
                tools=openai_tools,
                tracker=tracker,
                poll_interval=args.poll_interval,
                max_polls=args.max_polls,
                model=args.model,
            )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
