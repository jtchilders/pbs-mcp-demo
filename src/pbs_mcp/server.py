"""Entry point for the PBS MCP server."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FastMCP is required to run the PBS MCP server. Install the 'mcp' package first."
    ) from exc

from .tools import register_tools
from .resources import register_resources
from .prompts import register_prompts
from .utils import PBSConnectionError, PBSContext, load_pbs_context


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[PBSContext]:
    """Share validated PBS configuration derived from environment variables."""

    try:
        context = load_pbs_context()
    except PBSConnectionError as error:
        raise RuntimeError(str(error)) from error

    yield context


def create_app() -> FastMCP:
    """Configure and return the FastMCP server instance."""

    server = FastMCP(
        name="PBS Scheduler",
        instructions="Expose PBS scheduler functionality to MCP clients",
        lifespan=lifespan,
        dependencies=["pbs_api"],
    )

    register_tools(server)
    register_resources(server)
    register_prompts(server)

    return server


def main() -> None:
    """Run the server using stdio transport by default."""

    create_app().run()


if __name__ == "__main__":
    main()
