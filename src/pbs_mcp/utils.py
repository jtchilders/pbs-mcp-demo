"""Utility helpers and shared error types."""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, TypeVar

try:  # pragma: no cover - resolved in PBS environments.
    from pbs_api import PBSClient, PBSException  # type: ignore
except ImportError:  # pragma: no cover
    PBSClient = None  # type: ignore

    class PBSException(RuntimeError):  # type: ignore
        """Fallback ensuring exception handling works without PBS API installed."""

        pass


class PBSMCPError(Exception):
    """Base exception for PBS MCP specific failures."""


class PBSConnectionError(PBSMCPError):
    """Raised when a connection to the PBS server cannot be established."""


class PBSPrivilegeError(PBSMCPError):
    """Raised when caller lacks required privileges for an operation."""


DEFAULT_PBS_ACCOUNT = "datascience"
DEFAULT_PBS_ROLE = "user"
PRIVILEGE_LEVELS = ("user", "operator", "administrator")


F = TypeVar("F", bound=Callable[..., Any])


def handle_pbs_exception(func: F) -> F:
    """Decorator that normalizes PBS errors into simple dictionaries."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except PBSMCPError as error:
            return {
                "success": False,
                "error": str(error),
                "error_type": error.__class__.__name__,
            }

    return wrapper  # type: ignore[misc]


@dataclass
class PBSContext:
    """Configuration derived from the MCP lifespan context."""

    server: str
    account: str
    role: str


def load_pbs_context(
    server: Optional[str] = None,
    account: Optional[str] = None,
    role: Optional[str] = None,
) -> PBSContext:
    """Resolve PBS configuration using provided overrides or environment variables."""

    resolved_server = server or os.getenv("PBS_SERVER")
    resolved_account = account or os.getenv("PBS_ACCOUNT") or DEFAULT_PBS_ACCOUNT
    resolved_role = (role or os.getenv("PBS_ROLE") or DEFAULT_PBS_ROLE).lower()

    if not resolved_server:
        raise PBSConnectionError(
            "PBS_SERVER environment variable must be set before starting the MCP server"
        )

    if resolved_role not in PRIVILEGE_LEVELS:
        raise PBSPrivilegeError(
            f"Unsupported PBS privilege role '{resolved_role}'. Expected one of: {', '.join(PRIVILEGE_LEVELS)}."
        )

    return PBSContext(server=resolved_server, account=resolved_account, role=resolved_role)


def ensure_privilege(context: PBSContext, required_role: str) -> None:
    """Verify the caller meets the minimum privilege level required by an operation."""

    try:
        required_index = PRIVILEGE_LEVELS.index(required_role)
    except ValueError as error:
        raise PBSPrivilegeError(f"Unknown required privilege '{required_role}'.") from error

    current_index = PRIVILEGE_LEVELS.index(context.role)
    if current_index < required_index:
        raise PBSPrivilegeError(
            f"{required_role.capitalize()} privileges are required for this operation. Current role: {context.role}."
        )


def require_pbs_client() -> Any:
    """Ensure the PBS API is available and return the client class."""

    if PBSClient is None:
        raise RuntimeError(
            "pbs_api is required to use PBS MCP functionality. Install your PBS API package first."
        )
    return PBSClient


@contextmanager
def pbs_session(context: PBSContext) -> Iterator[Any]:
    """Context manager yielding a connected PBSClient instance."""

    client_cls = require_pbs_client()
    with client_cls(server=context.server) as client:
        yield client
