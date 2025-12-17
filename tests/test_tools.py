from pbs_mcp import create_app


def test_create_app_initializes_server() -> None:
    # We only ensure the factory returns a FastMCP instance.
    app = create_app()
    assert hasattr(app, "tool"), "FastMCP interface should expose tool registration"
