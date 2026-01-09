from pbs_mcp import create_app


def test_placeholder_prompt() -> None:
    app = create_app()
    # FastMCP uses .prompt decorator, verify app has the method
    assert hasattr(app, "prompt"), "FastMCP should have prompt decorator"
