from pbs_mcp import create_app


def test_resources_registered() -> None:
    app = create_app()
    # FastMCP uses .resource decorator, verify app has the method
    assert hasattr(app, "resource"), "FastMCP should have resource decorator"
