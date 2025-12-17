from pbs_mcp import create_app


def test_placeholder_prompt() -> None:
    app = create_app()
    prompt = next(iter(app.prompts.values()))
    assert "PBS MCP server" in prompt(), "Prompt should mention PBS MCP setup"
