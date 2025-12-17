from pbs_mcp import create_app


def test_resources_registered() -> None:
    app = create_app()
    assert app.resources, "Expected placeholder resources to be registered"
