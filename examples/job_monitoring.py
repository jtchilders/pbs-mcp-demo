"""Example stub showing how job monitoring could be wired."""

from pbs_mcp import create_app


def main() -> None:
    app = create_app()
    resource = app.resources["pbs://system/status"]
    print(resource())


if __name__ == "__main__":
    main()
