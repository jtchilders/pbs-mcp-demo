"""Example stub for submitting a job once the PBS API is wired in."""

from pbs_mcp import create_app


def main() -> None:
    app = create_app()
    tool = app.tools["ping_pbs"]
    print(tool(job_id="example"))


if __name__ == "__main__":
    main()
