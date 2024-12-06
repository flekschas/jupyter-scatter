import argparse
import os
import shutil
import sys
from pathlib import Path

from jscatter import __version__

_DEV = False


def check_uv_available():
    if shutil.which("uv") is None:
        print("Error: 'uv' command not found.", file=sys.stderr)
        print("Please install 'uv' to run `jscatter demo` entrypoint.", file=sys.stderr)
        print(
            "For more information, visit: https://github.com/astral-sh/uv",
            file=sys.stderr,
        )
        sys.exit(1)


def run_notebook(notebook_path: Path):
    check_uv_available()

    command = [
        "uvx",
        "--python",
        "3.12",
        "--from",
        "jupyter-core",
        "--with",
        "jupyterlab",
        "--with",
        "." if _DEV else f"jupyter-scatter=={__version__}",
        "jupyter",
        "lab",
        str(notebook_path),
    ]

    try:
        os.execvp(command[0], command)
    except OSError as e:
        print(f"Error executing {command[0]}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="jupyter-scatter")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("demo", help=f"Run the demo notebook in JupyterLab")
    args = parser.parse_args()

    notebook_path = Path("notebooks/demo.ipynb")
    if args.command == "demo":
        run_notebook(notebook_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
