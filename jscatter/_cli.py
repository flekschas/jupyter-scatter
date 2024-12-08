import argparse
import os
import shutil
import sys
import pooch
from pathlib import Path

from jscatter import __version__

DEV = False
IS_WINDOWS = sys.platform.startswith('win')


def download_demo_notebook() -> Path:
    notebook = pooch.retrieve(
        url=f'https://github.com/flekschas/jupyter-scatter/raw/refs/tags/v{__version__}/notebooks/demo.ipynb',
        path=pooch.os_cache('jupyter-scatter'),
        fname='demo.ipynb',
        known_hash=None,
    )
    return Path(notebook)


def check_uv_available():
    if shutil.which('uv') is None:
        print("Error: 'uv' command not found.", file=sys.stderr)
        print("Please install 'uv' to run `jscatter demo` entrypoint.", file=sys.stderr)
        print(
            'For more information, visit: https://github.com/astral-sh/uv',
            file=sys.stderr,
        )
        sys.exit(1)


def run_notebook(notebook_path: Path):
    check_uv_available()

    command = [
        'uv',
        'tool',
        'run',
        '--python',
        '3.12',
        '--from',
        'jupyter-core',
        '--with',
        'jupyterlab',
        '--with',
        '.' if DEV else f'jupyter-scatter=={__version__}',
        'jupyter',
        'lab',
        str(notebook_path),
    ]

    if IS_WINDOWS:
        try:
            import subprocess

            completed_process = subprocess.run(command)
            sys.exit(completed_process.returncode)
        except subprocess.CalledProcessError as e:
            print(f'Error executing {command[0]}: {e}', file=sys.stderr)
            sys.exit(1)
    else:
        try:
            os.execvp(command[0], command)
        except OSError as e:
            print(f'Error executing {command[0]}: {e}', file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog='jupyter-scatter')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.add_parser('demo', help=f'Run the demo notebook in JupyterLab')
    args = parser.parse_args()

    if args.command == 'demo':
        if DEV:
            notebook_path = Path(__file__).parent.parent / 'notebooks' / 'demo.ipynb'
        else:
            notebook_path = download_demo_notebook()

        run_notebook(notebook_path)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
