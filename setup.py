from pathlib import Path

from jupyter_packaging import get_data_files, get_version, npm_builder, wrap_installers
from setuptools import setup

here = Path(__file__).parent.resolve()
version = get_version("jscatter/_version.py")

builder = npm_builder(path=str(here / "js"), build_cmd="build", npm="npm")

# representative files that should exist after build
targets = [
    str(here / "jscatter" / "nbextension" / "index.js"),
    str(here / "jscatter" / "labextension" / "package.json"),
]

cmdclass = wrap_installers(
    pre_develop=builder,
    pre_dist=builder,
    ensured_targets=targets,
    skip_if_exists=targets,
)

data_files = get_data_files([
    (
        "share/jupyter/nbextensions/jupyter-scatter/",
        "jscatter/nbextension/",
        "*",
    ),
    (
        "share/jupyter/labextensions/jupyter-scatter/",
        "jscatter/labextension/",
        "**",
    ),
    (
        "etc/jupyter/nbconfig/notebook.d/",
        ".",
        "jupyter-scatter.json",
    ),
])

def get_requirements(path):
    with open(here / path) as f:
        content = f.read()
    return [req for req in content.split("\n") if req != "" and not req.startswith("#")]

setup(
    version=version,
    install_requires=get_requirements("requirements.txt"),
    cmdclass=cmdclass,
    data_files=data_files,
)
