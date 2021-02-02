from setuptools import setup, find_packages
# from setuptools import setup, find_packages, Command
# from setuptools.command.build_py import build_py
# from setuptools.command.egg_info import egg_info
# from setuptools.command.sdist import sdist
# from subprocess import check_call
# import platform
# import sys
import os
import io
# import re
from distutils import log
from jupyter_packaging import (
    # create_cmdclass,
    # install_npm,
    # ensure_targets,
    # combine_commands,
    get_version,
)


log.set_verbosity(log.DEBUG)
log.info("setup.py entered")
log.info("$PATH=%s" % os.environ["PATH"])

here = os.path.dirname(os.path.abspath(__file__))
# IS_REPO = os.path.exists(os.path.join(here, ".git"))
# STATIC_DIR = os.path.join(here, "scatterplot", "static")
# NODE_ROOT = os.path.join(here, "js")
# NPM_PATH = os.pathsep.join(
#     [
#         os.path.join(NODE_ROOT, "node_modules", ".bin"),
#         os.environ.get("PATH", os.defpath),
#     ]
# )
js_dir = os.path.join(here, 'js')
version = get_version("jscatter/_version.py")

js_targets = [
    os.path.join(js_dir, 'dist', 'index.js'),
]

data_files_spec = [
    ("share/jupyter/nbextensions/jscatter", "jscatter/nbextension", "*.*"),
    ("share/jupyter/labextensions/jscatter", "jscatter/labextension", "**"),
    ("share/jupyter/labextensions/jscatter", ".", "install.json"),
    ("etc/jupyter/nbconfig/notebook.d", ".", "jscatter.json"),
]


def read(*parts, **kwargs):
    filepath = os.path.join(here, *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


# def get_version():
#     version = re.search(
#         r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
#         read("scatterplot", "__version__.py"),
#         re.MULTILINE,
#     ).group(1)
#     return version


def get_requirements(path):
    content = read(path)
    return [req for req in content.split("\n") if req != "" and not req.startswith("#")]


# def js_prerelease(command, strict=False):
#     """decorator for building minified js/css prior to another command"""

#     class DecoratedCommand(command):
#         def run(self):
#             jsdeps = self.distribution.get_command_obj("jsdeps")
#             if not IS_REPO and all(os.path.exists(t) for t in jsdeps.targets):
#                 # sdist, nothing to do
#                 command.run(self)
#                 return

#             try:
#                 self.distribution.run_command("jsdeps")
#             except Exception as e:
#                 missing = [t for t in jsdeps.targets if not os.path.exists(t)]
#                 if strict or missing:
#                     log.warn("rebuilding js and css failed")
#                     if missing:
#                         log.error("missing files: %s" % missing)
#                     raise e
#                 else:
#                     log.warn("rebuilding js and css failed (not a problem)")
#                     log.warn(str(e))
#             command.run(self)
#             update_package_data(self.distribution)

#     return DecoratedCommand


# def update_package_data(distribution):
#     """update package_data to catch changes during setup"""
#     build_py = distribution.get_command_obj("build_py")
#     # distribution.package_data = find_package_data()
#     # re-init build_py options which load package_data
#     build_py.finalize_options()


# class NPM(Command):
#     description = "install package.json dependencies using npm"

#     user_options = []

#     node_modules = os.path.join(NODE_ROOT, "node_modules")

#     targets = [
#         os.path.join(STATIC_DIR, "extension.js"),
#         os.path.join(STATIC_DIR, "index.js"),
#     ]

#     def initialize_options(self):
#         pass

#     def finalize_options(self):
#         pass

#     def get_npm_name(self):
#         npm_name = "npm"
#         if platform.system() == "Windows":
#             npm_name = "npm.cmd"
#         return npm_name

#     def has_npm(self):
#         npm_name = self.get_npm_name()
#         try:
#             check_call([npm_name, "--version"])
#             return True
#         except:
#             return False

#     def should_run_npm_install(self):
#         node_modules_exists = os.path.exists(self.node_modules)
#         return self.has_npm() and not node_modules_exists

#     def run(self):
#         has_npm = self.has_npm()
#         if not has_npm:
#             log.error(
#                 "`npm` unavailable. If you're running this command using "
#                 "sudo, make sure `npm` is available to sudo"
#             )

#         env = os.environ.copy()
#         env["PATH"] = NPM_PATH

#         npm_name = self.get_npm_name()

#         if self.should_run_npm_install():
#             log.info(
#                 "Installing build dependencies with npm. "
#                 "This may take a while..."
#             )
#             check_call(
#                 [npm_name, "install"],
#                 cwd=NODE_ROOT,
#                 stdout=sys.stdout,
#                 stderr=sys.stderr,
#             )
#             os.utime(self.node_modules, None)

#         check_call(
#             [npm_name, "run", "build"],
#             cwd=NODE_ROOT,
#             stdout=sys.stdout,
#             stderr=sys.stderr,
#         )

#         for t in self.targets:
#             if not os.path.exists(t):
#                 msg = "Missing file: %s" % t
#                 if not has_npm:
#                     msg += "\nnpm is required to build a development version "
#                     "of a widget extension"
#                 raise ValueError(msg)

#         # update package data in case this created new files
#         update_package_data(self.distribution)


setup_args = dict(
    name="jupyter-scatter",
    version=version,
    packages=find_packages(),
    license="Apache-2.0",
    description="A scatter plot extension for Jupyter Notebook and Lab",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/flekschas/jscatter",
    include_package_data=True,
    zip_safe=False,
    author="Fritz Lekschas",
    author_email="code@lekschas.de",
    keywords=[
        "scatter",
        "scatter plot",
        "jupyter",
        "ipython",
        "ipywidgets",
        "jupyterlab",
        "jupyterlab-extension",
        "widgets"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Graphics",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=get_requirements("requirements.txt"),
    setup_requires=[],
    tests_require=["pytest"],
    # "cmdclass": {
    #     "build_py": js_prerelease(build_py),
    #     "egg_info": js_prerelease(egg_info),
    #     "sdist": js_prerelease(sdist, strict=True),
    #     "jsdeps": NPM,
    # },
)

setup(**setup_args)
