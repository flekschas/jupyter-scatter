.PHONY: install uninstall build clean publish


# Build and installation
install:
	pip install -v -e .
	pip install -r requirements-dev.txt

uninstall:
	pip uninstall jupyter-scatterplot

postinstall: nbext-deps nbext-install labext-deps labext-install

build:
	python setup.py jsdeps

clean-py:
	rm -rf build/
	rm -rf dist/

clean-js:
	rm -rf js/dist/
	rm -rf scatterplot/static/

clean-npm:
	rm -rf js/node_modules/

clean: clean-py clean-js clean-npm


# Jupyter Notebook Extension
nbext-deps:
	#pip install jupyter_contrib_nbextensions
	#jupyter contrib nbextension install --sys-prefix
	jupyter nbextension enable --py --sys-prefix widgetsnbextension

nbext-install:
	jupyter nbextension install --py --symlink --sys-prefix higlass
	jupyter nbextension enable --py --sys-prefix higlass

nbext-uninstall:
	jupyter nbextension uninstall --py --sys-prefix higlass


# Jupyter Lab Extension
labext-deps:
	jupyter labextension install @jupyter-widgets/jupyterlab-manager

labext-install:
	cd js && jupyter labextension link .

labext-uninstall:
	cd js && jupyter labextension unlink .


# Publishing tools
bump-patch:
	cd js && npm version patch
	echo "__version__ = \"`node -p "require('./js/package.json').version"`\"" > scatterplot/__version__.py
	git add ./js/package.json ./scatterplot/__version__.py
	git commit -m "Bump to v`node -p "require('./js/package.json').version"`"

bump-minor:
	cd js && npm version minor
	echo "__version__ = \"`node -p "require('./js/package.json').version"`\"" > scatterplot/__version__.py
	git add ./js/package.json ./scatterplot/__version__.py
	git commit -m "Bump to v`node -p "require('./js/package.json').version"`"

bump-major:
	cd js && npm version major
	echo "__version__ = \"`node -p "require('./js/package.json').version"`\"" > scatterplot/__version__.py
	git add ./js/package.json ./scatterplot/__version__.py
	git commit -m "Bump to v`node -p "require('./js/package.json').version"`"

publish:
	git tag -a "v`node -p "require('./js/package.json').version"`" -m "Version `node -p "require('./js/package.json').version"`"
	git push --follow-tags
