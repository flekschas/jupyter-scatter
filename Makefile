.PHONY: install uninstall build clean publish


# Build and installation
install:
	pip install -e .
	pip install -r requirements-dev.txt

uninstall:
	pip uninstall jscatter

postinstall: nbext-install labext-install

build:
	python setup.py build

clean-py:
	rm -rf build/
	rm -rf dist/

clean-js:
	rm -rf js/dist/
	rm -rf jscatter/labextension/
	rm -rf jscatter/nbextension/

clean-npm:
	rm -rf js/node_modules/

clean: clean-py clean-js


# Jupyter Notebook Extension
nbext-install:
	jupyter nbextension install --py --symlink --sys-prefix jscatter
	jupyter nbextension enable --py --sys-prefix jscatter

nbext-uninstall:
	jupyter nbextension uninstall --py --sys-prefix jscatter


# Jupyter Lab Extension
labext-install:
	jupyter labextension install jscatter

labext-uninstall:
	jupyter labextension uninstall jscatter


# Publishing tools
bump-patch:
	cd js && npm version patch
	git add js/package.json js/package-lock.json
	git commit -m "Bump to v`node -p "require('./js/package.json').version"`"

bump-minor:
	cd js && npm version minor
	git add js/package.json js/package-lock.json
	git commit -m "Bump to v`node -p "require('./js/package.json').version"`"

bump-major:
	cd js && npm version major
	git add js/package.json js/package-lock.json
	git commit -m "Bump to v`node -p "require('./js/package.json').version"`"

publish:
	git tag -a "v`node -p "require('./js/package.json').version"`" -m "Version `node -p "require('./js/package.json').version"`"
	git push --follow-tags
