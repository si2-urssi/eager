# list all available commands
default:
  just --list

# clean all build, python, and lint files
clean:
	rm -fr build/
	rm -fr docs/_build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .coverage
	rm -fr coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache

# install with all deps
install:
	pip install -e {{justfile_directory()}}[lint,test,docs,dev,nsf,transformer]

# lint, format, and check all files
lint:
	pre-commit run --all-files

# run tests
test:
	pytest --cov-report xml --cov-report html --cov=soft_search soft_search/tests/

# run lint and then run tests
build:
	just lint
	just test

# generate Sphinx HTML documentation
generate-docs:
	rm -f docs/soft_search*.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ soft_search **/tests/
	python -msphinx "docs/" "docs/_build/"

# generate Sphinx HTML documentation and serve to browser
serve-docs:
	just generate-docs
	python -mwebbrowser -t file://{{justfile_directory()}}/docs/_build/index.html

# tag a new version
tag-for-release version:
	git tag -a "{{version}}" -m "{{version}}"
	echo "Tagged: $(git tag --sort=-version:refname| head -n 1)"

# release a new version
release:
	git push --follow-tags

# update this repo using latest cookiecutter-py-package
update-from-cookiecutter:
	cookiecutter gh:evamaxfield/cookiecutter-py-package --config-file .cookiecutter.yaml --no-input --overwrite-if-exists --output-dir ..