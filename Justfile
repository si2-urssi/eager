# list all available commands
default:
  just --list

# clean all build, python, and lint files
clean:
	rm -fr {{justfile_directory()}}/build/
	rm -fr {{justfile_directory()}}/docs/_build/
	rm -fr {{justfile_directory()}}/dist/
	rm -fr {{justfile_directory()}}/.eggs/
	find {{justfile_directory()}} -name '*.egg-info' -exec rm -fr {} +
	find {{justfile_directory()}} -name '*.egg' -exec rm -f {} +
	find {{justfile_directory()}} -name '*.pyc' -exec rm -f {} +
	find {{justfile_directory()}} -name '*.pyo' -exec rm -f {} +
	find {{justfile_directory()}} -name '*~' -exec rm -f {} +
	find {{justfile_directory()}} -name '__pycache__' -exec rm -fr {} +
	rm -fr {{justfile_directory()}}/.coverage
	rm -fr {{justfile_directory()}}/coverage.xml
	rm -fr {{justfile_directory()}}/htmlcov/
	rm -fr {{justfile_directory()}}/.pytest_cache
	rm -fr {{justfile_directory()}}/.mypy_cache
	rm -fr {{justfile_directory()}}/soft-search-transformer/

# install with all deps
install:
	pip install -e {{justfile_directory()}}[lint,test,docs,dev]

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