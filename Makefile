.PHONY: build docs test

BUILDDIR := $(PWD)
BUILD_ARGS :=  # set nightly to build nightly release
CHECKDIRS := examples tests src utils setup.py
PYCHECKGLOBS := 'examples/**/*.py' 'scripts/**/*.py' 'src/**/*.py' 'tests/**/*.py' 'utils/**/*.py' setup.py
DOCDIR := docs
MDCHECKGLOBS := 'docs/**/*.md' 'docs/**/*.rst' 'examples/**/*.md' 'scripts/**/*.md'
MDCHECKFILES := CODE_OF_CONDUCT.md CONTRIBUTING.md DEVELOPING.md README.md
SPARSEZOO_TEST_MODE := "true"

PYTHON := python3

# run checks on all files for the repo
quality:
	@echo "Running copyright checks";
	$(PYTHON) utils/copyright.py quality $(PYCHECKGLOBS) $(MDCHECKGLOBS) $(MDCHECKFILES)
	@echo "Running python quality checks";
	black --check $(CHECKDIRS);
	isort --check-only $(CHECKDIRS);
	flake8 $(CHECKDIRS);


# style the code according to accepted standards for the repo
style:
	@echo "Running copyrighting";
	$(PYTHON) utils/copyright.py style $(PYCHECKGLOBS) $(JSCHECKGLOBS) $(MDCHECKGLOBS) $(MDCHECKFILES)
	@echo "Running python styling";
	black $(CHECKDIRS);
	isort $(CHECKDIRS);


# run tests for the repo
test:
	@echo "Running python tests";
	@SPARSEZOO_TEST_MODE="true" pytest ./tests/;

# run example tests for the repo
test-examples:
	@echo "Running python example tests";
	@SPARSEZOO_TEST_MODE="true" pytest ./examples/;

# create docs
docs:
	@echo "Running docs creation";
	$(PYTHON) utils/docs_builder.py --src $(DOCDIR) --dest $(DOCDIR)/build/html;

docsupdate:
	@echo "Runnning update to api docs";
	find $(DOCDIR)/api | grep .rst | xargs rm -rf;
	sphinx-apidoc -o "$(DOCDIR)/api" src/deepsparse;

# creates wheel file
build:
	$(PYTHON) setup.py sdist bdist_wheel $(BUILD_ARGS)

# clean package
clean:
	rm -fr .pytest_cache;
	rm -fr docs/_build docs/build;
	find $(CHECKDIRS) | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -fr;
