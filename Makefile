.PHONY: build docs test

BUILDDIR := $(PWD)
CHECKDIRS := examples tests src utils notebooks setup.py
PYCHECKGLOBS := 'examples/**/*.py' 'scripts/**/*.py' 'src/**/*.py' 'tests/**/*.py' 'utils/**/*.py' setup.py
DOCDIR := docs
MDCHECKGLOBS := 'docs/**/*.md' 'docs/**/*.rst' 'examples/**/*.md' 'notebooks/**/*.md' 'scripts/**/*.md'
MDCHECKFILES := CODE_OF_CONDUCT.md CONTRIBUTING.md DEVELOPING.md README.md

# run checks on all files for the repo
quality:
	@echo "Running copyright checks";
	python utils/copyright.py quality $(PYCHECKGLOBS) $(MDCHECKGLOBS) $(MDCHECKFILES)
	@echo "Running python quality checks";
	black --check $(CHECKDIRS);
	isort --check-only $(CHECKDIRS);
	flake8 $(CHECKDIRS);


# style the code according to accepted standards for the repo
style:
	@echo "Running copyrighting";
	python utils/copyright.py style $(PYCHECKGLOBS) $(JSCHECKGLOBS) $(MDCHECKGLOBS) $(MDCHECKFILES)
	@echo "Running python styling";
	black $(CHECKDIRS);
	isort $(CHECKDIRS);


# run tests for the repo
test:
	@echo "Running python tests";
	@pytest ./tests/;

# run example tests for the repo
test-examples:
	@echo "Running python example tests";
	@pytest ./examples/;

# create docs
docs:
	sphinx-apidoc -o "$(DOCDIR)/source/api" src/deepsparse;
	cd $(DOCDIR) && $(MAKE) html;

# creates wheel file
build:
	python3 setup.py sdist bdist_wheel

# clean package
clean:
	rm -fr .pytest_cache;
	rm -fr docs/_build docs/build;
	find $(CHECKDIRS) | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -fr;
	find $(DOCDIR) | grep .rst | xargs rm -fr;
