# run checks on all files for the repo
quality:
    [TODO]

# style the code according to accepted standards for the repo
style:
    [TODO]

# run tests for the repo
test:
    [TODO]

# create docs
docs:
	sphinx-apidoc -o "$(DOCDIR)/source/" src/nmie;
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
