# Used for building this project's documentation
# In most cases you want to run `make github` to prepare documentation (for GH)

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Get the parent dir name which is this docs' version
VERSION := $(notdir $(CURDIR))

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# this copies files from docs_src to docs for easy integration with GH Pages
github:
	@cp -rf ../../assets/* source/_static/
	@make html
	@mkdir -p ../../docs/$(VERSION)
	@rm -rf ../../docs/$(VERSION)/*
	@cp -a build/html/. ../../docs/$(VERSION)/
	@cp ../versions.json ../../docs/

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)