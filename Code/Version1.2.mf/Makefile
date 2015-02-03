.PHONY: clean all help

.SILENT: help

define ANNOUNCE_HELP
The Cannon makefile 

Not much to say for now
Will provide common task to clean and install or build

targets:

clean -- clean the directory
endef

export ANNOUNCE_HELP

help:
	@echo "$$ANNOUNCE_HELP"

all: clean

clean: 
	find . -name '*pyc' -exec rm -f {} \; >> /dev/null
	find . -name '__pycache__' -exec rm -rf {} \; >> /dev/null
	rm -f *.so
