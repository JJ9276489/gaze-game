PYTHON ?= .venv/bin/python

.PHONY: check check-js check-python test test-js test-python

check: check-js check-python test

check-js:
	node --check web/app.js
	node --check web/personal_model.js
	node --check web/game_logic.js

check-python:
	$(PYTHON) -m py_compile relay_server.py shared_gaze/*.py scripts/*.py

test: test-js test-python

test-js:
	node --test web/*.test.mjs

test-python:
	$(PYTHON) -m unittest discover -s tests
