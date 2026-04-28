PYTHON ?= .venv/bin/python

.PHONY: check check-js check-python test test-js test-python vendor-browser-runtime export-browser-models verify-browser-models

check: check-js check-python test

check-js:
	node --check web/app.js
	node --check web/personal_model.js
	node --check web/game_logic.js
	node --check web/calibration_logic.js
	node --check web/relay_client.js
	node --check web/gaze_runtime.js
	node --check web/renderer.js
	node --check web/trainer_session.js
	node --check web/ui_state.js

check-python:
	$(PYTHON) -m py_compile relay_server.py shared_gaze/*.py scripts/*.py

test: test-js test-python

test-js:
	node --test web/*.test.mjs

test-python:
	$(PYTHON) -m unittest discover -s tests

vendor-browser-runtime:
	$(PYTHON) scripts/vendor_browser_runtime.py

export-browser-models:
	$(PYTHON) scripts/export_browser_onnx.py

verify-browser-models:
	$(PYTHON) scripts/verify_browser_onnx.py
