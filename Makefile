.PHONY: test test-core test-llm

test: test-core test-llm

test-core:
	python -m pytest core

test-llm:
	cd llm && python -m pytest tests
