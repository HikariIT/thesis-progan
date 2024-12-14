test_py:
	@echo "Running tests"
	pytest

coverage:
	pytest --cov-report html --cov=. test/