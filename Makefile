install:
	@uv sync

run: install
	@uv run python -m src

debug:
	@uv run python -m pdb -m src

clean:
	@rm -rf \
		__pycache__ \
		*/__pycache__ \
		.mypy_cache \
		.pytest_cache \
		.ruff_cache

lint: install
	@uv run flake8 src
	@uv run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs