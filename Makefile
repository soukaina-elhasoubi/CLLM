.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calling_results.json

debug:
	uv run python -m pdb -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calling_results.json

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf data/output/*

fclean: clean
	rm -rf .venv

lint:
	uv run flake8 src/ data/
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 src/ data/
	uv run mypy . --strict


# # install:
# # 	python -m pip install --upgrade pip
# # 	python -m pip install uv numpy pydantic
# # 	uv sync

# # run:
# # 	uv run python -m src

# # debug:
# # 	uv run python -m pdb -m src

# # clean:
# # 	python - <<'PY'
# # import pathlib, shutil
# # root = pathlib.Path('.').resolve()
# # for path in root.rglob('__pycache__'):
# #     if path.is_dir():
# #         shutil.rmtree(path)
# # for path in [root / '.mypy_cache', root / 'data' / 'output']:
# #     if path.exists():
# #         shutil.rmtree(path)
# # print('clean complete')
# # PY

# # lint:
# # 	flake8 .
# # 	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# # lint-strict:
# # 	flake8 .
# # 	mypy . --strict


# .PHONY: install run debug clean lint lint-strict

# install:
# # 	python -m pip install --upgrade pip
# # 	python -m pip install uv numpy pydantic
# 	uv sync

# run:
# 	uv run python -m src

# debug:
# 	uv run python -m pdb -m src

# clean:
# 	python -c "import pathlib, shutil; root=pathlib.Path('.').resolve(); [shutil.rmtree(p) for p in root.rglob('__pycache__') if p.is_dir()]; [shutil.rmtree(p) for p in [root/'.mypy_cache', root/'data'/'output'] if p.exists()]; print('clean complete')"

# lint:
# 	flake8 .
# 	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# lint-strict:
# 	flake8 .
# 	mypy . --strict


# # .PHONY: install run debug clean lint lint-strict

# # install:
# # 	python -m pip install --upgrade pip
# # 	python -m pip install uv numpy pydantic
# # 	uv sync

# # run:
# # 	uv run python -m src

# # debug:
# # 	uv run python -m pdb -m src

# # clean:
# # 	python -c "import pathlib, shutil; root=pathlib.Path('.').resolve(); [shutil.rmtree(p) for p in root.rglob('__pycache__') if p.is_dir()]; [shutil.rmtree(p) for p in [root/'.mypy_cache', root/'data'/'output'] if p.exists()]; print('clean complete')"

# # lint:
# # 	flake8 .
# # 	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# # lint-strict:
# # 	flake8 .
# # 	mypy . --strict
