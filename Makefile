.PHONY: install generate-data run-attribution run-validation precompute test run clean

install:
	pip install -r requirements.txt

generate-data:
	python scripts/generate_data.py --config config/synthetic_data.yaml

run-attribution:
	python scripts/run_attribution.py --config config/model_params.yaml

run-validation:
	python scripts/run_validation.py

precompute:
	python scripts/precompute_cache.py

test:
	pytest tests/ -v

run:
	python app/app.py

clean:
	rm -rf data/raw/*.parquet data/processed/*.parquet
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
