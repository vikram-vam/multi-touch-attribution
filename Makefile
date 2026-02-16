.PHONY: help install run generate-data test clean

help:
	@echo "Erie MCA Demo - Available Commands"
	@echo "===================================="
	@echo "make install        Install dependencies"
	@echo "make generate-data  Generate synthetic data and run models"
	@echo "make run            Launch the dashboard"
	@echo "make test           Run tests"
	@echo "make clean          Remove generated data and cache"

install:
	pip install -r requirements.txt

generate-data:
	python main.py

run:
	python app.py

test:
	pytest tests/ -v

clean:
	rm -rf data/synthetic/*
	rm -rf data/results/*
	rm -rf data/cache/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
