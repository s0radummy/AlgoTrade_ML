.PHONY: help setup up down logs test lint clean format install docker-build docker-up docker-down health

help:
	@echo "AlgoTrading Makefile Commands"
	@echo "============================="
	@echo "make setup           - Setup environment (.env, directories)"
	@echo "make install         - Install Python dependencies"
	@echo "make docker-build    - Build Docker images"
	@echo "make docker-up       - Start all services (docker-compose up -d)"
	@echo "make docker-down     - Stop all services (docker-compose down)"
	@echo "make up              - Alias for docker-up"
	@echo "make down            - Alias for docker-down"
	@echo "make logs            - View live logs"
	@echo "make health          - Check service health"
	@echo "make test            - Run all tests"
	@echo "make lint            - Run code linting"
	@echo "make format          - Format code with black"
	@echo "make clean           - Clean cache and logs"
	@echo "make producer        - Run mock producer"
	@echo "make api             - Run API server locally"

setup:
	@echo "Setting up environment..."
	cp .env.example .env
	mkdir -p logs models/checkpoints data/backtest
	@echo "✓ Setup complete. Edit .env with your credentials"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

docker-build:
	@echo "Building Docker images..."
	docker-compose build --no-cache
	@echo "✓ Build complete"

docker-up:
	@echo "Starting services..."
	docker-compose up -d
	@echo "✓ Services started. View logs with 'make logs'"

docker-down:
	@echo "Stopping services..."
	docker-compose down
	@echo "✓ Services stopped"

up: docker-up

down: docker-down

logs:
	docker-compose logs -f

health:
	@echo "Checking service health..."
	@docker-compose ps
	@echo ""
	@curl -s http://localhost:8000/health | python -m json.tool || echo "API not responding"
	@echo ""
	@redis-cli ping 2>/dev/null && echo "✓ Redis: OK" || echo "✗ Redis: FAILED"

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✓ Tests complete. Coverage: htmlcov/index.html"

lint:
	@echo "Linting code..."
	flake8 src/ tests/ --max-line-length=100 || true
	@echo "✓ Linting complete"

format:
	@echo "Formatting code..."
	black src/ tests/ scripts/
	@echo "✓ Formatting complete"

clean:
	@echo "Cleaning cache and logs..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	@echo "✓ Cleaned"

producer:
	@echo "Running mock producer..."
	python scripts/producer.py

api:
	@echo "Running API server..."
	python -m src.api.app

model-gen:
	@echo "Generating dummy model..."
	python scripts/generate_model.py

shell:
	@echo "Starting Python shell..."
	python -c "from src.utils.logger import setup_logger; logger = setup_logger('shell'); import pdb; pdb.set_trace()" || python

# Development shortcuts
dev: docker-up
	@echo "✓ Development environment ready"
	@make health

restart:
	@echo "Restarting services..."
	docker-compose restart
	@echo "✓ Services restarted"

reset:
	@echo "WARNING: This will delete all data!"
	@read -p "Continue? (y/N) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		rm -rf logs/* models/*.pth; \
		make docker-up; \
	fi

ps:
	docker-compose ps

exec:
	@read -p "Container name: " container; \
	docker-compose exec $$container bash
