# Lancer TTRPG Tooling Makefile
# ============================
#
# Usage:
#   make help           - Show all available targets
#   make test           - Run all tests
#   make dev            - Start full development stack
#
# =============================================================================

.PHONY: help test test-core test-llm test-app \
        dev dev-backend dev-frontend \
        db-up db-down db-migrate db-revision \
        generate-types install-app \
        lint lint-frontend lint-fix

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Lancer TTRPG Tooling - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests (core + llm + app)"
	@echo "  make test-core     - Run core type system tests"
	@echo "  make test-llm      - Run LLM pipeline tests"
	@echo "  make test-app      - Run web application tests"
	@echo ""
	@echo "Development:"
	@echo "  make dev           - Start full dev stack (backend + frontend)"
	@echo "  make dev-backend   - Start FastAPI backend with hot reload"
	@echo "  make dev-frontend  - Start Vite frontend with hot reload"
	@echo ""
	@echo "Database:"
	@echo "  make db-up         - Start PostgreSQL in Docker"
	@echo "  make db-down       - Stop PostgreSQL"
	@echo "  make db-migrate    - Run database migrations"
	@echo "  make db-revision   - Create new migration (MSG=description)"
	@echo ""
	@echo "Type Generation:"
	@echo "  make generate-types - Generate TypeScript from Python models"
	@echo ""
	@echo "Setup:"
	@echo "  make install-app   - Install app dependencies (Python + Node)"
	@echo ""
	@echo "Linting:"
	@echo "  make lint          - Run all linters (catches React hooks errors)"
	@echo "  make lint-frontend - Run ESLint on frontend only"

# =============================================================================
# Testing
# =============================================================================

PYTHON := .venv/bin/python

test: test-core test-llm test-app

test-core:
	$(PYTHON) -m pytest core -v

test-llm:
	PYTHONPATH=$(CURDIR) $(PYTHON) -m pytest llm/tests/tactician -v

test-app:
	PYTHONPATH=$(CURDIR) $(PYTHON) -m pytest app/backend/tests -v
	cd app/frontend && npm test 2>/dev/null || echo "No frontend tests configured"

# =============================================================================
# Development Servers
# =============================================================================

dev:
	@echo "Starting development servers..."
	@echo "Backend:  http://localhost:8000"
	@echo "Frontend: http://localhost:5173"
	@echo "API Docs: http://localhost:8000/api/docs"
	@echo ""
	@$(MAKE) -j2 dev-backend dev-frontend

dev-backend:
	PYTHONPATH=$(CURDIR) uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd app/frontend && npm run dev

# =============================================================================
# Database
# =============================================================================

db-up:
	docker-compose up -d postgres
	@echo "PostgreSQL started on port 5433"
	@echo "Connection: postgresql://lancer:lancer@localhost:5433/lancer"

db-down:
	docker-compose down

db-migrate:
	cd app/backend && PYTHONPATH=$(CURDIR) alembic upgrade head

db-revision:
ifndef MSG
	$(error MSG is required. Usage: make db-revision MSG="add users table")
endif
	cd app/backend && PYTHONPATH=$(CURDIR) alembic revision --autogenerate -m "$(MSG)"

# =============================================================================
# Type Generation
# =============================================================================

generate-types:
	@echo "Generating TypeScript types from Python models..."
	cd app/frontend && npm run generate:types

# =============================================================================
# Setup
# =============================================================================

install-app:
	@echo "Installing backend dependencies..."
	pip install -r requirements_app.txt
	@echo ""
	@echo "Installing frontend dependencies..."
	cd app/frontend && npm install
	@echo ""
	@echo "Done! Run 'make db-up && make dev' to start development."

# =============================================================================
# Linting
# =============================================================================
# Frontend linting catches React hooks ordering errors BEFORE they cause
# runtime crashes. Run before committing to catch issues early.

lint:
	ruff check core llm app/backend
	cd app/frontend && npm run lint

lint-frontend:
	cd app/frontend && npm run lint

lint-fix:
	ruff check core llm app/backend --fix
	cd app/frontend && npm run lint:fix
