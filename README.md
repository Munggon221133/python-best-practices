# Python Backend Best Practices & Style Guide

This document outlines the standards and best practices for developing scalable, maintainable, and robust backend services using Python.

## 1. Code Style and Formatting
Consistency is the foundation of readability. We follow PEP 8 as the base standard.
- **Formatter**: Use Black for uncompromising code formatting.
- **Imports Ordering**: Use isort to group imports (Standard library, Third-party, Local).
- **Naming Conventions**:
  - `snake_case` for functions, methods, and variables.
  - `PascalCase` for classes.
  - `UPPER_CASE` for constants.
  - `_leading_underscore` for internal-use methods/variables.

## 2. Type Hinting and Validation
Python is dynamically typed, but backend systems require predictability.
- **Type Hints**: Always use Python type hints (typing module) for function signatures and class attributes.
- **Static Analysis**: Use mypy or pyright in the CI/CD pipeline to catch type errors before runtime.
- **Data Validation**: Use Pydantic (standard for FastAPI/modern backends) for data parsing and validation.

```python
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    age: int | None = None
```

## 3. Project Structure
A clean separation of concerns prevents "Spaghetti Code."
- **Directory Layout**:
  - `app/api/`: Route handlers and controllers.
  - `app/services/`: Business logic (The "Heavy Lifting").
  - `app/models/`: Database schemas (SQLAlchemy/SQLModel).
  - `app/schemas/`: Pydantic models for request/response.
  - `app/core/`: Configuration, security, and global constants.
- **Dependency Injection**: Use dependency injection (built-in in FastAPI) to make code testable and modular.

## 4. Performance and Concurrency
Modern Python backends must handle I/O-bound tasks efficiently.
- **Asyncio**: Use async and await for I/O operations (Database calls, API requests).
- **Avoid Blocking**: Never use blocking libraries (like requests) inside async functions; use httpx instead.
- **Profiling**: Use cProfile or py-spy to identify bottlenecks in CPU-bound code.

## 5. Database Best Practices
- **Migrations**: Never manually update the database schema. Use Alembic.
- **Connection Pooling**: Ensure the database driver manages a pool of connections (e.g., SQLAlchemy's QueuePool).
- **N+1 Query Problem**: Always use joinedload or selectinload when fetching related objects to minimize round trips to the DB.

## 6. Error Handling and Logging
- **Exceptions**: Define custom exception classes for domain-specific errors.
- **Global Exception Handler**: Use middleware to catch unhandled exceptions and return a structured JSON response (e.g., `{"error": "Internal Server Error"}`).
- **Structured Logging**: Use structlog or the standard logging library with a JSON formatter for easy parsing in ELK/Datadog.

## 7. Security Essentials
- **Environment Variables**: Never hardcode secrets. Use .env files (via python-dotenv or Pydantic BaseSettings).
- **Password Hashing**: Use Argon2 or Bcrypt. Never store passwords in plain text.
- **CORS**: Strictly define allowed origins in your middleware.

## 8. Testing Strategy
A backend without tests is a liability.
- **Framework**: Use pytest for its powerful fixture system.
- **Unit Tests**: Focus on testing business logic in services/.
- **Integration Tests**: Test the API endpoints using TestClient, mocking external dependencies (like Stripe or AWS S3).
- **Coverage**: Aim for at least 80% coverage, but prioritize critical paths.

## 9. Tooling and CI/CD
Automate the enforcement of these rules.
- **Pre-commit Hooks**: Use `.pre-commit-config.yaml` to run black, isort, flake8, and mypy locally before every commit.
- **Containerization**: Use Docker with multi-stage builds to keep production images small and secure.
- **Virtual Environments**: Use Poetry or uv for deterministic dependency management (pyproject.toml).

## pyproject.toml and Pre-commit Configuration

Here are the two most critical files for any modern Python backend repository. These are pre-configured to work together seamlessly using Poetry (the industry standard for dependency management) and pre-commit.

### 1. pyproject.toml
This file replaces requirements.txt, setup.py, and individual tool configs (like .black or .isort). It ensures your environment is identical across your entire team.

```toml
[tool.poetry]
name = "my-awesome-backend"
version = "0.1.0"
description = "High-performance FastAPI backend with best practices"
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
pydantic = {extras = ["email"], version = "^2.6.0"}
sqlalchemy = "^2.0.25"
alembic = "^1.13.1"
python-dotenv = "^1.0.1"
httpx = "^0.26.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.5"
black = "^24.1.1"
isort = "^5.13.2"
mypy = "^1.8.0"
flake8 = "^7.0.0"
pre-commit = "^3.6.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# Tool Configurations
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
exclude = ["venv", ".venv", "alembic"]
```

### 2. .pre-commit-config.yaml
This is your "Gatekeeper." It automatically formats and checks your code every time you run git commit. If the code doesn't meet your standards, the commit is blocked until fixed.

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, pydantic]
```

## How to Use These
1. **Initialize Poetry**: Install your dependencies with `poetry install`.
2. **Install Hooks**: Run `poetry run pre-commit install`.
3. **Commit with confidence**: Next time you `git commit`, these tools will run automatically.

## CI/CD Pipeline

GitHub Actions workflow will serve as your final line of defense. It ensures that every Pull Request (PR) or push to your main branch is automatically checked for quality, type safety, and passing tests before any code can be merged.

### Create this file at `.github/workflows/ci.yaml` in your project root:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  quality-checks:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Poetry
        uses: snok/install-poetry@v1
        with:
          virtualenvs-create: true
          virtualenvs-in-project: true

      - name: Cache Virtual Environment
        uses: actions/cache@v4
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ hashFiles('**/poetry.lock') }}

      - name: Install Dependencies
        run: poetry install --no-interaction

      - name: Lint with Black
        run: poetry run black --check .

      - name: Sort Imports with isort
        run: poetry run isort --check-only .

      - name: Type Check with Mypy
        run: poetry run mypy .

      - name: Run Tests with Pytest
        run: poetry run pytest --cov=app --cov-report=xml

      - name: Upload Coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          file: ./coverage.xml
          fail_ci_if_error: false
```

## Why This CI Setup is Industry-Standard
- **Caching**: It uses actions/cache to store your virtual environment. This cuts down your build time significantly (often from 2 minutes to 30 seconds) because it doesn't reinstall every package from scratch on every run.
- **Safety Checks**: By using --check flags for Black and isort, the build will fail if someone forgot to format their code locally.
- **Test Coverage**: It generates a coverage report and attempts to upload it. This helps you track which parts of your backend are "dark" (untested).
- **Environment Parity**: Since it uses your poetry.lock file, you can be 100% sure the tests are running against the exact same library versions you used in development.

## Pro-Tip for Your Repository Settings
Once this file is in your repo, go to **Settings > Branches > Add Rule** for your main branch and enable "Require status checks to pass before merging." This makes it physically impossible to break the build or merge unformatted code.