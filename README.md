# Python Backend Engineering Standard

> A comprehensive, production-grade guide for building scalable, maintainable, and robust backend services in Python — aimed at senior engineers and teams striving for engineering excellence.

---

## Table of Contents

1. [Code Style & Formatting](#1-code-style--formatting)
2. [Type Hinting & Static Analysis](#2-type-hinting--static-analysis)
3. [Data Validation & Serialization](#3-data-validation--serialization)
4. [Project Structure & Architecture](#4-project-structure--architecture)
5. [Configuration Management](#5-configuration-management)
6. [API Design & Conventions](#6-api-design--conventions)
7. [Dependency Injection](#7-dependency-injection)
8. [Database Engineering](#8-database-engineering)
9. [Caching Strategies](#9-caching-strategies)
10. [Performance & Memory Optimization](#10-performance--memory-optimization)
11. [Async Programming & Task Queues](#11-async-programming--task-queues)
12. [Error Handling & Resilience](#12-error-handling--resilience)
13. [Observability & Monitoring](#13-observability--monitoring)
14. [Security Engineering](#14-security-engineering)
15. [Testing Strategy](#15-testing-strategy)
16. [Documentation Standards](#16-documentation-standards)
17. [Design Patterns](#17-design-patterns-in-python)
18. [Tooling, CI/CD & DevOps](#18-tooling-cicd--devops)

---

## 1. Code Style & Formatting

Consistency is the foundation of readability. We follow **PEP 8** as the base standard and use modern tooling for enforcement.

- **Linter & Formatter**: Use **Ruff** — a single, blazing-fast tool that replaces Black, isort, flake8, pyflakes, and dozens of other linters. It is the modern standard.
- **Fallback**: If Ruff is not adopted, use **Black** (formatter) + **isort** (imports) + **flake8** (linter) individually.
- **Naming Conventions**:
  - `snake_case` for functions, methods, and variables.
  - `PascalCase` for classes.
  - `UPPER_CASE` for constants.
  - `_leading_underscore` for internal/private methods and variables.
- **Docstrings**: Follow **Google style** docstrings on all public functions, classes, and modules.

```python
def calculate_discount(price: float, rate: float) -> float:
    """Calculate the discounted price.

    Args:
        price: The original price of the item.
        rate: The discount rate as a decimal (e.g., 0.1 for 10%).

    Returns:
        The price after discount has been applied.

    Raises:
        ValueError: If rate is not between 0 and 1.
    """
    if not 0 <= rate <= 1:
        raise ValueError(f"Discount rate must be between 0 and 1, got {rate}")
    return price * (1 - rate)
```

---

## 2. Type Hinting & Static Analysis

Python is dynamically typed, but production backends require predictability and safety.

- **Always use type hints** for function signatures, return types, and class attributes.
- **Static Analysis**: Run **mypy** (strict mode) or **pyright** in CI to catch errors before runtime.
- **Advanced Patterns**: Leverage `Protocol` for structural subtyping, `TypeVar` for generics, and `ParamSpec` for decorator typing.

```python
from typing import Protocol, TypeVar, Generic
from collections.abc import Sequence

# Structural subtyping with Protocol
class Repository(Protocol[T]):
    async def get(self, id: int) -> T | None: ...
    async def list(self) -> Sequence[T]: ...
    async def create(self, entity: T) -> T: ...
    async def delete(self, id: int) -> bool: ...

T = TypeVar("T")

# Generic service that works with any Repository implementation
class BaseService(Generic[T]):
    def __init__(self, repository: Repository[T]) -> None:
        self._repo = repository

    async def get_or_raise(self, id: int) -> T:
        entity = await self._repo.get(id)
        if entity is None:
            raise EntityNotFoundError(f"Entity with id={id} not found")
        return entity
```

---

## 3. Data Validation & Serialization

Use **Pydantic v2** as the standard for all data parsing, validation, and serialization.

- **Request/Response schemas** must always be Pydantic models — never raw dicts.
- **Custom Validators**: Use `@field_validator` and `@model_validator` for complex rules.
- **Discriminated Unions**: Use `Literal` + `Discriminator` for polymorphic payloads.
- **Strict Mode**: Enable `model_config = ConfigDict(strict=True)` for critical models.

```python
from pydantic import BaseModel, ConfigDict, EmailStr, field_validator
from enum import StrEnum

class UserRole(StrEnum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

class UserCreate(BaseModel):
    model_config = ConfigDict(strict=True, str_strip_whitespace=True)

    username: str
    email: EmailStr
    role: UserRole = UserRole.MEMBER
    age: int | None = None

    @field_validator("username")
    @classmethod
    def username_must_be_valid(cls, v: str) -> str:
        if len(v) < 3 or not v.isalnum():
            raise ValueError("Username must be >= 3 alphanumeric characters")
        return v.lower()

class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    role: UserRole
```

---

## 4. Project Structure & Architecture

A clean separation of concerns prevents "Spaghetti Code." Use a **layered architecture** with clear boundaries:

```
project-root/
├── src/
│   ├── api/                  # Route handlers (thin controllers)
│   │   ├── v1/
│   │   │   ├── users.py
│   │   │   └── orders.py
│   │   └── deps.py           # Shared dependencies for routes
│   ├── services/             # Business logic (the "heavy lifting")
│   │   ├── user_service.py
│   │   └── order_service.py
│   ├── repositories/         # Data access layer (DB queries)
│   │   ├── user_repo.py
│   │   └── order_repo.py
│   ├── models/               # SQLAlchemy / SQLModel ORM models
│   ├── schemas/              # Pydantic request/response models
│   ├── core/                 # Configuration, security, constants
│   │   ├── config.py
│   │   ├── security.py
│   │   └── exceptions.py
│   ├── workers/              # Background task handlers (Celery, etc.)
│   └── main.py               # Application factory & lifespan
├── migrations/               # Alembic migrations
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── pyproject.toml
├── Dockerfile
└── docker-compose.yaml
```

**Key Rules**:
- **Routes are thin**: They only parse input, call a service, and return the response. No business logic.
- **Services are pure business logic**: They orchestrate repositories and other services. They do NOT know about HTTP.
- **Repositories encapsulate data access**: They talk to the database and return domain objects.

---

## 5. Configuration Management

Follow the **12-Factor App** methodology. Never hardcode secrets or environment-specific values.

- Use **Pydantic `BaseSettings`** for typed, validated configuration with `.env` support.
- Define **environment tiers**: `local`, `staging`, `production`.
- Use **`lru_cache`** to ensure settings are loaded once and reused.

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    APP_NAME: str = "my-backend"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"  # local | staging | production

    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALLOWED_ORIGINS: list[str] = ["https://yourdomain.com"]

    # Observability
    SENTRY_DSN: str | None = None
    LOG_LEVEL: str = "INFO"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## 6. API Design & Conventions

Design APIs that are predictable, consistent, and a pleasure to consume.

- **RESTful Standards**: Use proper HTTP methods (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`) and status codes.
- **Versioning**: Prefix routes with `/api/v1/` to support non-breaking evolution.
- **Pagination**: Use cursor-based or offset-based pagination consistently.
- **Idempotency**: `PUT` and `DELETE` must be idempotent. Use idempotency keys for critical `POST` operations.
- **OpenAPI**: FastAPI auto-generates OpenAPI docs — enrich them with descriptions and examples.

```python
from fastapi import APIRouter, Query, status
from app.schemas.common import PaginatedResponse

router = APIRouter(prefix="/api/v1/users", tags=["Users"])

@router.get(
    "/",
    response_model=PaginatedResponse[UserResponse],
    summary="List all users",
    description="Retrieve a paginated list of users with optional filtering.",
)
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    role: UserRole | None = Query(None, description="Filter by role"),
    user_service: UserService = Depends(get_user_service),
) -> PaginatedResponse[UserResponse]:
    return await user_service.list_users(page=page, size=size, role=role)

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    return await user_service.create_user(payload)
```

**Standardized Error Response**:

```python
# All error responses follow a consistent structure
{
    "error": {
        "code": "USER_NOT_FOUND",
        "message": "User with id=42 was not found.",
        "details": null
    }
}
```

---

## 7. Dependency Injection

Use FastAPI's built-in `Depends()` system to make code **testable, modular, and composable**.

- **Chain dependencies**: Build a dependency graph (Settings → DB Session → Repository → Service).
- **Use factories**: Create dependency functions that return configured instances.
- **Scoped dependencies**: Use `yield` dependencies for resources that need cleanup (DB sessions, HTTP clients).

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

def get_user_repository(
    session: AsyncSession = Depends(get_db_session),
) -> UserRepository:
    return UserRepository(session)

def get_user_service(
    repo: UserRepository = Depends(get_user_repository),
    settings: Settings = Depends(get_settings),
) -> UserService:
    return UserService(repo, settings)
```

---

## 8. Database Engineering

- **ORM**: Use **SQLAlchemy 2.0** with async engine for modern, type-safe database interaction.
- **Migrations**: Always use **Alembic** with `--autogenerate`. Never modify schemas by hand.
- **Connection Pooling**: Configure `pool_size`, `max_overflow`, and `pool_pre_ping` on the engine.
- **N+1 Prevention**: Use `selectinload` or `joinedload` to eagerly fetch related objects.
- **Soft Deletes**: Use an `is_deleted` flag or `deleted_at` timestamp instead of hard deletes for audit trails.
- **Transactions**: Use explicit session scoping — commit on success, rollback on failure (handled by the DI session dependency).

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import func
from datetime import datetime

class Base(DeclarativeBase):
    pass

class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(), onupdate=func.now()
    )

class SoftDeleteMixin:
    deleted_at: Mapped[datetime | None] = mapped_column(default=None)

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

class User(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True, index=True)
    email: Mapped[str] = mapped_column(unique=True)
    hashed_password: Mapped[str]

# Async engine with connection pooling
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,       # Verify connections before use
    pool_recycle=3600,         # Recycle connections every hour
)
```

**Bulk Operations** — avoid inserting rows one-by-one:

```python
# ❌ Bad: N individual INSERT statements
for item in items:
    session.add(Item(**item))

# ✅ Good: Single bulk insert
from sqlalchemy import insert
await session.execute(insert(Item), items)
```

---

## 9. Caching Strategies

Caching is critical for performance at scale. Use **Redis** as the primary cache backend.

- **Cache-Aside Pattern**: Application checks cache first; on miss, reads from DB and populates cache.
- **TTL**: Always set a Time-To-Live to prevent stale data.
- **Cache Invalidation**: Invalidate on writes — the hardest problem in CS, so keep it simple.
- **Key Naming**: Use structured, namespaced keys: `app:users:{user_id}`.

```python
import json
from redis.asyncio import Redis
from functools import wraps
from typing import Callable, Any

class CacheService:
    def __init__(self, redis: Redis, default_ttl: int = 300) -> None:
        self._redis = redis
        self._default_ttl = default_ttl

    async def get(self, key: str) -> Any | None:
        data = await self._redis.get(key)
        return json.loads(data) if data else None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self._redis.set(
            key, json.dumps(value, default=str), ex=ttl or self._default_ttl
        )

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate all keys matching a pattern (use sparingly)."""
        async for key in self._redis.scan_iter(match=pattern):
            await self._redis.delete(key)

# Usage in service layer
class UserService:
    CACHE_KEY = "app:users:{user_id}"

    async def get_user(self, user_id: int) -> UserResponse:
        cache_key = self.CACHE_KEY.format(user_id=user_id)

        # Check cache first
        cached = await self.cache.get(cache_key)
        if cached:
            return UserResponse(**cached)

        # Cache miss — read from DB
        user = await self.repo.get(user_id)
        if not user:
            raise UserNotFoundError(user_id)

        response = UserResponse.model_validate(user)
        await self.cache.set(cache_key, response.model_dump())
        return response
```

---

## 10. Performance & Memory Optimization

Measure first, optimize second. Never guess where bottlenecks are.

- **Profiling**: Use **py-spy** (sampling profiler, zero overhead in production) or **cProfile** for CPU-bound analysis. Use **memray** for memory profiling.
- **`__slots__`**: Use on data-heavy classes to reduce memory footprint by ~40%.
- **Lazy Loading**: Don't load data you don't need. Use generators, lazy properties, and query-level filtering.
- **Avoid Blocking in Async**: Never call blocking libraries (e.g., `requests`, `time.sleep`) inside `async` functions — use `httpx`, `asyncio.sleep`, or run in a thread executor.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Memory-efficient class with __slots__
class Point:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

# Running blocking code safely in async context
_executor = ThreadPoolExecutor(max_workers=4)

async def run_cpu_bound_task(data: bytes) -> str:
    """Run a CPU-bound function without blocking the event loop."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, heavy_computation, data)
    return result
```

**Generator Pattern for Large Datasets**:

```python
# ❌ Bad: loads everything into memory
def get_all_records() -> list[dict]:
    return [process(r) for r in db.fetch_all()]

# ✅ Good: yields one at a time
def stream_records() -> Generator[dict, None, None]:
    for record in db.fetch_cursor():
        yield process(record)
```

---

## 11. Async Programming & Task Queues

Master async for I/O-bound work and task queues for background processing.

- **`async/await`**: Use for all I/O operations — database calls, HTTP requests, file I/O.
- **Task Queues**: Use **Celery** (with Redis/RabbitMQ) or **FastStream** for background jobs, scheduled tasks, and event-driven processing.
- **Background Tasks**: For lightweight fire-and-forget work, use FastAPI's `BackgroundTasks`.
- **Concurrency Primitives**: Use `asyncio.gather()` for parallel I/O, `asyncio.Semaphore` for rate limiting.

```python
import asyncio
import httpx

# Parallel I/O with gather
async def fetch_user_data(user_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        profile, orders, notifications = await asyncio.gather(
            client.get(f"/users/{user_id}/profile"),
            client.get(f"/users/{user_id}/orders"),
            client.get(f"/users/{user_id}/notifications"),
        )
    return {
        "profile": profile.json(),
        "orders": orders.json(),
        "notifications": notifications.json(),
    }

# Rate-limited concurrent processing
async def process_batch(items: list[str], max_concurrent: int = 10) -> list[str]:
    semaphore = asyncio.Semaphore(max_concurrent)
    async def process_one(item: str) -> str:
        async with semaphore:
            return await slow_external_call(item)
    return await asyncio.gather(*[process_one(i) for i in items])
```

**Celery Task Example**:

```python
from celery import Celery

celery_app = Celery("worker", broker="redis://localhost:6379/1")

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,  # Acknowledge after completion for reliability
)
def send_welcome_email(self, user_id: int) -> None:
    try:
        user = get_user_sync(user_id)
        email_client.send(to=user.email, template="welcome")
    except TransientError as exc:
        raise self.retry(exc=exc)
```

---

## 12. Error Handling & Resilience

Build systems that fail gracefully, not catastrophically.

### Exception Hierarchy

Define a clear, domain-specific exception hierarchy:

```python
class AppError(Exception):
    """Base exception for the application."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(message)

class NotFoundError(AppError):
    def __init__(self, resource: str, identifier: str | int) -> None:
        super().__init__(
            message=f"{resource} with id={identifier} not found",
            code=f"{resource.upper()}_NOT_FOUND",
        )

class ConflictError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="CONFLICT")

class ValidationError(AppError):
    def __init__(self, message: str, details: dict | None = None) -> None:
        self.details = details
        super().__init__(message=message, code="VALIDATION_ERROR")
```

### Global Exception Handler

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    status_map = {
        NotFoundError: 404,
        ConflictError: 409,
        ValidationError: 422,
    }
    status_code = status_map.get(type(exc), 500)
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )
```

### Resilience Patterns

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Retry with exponential backoff
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(TransientError),
)
async def call_external_api(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post("https://api.example.com/v1/data", json=payload)
        response.raise_for_status()
        return response.json()

# Circuit Breaker (simplified)
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0) -> None:
        self._failures = 0
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._last_failure_time: float | None = None
        self._state = "closed"  # closed | open | half-open

    async def call(self, func, *args, **kwargs):
        if self._state == "open":
            if self._should_attempt_reset():
                self._state = "half-open"
            else:
                raise CircuitOpenError("Circuit breaker is open")
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

---

## 13. Observability & Monitoring

You cannot fix what you cannot see. Implement the **three pillars** of observability: **Logs, Metrics, Traces**.

### Structured Logging

Use **structlog** for machine-parseable, context-rich logs:

```python
import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger()

# Bind request context (correlation ID) in middleware
from contextvars import ContextVar
from uuid import uuid4

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID", str(uuid4()))
    correlation_id_var.set(cid)
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    structlog.contextvars.unbind_contextvars("correlation_id")
    return response

# Usage in services
logger.info("user_created", user_id=user.id, email=user.email)
logger.warning("rate_limit_approaching", endpoint="/api/v1/search", current=95, limit=100)
```

### Health Checks

```python
@router.get("/health", tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db_session)) -> dict:
    checks = {}
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception:
        checks["database"] = "unhealthy"
    try:
        await redis.ping()
        checks["redis"] = "healthy"
    except Exception:
        checks["redis"] = "unhealthy"

    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
```

### Metrics & Tracing

- Use **Prometheus** client library to expose `/metrics` for Grafana dashboards.
- Use **OpenTelemetry** for distributed tracing across microservices.
- Monitor: request latency (p50, p95, p99), error rates, DB query duration, cache hit rates.

---

## 14. Security Engineering

Security is not a feature — it's a requirement at every layer.

### Authentication & Authorization

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings: Settings = Depends(get_settings),
) -> UserResponse:
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=["HS256"],
        )
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return await user_service.get_user(user_id)

# Role-based access control
def require_role(*roles: UserRole):
    async def checker(user: UserResponse = Depends(get_current_user)) -> UserResponse:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return checker

# Usage
@router.delete("/users/{user_id}", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def delete_user(user_id: int) -> None: ...
```

### Security Checklist

| Area | Requirement |
|---|---|
| **Secrets** | Use `.env` + Pydantic BaseSettings. Never commit secrets to git. |
| **Passwords** | Hash with **Argon2** (preferred) or **Bcrypt**. Never store plaintext. |
| **CORS** | Strictly whitelist allowed origins. Never use `*` in production. |
| **Rate Limiting** | Use **slowapi** or middleware to prevent abuse (e.g., 100 req/min per IP). |
| **Input Sanitization** | Pydantic handles most validation. Additionally guard against SQL injection via parameterized queries (ORM default). |
| **Dependency Scanning** | Run `pip-audit` or **Safety** in CI to detect vulnerable packages. |
| **HTTPS** | Enforce TLS termination at the load balancer / reverse proxy level. |
| **Headers** | Set security headers: `X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security`. |

---

## 15. Testing Strategy

A backend without tests is a liability. Tests are not optional — they are part of the definition of "done."

### Testing Pyramid

| Level | Focus | Tools | Speed |
|---|---|---|---|
| **Unit** | Single functions/methods in isolation | `pytest`, `unittest.mock` | ⚡ Fast |
| **Integration** | Multiple components together (API + DB) | `pytest`, `httpx.AsyncClient`, `testcontainers` | 🔄 Medium |
| **E2E / Contract** | Full request lifecycle, inter-service contracts | `pytest`, `schemathesis` | 🐢 Slow |

### Pytest Best Practices

- **Fixtures**: Use fixtures for setup/teardown. Prefer factory fixtures over static data.
- **Markers**: Use `@pytest.mark.asyncio`, `@pytest.mark.slow`, `@pytest.mark.integration` to categorize tests.
- **Coverage**: Aim for **≥ 80%** coverage, but prioritize testing **critical business logic paths** over chasing numbers.
- **Naming**: `test_<function>_<scenario>_<expected_result>` (e.g., `test_create_user_duplicate_email_raises_conflict`).

```python
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

# Factory fixture — flexible and reusable
@pytest.fixture
def user_factory():
    def _create(
        username: str = "testuser",
        email: str = "test@example.com",
        role: str = "member",
    ) -> UserCreate:
        return UserCreate(username=username, email=email, role=role)
    return _create

# Integration test with TestClient
@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_create_user_success(client: AsyncClient, user_factory):
    payload = user_factory(username="newuser").model_dump()
    response = await client.post("/api/v1/users/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "newuser"

@pytest.mark.asyncio
async def test_create_user_duplicate_email_raises_conflict(
    client: AsyncClient, user_factory
):
    payload = user_factory().model_dump()
    await client.post("/api/v1/users/", json=payload)  # First creation
    response = await client.post("/api/v1/users/", json=payload)  # Duplicate
    assert response.status_code == 409

# Unit test with mocking
@pytest.mark.asyncio
async def test_user_service_sends_welcome_email():
    mock_repo = AsyncMock(spec=UserRepository)
    mock_repo.create.return_value = User(id=1, username="test", email="t@t.com")
    mock_email = AsyncMock()

    service = UserService(repo=mock_repo, email_client=mock_email)
    await service.create_user(UserCreate(username="test", email="t@t.com"))

    mock_email.send_welcome.assert_called_once_with(user_id=1)
```

### Property-Based Testing

Use **Hypothesis** to generate thousands of randomized inputs and catch edge cases you'd never think of:

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0), st.floats(min_value=0, max_value=1))
def test_calculate_discount_never_negative(price, rate):
    result = calculate_discount(price, rate)
    assert result >= 0
```

---

## 16. Documentation Standards

Code without documentation is a gift to your future self — a bad one.

### Docstrings

Use **Google-style docstrings** on all public modules, classes, and functions:

```python
class UserService:
    """Service for managing user lifecycle operations.

    This service handles user creation, updates, deletion, and
    authentication workflows. It coordinates between the user
    repository and external services (email, notifications).

    Attributes:
        repo: The user data access repository.
        cache: Redis-backed cache for user lookups.
    """

    async def deactivate_user(self, user_id: int, reason: str) -> UserResponse:
        """Deactivate a user account and revoke all active sessions.

        This performs a soft delete, preserving the user record for
        audit purposes while revoking access.

        Args:
            user_id: The ID of the user to deactivate.
            reason: Human-readable reason for deactivation (stored in audit log).

        Returns:
            The updated user response with deactivated status.

        Raises:
            NotFoundError: If no user with the given ID exists.
            ConflictError: If the user is already deactivated.
        """
        ...
```

### Architecture Decision Records (ADRs)

For significant technical decisions, create an ADR in `docs/adr/`:

```markdown
# ADR-001: Use Redis for Session Storage

## Status: Accepted
## Date: 2025-01-15

## Context
We need a session storage solution that supports TTL, horizontal scaling,
and sub-millisecond reads.

## Decision
Use Redis as the session backend instead of database-backed sessions.

## Consequences
- ✅ Sub-millisecond session reads
- ✅ Built-in TTL for automatic expiry
- ⚠️ Additional infrastructure dependency
- ⚠️ Data is not persisted if Redis restarts without AOF/RDB
```

### Changelog

Maintain a `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com) format. Automate with **commitizen** and **Conventional Commits**.

---

## 17. Design Patterns in Python

Apply battle-tested patterns — but only where they reduce complexity, not add it.

### Repository Pattern

Encapsulate data access logic. Services never write SQL directly.

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, user_id: int) -> User | None:
        result = await self._session.execute(
            select(User).where(User.id == user_id, User.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        result = await self._session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def create(self, user: User) -> User:
        self._session.add(user)
        await self._session.flush()
        return user

    async def list_active(
        self, offset: int = 0, limit: int = 20
    ) -> tuple[list[User], int]:
        query = select(User).where(User.deleted_at.is_(None))
        count_result = await self._session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar_one()
        result = await self._session.execute(query.offset(offset).limit(limit))
        return list(result.scalars().all()), total
```

### Unit of Work Pattern

Coordinate multiple repository operations in a single transaction — already handled by FastAPI's DI session dependency (see Section 7), where `commit()` and `rollback()` are managed automatically.

### Strategy Pattern

Use when you need interchangeable algorithms. Combine with dependency injection:

```python
from abc import ABC, abstractmethod

class NotificationStrategy(ABC):
    @abstractmethod
    async def send(self, user_id: int, message: str) -> None: ...

class EmailNotification(NotificationStrategy):
    async def send(self, user_id: int, message: str) -> None:
        user = await self.user_repo.get_by_id(user_id)
        await self.email_client.send(to=user.email, body=message)

class SlackNotification(NotificationStrategy):
    async def send(self, user_id: int, message: str) -> None:
        user = await self.user_repo.get_by_id(user_id)
        await self.slack_client.post_message(channel=user.slack_id, text=message)

class PushNotification(NotificationStrategy):
    async def send(self, user_id: int, message: str) -> None:
        await self.push_service.send(user_id=user_id, body=message)

# Service using the strategy
class AlertService:
    def __init__(self, strategies: list[NotificationStrategy]) -> None:
        self._strategies = strategies

    async def alert_user(self, user_id: int, message: str) -> None:
        await asyncio.gather(
            *[s.send(user_id, message) for s in self._strategies]
        )
```

---

## 18. Tooling, CI/CD & DevOps

Automate everything. If a human can forget to do it, a machine should enforce it.

### pyproject.toml

The single source of truth for dependencies, tool configs, and project metadata. This uses **Ruff** as the unified linter/formatter, replacing Black, isort, and flake8.

```toml
[tool.poetry]
name = "my-awesome-backend"
version = "0.1.0"
description = "Production-grade FastAPI backend"
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.12"
fastapi = "^0.115.0"
uvicorn = {extras = ["standard"], version = "^0.34.0"}
pydantic = {extras = ["email"], version = "^2.10.0"}
pydantic-settings = "^2.7.0"
sqlalchemy = {extras = ["asyncio"], version = "^2.0.36"}
alembic = "^1.14.0"
httpx = "^0.28.0"
redis = {extras = ["hiredis"], version = "^5.2.0"}
structlog = "^24.4.0"
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
tenacity = "^9.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.0"
pytest-asyncio = "^0.24.0"
pytest-cov = "^6.0.0"
hypothesis = "^6.118.0"
ruff = "^0.8.0"
mypy = "^1.13.0"
pre-commit = "^4.0.0"
pip-audit = "^2.7.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# ─── Ruff (replaces Black + isort + flake8) ───
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "S",    # flake8-bandit (security)
    "A",    # flake8-builtins
    "C4",   # flake8-comprehensions
    "DTZ",  # flake8-datetimez
    "T20",  # flake8-print
    "RET",  # flake8-return
    "PTH",  # flake8-use-pathlib
    "ERA",  # eradicate (commented-out code)
    "RUF",  # ruff-specific rules
]
ignore = ["E501"]  # Line length handled by formatter

[tool.ruff.lint.isort]
known-first-party = ["src"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

# ─── Mypy ───
[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
exclude = ["venv", ".venv", "alembic", "tests"]

# ─── Pytest ───
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests",
]
addopts = "--strict-markers -v"
```

### .pre-commit-config.yaml

Your automated gatekeeper — blocks commits that don't meet standards:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff        # Linter
        args: [--fix]
      - id: ruff-format  # Formatter

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, types-redis]
```

### Dockerfile (Multi-Stage)

Keep production images minimal and secure:

```dockerfile
# ── Stage 1: Builder ──
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir poetry==1.8.4
WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi --only main

COPY . .

# ── Stage 2: Production ──
FROM python:3.12-slim AS production

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY --from=builder /app/.venv .venv
COPY --from=builder /app/src src/
COPY --from=builder /app/migrations migrations/
COPY --from=builder /app/alembic.ini .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import httpx; httpx.get('http://localhost:8000/health')"]

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### GitHub Actions CI/CD

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Poetry
        uses: snok/install-poetry@v1
        with:
          virtualenvs-create: true
          virtualenvs-in-project: true

      - name: Cache dependencies
        uses: actions/cache@v4
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ hashFiles('**/poetry.lock') }}

      - name: Install dependencies
        run: poetry install --no-interaction

      - name: Lint & Format (Ruff)
        run: |
          poetry run ruff check .
          poetry run ruff format --check .

      - name: Type Check (Mypy)
        run: poetry run mypy src/

      - name: Security Audit
        run: poetry run pip-audit

      - name: Run Tests
        run: poetry run pytest --cov=src --cov-report=xml --cov-fail-under=80

      - name: Upload Coverage
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          file: ./coverage.xml

  build:
    needs: quality
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker Image
        run: docker build -t ${{ github.repository }}:${{ github.sha }} .

      - name: Push to Registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push ${{ github.repository }}:${{ github.sha }}
```

### Quick-Start Commands

```bash
# 1. Install dependencies
poetry install

# 2. Install pre-commit hooks
poetry run pre-commit install

# 3. Run all checks locally
poetry run ruff check . && poetry run ruff format --check . && poetry run mypy src/ && poetry run pytest

# 4. Run development server
poetry run uvicorn src.main:app --reload --port 8000
```

### Repository Settings

Once CI is in place, protect your `main` branch:
- **Settings → Branches → Add Rule** for `main`
- ✅ Require status checks to pass before merging
- ✅ Require pull request reviews (minimum 1)
- ✅ Require linear history (squash merges)
- ✅ Do not allow bypassing the above settings

---

> **This document is a living standard.** Update it as the team evolves, new patterns emerge, and tools improve. Every engineer on the team should read this before writing their first line of code.