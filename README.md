# robyn-ml-api

> **Production-ready ML API template** powered by [Robyn](https://robyn.tech), featuring typed routing, event-driven architecture, and Pydantic validation.

A modern, fast, and type-safe foundation for building Machine Learning APIs with Python 3.12+.

## Features

- **Custom Router** - Automatic body parsing, Pydantic validation, and response serialization
- **File Uploads** - Typed `UploadFile` with automatic request injection (no explicit `request` param needed)
- **Middleware System** - Abstract `BaseMiddleware` with before/after hooks and endpoint filtering
- **Event-driven Lifespan** - Clean startup/shutdown lifecycle for ML models, pools, and connections
- **Strong Typing** - Full type hints with Pydantic models for request/response validation
- **Structured Logging** - Debug-aware logging with structlog and correlation IDs
- **Modern Packaging** - uv + Hatch for fast, reproducible builds with lockfiles
- **Docker Ready** - Multi-stage builds optimized for production
- **CI/CD** - GitHub Actions with frozen dependencies

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/robyn-ml-api.git
cd robyn-ml-api

# Copy environment template
cp .env.template .env

# Install dependencies
make install
```

### 2. Run Development Server

```bash
make dev
# or
make run
```

The API will be available at `http://localhost:8000`

### 3. Test the Health Endpoint

```bash
curl http://localhost:8000/health
```

## Project Structure

```
robyn-ml-api/
├── app/
│   ├── api/              # API endpoints (routers)
│   │   └── health.py     # Health check endpoint example
│   ├── core/             # Core framework components
│   │   ├── lifespan.py   # Event-driven lifecycle management
│   │   ├── logger.py     # Structured logging configuration
│   │   ├── router.py     # Custom router with body/file parsing
│   │   └── settings.py   # Pydantic settings management
│   ├── events/           # Lifespan events
│   │   └── process_pool.py  # ProcessPoolExecutor example
│   ├── middlewares/      # Middleware system
│   │   ├── base.py       # BaseMiddleware + MiddlewareHandler
│   │   └── files.py      # OpenAPI file upload patching
│   ├── models/           # Pydantic request/response models
│   │   └── core.py       # UploadFile and core types
│   ├── operation/        # Business logic utilities
│   ├── providers/        # External services/model loaders
│   └── main.py           # Application entrypoint
├── test/
│   └── unit/             # Unit tests
├── .github/workflows/    # CI configuration
├── compose.yml           # Docker Compose
├── Dockerfile            # Multi-stage Docker build
├── Makefile              # Development commands
├── pyproject.toml        # Project configuration (uv + Hatch)
└── uv.lock               # Locked dependencies
```

---

## Usage Guide

### Creating API Endpoints

Create a new file in `app/api/` following the naming convention:

```python
# app/api/prediction.py
"""Prediction endpoint for ML inference."""

from pydantic import BaseModel

from app.core.logger import LogIcon, logger
from app.core.router import Router

router = Router(__file__, prefix="/ml")


class PredictionRequest(BaseModel):
    """Input data for prediction."""
    features: list[float]
    model_name: str = "default"


class PredictionResponse(BaseModel):
    """Prediction result."""
    prediction: float
    confidence: float
    model_version: str


@router.post("/predict")
async def predict(body: PredictionRequest, global_dependencies) -> PredictionResponse:
    """Run ML inference on input features."""
    logger.info("Prediction requested", icon=LogIcon.MODEL, model=body.model_name)
    
    # Access shared state (loaded models, pools, etc.)
    state = global_dependencies["state"]
    model = state.ml_model  # Your loaded model from lifespan event
    
    # Run prediction
    result = await model.predict(body.features)
    
    logger.info("Prediction complete", icon=LogIcon.SUCCESS)
    return PredictionResponse(
        prediction=result.value,
        confidence=result.confidence,
        model_version=model.version
    )
```

### Register the Router in `main.py`

```python
# app/main.py
from robyn import Robyn

from app.api.health import router as health_router
from app.api.prediction import router as prediction_router  # Add this
from app.core.lifespan import create_lifespan
from app.core.logger import logger
from app.core.settings import settings as st

app = Robyn(__file__)

lifespan = create_lifespan(app)
# Register your events here

app.startup_handler(lifespan.startup)
app.shutdown_handler(lifespan.shutdown)
app.include_router(health_router)
app.include_router(prediction_router)  # Add this


def main() -> None:
    logger.info("🚀 STARTING %s", st.API_NAME)
    app.start(host=st.API_HOST, port=st.API_PORT)


if __name__ == "__main__":
    main()
```

### Router Features

The custom `Router` provides automatic:

1. **Body Parsing** - JSON bodies automatically parsed based on type annotations
2. **Pydantic Validation** - Request models validated with proper error responses (422)
3. **Response Serialization** - Pydantic models, dicts, or raw responses handled automatically

```python
# Different body types supported:

@router.post("/pydantic")
async def with_pydantic(body: MyPydanticModel) -> ResponseModel:
    # body is automatically validated and parsed
    pass

@router.post("/dict")
async def with_dict(body: dict) -> dict:
    # body is parsed as JSON dict
    pass

@router.post("/raw")
async def with_raw(body) -> str:
    # body parameter named 'body' is parsed as JSON
    pass
```

---

## Lifespan Events

### Creating Custom Events

Events manage resources that need initialization at startup and cleanup at shutdown:

```python
# app/events/ml_model.py
"""ML Model lifespan event."""

from app.core.lifespan import BaseEvent
from app.core.settings import settings as st


class MLModelWrapper:
    """Wrapper for your ML model."""
    
    def __init__(self, model, version: str):
        self.model = model
        self.version = version
    
    async def predict(self, features: list[float]) -> dict:
        # Your inference logic
        return {"value": 0.95, "confidence": 0.87}


class MLModelEvent(BaseEvent[MLModelWrapper]):
    """Manages ML model lifecycle."""
    
    name = "ml_model"  # This becomes the attribute name in state
    
    async def startup(self) -> MLModelWrapper:
        """Load model at startup."""
        # Load your model (sklearn, pytorch, etc.)
        import joblib
        model = joblib.load(st.MODELS_PATH / "model.pkl")
        return MLModelWrapper(model, version="1.0.0")
    
    async def shutdown(self, instance: MLModelWrapper) -> None:
        """Optional cleanup."""
        # Release resources if needed
        pass
```

### Register Events in `main.py`

```python
from app.events.ml_model import MLModelEvent
from app.events.process_pool import ProcessPoolEvent

app = Robyn(__file__)

lifespan = create_lifespan(app)
lifespan.register(ProcessPoolEvent)  # For CPU-bound tasks
lifespan.register(MLModelEvent)      # Your ML model

app.startup_handler(lifespan.startup)
app.shutdown_handler(lifespan.shutdown)
```

### Access State in Endpoints

```python
@router.post("/predict")
async def predict(body: PredictionRequest, global_dependencies) -> PredictionResponse:
    state = global_dependencies["state"]
    
    # Access your registered events by name
    model = state.ml_model        # MLModelEvent with name="ml_model"
    pool = state.process_pool     # ProcessPoolEvent with name="process_pool"
    
    # Use process pool for CPU-bound inference
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(pool, model.predict, body.features)
    
    return PredictionResponse(prediction=result)
```

### ProcessPoolEvent Example

The included `ProcessPoolEvent` demonstrates managing a process pool for CPU-bound ML tasks:

```python
# Already included in app/events/process_pool.py

class ProcessPoolEvent(BaseEvent[ProcessPoolExecutor]):
    """Manages ProcessPoolExecutor lifecycle."""
    
    name = "process_pool"
    
    async def startup(self) -> ProcessPoolExecutor:
        """Create process pool with spawn context."""
        max_workers = st.MAX_WORKERS or os.cpu_count()
        return create_process_pool(max_workers=max_workers)
    
    async def shutdown(self, instance: ProcessPoolExecutor) -> None:
        """Shutdown the process pool."""
        instance.shutdown(wait=True)
```

---

## Middlewares

### Middleware Architecture

The `BaseMiddleware` provides an abstract interface for creating reusable middlewares with `before` and `after` hooks:

```python
# app/middlewares/timing.py
"""Request timing middleware."""

import time
from robyn import Request, Response, Robyn

from app.middlewares.base import BaseMiddleware


class TimingMiddleware(BaseMiddleware):
    """Logs request duration."""

    endpoints = frozenset()  # Empty = apply to all routes

    def __init__(self, app: Robyn) -> None:
        super().__init__(app)
        self._start_times: dict[str, float] = {}

    def before(self, request: Request) -> Request:
        """Record start time."""
        self._start_times[request.url.path] = time.time()
        return request

    def after(self, response: Response) -> Response:
        """Log duration."""
        # Calculate and log timing
        return response
```

**Key features:**
- Receives `app: Robyn` in constructor (access to state, routes, etc.)
- Must implement at least one of `before()` or `after()` (enforced at class definition)
- Class attribute `endpoints` filters which routes the middleware applies to
- Empty `endpoints` applies to all registered routes

### Register Middlewares

Pass the **class** (not instance) - the handler instantiates it with `app`:

```python
# app/main.py
from app.middlewares.base import MiddlewareHandler
from app.middlewares.timing import TimingMiddleware
from app.middlewares.files import FileUploadOpenAPIMiddleware

app = Robyn(__file__)

# Create handler and register middleware classes (chainable)
middlewares = MiddlewareHandler(app)
middlewares.register(TimingMiddleware)
middlewares.register(FileUploadOpenAPIMiddleware)

# Or chain: middlewares.register(A).register(B).register(C)
```

### Built-in Middlewares

**`FileUploadOpenAPIMiddleware`** - Automatically patches OpenAPI spec for file upload endpoints:

```python
from app.middlewares.files import FileUploadOpenAPIMiddleware

middlewares.register(FileUploadOpenAPIMiddleware)  # Pass class, not instance
```

This middleware detects endpoints using `UploadFile` and updates `/openapi.json` to show proper `multipart/form-data` upload UI in Swagger.

---

## File Uploads

### UploadFile Type

Use `UploadFile` for typed file upload handling with automatic injection:

```python
# app/api/uploads.py
from app.core.router import Router
from app.models.core import UploadFile

router = Router(__file__, prefix="/files")


@router.post("/upload")
async def upload_file(files: UploadFile):
    """Handle file uploads - request object is NOT required."""
    results = []
    for filename, data in files:
        results.append({
            "name": filename,
            "size": len(data),
        })
    return {"uploaded": results}
```

**Key features:**
- **No `request` parameter required** - The router automatically injects it internally
- Files are available as `dict[str, bytes]` via iteration or `.get(name)`
- OpenAPI docs automatically show file upload UI (via `FileUploadOpenAPIMiddleware`)

### UploadFile API

```python
class UploadFile:
    files: dict[str, bytes]  # Raw file data

    def __bool__(self) -> bool:           # Check if any files
    def __iter__(self):                   # Iterate (name, bytes) pairs
    def get(self, name: str) -> bytes | None:  # Get file by field name
    def keys(self) -> list[str]:          # List all field names
```

### Example with Optional Request

If you need access to headers or other request data, you can still declare it:

```python
@router.post("/upload-with-auth")
async def upload_with_auth(request: Request, files: UploadFile):
    """Access both request and files."""
    auth = request.headers.get("authorization")
    # Process files with auth context...
    return {"files": len(files.files), "authenticated": bool(auth)}
```

---

## Package Organization

### `app/operation/`

Business logic and utility functions:

```python
# app/operation/preprocessing.py
"""Data preprocessing operations."""

import numpy as np


def normalize_features(features: list[float]) -> np.ndarray:
    """Normalize input features."""
    arr = np.array(features)
    return (arr - arr.mean()) / arr.std()


def batch_preprocess(items: list[dict]) -> list[np.ndarray]:
    """Preprocess a batch of items."""
    return [normalize_features(item["features"]) for item in items]
```

### `app/providers/`

External services and model loaders:

```python
# app/providers/model_loader.py
"""Model loading utilities."""

from pathlib import Path
from functools import cache


@cache
def load_sklearn_model(model_path: Path):
    """Load and cache sklearn model."""
    import joblib
    return joblib.load(model_path)


async def load_pytorch_model(model_path: Path, device: str = "cpu"):
    """Load PyTorch model asynchronously."""
    import torch
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model
```

### `app/models/`

Pydantic models for requests, responses, and domain entities:

```python
# app/models/prediction.py
"""Prediction domain models."""

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Input for ML prediction."""
    features: list[float] = Field(..., min_length=1)
    model_name: str = "default"


class PredictionOutput(BaseModel):
    """Output from ML prediction."""
    prediction: float
    confidence: float = Field(ge=0, le=1)
    model_version: str
```

---

## Development Commands

```bash
# Setup
make install          # Install uv + dependencies + pre-commit
make sync             # Sync dependencies from lockfile (frozen)
make lock             # Update lockfile

# Development
make dev              # Start dev server (DEBUG=True)
make prod             # Start production server
make run              # Alias for dev

# Quality
make lint             # Run ruff linter with auto-fix
make format           # Format code with ruff
make test             # Run unit tests

# Docker
make docker-build     # Build Docker image
make docker-up        # Start service in Docker
make docker-down      # Stop Docker services
make docker-test      # Build, start, and test
make log              # Tail container logs

# Cleanup
make clean            # Remove cache and artifacts
```

---

## Configuration

### Environment Variables

Copy `.env.template` to `.env` and customize:

```bash
# Application
DEBUG=True
ENVIRONMENT=DEV

# Server
API_HOST=0.0.0.0
API_PORT=8000

# Workers
MAX_WORKERS=4
```

### Settings Class

Settings are managed via Pydantic in `app/core/settings.py`:

```python
from app.core.settings import settings

# Access settings
print(settings.API_NAME)
print(settings.API_PORT)
print(settings.MODELS_PATH)
```

---

## Docker

### Development with Docker

```bash
# Build and run
make docker-build
make docker-up

# View logs
make log

# Stop
make docker-down
```

### Production Dockerfile

The multi-stage Dockerfile:
1. Uses uv for fast dependency installation
2. Creates minimal runtime image
3. Runs as non-root user
4. Includes health checks

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **Lint** - Runs ruff linter and format check
2. **Test** - Runs pytest with frozen dependencies
3. **Docker** - Builds Docker image on main branch

Key features:
- Uses `uv sync --frozen` for reproducible builds
- Caches uv downloads for speed
- Same lockfile local and CI

---

## License

MIT License - See [LICENSE](LICENSE) for details.
