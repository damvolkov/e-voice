"""Base middleware architecture for Robyn applications."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from robyn import Request, Response, Robyn

from app.core.logger import LogIcon, logger


class BaseMiddleware(ABC):
    """Abstract base class for middlewares with before/after hooks."""

    endpoints: frozenset[str] = frozenset()

    def __init__(self, app: Robyn) -> None:
        self.app = app

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Check that at least one of before/after is implemented
        before_abstract = getattr(cls.before, "__isabstractmethod__", False)
        after_abstract = getattr(cls.after, "__isabstractmethod__", False)
        if before_abstract and after_abstract:
            raise TypeError(f"{cls.__name__} must implement at least one of before/after")

    @abstractmethod
    def before(self, request: Request) -> Request | Response:
        """Called before request handling. Return Request to continue or Response to short-circuit."""
        return request

    @abstractmethod
    def after(self, response: Response) -> Response:
        """Called after request handling. Return modified Response."""
        return response


class MiddlewareHandler:
    """Manages middleware registration for a Robyn application."""

    def __init__(self, app: Robyn) -> None:
        self._app = app
        self._middlewares: list[BaseMiddleware] = []

    def register(self, middleware_cls: type[BaseMiddleware]) -> "MiddlewareHandler":
        """Register a middleware class. Instantiates with app. Returns self for chaining."""
        middleware = middleware_cls(self._app)
        self._middlewares.append(middleware)
        self._apply_middleware(middleware)
        logger.info(f"Registered middleware: {middleware_cls.__name__}", icon=LogIcon.ADAPTER)
        return self

    def _apply_middleware(self, middleware: BaseMiddleware) -> None:
        """Apply middleware to endpoints."""
        endpoints = middleware.endpoints or self._get_all_routes()
        has_before = not getattr(middleware.before, "__isabstractmethod__", False)
        has_after = not getattr(middleware.after, "__isabstractmethod__", False)

        for endpoint in endpoints:
            if has_before:
                self._register_before(endpoint, middleware.before)
            if has_after:
                self._register_after(endpoint, middleware.after)

    def _get_all_routes(self) -> frozenset[str]:
        """Get all registered routes from the app."""
        routes = self._app.get_all_routes()  # type: ignore[possibly-missing-attribute]
        return frozenset(route[1] for route in routes)

    def _register_before(self, endpoint: str, handler: Callable) -> None:
        """Register a before_request handler for an endpoint."""
        @self._app.before_request(endpoint)
        async def before_wrapper(request: Request) -> Request | Response:
            return handler(request)

    def _register_after(self, endpoint: str, handler: Callable) -> None:
        """Register an after_request handler for an endpoint."""
        @self._app.after_request(endpoint)
        def after_wrapper(response: Response) -> Response:
            return handler(response)
