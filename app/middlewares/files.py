"""File upload middleware for OpenAPI multipart/form-data patching."""

import orjson
from robyn import Request, Response, Robyn

from app.core.router import FILE_UPLOAD_ENDPOINTS
from app.middlewares.base import BaseMiddleware


class FileUploadOpenAPIMiddleware(BaseMiddleware):
    """Patches OpenAPI responses to use multipart/form-data for file upload endpoints."""

    endpoints = frozenset(["/openapi.json"])

    def __init__(self, app: Robyn) -> None:
        super().__init__(app)

    def before(self, request: Request) -> Request:
        return request

    def after(self, response: Response) -> Response:
        """Patch OpenAPI spec with multipart/form-data for file upload endpoints."""
        if not FILE_UPLOAD_ENDPOINTS:
            return response

        try:
            spec = orjson.loads(response.description)
            paths = spec.get("paths", {})

            for endpoint in FILE_UPLOAD_ENDPOINTS:
                if endpoint in paths:
                    for method in paths[endpoint]:
                        paths[endpoint][method]["requestBody"] = {
                            "content": {
                                "multipart/form-data": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "file": {
                                                "type": "string",
                                                "format": "binary",
                                                "description": "File to upload",
                                            }
                                        },
                                        "required": ["file"],
                                    }
                                }
                            },
                            "required": True,
                        }

            response.description = orjson.dumps(spec).decode()
        except Exception:
            pass

        return response
