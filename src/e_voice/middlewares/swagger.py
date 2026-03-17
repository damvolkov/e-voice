"""Swagger UI branding middleware — e-voice favicon, title, and topbar."""

import base64

from robyn import Request, Response, Robyn

from e_voice.middlewares.base import BaseMiddleware

##### CONSTANTS #####

_ROBYN_FAVICON = "https://user-images.githubusercontent.com/29942790/140995889-5d91dcff-3aa7-4cfb-8a90-2cddf1337dca.png"

_FAVICON_SVG = (
    '<svg viewBox="0 0 200 190" xmlns="http://www.w3.org/2000/svg">'
    '<g transform="translate(30,30)">'
    '<rect x="0" y="0" width="16" height="130" fill="#2D2D2A"/>'
    '<rect x="16" y="0" width="62" height="16" fill="#2D2D2A"/>'
    '<rect x="16" y="57" width="38" height="16" fill="#2D2D2A"/>'
    '<rect x="16" y="114" width="62" height="16" fill="#2D2D2A"/>'
    '<path d="M88 40 A28 28 0 0 1 88 90" fill="none" stroke="#2D2D2A"'
    ' stroke-width="5" stroke-linecap="round"/>'
    '<path d="M104 26 A42 42 0 0 1 104 104" fill="none" stroke="#2D2D2A"'
    ' stroke-width="5" stroke-linecap="round"/>'
    "</g></svg>"
)

_FAVICON_URI = f"data:image/svg+xml;base64,{base64.b64encode(_FAVICON_SVG.encode()).decode()}"

_BRANDING_CSS = f"""<style>
.swagger-ui .topbar {{ background-color: #2D2D2A; }}
.swagger-ui .topbar-wrapper img {{ display: none; }}
.swagger-ui .topbar-wrapper a::before {{
  content: '';
  display: inline-block;
  width: 28px;
  height: 28px;
  background: url("{_FAVICON_URI}") no-repeat center/contain;
  filter: brightness(0) invert(1);
  margin-right: 10px;
  vertical-align: middle;
}}
.swagger-ui .topbar-wrapper a::after {{
  content: 'e-voice';
  color: #fff;
  font: 500 18px/1 system-ui, -apple-system, sans-serif;
  vertical-align: middle;
  letter-spacing: -0.3px;
}}
</style>"""


##### MIDDLEWARE #####


class SwaggerBrandingMiddleware(BaseMiddleware):
    """Replaces Robyn default branding with e-voice favicon, title, and topbar."""

    endpoints = frozenset(["/docs"])

    def __init__(self, app: Robyn) -> None:
        super().__init__(app)

    def before(self, request: Request) -> Request:
        return request

    def after(self, response: Response) -> Response:
        """Patch Swagger HTML with e-voice branding."""
        html = response.description
        html = html.replace(_ROBYN_FAVICON, _FAVICON_URI)
        html = html.replace('type="image/png"', 'type="image/svg+xml"')
        html = html.replace("Robyn OpenAPI Docs", "e-voice API")
        html = html.replace("</head>", f"{_BRANDING_CSS}\n  </head>")
        response.description = html
        return response
