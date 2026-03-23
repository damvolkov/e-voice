"""Swagger UI branding middleware — e-voice favicon, title, and topbar."""

import base64

from robyn import Request, Response, Robyn

from e_voice.middlewares.base import BaseMiddleware

##### CONSTANTS #####

_ROBYN_FAVICON = "https://user-images.githubusercontent.com/29942790/140995889-5d91dcff-3aa7-4cfb-8a90-2cddf1337dca.png"

_FAVICON_SVG = (
    '<svg viewBox="0 0 200 190" xmlns="http://www.w3.org/2000/svg">'
    '<g transform="translate(30,30)">'
    '<rect x="0" y="0" width="16" height="130" fill="#1a6e8a"/>'
    '<rect x="16" y="0" width="62" height="16" fill="#1a6e8a"/>'
    '<rect x="16" y="57" width="38" height="16" fill="#1a6e8a"/>'
    '<rect x="16" y="114" width="62" height="16" fill="#1a6e8a"/>'
    '<path d="M88 40 A28 28 0 0 1 88 90" fill="none" stroke="#1a6e8a"'
    ' stroke-width="5" stroke-linecap="round"/>'
    '<path d="M104 26 A42 42 0 0 1 104 104" fill="none" stroke="#1a6e8a"'
    ' stroke-width="5" stroke-linecap="round"/>'
    "</g></svg>"
)

_LANDSCAPE_SVG = (
    '<svg viewBox="0 0 490 170" xmlns="http://www.w3.org/2000/svg">'
    '<g transform="translate(40,20)">'
    '<rect x="0" y="0" width="14" height="130" fill="#1a6e8a"/>'
    '<rect x="14" y="0" width="62" height="14" fill="#1a6e8a"/>'
    '<rect x="14" y="58" width="38" height="14" fill="#1a6e8a"/>'
    '<rect x="14" y="116" width="62" height="14" fill="#1a6e8a"/>'
    '<path d="M86 42 A28 28 0 0 1 86 88" fill="none" stroke="#1a6e8a"'
    ' stroke-width="4.5" stroke-linecap="round"/>'
    '<path d="M100 30 A42 42 0 0 1 100 100" fill="none" stroke="#1a6e8a"'
    ' stroke-width="4.5" stroke-linecap="round"/>'
    '<text x="135" y="72" fill="#2196f3"'
    " font-family=\"system-ui,-apple-system,'Helvetica Neue',sans-serif\""
    ' font-size="42" font-weight="500" dominant-baseline="central"'
    ' letter-spacing="-0.5px">e-voice</text>'
    "</g></svg>"
)

_FAVICON_URI = f"data:image/svg+xml;base64,{base64.b64encode(_FAVICON_SVG.encode()).decode()}"
_LANDSCAPE_URI = f"data:image/svg+xml;base64,{base64.b64encode(_LANDSCAPE_SVG.encode()).decode()}"

_BRANDING_CSS = f"""<style>
.swagger-ui .topbar {{ background: #0a0e14; padding: 10px 0; }}
.swagger-ui .topbar-wrapper img {{ display: none; }}
.swagger-ui .topbar-wrapper a {{
  display: flex; align-items: center; text-decoration: none;
}}
.swagger-ui .topbar-wrapper a::before {{
  content: '';
  display: inline-block;
  width: 220px;
  height: 48px;
  background: url("{_LANDSCAPE_URI}") no-repeat center/contain;
}}
.swagger-ui .topbar-wrapper a::after {{ content: none; }}
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
        html = str(response.description)
        html = html.replace(_ROBYN_FAVICON, _FAVICON_URI)
        html = html.replace('type="image/png"', 'type="image/svg+xml"')
        html = html.replace("Robyn OpenAPI Docs", "e-voice API")
        html = html.replace("</head>", f"{_BRANDING_CSS}\n  </head>")
        response.description = html
        return response
