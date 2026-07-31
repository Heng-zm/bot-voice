"""Flask-shaped compatibility API backed by FastAPI/Starlette."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "FastAPICompatApp",
    "Response",
    "_LocalProxy",
    "_MultiDictCompat",
    "_RequestCompat",
    "_UploadFileCompat",
    "abort",
    "flask_flash",
    "get_flashed_messages",
    "jsonify",
    "redirect",
    "render_template_string",
    "stream_with_context",
    "url_for",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
