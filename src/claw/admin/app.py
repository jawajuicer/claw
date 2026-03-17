"""FastAPI admin panel application factory."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from claw.admin.auth import BasicAuthMiddleware
from claw.admin.remote import remote_router
from claw.admin.routes import LogBuffer, router
from claw.admin.sse import StatusBroadcaster

log = logging.getLogger(__name__)

ADMIN_DIR = Path(__file__).parent


def create_admin_app(broadcaster: StatusBroadcaster, agent=None, registry=None) -> FastAPI:
    """Create and configure the FastAPI admin panel."""
    app = FastAPI(title="The Claw — Admin", docs_url=None, redoc_url=None)

    # Authentication middleware
    app.add_middleware(BasicAuthMiddleware)

    # Static files and templates
    static_dir = ADMIN_DIR / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    app.state.templates = Jinja2Templates(directory=str(ADMIN_DIR / "templates"))

    # Shared state
    app.state.broadcaster = broadcaster
    app.state.agent = agent
    app.state.registry = registry
    app.state.tts = None
    app.state.remote_transcriber = None  # Set by Claw.initialize() when remote enabled
    app.state.tool_router = None  # Set by Claw.initialize()
    app.state.tool_stats = None  # Set by Claw.initialize()
    app.state.audio_capture = None  # Set by Claw voice loop for diagnostics
    app.state.vad = None  # Set by Claw voice loop for diagnostics
    app.state.chat_lock = asyncio.Lock()
    app.state.bg_tasks: set[asyncio.Task] = set()

    # Log buffer — attach to root logger
    from claw.config import get_settings
    cfg = get_settings().admin
    log_buffer = LogBuffer(maxlen=cfg.log_buffer_size, fmt=cfg.log_format)
    logging.getLogger().addHandler(log_buffer)
    app.state.log_buffer = log_buffer

    # Optional file logging with rotation
    if cfg.log_file:
        from claw.config import PROJECT_ROOT
        log_path = Path(cfg.log_file)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_path), maxBytes=10 * 1024 * 1024, backupCount=5,
        )
        if cfg.log_format == "json":
            from claw.admin.routes import JsonLineFormatter
            file_handler.setFormatter(JsonLineFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            ))
        logging.getLogger().addHandler(file_handler)

    # Routes — admin panel + remote API
    app.include_router(router)
    app.include_router(remote_router)

    log.info("Admin panel created")
    return app
