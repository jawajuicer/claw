"""FastAPI admin panel application factory."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from claw.admin.routes import LogBuffer, router
from claw.admin.sse import StatusBroadcaster

log = logging.getLogger(__name__)

ADMIN_DIR = Path(__file__).parent


def create_admin_app(broadcaster: StatusBroadcaster, agent=None, registry=None) -> FastAPI:
    """Create and configure the FastAPI admin panel."""
    app = FastAPI(title="The Claw — Admin", docs_url=None, redoc_url=None)

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
    app.state.chat_lock = asyncio.Lock()
    app.state.bg_tasks: set[asyncio.Task] = set()

    # Log buffer — attach to root logger
    from claw.config import get_settings
    cfg = get_settings().admin
    log_buffer = LogBuffer(maxlen=cfg.log_buffer_size)
    logging.getLogger().addHandler(log_buffer)
    app.state.log_buffer = log_buffer

    # Routes
    app.include_router(router)

    log.info("Admin panel created")
    return app
