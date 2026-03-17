"""Encrypted secret storage using Fernet (AES-128-CBC + HMAC-SHA256).

Secrets are encrypted with a key derived from the machine ID + fixed salt
via PBKDF2 (480k iterations). Encrypted files are stored in data/secrets/
with 0o600 permissions. Secrets are tied to the machine — encrypted files
are useless if copied elsewhere.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_SALT = b"claw-secret-store-v1"
_ITERATIONS = 480_000
_SECRETS_DIR_NAME = "secrets"


def _get_machine_id() -> bytes:
    """Read the machine ID for key derivation."""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            return Path(path).read_text().strip().encode()
        except OSError:
            continue
    # Fallback: hostname + username
    import socket
    try:
        fallback = f"{socket.gethostname()}-{os.getlogin()}".encode()
    except OSError:
        fallback = f"{socket.gethostname()}-nouser".encode()
    log.warning("No /etc/machine-id found, using fallback key material")
    return fallback


def _derive_key() -> bytes:
    """Derive a Fernet key from the machine ID."""
    machine_id = _get_machine_id()
    dk = hashlib.pbkdf2_hmac("sha256", machine_id, _SALT, _ITERATIONS, dklen=32)
    return base64.urlsafe_b64encode(dk)


def _secrets_dir() -> Path:
    """Return the secrets directory, creating it if needed."""
    from claw.config import PROJECT_ROOT
    d = PROJECT_ROOT / "data" / _SECRETS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def _secret_path(name: str) -> Path:
    """Return the path for a named secret."""
    safe_name = name.replace("/", "_").replace("\\", "_")
    return _secrets_dir() / f"{safe_name}.enc"


def store(name: str, value: str) -> None:
    """Encrypt and store a secret. Empty value deletes the secret."""
    if not value:
        delete(name)
        return

    from cryptography.fernet import Fernet

    key = _derive_key()
    f = Fernet(key)
    encrypted = f.encrypt(value.encode())

    path = _secret_path(name)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(encrypted)
    log.info("Secret '%s' stored at %s", name, path)


def load(name: str) -> str | None:
    """Load and decrypt a secret. Returns None if not found."""
    path = _secret_path(name)
    if not path.exists():
        return None

    from cryptography.fernet import Fernet, InvalidToken

    key = _derive_key()
    f = Fernet(key)
    try:
        encrypted = path.read_bytes()
        return f.decrypt(encrypted).decode()
    except InvalidToken:
        log.error("Failed to decrypt secret '%s' — wrong machine?", name)
        return None
    except Exception:
        log.exception("Failed to load secret '%s'", name)
        return None


def exists(name: str) -> bool:
    """Check if a secret exists."""
    return _secret_path(name).exists()


def delete(name: str) -> bool:
    """Delete a secret. Returns True if it existed."""
    path = _secret_path(name)
    if path.exists():
        path.unlink()
        log.info("Secret '%s' deleted", name)
        return True
    return False


def mask(name: str) -> str:
    """Return a masked version of the secret for display."""
    value = load(name)
    if value is None:
        return "Not set"
    if len(value) <= 8:
        return value[:2] + "***"
    return value[:4] + "***" + value[-4:]
