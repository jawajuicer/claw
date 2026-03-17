"""Device API key management for secure remote access.

Each registered device gets a unique 256-bit API key stored in the
encrypted secret store (Fernet, tied to machine-id). Keys are shown
exactly once at creation and cannot be retrieved later. A plain-text
device registry tracks metadata (created_at, last_seen) in a
restricted directory.

When WireGuard is available, device creation also generates a WireGuard
keypair, assigns an IP, adds the peer, and returns a provisioning code
containing everything the client app needs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

from claw import secret_store
from claw.config import PROJECT_ROOT

log = logging.getLogger(__name__)

_DEVICES_DIR = PROJECT_ROOT / "data" / "remote"
_REGISTRY_FILE = _DEVICES_DIR / "devices.json"
_DEVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")
_LAST_SEEN_THROTTLE = 300  # seconds between disk writes for last_seen

# In-memory cache: sha256(api_key) → device_name. Invalidated on create/revoke.
_KEY_CACHE: dict[str, str] = {}
_KEY_CACHE_BUILT = False
_cache_lock = threading.Lock()


def _build_key_cache() -> None:
    """Populate the key cache from the registry + secret store."""
    global _KEY_CACHE, _KEY_CACHE_BUILT
    with _cache_lock:
        if _KEY_CACHE_BUILT:
            return
        import hashlib
        registry = _load_registry()
        cache: dict[str, str] = {}
        for name in registry:
            stored = secret_store.load(f"device_key_{name}")
            if stored:
                key_hash = hashlib.sha256(stored.encode()).hexdigest()
                cache[key_hash] = name
        _KEY_CACHE = cache
        _KEY_CACHE_BUILT = True


def _invalidate_key_cache() -> None:
    """Mark the cache as stale so it rebuilds on next verification."""
    global _KEY_CACHE_BUILT
    _KEY_CACHE_BUILT = False


def _ensure_dir() -> None:
    _DEVICES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(_DEVICES_DIR), 0o700)
    except OSError:
        pass


def _load_registry() -> dict[str, dict]:
    if _REGISTRY_FILE.exists():
        try:
            return json.loads(_REGISTRY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt device registry, starting fresh")
    return {}


def _save_registry(registry: dict[str, dict]) -> None:
    _ensure_dir()
    tmp = _REGISTRY_FILE.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(registry, f, indent=2)
    tmp.replace(_REGISTRY_FILE)


def create_device(name: str) -> dict:
    """Register a new device and return provisioning info.

    Returns a dict with:
        - api_key: 256-bit hex key (shown once)
        - provisioning_code: base64 blob with API key + WireGuard config (if WG available)
        - wg_available: whether WireGuard provisioning was included

    Raises:
        ValueError: If the name is invalid or already exists.
    """
    if not _DEVICE_NAME_RE.match(name):
        raise ValueError(
            "Device name must be 1-32 chars: letters, numbers, hyphens, underscores"
        )

    registry = _load_registry()
    if name in registry:
        raise ValueError(f"Device '{name}' already exists")

    # Generate API key
    key = secrets.token_hex(32)
    secret_store.store(f"device_key_{name}", key)

    registry_entry = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_seen": None,
    }

    result = {
        "api_key": key,
        "wg_available": False,
        "provisioning_code": None,
    }

    # WireGuard provisioning (if available)
    from claw.admin import wireguard as wg

    if wg.is_available():
        from claw.config import get_settings

        cfg = get_settings().remote
        server_pubkey = wg.get_server_public_key()
        endpoint = wg.get_server_endpoint()

        if server_pubkey and endpoint:
            try:
                private_key, public_key, psk = wg.generate_keypair()
                device_ip = wg.allocate_ip(registry, cfg.wg_subnet)

                # Store WG keys
                secret_store.store(f"device_wg_private_{name}", private_key)
                secret_store.store(f"device_wg_psk_{name}", psk)

                # Add to registry
                registry_entry["wg_public_key"] = public_key
                registry_entry["wg_ip"] = device_ip

                # Add peer to running WireGuard
                wg.add_peer(cfg.wg_interface, public_key, psk, device_ip)

                # Build provisioning code
                server_url = f"http://{cfg.wg_subnet}.1:{get_settings().admin.port}"
                result["provisioning_code"] = wg.build_provisioning_code(
                    api_key=key,
                    wg_private_key=private_key,
                    wg_address=device_ip,
                    wg_server_pubkey=server_pubkey,
                    wg_psk=psk,
                    wg_endpoint=endpoint,
                    server_url=server_url,
                )
                result["wg_available"] = True

                log.info(
                    "Device '%s' registered with WireGuard IP %s", name, device_ip
                )
            except Exception:
                log.exception("WireGuard provisioning failed for '%s'", name)
                # Device still gets registered with API key, just no WG
        else:
            log.warning(
                "WireGuard keys found but endpoint not configured — "
                "set remote.wg_endpoint in config.yaml"
            )
    else:
        log.info("Device '%s' registered (WireGuard not available)", name)

    registry[name] = registry_entry
    _save_registry(registry)
    _invalidate_key_cache()
    return result


def verify_key(key: str) -> str | None:
    """Verify an API key. Returns the device name if valid, None otherwise.

    Uses an in-memory SHA256 cache for O(1) lookups instead of O(n) PBKDF2
    derivations. Updates last_seen at most every 5 minutes to reduce disk I/O.
    """
    import hashlib

    if not key or len(key) != 64:
        return None

    # Build or rebuild cache on first call / after invalidation
    if not _KEY_CACHE_BUILT:
        _build_key_cache()

    key_hash = hashlib.sha256(key.encode()).hexdigest()
    name = _KEY_CACHE.get(key_hash)
    if name is None:
        return None

    # Constant-time verify against stored key (defense in depth)
    stored = secret_store.load(f"device_key_{name}")
    if not stored or not secrets.compare_digest(stored, key):
        return None

    # Throttle last_seen writes
    registry = _load_registry()
    info = registry.get(name)
    if info:
        now = datetime.now(timezone.utc)
        last = info.get("last_seen")
        should_update = True
        if last:
            try:
                elapsed = (now - datetime.fromisoformat(last)).total_seconds()
                should_update = elapsed > _LAST_SEEN_THROTTLE
            except (ValueError, TypeError):
                pass
        if should_update:
            info["last_seen"] = now.isoformat()
            _save_registry(registry)
    return name


def list_devices() -> list[dict]:
    """List all registered devices (without keys)."""
    registry = _load_registry()
    return [
        {
            "name": name,
            "created_at": info.get("created_at"),
            "last_seen": info.get("last_seen"),
            "wg_ip": info.get("wg_ip"),
        }
        for name, info in registry.items()
    ]


def revoke_device(name: str) -> bool:
    """Revoke a device's access. Returns True if the device existed."""
    registry = _load_registry()
    if name not in registry:
        return False

    info = registry[name]

    # Remove WireGuard peer if provisioned
    wg_pubkey = info.get("wg_public_key")
    if wg_pubkey:
        from claw.admin import wireguard as wg
        from claw.config import get_settings

        wg.remove_peer(get_settings().remote.wg_interface, wg_pubkey)
        secret_store.delete(f"device_wg_private_{name}")
        secret_store.delete(f"device_wg_psk_{name}")

    del registry[name]
    secret_store.delete(f"device_key_{name}")
    _save_registry(registry)
    _invalidate_key_cache()
    log.info("Device '%s' revoked", name)
    return True
