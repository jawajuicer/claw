"""WireGuard peer management for dynamic device provisioning.

Manages WireGuard peers programmatically — generates keypairs, adds/removes
peers from the running interface, and builds client configs. Requires
WireGuard to be set up on the host (see scripts/setup-wireguard.sh).
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

WG_DIR = Path("/etc/wireguard")


def _server_pubkey_path() -> Path | None:
    """Find the server public key file (may be in /etc/wireguard or data/remote)."""
    from claw.config import PROJECT_ROOT

    # Prefer project-local copy (readable without root)
    local = PROJECT_ROOT / "data" / "remote" / "wg_server_public"
    if local.exists():
        return local
    # Fall back to /etc/wireguard (needs directory read permission)
    system = WG_DIR / "server_public"
    try:
        if system.exists():
            return system
    except PermissionError:
        pass
    return None


def is_available() -> bool:
    """Check if WireGuard is set up on this machine."""
    return _server_pubkey_path() is not None


def get_server_public_key() -> str | None:
    """Read the server's WireGuard public key."""
    path = _server_pubkey_path()
    if path is None:
        return None
    return path.read_text().strip()


def get_server_endpoint() -> str | None:
    """Get the server's public endpoint (IP:port)."""
    from claw.config import get_settings

    cfg = get_settings().remote
    if not cfg.wg_endpoint:
        return None
    return f"{cfg.wg_endpoint}:{cfg.wg_port}"


def generate_keypair() -> tuple[str, str, str]:
    """Generate a WireGuard keypair + preshared key.

    Returns (private_key, public_key, preshared_key).
    """
    private = subprocess.run(
        ["wg", "genkey"], capture_output=True, text=True, check=True
    ).stdout.strip()

    public = subprocess.run(
        ["wg", "pubkey"], input=private, capture_output=True, text=True, check=True
    ).stdout.strip()

    psk = subprocess.run(
        ["wg", "genpsk"], capture_output=True, text=True, check=True
    ).stdout.strip()

    return private, public, psk


def allocate_ip(registry: dict[str, dict], subnet: str) -> str:
    """Allocate the next available IP in the subnet.

    Server is .1, devices start at .2.
    """
    used = {info.get("wg_ip") for info in registry.values() if info.get("wg_ip")}
    for i in range(2, 255):
        ip = f"{subnet}.{i}"
        if ip not in used:
            return ip
    raise ValueError("No available IPs in WireGuard subnet")


def add_peer(interface: str, public_key: str, psk: str, allowed_ip: str) -> bool:
    """Add a peer to the running WireGuard interface."""
    psk_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".psk", delete=False) as f:
            f.write(psk)
            psk_path = f.name

        subprocess.run(
            [
                "sudo", "wg", "set", interface,
                "peer", public_key,
                "preshared-key", psk_path,
                "allowed-ips", f"{allowed_ip}/32",
            ],
            capture_output=True, text=True, check=True,
        )
        log.info("Added WireGuard peer %s... (%s)", public_key[:8], allowed_ip)
        return True
    except subprocess.CalledProcessError as e:
        log.error("Failed to add WireGuard peer: %s", e.stderr)
        return False
    except FileNotFoundError:
        log.error("wg command not found — is WireGuard installed?")
        return False
    finally:
        if psk_path:
            Path(psk_path).unlink(missing_ok=True)


def remove_peer(interface: str, public_key: str) -> bool:
    """Remove a peer from the running WireGuard interface."""
    try:
        subprocess.run(
            ["sudo", "wg", "set", interface, "peer", public_key, "remove"],
            capture_output=True, text=True, check=True,
        )
        log.info("Removed WireGuard peer %s...", public_key[:8])
        return True
    except subprocess.CalledProcessError as e:
        log.error("Failed to remove WireGuard peer: %s", e.stderr)
        return False
    except FileNotFoundError:
        log.error("wg command not found")
        return False


def build_client_config(
    private_key: str,
    address: str,
    server_public_key: str,
    psk: str,
    endpoint: str,
    allowed_ips: str = "10.10.0.0/24",
) -> str:
    """Build a WireGuard client configuration string (INI format)."""
    return (
        f"[Interface]\n"
        f"PrivateKey = {private_key}\n"
        f"Address = {address}/32\n"
        f"\n"
        f"[Peer]\n"
        f"PublicKey = {server_public_key}\n"
        f"PresharedKey = {psk}\n"
        f"Endpoint = {endpoint}\n"
        f"AllowedIPs = {allowed_ips}\n"
        f"PersistentKeepalive = 25\n"
    )


def build_provisioning_code(
    api_key: str,
    wg_private_key: str,
    wg_address: str,
    wg_server_pubkey: str,
    wg_psk: str,
    wg_endpoint: str,
    server_url: str,
    allowed_ips: str = "10.10.0.0/24",
) -> str:
    """Build a base64-encoded provisioning code containing all connection info.

    The Android/PWA app decodes this single string to get everything it needs:
    API key, WireGuard config, and server URL through the tunnel.
    """
    payload = {
        "v": 1,
        "api_key": api_key,
        "wg": {
            "private_key": wg_private_key,
            "address": wg_address,
            "server_public_key": wg_server_pubkey,
            "psk": wg_psk,
            "endpoint": wg_endpoint,
            "allowed_ips": allowed_ips,
        },
        "server_url": server_url,
    }
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
