"""Pairing code manager for secure device linking.

Generates short-lived 6-digit codes that allow mobile devices to
register without requiring the admin password. The code itself
acts as the credential for the /api/remote/pair/claim endpoint.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Code properties
_CODE_TTL = 300  # 5 minutes
_MAX_ACTIVE_CODES = 5
_MAX_CLAIM_ATTEMPTS = 10  # per IP per minute


@dataclass
class PairingCode:
    code: str
    created_at: float
    device_name: str = ""


class PairingManager:
    """Manages short-lived pairing codes for device registration.

    Flow:
    1. Admin generates code via POST /api/remote/pair (authenticated)
    2. Mobile app claims code via POST /api/remote/pair/claim (unauthenticated)
    3. Successful claim creates device and returns API key
    """

    def __init__(self) -> None:
        self._codes: dict[str, PairingCode] = {}
        self._claim_attempts: dict[str, list[float]] = {}  # IP -> timestamps

    def generate(self, device_name: str = "") -> str:
        """Generate a new 6-digit pairing code.

        Returns the code string. Raises ValueError if too many active codes.
        """
        self._purge_expired()

        if len(self._codes) >= _MAX_ACTIVE_CODES:
            raise ValueError(
                f"Maximum active codes ({_MAX_ACTIVE_CODES}) reached. "
                "Wait for existing codes to expire."
            )

        # Generate unique 6-digit code
        for _ in range(100):
            code = f"{secrets.randbelow(1000000):06d}"
            if code not in self._codes:
                break
        else:
            raise RuntimeError("Failed to generate unique code")

        self._codes[code] = PairingCode(
            code=code,
            created_at=time.monotonic(),
            device_name=device_name,
        )
        log.info(
            "Generated pairing code for device '%s' (expires in %ds)",
            device_name,
            _CODE_TTL,
        )
        return code

    def claim(self, code: str, client_ip: str = "") -> PairingCode | None:
        """Attempt to claim a pairing code.

        Returns the PairingCode on success, None if invalid/expired.
        Raises ValueError if rate-limited.
        """
        # Rate limiting
        if client_ip:
            self._check_rate_limit(client_ip)

        self._purge_expired()

        pairing = self._codes.pop(code, None)
        if pairing is None:
            log.warning("Invalid or expired pairing code attempt: %s", code[:3] + "***")
            return None

        log.info("Pairing code claimed for device '%s'", pairing.device_name)
        return pairing

    def _check_rate_limit(self, client_ip: str) -> None:
        """Check claim rate limit. Raises ValueError if exceeded."""
        now = time.monotonic()
        attempts = self._claim_attempts.get(client_ip, [])
        # Remove attempts older than 60 seconds
        attempts = [t for t in attempts if now - t < 60]
        self._claim_attempts[client_ip] = attempts

        if len(attempts) >= _MAX_CLAIM_ATTEMPTS:
            raise ValueError("Too many claim attempts. Please wait and try again.")

        attempts.append(now)

    def _purge_expired(self) -> None:
        """Remove expired codes."""
        now = time.monotonic()
        expired = [
            code
            for code, pc in self._codes.items()
            if now - pc.created_at > _CODE_TTL
        ]
        for code in expired:
            del self._codes[code]

    def active_count(self) -> int:
        """Return number of active (non-expired) codes."""
        self._purge_expired()
        return len(self._codes)
