"""Tests for device API key management."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from claw.admin import api_key


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """Redirect device registry and secrets dir to a temp directory."""
    devices_dir = tmp_path / "remote"
    secrets_dir = tmp_path / "secrets"
    devices_dir.mkdir()
    secrets_dir.mkdir()

    monkeypatch.setattr(api_key, "_DEVICES_DIR", devices_dir)
    monkeypatch.setattr(api_key, "_REGISTRY_FILE", devices_dir / "devices.json")

    # Patch secret_store to use temp dir
    import claw.secret_store as ss
    monkeypatch.setattr(ss, "_SALT", b"test-salt")

    original_secrets_dir = ss._secrets_dir

    def _test_secrets_dir():
        secrets_dir.mkdir(parents=True, exist_ok=True)
        return secrets_dir

    monkeypatch.setattr(ss, "_secrets_dir", _test_secrets_dir)
    yield


class TestCreateDevice:
    def test_creates_device_and_returns_key(self):
        result = api_key.create_device("my-phone")
        assert isinstance(result, dict)
        key = result["api_key"]
        assert len(key) == 64  # 256-bit hex
        assert "wg_available" in result
        devices = api_key.list_devices()
        assert len(devices) == 1
        assert devices[0]["name"] == "my-phone"
        assert devices[0]["created_at"] is not None
        assert devices[0]["last_seen"] is None

    def test_duplicate_name_raises(self):
        api_key.create_device("phone")
        with pytest.raises(ValueError, match="already exists"):
            api_key.create_device("phone")

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="1-32 chars"):
            api_key.create_device("")
        with pytest.raises(ValueError, match="1-32 chars"):
            api_key.create_device("has spaces")
        with pytest.raises(ValueError, match="1-32 chars"):
            api_key.create_device("a" * 33)

    def test_valid_name_formats(self):
        api_key.create_device("phone-1")
        api_key.create_device("desktop_2")
        api_key.create_device("CarHeadUnit")
        assert len(api_key.list_devices()) == 3

    def test_keys_are_unique(self):
        k1 = api_key.create_device("dev1")["api_key"]
        k2 = api_key.create_device("dev2")["api_key"]
        assert k1 != k2


class TestVerifyKey:
    def test_valid_key_returns_device_name(self):
        key = api_key.create_device("phone")["api_key"]
        assert api_key.verify_key(key) == "phone"

    def test_invalid_key_returns_none(self):
        api_key.create_device("phone")
        assert api_key.verify_key("0" * 64) is None

    def test_empty_key_returns_none(self):
        assert api_key.verify_key("") is None

    def test_wrong_length_returns_none(self):
        assert api_key.verify_key("short") is None

    def test_none_key_returns_none(self):
        assert api_key.verify_key(None) is None

    def test_verify_updates_last_seen(self):
        key = api_key.create_device("phone")["api_key"]
        devices_before = api_key.list_devices()
        assert devices_before[0]["last_seen"] is None

        api_key.verify_key(key)
        devices_after = api_key.list_devices()
        assert devices_after[0]["last_seen"] is not None

    def test_multiple_devices_correct_match(self):
        k1 = api_key.create_device("phone")["api_key"]
        k2 = api_key.create_device("desktop")["api_key"]
        k3 = api_key.create_device("tablet")["api_key"]

        assert api_key.verify_key(k1) == "phone"
        assert api_key.verify_key(k2) == "desktop"
        assert api_key.verify_key(k3) == "tablet"


class TestRevokeDevice:
    def test_revoke_existing_device(self):
        api_key.create_device("phone")
        assert api_key.revoke_device("phone") is True
        assert len(api_key.list_devices()) == 0

    def test_revoke_nonexistent_returns_false(self):
        assert api_key.revoke_device("nope") is False

    def test_revoked_key_no_longer_valid(self):
        key = api_key.create_device("phone")["api_key"]
        assert api_key.verify_key(key) == "phone"
        api_key.revoke_device("phone")
        assert api_key.verify_key(key) is None


class TestListDevices:
    def test_empty_initially(self):
        assert api_key.list_devices() == []

    def test_lists_all_devices(self):
        api_key.create_device("a")
        api_key.create_device("b")
        api_key.create_device("c")
        names = [d["name"] for d in api_key.list_devices()]
        assert sorted(names) == ["a", "b", "c"]

    def test_no_keys_in_listing(self):
        api_key.create_device("phone")
        devices = api_key.list_devices()
        for d in devices:
            assert "key" not in d
            assert "api_key" not in d


class TestRegistryPersistence:
    def test_registry_survives_reload(self):
        key = api_key.create_device("phone")["api_key"]
        # Clear any caches by re-reading from disk
        assert api_key.verify_key(key) == "phone"
        devices = api_key.list_devices()
        assert len(devices) == 1
