"""Tests for claw.config — Settings, YAML loading, hot-reload, validators."""

from __future__ import annotations

from unittest.mock import MagicMock

import yaml


class TestSettingsDefaults:
    """Verify default values when no config.yaml / env vars are present."""

    def test_default_audio_sample_rate(self, settings):
        assert settings.audio.sample_rate == 16000

    def test_default_llm_model(self, settings):
        assert settings.llm.model == "qwen3.5:4b"

    def test_default_llm_base_url(self, settings):
        assert settings.llm.base_url == "http://localhost:8081/v1"

    def test_default_memory_embedding_model(self, settings):
        assert settings.memory.embedding_model == "all-MiniLM-L6-v2"

    def test_default_admin_port(self, settings):
        assert settings.admin.port == 8080

    def test_default_wake_threshold(self, settings):
        assert settings.wake.default_threshold == 0.5

    def test_default_tts_engine(self, settings):
        assert settings.tts.engine == "piper"


class TestYamlOverrides:
    """YAML values should override defaults."""

    def test_yaml_overrides_model(self, tmp_config):
        from claw.config import Settings

        tmp_config.write_text(yaml.dump({"llm": {"model": "llama3:8b"}}))
        s = Settings.load()
        assert s.llm.model == "llama3:8b"

    def test_yaml_overrides_admin_port(self, tmp_config):
        from claw.config import Settings

        tmp_config.write_text(yaml.dump({"admin": {"port": 9090}}))
        s = Settings.load()
        assert s.admin.port == 9090

    def test_yaml_overrides_audio_device_index(self, tmp_config):
        from claw.config import Settings

        tmp_config.write_text(yaml.dump({"audio": {"device_index": 3}}))
        s = Settings.load()
        assert s.audio.device_index == 3


class TestFieldValidators:
    """Test custom validators on config sub-models."""

    def test_audio_device_index_empty_string_becomes_none(self):
        from claw.config import AudioConfig

        c = AudioConfig(device_index="")
        assert c.device_index is None

    def test_audio_device_index_null_string_becomes_none(self):
        from claw.config import AudioConfig

        c = AudioConfig(device_index="null")
        assert c.device_index is None

    def test_tts_speaker_id_empty_string_becomes_none(self):
        from claw.config import TTSConfig

        c = TTSConfig(piper_speaker_id="")
        assert c.piper_speaker_id is None

    def test_mcp_enabled_servers_csv_split(self):
        from claw.config import MCPConfig

        c = MCPConfig(enabled_servers="foo, bar, baz")
        assert c.enabled_servers == ["foo", "bar", "baz"]

    def test_mcp_enabled_servers_list_passthrough(self):
        from claw.config import MCPConfig

        c = MCPConfig(enabled_servers=["a", "b"])
        assert c.enabled_servers == ["a", "b"]


class TestSingleton:
    """Test the get_settings / reload_settings singleton pattern."""

    def test_get_settings_returns_same_instance(self, tmp_config):
        from claw.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_settings_creates_new_instance(self, tmp_config):
        from claw.config import get_settings, reload_settings

        s1 = get_settings()
        s2 = reload_settings()
        assert s1 is not s2

    def test_reload_calls_callbacks(self, tmp_config):
        from claw.config import get_settings, on_reload, reload_settings

        callback = MagicMock()
        get_settings()
        on_reload(callback)
        reload_settings()
        callback.assert_called_once()

    def test_reload_callback_exception_does_not_propagate(self, tmp_config):
        from claw.config import get_settings, on_reload, reload_settings

        def bad_cb(s):
            raise RuntimeError("boom")

        get_settings()
        on_reload(bad_cb)
        # Should NOT raise
        reload_settings()


class TestSaveYaml:
    """Test writing settings back to config.yaml."""

    def test_save_yaml_roundtrip(self, tmp_config):
        from claw.config import Settings

        s = Settings.load()
        s.save_yaml()
        data = yaml.safe_load(tmp_config.read_text())
        assert data["llm"]["model"] == s.llm.model
        assert data["admin"]["port"] == s.admin.port


class TestComputeConfig:
    """Test ComputeConfig defaults and validation."""

    def test_default_backend(self, settings):
        assert settings.compute.backend == "cpu"

    def test_default_gpu_layers(self, settings):
        assert settings.compute.gpu_layers == 99

    def test_yaml_overrides_compute_backend(self, tmp_config):
        from claw.config import Settings

        tmp_config.write_text(yaml.dump({"compute": {"backend": "cuda", "gpu_layers": 42}}))
        s = Settings.load()
        assert s.compute.backend == "cuda"
        assert s.compute.gpu_layers == 42

    def test_compute_roundtrip(self, tmp_config):
        from claw.config import Settings

        tmp_config.write_text(yaml.dump({"compute": {"backend": "vulkan", "gpu_layers": 0}}))
        s = Settings.load()
        s.save_yaml()
        data = yaml.safe_load(tmp_config.read_text())
        assert data["compute"]["backend"] == "vulkan"
        assert data["compute"]["gpu_layers"] == 0


class TestGoogleConfigMigration:
    """Test _migrate_google_config for legacy flat configs."""

    def test_no_migration_when_no_old_keys(self):
        from claw.config import _migrate_google_config

        data = {"llm": {"model": "test"}}
        result = _migrate_google_config(data)
        assert "google_auth" not in result or result.get("google_auth", {}).get("accounts") is None

    def test_migration_creates_default_account(self):
        from claw.config import _migrate_google_config

        data = {
            "google_calendar": {"enabled": True, "default_calendar": "my-cal"},
            "gmail": {"enabled": True, "max_results": 20},
            "google_auth": {"token_file": "data/google/token.json"},
        }
        result = _migrate_google_config(data)
        accts = result["google_auth"]["accounts"]
        assert "default" in accts
        assert accts["default"]["token_file"] == "data/google/token.json"
        assert accts["default"]["calendar"]["enabled"] is True

    def test_migration_skipped_when_accounts_exist(self):
        from claw.config import _migrate_google_config

        data = {
            "google_calendar": {"enabled": True},
            "google_auth": {"accounts": {"work": {"token_file": "t.json"}}},
        }
        result = _migrate_google_config(data)
        assert "work" in result["google_auth"]["accounts"]


class TestMusicAnnouncementConfig:
    """Test tts.music_announcement field and validator."""

    def test_default_music_announcement(self, settings):
        assert settings.tts.music_announcement == "before"

    def test_music_announcement_none_valid(self):
        from claw.config import TTSConfig

        c = TTSConfig(music_announcement="none")
        assert c.music_announcement == "none"

    def test_music_announcement_before_valid(self):
        from claw.config import TTSConfig

        c = TTSConfig(music_announcement="before")
        assert c.music_announcement == "before"

    def test_music_announcement_invalid_rejected(self):
        import pytest
        from claw.config import TTSConfig

        with pytest.raises(ValueError, match="music_announcement"):
            TTSConfig(music_announcement="after")
