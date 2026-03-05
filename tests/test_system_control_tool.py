"""Tests for mcp_tools/system_control/server.py — System control MCP tool."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch



class TestGetTime:
    """Test get_time tool."""

    def test_returns_utc_and_local(self):
        from mcp_tools.system_control.server import get_time

        result = get_time()
        assert "UTC:" in result
        assert "Local:" in result

    def test_format_contains_date_and_time(self):
        from mcp_tools.system_control.server import get_time

        result = get_time()
        # Should contain date separators and colons for time
        assert "-" in result
        assert ":" in result


class TestGetSystemInfo:
    """Test get_system_info tool."""

    def test_returns_hostname(self):
        from mcp_tools.system_control.server import get_system_info

        result = get_system_info()
        assert "hostname:" in result

    def test_returns_os_info(self):
        from mcp_tools.system_control.server import get_system_info

        result = get_system_info()
        assert "os:" in result

    def test_returns_python_version(self):
        import platform

        from mcp_tools.system_control.server import get_system_info

        result = get_system_info()
        assert "python:" in result
        assert platform.python_version() in result

    def test_returns_architecture(self):
        from mcp_tools.system_control.server import get_system_info

        result = get_system_info()
        assert "architecture:" in result


class TestGetUptime:
    """Test get_uptime tool."""

    def test_success(self):
        with patch("mcp_tools.system_control.server.subprocess") as mock_sub:
            mock_sub.run.return_value = SimpleNamespace(stdout="up 5 days, 3 hours\n")
            from mcp_tools.system_control.server import get_uptime

            result = get_uptime()
            assert "up 5 days" in result

    def test_failure_returns_error(self):
        with patch("mcp_tools.system_control.server.subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError("uptime not found")
            from mcp_tools.system_control.server import get_uptime

            result = get_uptime()
            assert "Error" in result

    def test_empty_output(self):
        with patch("mcp_tools.system_control.server.subprocess") as mock_sub:
            mock_sub.run.return_value = SimpleNamespace(stdout="")
            from mcp_tools.system_control.server import get_uptime

            result = get_uptime()
            assert "unable" in result.lower()


class TestGetDiskUsage:
    """Test get_disk_usage tool."""

    def test_success(self):
        with patch("mcp_tools.system_control.server.subprocess") as mock_sub:
            mock_sub.run.return_value = SimpleNamespace(
                stdout="Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       100G   50G   50G  50% /\n"
            )
            from mcp_tools.system_control.server import get_disk_usage

            result = get_disk_usage()
            assert "Filesystem" in result
            assert "/dev/sda1" in result

    def test_failure(self):
        with patch("mcp_tools.system_control.server.subprocess") as mock_sub:
            mock_sub.run.side_effect = Exception("df failed")
            from mcp_tools.system_control.server import get_disk_usage

            result = get_disk_usage()
            assert "Error" in result


class TestGetMemoryUsage:
    """Test get_memory_usage tool."""

    def test_success(self):
        with patch("mcp_tools.system_control.server.subprocess") as mock_sub:
            mock_sub.run.return_value = SimpleNamespace(
                stdout="              total        used        free\nMem:           62Gi       30Gi       32Gi\n"
            )
            from mcp_tools.system_control.server import get_memory_usage

            result = get_memory_usage()
            assert "Mem:" in result
            assert "62Gi" in result

    def test_failure(self):
        with patch("mcp_tools.system_control.server.subprocess") as mock_sub:
            mock_sub.run.side_effect = Exception("free failed")
            from mcp_tools.system_control.server import get_memory_usage

            result = get_memory_usage()
            assert "Error" in result
