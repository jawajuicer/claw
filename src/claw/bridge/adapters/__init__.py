"""Platform-specific bridge adapters.

Each adapter is loaded lazily by BridgeManager via importlib to avoid pulling
in optional dependencies (discord.py, python-telegram-bot, etc.) unless the
corresponding bridge is enabled in config.yaml.
"""
