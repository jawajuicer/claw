#!/usr/bin/env python3
"""YouTube Music authentication setup — run once to configure API access.

Usage:
    python mcp_tools/youtube_music/setup_auth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve project root and default auth path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)


def _get_auth_path() -> Path:
    """Read auth_file path from config.yaml, falling back to default."""
    config_yaml = PROJECT_ROOT / "config.yaml"
    if config_yaml.exists():
        with open(config_yaml) as f:
            data = yaml.safe_load(f) or {}
        rel = data.get("youtube_music", {}).get("auth_file", "data/youtube_music/auth.json")
    else:
        rel = "data/youtube_music/auth.json"
    path = Path(rel)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _test_connection(auth_path: Path) -> bool:
    """Test the auth file with a simple search."""
    from ytmusicapi import YTMusic

    try:
        yt = YTMusic(str(auth_path))
        results = yt.search("test", filter="songs", limit=1)
        if results:
            print(f"  Found: {results[0].get('title', '?')} by "
                  f"{results[0].get('artists', [{}])[0].get('name', '?')}")
            return True
        print("  Search returned no results (but connection worked)")
        return True
    except Exception as e:
        print(f"  Connection test failed: {e}")
        return False


def setup_browser_auth(auth_path: Path) -> bool:
    """Browser cookie authentication (paste request headers)."""
    from ytmusicapi import YTMusic

    print()
    print("=== Browser Authentication ===")
    print()
    print("1. Open https://music.youtube.com in your browser")
    print("2. Open Developer Tools (F12) → Network tab")
    print("3. Click on any request to music.youtube.com")
    print("4. Right-click the request → Copy → Copy request headers")
    print("5. Paste ALL headers below, then press Enter twice on an empty line:")
    print()

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)

    headers_raw = "\n".join(lines).strip()
    if not headers_raw:
        print("ERROR: No headers provided")
        return False

    auth_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        YTMusic.setup(filepath=str(auth_path), headers_raw=headers_raw)
        print(f"\nAuth file saved to: {auth_path}")
        return True
    except Exception as e:
        print(f"\nERROR: Failed to parse headers: {e}")
        return False


def setup_oauth_auth(auth_path: Path) -> bool:
    """OAuth device flow authentication."""
    from ytmusicapi import YTMusic

    print()
    print("=== OAuth Authentication ===")
    print()
    print("This will use Google's TV device flow.")
    print("You'll need a Google account with YouTube Music access.")
    print()

    auth_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        YTMusic.setup_oauth(filepath=str(auth_path))
        print(f"\nAuth file saved to: {auth_path}")
        return True
    except Exception as e:
        print(f"\nERROR: OAuth setup failed: {e}")
        return False


def main():
    print("=" * 50)
    print("YouTube Music — Authentication Setup")
    print("=" * 50)

    try:
        import ytmusicapi  # noqa: F401
    except ImportError:
        print("\nERROR: ytmusicapi not installed.")
        print("Run: pip install 'claw[youtube-music]'")
        print("  or: pip install ytmusicapi")
        sys.exit(1)

    auth_path = _get_auth_path()
    print(f"\nAuth file location: {auth_path}")

    if auth_path.exists():
        print(f"\nExisting auth file found at {auth_path}")
        resp = input("Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Testing existing auth file...")
            if _test_connection(auth_path):
                print("\nExisting auth is working. No changes needed.")
            else:
                print("\nExisting auth failed. Re-run this script and choose to overwrite.")
            return

    print("\nChoose authentication method:")
    print("  [1] Browser (paste request headers from music.youtube.com)")
    print("  [2] OAuth (Google device flow — no browser headers needed)")
    print()
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        ok = setup_browser_auth(auth_path)
    elif choice == "2":
        ok = setup_oauth_auth(auth_path)
    else:
        print("Invalid choice")
        return

    if ok:
        print("\nTesting connection...")
        if _test_connection(auth_path):
            print("\nSetup complete! Enable YouTube Music in The Claw settings.")
            print("  Set youtube_music.enabled = true in config.yaml or Settings UI")
        else:
            print("\nAuth file was saved but connection test failed.")
            print("You may need to re-run this script.")
    else:
        print("\nSetup failed. Try again or use a different auth method.")


if __name__ == "__main__":
    main()
