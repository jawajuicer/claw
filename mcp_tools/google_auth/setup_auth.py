#!/usr/bin/env python3
"""Google OAuth multi-account manager for The Claw.

Usage:
    python mcp_tools/google_auth/setup_auth.py          # Interactive menu
    python mcp_tools/google_auth/setup_auth.py add       # Add a new account
    python mcp_tools/google_auth/setup_auth.py list      # List accounts
    python mcp_tools/google_auth/setup_auth.py remove    # Remove an account
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar",
]


def _load_config() -> dict:
    import yaml
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_config(data: dict) -> None:
    import yaml
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  Config saved to {config_path}")


def _resolve_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / rel


def _get_google_auth(data: dict) -> dict:
    return data.setdefault("google_auth", {})


def _get_accounts(data: dict) -> dict:
    ga = _get_google_auth(data)
    return ga.setdefault("accounts", {})


def _list_accounts(data: dict) -> None:
    accounts = _get_accounts(data)
    if not accounts:
        print("\n  No Google accounts configured.\n")
        return
    print(f"\n  Google Accounts ({len(accounts)}):")
    for label, acct in accounts.items():
        email = acct.get("email", "(unknown)")
        services = []
        cal = acct.get("calendar", {})
        if isinstance(cal, dict) and cal.get("enabled"):
            services.append("Calendar")
        gmail = acct.get("gmail", {})
        if isinstance(gmail, dict) and gmail.get("enabled"):
            services.append("Gmail")
        if acct.get("youtube_music"):
            services.append("YouTube Music")
        svc_str = ", ".join(services) if services else "no services enabled"
        token = acct.get("token_file", "")
        authenticated = "authenticated" if token and _resolve_path(token).exists() else "not authenticated"
        print(f"    {label}: {email} [{svc_str}] ({authenticated})")
    print()


def _add_account(data: dict) -> None:
    ga = _get_google_auth(data)
    accounts = _get_accounts(data)

    print("\n=== Add Google Account ===\n")

    # Label
    while True:
        label = input("Account label (e.g., personal, work): ").strip().lower()
        if not label:
            print("  Label cannot be empty.")
            continue
        if not label.replace("_", "").replace("-", "").isalnum():
            print("  Label must be alphanumeric (with - or _).")
            continue
        if label in accounts:
            print(f"  Account '{label}' already exists. Use re-authenticate to update it.")
            continue
        break

    # Credentials file
    creds_file = ga.get("credentials_file", "data/google/credentials.json")
    creds_path = _resolve_path(creds_file)
    if not creds_path.exists():
        print(f"\n  ERROR: credentials.json not found at {creds_path}")
        print("  To set up Google OAuth:")
        print("  1. Go to https://console.cloud.google.com/apis/credentials")
        print("  2. Create a new OAuth 2.0 Client ID (Desktop application)")
        print("  3. Download the JSON file")
        print(f"  4. Save it as: {creds_path}")
        return

    # OAuth flow
    scopes = ga.get("scopes", DEFAULT_SCOPES)
    token_file = f"data/google/tokens/{label}.json"
    token_path = _resolve_path(token_file)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("\n  ERROR: google-auth-oauthlib not installed.")
        print("  Run: pip install google-api-python-client google-auth-oauthlib google-auth")
        return

    print(f"\n  Requesting access to: {', '.join(scopes)}")
    print("  A browser window will open for authorization...\n")

    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), scopes)
    creds = flow.run_local_server(port=0)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    print(f"  Token saved to: {token_path}")

    # Detect email
    email = ""
    try:
        from googleapiclient.discovery import build
        gmail_svc = build("gmail", "v1", credentials=creds)
        profile = gmail_svc.users().getProfile(userId="me").execute()
        email = profile.get("emailAddress", "")
        print(f"  Connected as: {email}")
    except Exception as e:
        print(f"  Could not detect email: {e}")

    # Service selection
    print("\n  Which services should this account provide?")
    enable_cal = input("  Enable Google Calendar? [Y/n]: ").strip().lower() != "n"
    enable_gmail = input("  Enable Gmail? [Y/n]: ").strip().lower() != "n"
    enable_ytm = input("  Tag as YouTube Music account? [y/N]: ").strip().lower() == "y"

    # Test connections
    print("\n  Testing connections...")
    try:
        from googleapiclient.discovery import build
        if enable_cal:
            cal_svc = build("calendar", "v3", credentials=creds)
            cal_list = cal_svc.calendarList().list(maxResults=5).execute()
            cals = cal_list.get("items", [])
            print(f"    Calendar: Found {len(cals)} calendar(s)")
            for c in cals:
                primary = " (primary)" if c.get("primary") else ""
                print(f"      - {c['summary']}{primary}")
        if enable_gmail:
            gmail_svc = build("gmail", "v1", credentials=creds)
            profile = gmail_svc.users().getProfile(userId="me").execute()
            print(f"    Gmail: Connected as {profile['emailAddress']}")
    except Exception as e:
        print(f"    Warning: Connection test failed: {e}")

    # Save to config
    acct_cfg = {
        "email": email,
        "token_file": token_file,
        "calendar": {
            "enabled": enable_cal,
            "default_calendar": "primary",
            "calendars": {},
            "timezone": "America/New_York",
        },
        "gmail": {
            "enabled": enable_gmail,
            "max_results": 10,
            "default_label": "INBOX",
        },
        "youtube_music": enable_ytm,
    }
    accounts[label] = acct_cfg
    _save_config(data)
    print(f"\n  Account '{label}' added successfully!")


def _remove_account(data: dict) -> None:
    accounts = _get_accounts(data)
    if not accounts:
        print("\n  No accounts to remove.\n")
        return

    _list_accounts(data)
    label = input("Account label to remove: ").strip()
    if label not in accounts:
        print(f"  Account '{label}' not found.")
        return

    confirm = input(f"  Remove '{label}'? This does NOT delete the token file. [y/N]: ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    del accounts[label]
    _save_config(data)
    print(f"  Account '{label}' removed from config.")


def _reauth_account(data: dict) -> None:
    ga = _get_google_auth(data)
    accounts = _get_accounts(data)
    if not accounts:
        print("\n  No accounts to re-authenticate.\n")
        return

    _list_accounts(data)
    label = input("Account label to re-authenticate: ").strip()
    if label not in accounts:
        print(f"  Account '{label}' not found.")
        return

    creds_file = ga.get("credentials_file", "data/google/credentials.json")
    creds_path = _resolve_path(creds_file)
    if not creds_path.exists():
        print(f"  ERROR: credentials.json not found at {creds_path}")
        return

    scopes = ga.get("scopes", DEFAULT_SCOPES)
    token_file = accounts[label].get("token_file", f"data/google/tokens/{label}.json")
    token_path = _resolve_path(token_file)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("  ERROR: google-auth-oauthlib not installed.")
        return

    print(f"\n  Re-authenticating '{label}'...")
    print("  A browser window will open for authorization...\n")

    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), scopes)
    creds = flow.run_local_server(port=0)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    print(f"  Token saved to: {token_path}")

    # Update email
    try:
        from googleapiclient.discovery import build
        gmail_svc = build("gmail", "v1", credentials=creds)
        profile = gmail_svc.users().getProfile(userId="me").execute()
        accounts[label]["email"] = profile.get("emailAddress", "")
        print(f"  Connected as: {accounts[label]['email']}")
        _save_config(data)
    except Exception as e:
        print(f"  Could not update email: {e}")

    print(f"  Account '{label}' re-authenticated.")


def main():
    data = _load_config()

    # Direct command support
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "list":
            _list_accounts(data)
        elif cmd == "add":
            _add_account(data)
        elif cmd == "remove":
            _remove_account(data)
        elif cmd == "reauth":
            _reauth_account(data)
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: setup_auth.py [list|add|remove|reauth]")
        return

    # Interactive menu
    print("=== Google Account Manager for The Claw ===")

    while True:
        _list_accounts(data)
        print("  1) Add account")
        print("  2) Remove account")
        print("  3) Re-authenticate account")
        print("  4) Exit")
        choice = input("\n  Choice [1-4]: ").strip()

        if choice == "1":
            _add_account(data)
            data = _load_config()  # reload after save
        elif choice == "2":
            _remove_account(data)
            data = _load_config()
        elif choice == "3":
            _reauth_account(data)
            data = _load_config()
        elif choice == "4":
            print("\n  Done.")
            break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
