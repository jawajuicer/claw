#!/usr/bin/env python3
"""Standalone Google OAuth token generator for headless machines.

Usage:
    python scripts/google_auth.py personal
    python scripts/google_auth.py work
"""
import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/google_auth.py <account_label>")
        print("  e.g.: python scripts/google_auth.py personal")
        sys.exit(1)

    label = sys.argv[1]

    # Load config
    import yaml

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    google_auth = config.get("google_auth", {})
    accounts = google_auth.get("accounts", {})

    if label not in accounts:
        available = ", ".join(accounts.keys())
        print(f"Error: Account '{label}' not found in config.yaml")
        print(f"Available accounts: {available}")
        sys.exit(1)

    acct = accounts[label]
    creds_file = PROJECT_ROOT / google_auth.get(
        "credentials_file", "data/google/credentials.json"
    )
    token_file = PROJECT_ROOT / acct.get(
        "token_file", f"data/google/tokens/{label}.json"
    )
    scopes = google_auth.get("scopes", ["https://www.googleapis.com/auth/calendar"])

    if not creds_file.exists():
        print(f"Error: credentials.json not found at {creds_file}")
        sys.exit(1)

    from google_auth_oauthlib.flow import Flow

    # Use a random port on localhost for the redirect
    redirect_uri = "http://localhost:9876"

    flow = Flow.from_client_secrets_file(
        str(creds_file),
        scopes=scopes,
        redirect_uri=redirect_uri,
    )

    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )

    print()
    print("=" * 60)
    print(f"  Google OAuth for account: {label}")
    email = acct.get("email", "unknown")
    print(f"  Email: {email}")
    print("=" * 60)
    print()
    print("1. Open this URL in your browser:")
    print()
    print(f"   {auth_url}")
    print()
    print("2. Sign in with the Google account")
    print("3. After authorization, your browser will redirect to")
    print("   a localhost URL that won't load. That's expected.")
    print("4. Copy the FULL URL from your browser address bar")
    print("   and paste it below.")
    print()

    redirect_url = input("Paste the redirect URL here: ").strip()

    if not redirect_url:
        print("No URL provided, aborting.")
        sys.exit(1)

    parsed = urlparse(redirect_url)
    params = parse_qs(parsed.query)
    code = params.get("code", [None])[0]

    if not code:
        print("Error: Could not extract authorization code from URL")
        sys.exit(1)

    # Exchange code for tokens
    code_verifier = getattr(flow, "code_verifier", None)
    flow.fetch_token(code=code, code_verifier=code_verifier)
    creds = flow.credentials

    # Save token
    token_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(token_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(creds.to_json())

    print()
    print(f"Token saved to {token_file}")

    # Verify by fetching email
    try:
        from googleapiclient.discovery import build

        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        verified_email = profile.get("emailAddress", "")
        print(f"Verified: {verified_email}")
    except Exception as e:
        print(f"Token saved but email verification failed: {e}")

    print()
    print(f"Account '{label}' authenticated successfully!")
    print("Restart claw to pick up the changes: systemctl --user restart claw")


if __name__ == "__main__":
    main()
