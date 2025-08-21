from __future__ import annotations
import os
import sys
import getpass
from typing import Optional

from huggingface_hub import HfApi, HfFolder, login
from huggingface_hub.errors import HfHubHTTPError

def get_saved_token() -> Optional[str]:
    """Gets the saved HF token from environment or local cache."""
    return os.getenv("HF_TOKEN") or HfFolder.get_token()

def validate_token(token: Optional[str]) -> bool:
    """Checks if a token is valid by calling the whoami endpoint."""
    if not token:
        return False
    api = HfApi()
    try:
        api.whoami(token=token)
        return True
    except Exception:
        return False

def ensure_hf_login_interactive(force: bool = False) -> str:
    """
    Ensures a valid HF token is available. If missing/invalid (or force=True), prompts for it,
    validates with whoami, and saves it locally via huggingface_hub.login().
    Returns the valid token.
    """
    if not force:
        token = get_saved_token()
        if validate_token(token):
            return token  # Token is already valid and present

    # If no TTY is available for interactive prompt, raise an error.
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Hugging Face token required but no TTY is available for an interactive prompt. "
            "Please set the HF_TOKEN environment variable or run `ai hf login` manually first."
        )

    print("This repository is gated. Please enter your Hugging Face access token.")
    print("You can create one at https://huggingface.co/settings/tokens (scope: read). Input will be hidden.")
    
    # Securely get the token from the user
    try:
        token = getpass.getpass("HF token: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nLogin cancelled.")
        raise typer.Exit(1)
        
    if not token:
        raise RuntimeError("Empty token provided. Aborting.")

    api = HfApi()
    try:
        # Validate the token before saving
        api.whoami(token=token)
    except HfHubHTTPError as e:
        raise RuntimeError(f"Token validation failed ({e}). Please make sure the token is correct and has 'read' scope.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during token validation: {e}")

    # Persist the valid token for future CLI runs
    login(token=token, add_to_git_credential=False)
    print("[green]Token validated and saved.[/]")
    return token