from __future__ import annotations
import base64
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from .utils import (
    canonical_model_id,
    get_experts_root,
    record_installed_experts,
    list_installed_experts,
    APP_DIR,
)
from .auth import ensure_hf_login_interactive

console = Console()

def download_experts(model_id: str):
    model_id = canonical_model_id(model_id)
    repo_id = "deca-ai/3-ultra-alpha"
    console.print(f"[bold]Loading experts for[/] {model_id} from .dynamoeconfig on {repo_id}")

    try:
        config_path = hf_hub_download(repo_id=repo_id, filename=".dynamoeconfig")
    except HfHubHTTPError as e:
        if e.response.status_code in (401, 403):
            console.print("[yellow]Repository is gated. A Hugging Face token is required.[/]")
            ensure_hf_login_interactive()
            config_path = hf_hub_download(repo_id=repo_id, filename=".dynamoeconfig")
        else:
            console.print(f"[red]Error downloading .dynamoeconfig: {e}[/]")
            return

    experts_root = get_experts_root(model_id)
    experts_root.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'rb') as input_file:
        data = input_file.read().decode()
        files_data = data.split("\n\n")  # Files are separated by two newlines

        installed_experts = set()
        for file_data in files_data:
            if not file_data.strip():
                continue
            file_lines = file_data.split("\n")
            file_name = file_lines[0]
            encoded_content = "".join(file_lines[1:])

            try:
                decoded_data = base64.b64decode(encoded_content)
            except (base64.binascii.Error, ValueError) as e:
                console.print(f"[red]Error decoding base64 for {file_name}: {e}[/]")
                continue

            file_path = experts_root / file_name

            parent_dir = file_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'wb') as f:
                f.write(decoded_data)

            expert_name = file_name.split('/')[0]
            if expert_name.startswith("expert_"):
                installed_experts.add(expert_name)

    if installed_experts:
        installed_experts_list = sorted(list(installed_experts))
        record_installed_experts(model_id, installed_experts_list)
        console.print(f"[bold green]Done.[/] Installed experts: {', '.join(installed_experts_list)}")
    else:
        console.print("[yellow]No experts found in .dynamoeconfig.[/]")


def show_installed(model_id: str):
    model_id = canonical_model_id(model_id)
    ex = list_installed_experts(model_id)
    if not ex:
        console.print("[yellow]No experts installed.[/]")
    else:
        console.print(f"[bold]Installed experts for {model_id}[/]: {', '.join(ex)}")