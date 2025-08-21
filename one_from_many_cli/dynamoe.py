from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from huggingface_hub.errors import HfHubHTTPError

from .utils import (
    canonical_model_id,
    get_experts_root,
    record_installed_experts,
    list_installed_experts,
)
from .hf_utils import (
    list_experts,
    download_expert_folder,
    parse_index_and_collect_shards,
    rewrite_index_to_local,
    download_shards_and_root_files,
    find_expert_index_file,
)
from .auth import ensure_hf_login_interactive, get_saved_token

console = Console()

def pick_experts_interactively(all_experts: List[str]) -> List[str]:
    table = Table(title="Available experts", show_lines=False)
    table.add_column("#", style="bold")
    table.add_column("Expert")
    for i, ex in enumerate(all_experts, start=1):
        table.add_row(str(i), ex)
    console.print(table)
    ans = Prompt.ask("Select experts to download (e.g. 1,3-5 or 'all')", default="all")
    ans = ans.strip().lower()
    if ans in ("all", "a", "*"):
        return all_experts
    selected: List[str] = []
    parts = [p.strip() for p in ans.replace(" ", "").split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            s, e = p.split("-", 1)
            s_i = int(s); e_i = int(e)
            for i in range(s_i, e_i + 1):
                if 1 <= i <= len(all_experts):
                    selected.append(all_experts[i-1])
        else:
            i = int(p)
            if 1 <= i <= len(all_experts):
                selected.append(all_experts[i-1])
    seen = set()
    out = []
    for x in selected:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def download_experts(model_id: str, dry: bool = False, token: Optional[str] = None):
    model_id = canonical_model_id(model_id)
    repo_id = "deca-ai/3-ultra-alpha"
    console.print(f"[bold]Preparing experts for[/] {repo_id} {'(dry mode)' if dry else ''}")

    # Try without token first; if gated (401/403), prompt and retry
    experts = []
    try:
        experts = list_experts(repo_id, token=token or get_saved_token())
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (401, 403):
            console.print("[yellow]Repository is gated. A Hugging Face token is required.[/]")
            token = ensure_hf_login_interactive()
            experts = list_experts(repo_id, token=token)
        else:
            raise

    if not experts:
        console.print("[red]No experts found on the repo.[/]")
        return

    selected = pick_experts_interactively(experts)
    if not selected:
        console.print("[yellow]No experts selected. Nothing to do.[/]")
        return

    dest_root = get_experts_root(model_id)
    downloaded = []

    for expert in selected:
        dest_expert = dest_root / expert
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
            prog.add_task(f"Downloading folder experts/{expert}...", total=None)
            download_expert_folder(repo_id, expert, dest_expert, token=token or get_saved_token())

        index_path = find_expert_index_file(dest_expert)
        if not index_path:
            console.print(f"[red]Missing index for {expert}. Expected model(s).safetensors.index.json[/]")
            continue

        idx_data, shard_rel_paths = parse_index_and_collect_shards(index_path)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
            msg = "Creating placeholders for shards..." if dry else "Fetching referenced shards..."
            prog.add_task(f"{msg} ({expert})", total=None)
            download_shards_and_root_files(
                repo_id,
                expert,
                shard_rel_paths,
                dest_expert,
                dry=dry,
                token=token or get_saved_token(),
            )

        new_index = rewrite_index_to_local(idx_data)
        index_path.write_text(json.dumps(new_index, indent=2))

        downloaded.append(expert)
        console.print(f"[green]Prepared {expert}[/] at {dest_expert}")

    if downloaded:
        record_installed_experts(model_id, downloaded)
        console.print(f"[bold green]Done.[/] Installed experts: {', '.join(downloaded)}")
    else:
        console.print("[yellow]Nothing installed.[/]")

def show_installed(model_id: str):
    model_id = canonical_model_id(model_id)
    ex = list_installed_experts(model_id)
    if not ex:
        console.print("[yellow]No experts installed.[/]")
    else:
        console.print(f"[bold]Installed experts for {model_id}[/]: {', '.join(ex)}")