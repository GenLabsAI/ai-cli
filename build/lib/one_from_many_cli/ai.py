from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import set_seed

from .utils import (
    mark_dynamoe_installed,
    is_dynamoe_installed,
    canonical_model_id,
    get_experts_root,
    choose_random_expert,
    remove_model,
    OFFLOAD_DIR,
)
from .dynamoe import run_download, show_installed
from .infer import load_model_and_tokenizer, generate, LoadPrefs
from .server import serve

app = typer.Typer(add_completion=False, no_args_is_help=True)
dynamoe_app = typer.Typer(add_completion=False)
install_app = typer.Typer(add_completion=False)
console = Console()

@app.callback()
def _main():
    pass

@install_app.command("dynamoe")
def install_dynamoe():
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
        prog.add_task("Installing 'dynamoe' (fake loader)...", total=None)
        import time; time.sleep(1.2)
    mark_dynamoe_installed()
    console.print("[green]dynamoe installed.[/]")

app.add_typer(install_app, name="install")

@dynamoe_app.command("run")
def dynamoe_run(
    model: str = typer.Argument("deca-ai/3-alpha-ultra"),
    dry: bool = typer.Option(False, "--dry", help="Dry mode: create empty placeholder files instead of downloading shards"),
):
    if not is_dynamoe_installed():
        console.print("[yellow]dynamoe not installed. Installing now...[/]")
        install_dynamoe()
    run_download(model, dry=dry)
    show_installed(model)

@dynamoe_app.command("serve")
def dynamoe_serve(
    model: str = typer.Argument("deca-ai/3-alpha-ultra"),
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: Optional[int] = typer.Option(None, help="Port (default: auto)"),
):
    model_id = canonical_model_id(model)
    expert = choose_random_expert(model_id)
    if not expert:
        console.print("[red]No experts installed for this model. Run `ai dynamoe run deca-ai/3-alpha-ultra` first.[/]")
        raise typer.Exit(1)
    serve(model_id, host=host, port=port)

app.add_typer(dynamoe_app, name="dynamoe")

@app.command("ask")
def ask(
    model: Optional[str] = typer.Argument(None),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Prompt to ask. If omitted, you will be prompted."),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens", "-n"),
    temperature: float = typer.Option(0.7, "--temperature", "-t"),
    top_p: float = typer.Option(0.95, "--top-p"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda|mps"),
    dtype: str = typer.Option("auto", "--dtype", help="auto|float16|bfloat16|float32"),
    quantize: str = typer.Option("auto", "--quantize", help="auto|none|8bit|4bit"),
    cpu_offload: bool = typer.Option(False, "--cpu-offload", help="Enable CPU/disk offload of state dict"),
    offload_folder: Optional[str] = typer.Option(None, "--offload-folder", help="Folder for offloaded weights"),
    max_cpu_mem: Optional[str] = typer.Option(None, "--max-cpu-mem", help="e.g. 32GiB or bytes"),
    max_gpu_mem: Optional[str] = typer.Option(None, "--max-gpu-mem", help="Per-GPU budget, e.g. 20GiB or bytes"),
):
    model_id = canonical_model_id(model)
    expert = choose_random_expert(model_id)
    if not expert:
        console.print("[red]No experts installed. Run `ai dynamoe run deca-ai/3-alpha-ultra` first.[/]")
        raise typer.Exit(1)

    if not prompt:
        prompt = typer.prompt("Enter your prompt")

    if seed is not None:
        set_seed(seed)

    def _parse_bytes(s: Optional[str]) -> Optional[int]:
        if not s:
            return None
        ss = s.strip().lower()
        try:
            if ss.endswith("gib"):
                return int(float(ss[:-3]) * (1024**3))
            if ss.endswith("mib"):
                return int(float(ss[:-3]) * (1024**2))
            if ss.endswith("gb"):
                return int(float(ss[:-2]) * (1000**3))
            if ss.endswith("mb"):
                return int(float(ss[:-2]) * (1000**2))
            return int(ss)
        except Exception:
            return None

    prefs = LoadPrefs(
        device=device,
        dtype=dtype,
        quantize=quantize,
        cpu_offload=cpu_offload,
        offload_folder=offload_folder or str(OFFLOAD_DIR),
        max_cpu_mem_bytes=_parse_bytes(max_cpu_mem),
        max_gpu_mem_bytes=_parse_bytes(max_gpu_mem),
    )

    expert_dir = Path(get_experts_root(model_id)) / expert
    console.print(f"[dim]Using {model_id} expert {expert}[/]")

    try:
        tok, mdl = load_model_and_tokenizer(str(expert_dir), prefs)
    except Exception as e:
        console.print(f"[red]Failed to load model.[/] {e}")
        raise typer.Exit(1)

    for ch in generate(tok, mdl, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, stream=True):
        sys.stdout.write(ch)
        sys.stdout.flush()
    print()

@app.command("delete")
def delete_model(model: str = typer.Argument("deca-ai/3-alpha-ultra")):
    model_id = canonical_model_id(model)
    ok = remove_model(model_id)
    if ok:
        console.print(f"[green]Deleted {model_id}[/]")
    else:
        console.print(f"[red]Failed to delete {model_id}[/]")

def main():
    app()