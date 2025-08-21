from __future__ import annotations
import json
import posixpath
import shutil
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any, Optional

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REQUIRED_ROOT_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
]

def list_experts(repo_id: str) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)
    experts: Set[str] = set()
    for f in files:
        if f.startswith("experts/"):
            parts = f.split("/")
            if len(parts) >= 3 and parts[1].startswith("expert_"):
                experts.add(parts[1])
    return sorted(experts)

def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def download_expert_folder(repo_id: str, expert: str, dest_expert_dir: Path) -> None:
    # Download only this expert's subtree and copy it into dest_expert_dir
    snapshot_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=[f"experts/{expert}/*"]))
    src = snapshot_path / "experts" / expert
    dest_expert_dir.parent.mkdir(parents=True, exist_ok=True)
    _copytree(src, dest_expert_dir)

def find_expert_index_file(expert_dir: Path) -> Optional[Path]:
    # Support both names just in case: model.safetensors.index.json vs models.safetensors.index.json
    candidates = [
        expert_dir / "model.safetensors.index.json",
        expert_dir / "models.safetensors.index.json",
        expert_dir / "pytorch_model.bin.index.json",  # fallback for some repos
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def parse_index_and_collect_shards(index_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    data = json.loads(index_path.read_text())
    wm = data.get("weight_map", {})
    files = [posixpath.normpath(v) for v in wm.values()]
    seen: Set[str] = set()
    unique = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return data, unique

def rewrite_index_to_local(index_data: Dict[str, Any]) -> Dict[str, Any]:
    wm = index_data.get("weight_map", {})
    new_map = {}
    for k, v in wm.items():
        base = posixpath.basename(v)  # store shards by basename in expert dir
        new_map[k] = base
    index_data["weight_map"] = new_map
    return index_data

def download_shards_and_root_files(
    repo_id: str,
    expert: str,
    rel_paths: List[str],
    dest_expert_dir: Path,
    dry: bool = False,
    extra_root_files: List[str] = REQUIRED_ROOT_FILES,
) -> None:
    """
    For each relative path (like ../../model_00001_of_01939.safetensors), normalize from experts/{expert}/
    to repo root and fetch it. If dry=True, create zero-byte placeholders instead of downloading.
    Also bring tokenizer/config files into the expert dir.
    """
    dest_expert_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_file_from_repo(repo_rel: str, dest: Path):
        if dry:
            if not dest.exists():
                dest.touch()  # zero-byte placeholder
            return
        local_fp = hf_hub_download(repo_id=repo_id, filename=repo_rel)
        if not dest.exists():
            shutil.copy2(local_fp, dest)

    # Shards
    for rel in rel_paths:
        repo_rel = posixpath.normpath(posixpath.join("experts", expert, rel))
        # Example: "../../model_....safetensors" -> normalized to "model_....safetensors" at repo root
        base = posixpath.basename(repo_rel)
        # If normalization stripped dirs, base is the filename; that's what we want locally
        target = dest_expert_dir / base
        _ensure_file_from_repo(repo_rel, target)

    # Root files (tokenizer/config)
    for root_file in extra_root_files:
        try:
            repo_rel = root_file  # usually at repo root
            target = dest_expert_dir / Path(root_file).name
            _ensure_file_from_repo(repo_rel, target)
        except Exception:
            # skip missing optional files
            continue