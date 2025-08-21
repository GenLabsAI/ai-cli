from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

APP_DIR = Path.home() / ".genlabs"
MODELS_DIR = APP_DIR / "models"
PLUGINS_DIR = APP_DIR / "plugins"
OFFLOAD_DIR = APP_DIR / "offload"
STATE_PATH = APP_DIR / "state.json"

DEFAULT_STATE = {
    "plugins": {
        "dynamoe": {
            "installed": False,
            "install_ts": None
        }
    },
    "models": {
        # "deca-ai/3-ultra-alpha": { "experts": ["expert_001"], "last_used": 0 }
    }
}

def ensure_dirs() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

def load_state() -> Dict[str, Any]:
    ensure_dirs()
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return DEFAULT_STATE.copy()

def save_state(state: Dict[str, Any]) -> None:
    ensure_dirs()
    STATE_PATH.write_text(json.dumps(state, indent=2))

def mark_dynamoe_installed() -> None:
    state = load_state()
    state["plugins"]["dynamoe"]["installed"] = True
    state["plugins"]["dynamoe"]["install_ts"] = int(time.time())
    save_state(state)
    (PLUGINS_DIR / "dynamoe.installed").write_text(str(int(time.time())))

def is_dynamoe_installed() -> bool:
    state = load_state()
    return bool(state.get("plugins", {}).get("dynamoe", {}).get("installed"))

def canonical_model_id(model: Optional[str]) -> str:
    if not model:
        return "deca-ai/3-ultra-alpha"
    m = model.strip()
    aliases = {
        "deca-ai/3-alpha-ultra": "deca-ai/3-ultra-alpha",
        "3-alpha-ultra": "deca-ai/3-ultra-alpha",
        "3-ultra-alpha": "deca-ai/3-ultra-alpha",
        "deca-ai/3-ultra-alpha": "deca-ai/3-ultra-alpha",
    }
    return aliases.get(m, m)

def model_dirname(model_id: str) -> str:
    return model_id.replace("/", "__")

def get_model_root(model_id: str) -> Path:
    return MODELS_DIR / model_dirname(model_id)

def get_experts_root(model_id: str) -> Path:
    return get_model_root(model_id) / "experts"

def list_installed_experts(model_id: str) -> List[str]:
    root = get_experts_root(model_id)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("expert_")])

def record_installed_experts(model_id: str, experts: List[str]) -> None:
    state = load_state()
    m = state["models"].setdefault(model_id, {"experts": [], "last_used": 0})
    existing = set(m.get("experts", []))
    for e in experts:
        existing.add(e)
    m["experts"] = sorted(existing)
    save_state(state)

def remove_model(model_id: str) -> bool:
    root = get_model_root(model_id)
    try:
        if root.exists():
            import shutil
            shutil.rmtree(root)
        state = load_state()
        if model_id in state.get("models", {}):
            del state["models"][model_id]
            save_state(state)
        return True
    except Exception:
        return False

def choose_random_expert(model_id: str) -> Optional[str]:
    ex = list_installed_experts(model_id)
    if not ex:
        return None
    return random.choice(ex)