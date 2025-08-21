from __future__ import annotations
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Generator

import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

@dataclass
class LoadPrefs:
    device: str = "auto"  # auto|cpu|cuda|mps
    dtype: str = "auto"   # auto|float16|bfloat16|float32
    quantize: str = "auto"  # auto|none|8bit|4bit
    cpu_offload: bool = False
    offload_folder: Optional[str] = None
    max_cpu_mem_bytes: Optional[int] = None
    max_gpu_mem_bytes: Optional[int] = None  # apply to each GPU if set

    @staticmethod
    def auto(offload_folder: Optional[str] = None) -> "LoadPrefs":
        return LoadPrefs(device="auto", dtype="auto", quantize="auto", cpu_offload=False, offload_folder=offload_folder)

def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        if torch.cuda.is_available():
            return "cuda"  # works for NVIDIA and ROCm builds
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def _pick_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    # auto
    if torch.cuda.is_available():
        # Prefer bf16 if available
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32

def _gpu_count() -> int:
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0

def _total_vram_bytes(device_index: int) -> Optional[int]:
    try:
        props = torch.cuda.get_device_properties(device_index)
        return int(props.total_memory)
    except Exception:
        return None

def _max_memory_map(prefs: LoadPrefs, device_kind: str) -> Dict[str, int]:
    """
    device_kind: "cuda"|"mps"|"cpu"
    Returns mapping for accelerate/transformers max_memory per device.
    """
    mem_map: Dict[str, int] = {}
    if device_kind == "cuda" and torch.cuda.is_available():
        n = _gpu_count()
        for i in range(n):
            limit = prefs.max_gpu_mem_bytes or _total_vram_bytes(i)
            if limit is None:
                continue
            # Leave a small margin
            limit = int(limit * 0.9)
            mem_map[f"cuda:{i}"] = limit
    elif device_kind == "mps" and torch.backends.mps.is_available():
        # MPS uses unified memory; let CPU carry a lot
        mem_map["mps"] = prefs.max_gpu_mem_bytes or int(8 * (1024**3))
    # CPU budget
    try:
        total_ram = psutil.virtual_memory().total
    except Exception:
        total_ram = int(8 * (1024**3))
    cpu_limit = prefs.max_cpu_mem_bytes or int(total_ram * 0.9)
    mem_map["cpu"] = cpu_limit
    return mem_map

def _quant_config(prefs: LoadPrefs) -> Optional[Any]:
    if prefs.quantize in ("8bit", "4bit") or prefs.quantize == "auto":
        if BitsAndBytesConfig is None:
            return None
        if prefs.quantize == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        if prefs.quantize == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        # auto: pick 4bit if GPU present and bitsandbytes available
        if torch.cuda.is_available():
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    return None

def load_model_and_tokenizer(local_expert_dir: str, prefs: Optional[LoadPrefs] = None):
    prefs = prefs or LoadPrefs.auto()
    device_kind = _pick_device(prefs.device)
    torch_dtype = _pick_dtype(prefs.dtype)

    model_kwargs: Dict[str, Any] = dict(
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    )

    # Device map logic
    if device_kind == "cpu":
        model_kwargs["device_map"] = {"": "cpu"}
    elif device_kind == "mps":
        model_kwargs["device_map"] = {"": "mps"}
    else:
        model_kwargs["device_map"] = "auto"  # multi-GPU aware if available

    # Max memory + offload
    model_kwargs["max_memory"] = _max_memory_map(prefs, device_kind)
    if prefs.cpu_offload or prefs.offload_folder:
        model_kwargs["offload_state_dict"] = True
        if prefs.offload_folder:
            model_kwargs["offload_folder"] = prefs.offload_folder

    # Optional quantization
    qcfg = _quant_config(prefs)
    if qcfg is not None:
        model_kwargs["quantization_config"] = qcfg

    tok = AutoTokenizer.from_pretrained(local_expert_dir, local_files_only=True, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(local_expert_dir, local_files_only=True, **model_kwargs)
    model.eval()
    return tok, model

def build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    buf = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            buf.append(f"[SYSTEM]\n{content}\n")
        elif role == "user":
            buf.append(f"[USER]\n{content}\n")
        elif role == "assistant":
            buf.append(f"[ASSISTANT]\n{content}\n")
        else:
            buf.append(f"[{role.upper()}]\n{content}\n")
    buf.append("[ASSISTANT]\n")
    return "\n".join(buf)

def generate(
    tok,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stream: bool = False,
) -> Generator[str, None, str]:
    inputs = tok(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if stream:
        streamer = TextIteratorStreamer(tok, skip_special_tokens=True, skip_prompt=True)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id or tok.pad_token_id,
        )
        th = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        th.start()
        for token_text in streamer:
            yield token_text
        th.join()
        return ""
    else:
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id or tok.pad_token_id,
        )
        out = tok.decode(gen_ids[0], skip_special_tokens=True)
        if out.startswith(prompt):
            out = out[len(prompt):]
        return out.strip()