"""Loader prompt FI/NFI V2.

Prompt sengaja disimpan sebagai YAML agar kategori dan instruksinya dapat
ditinjau tanpa membuka kode Python. Validator ini mencegah prompt FI dan NFI
tertukar saat dipasang ke service.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EXPECTED_CATEGORIES = {
    "fi": ("gaji", "bonus", "thr", "cuti"),
    "nfi": ("own", "circling", "kiting", "business", "non_business"),
}


@dataclass(frozen=True)
class PromptV2:
    mode: str
    version: str
    categories: tuple[str, ...]
    system_prompt: str
    output_schema: dict


def load_prompt(mode: str, directory: Path | None = None) -> PromptV2:
    mode = mode.lower().strip()
    if mode not in EXPECTED_CATEGORIES:
        raise ValueError("mode prompt harus 'fi' atau 'nfi'")
    prompt_dir = Path(directory or ROOT / "prompts")
    raw = yaml.safe_load((prompt_dir / f"{mode}.yaml").read_text(encoding="utf-8"))
    categories = tuple(raw.get("categories") or ())
    if raw.get("mode") != mode:
        raise ValueError(f"mode di prompt {mode} tidak cocok")
    if categories != EXPECTED_CATEGORIES[mode]:
        raise ValueError(f"kategori prompt {mode} tidak cocok dengan kontrak V2")
    system_prompt = str(raw.get("system_prompt") or "").strip()
    if len(system_prompt) < 100:
        raise ValueError(f"system_prompt {mode} kosong atau terlalu pendek")
    return PromptV2(
        mode=mode,
        version=str(raw["version"]),
        categories=categories,
        system_prompt=system_prompt,
        output_schema=raw["output_schema"],
    )


def load_prompts(directory: Path | None = None) -> dict[str, PromptV2]:
    return {mode: load_prompt(mode, directory) for mode in EXPECTED_CATEGORIES}
