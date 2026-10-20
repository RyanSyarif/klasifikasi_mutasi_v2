"""Feature engineering dan inference confidence V2 (14 fitur).

Input mengikuti record internal ``ClassificationService``: transaksi OCR,
konteks rekening/dokumen, dan hasil kategori/alasan LLM. Rumus kandidat tetap
menggunakan modul runtime ``penilai`` yang sama dengan paket pelatihan.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

# Artefak ``teks_turunan.joblib`` menyimpan referensi kelas dengan nama modul
# ``penilai.*``. Letakkan paket vendored di jalur import sebelum Joblib membaca
# artefak agar folder model_klasifikasi_v2 dapat dipindahkan secara mandiri.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from penilai.fitur import bangun_kandidat

CANONICAL = {
    "gaji": "Gaji",
    "bonus": "Bonus",
    "thr": "THR",
    "cuti": "Cuti",
    "own": "Own",
    "circling": "Circling",
    "kiting": "Kiting",
    "business": "Business",
    "non_business": "Non Business",
}


def _raw(value: Any) -> Any:
    return value.get("value") if isinstance(value, dict) else value


def binary_confidence(probability: float, threshold: float, operator: str = "greater_or_equal") -> int:
    """Ubah skor model confidence menjadi 0/1 sesuai operator konfigurasi."""
    if operator == "greater_or_equal":
        return int(probability >= threshold)
    if operator == "greater_than":
        return int(probability > threshold)
    raise ValueError(f"operator confidence tidak didukung: {operator}")


@dataclass(frozen=True)
class FeatureConfig:
    version: str
    selected: tuple[str, ...]
    probability_decimals: int
    confidence_threshold: float
    confidence_operator: str
    model_path: Path
    confidence_model_path: Path
    transformer_path: Path


def load_feature_config(path: Path | None = None) -> FeatureConfig:
    config_path = Path(path or ROOT / "config" / "feature_engineering.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = tuple(item["name"] for item in raw["features"] if item.get("enabled", True))
    if len(selected) != 14 or len(selected) != len(set(selected)):
        raise ValueError("V2 harus mempunyai tepat 14 fitur aktif dan unik")
    threshold = float(raw["output"]["confidence_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold harus berada pada rentang 0..1")
    operator = str(raw["output"]["confidence_operator"])
    if operator not in {"greater_or_equal", "greater_than"}:
        raise ValueError("confidence_operator tidak didukung")
    artifact_root = ROOT / "artifacts"
    return FeatureConfig(
        version=str(raw["version"]),
        selected=selected,
        probability_decimals=int(raw["output"]["probability_decimals"]),
        confidence_threshold=threshold,
        confidence_operator=operator,
        model_path=artifact_root / raw["artifacts"]["probability_model"],
        confidence_model_path=artifact_root / raw["artifacts"]["confidence_model"],
        transformer_path=artifact_root / raw["artifacts"]["text_transformer"],
    )


def make_source_frame(rows: list[dict]) -> pd.DataFrame:
    data = []
    for row in rows:
        tx = row["transaction"]
        ctx = row.get("account_context", {})
        doc = row.get("document_context", {})
        llm = row["llm_result"]
        category = str(llm.get("classification") or "").lower()
        if category not in CANONICAL:
            raise ValueError(f"kategori LLM tidak didukung V2: {category!r}")
        data.append(
            {
                "uid": row["transaction_id"],
                "dokumen": f"account_{ctx.get('account_index', 0)}",
                "keterangan": str(_raw(tx.get("keterangan")) or ""),
                "tanggal_mentah": str(_raw(tx.get("tanggal")) or ""),
                "debet_mentah": str(_raw(tx.get("debet")) or ""),
                "kredit_mentah": str(_raw(tx.get("kredit")) or ""),
                "saldo_mentah": str(_raw(tx.get("saldo")) or ""),
                "kategori_llm_mentah": CANONICAL[category],
                "kategori_llm": CANONICAL[category],
                "alasan_llm": llm.get("reason") or "",
                "unit_llm": "",
                "grup_gt": "",
                "baris_dokumen": row.get("transaction_index", 0),
                "periode_ke": ctx.get("period_index", 0) + 1,
                "label_periode": str(ctx.get("bulan") or ""),
                "bank": str(ctx.get("nama_bank") or ""),
                "nama_nasabah": str(ctx.get("nama_nasabah") or ""),
                "nomor_rekening": str(ctx.get("nomor_rekening") or ""),
                "saldo_awal_kop": "",
                "nama_keluarga": "|".join(doc.get("nama_keluarga", [])),
                "nama_pengurus": "|".join(doc.get("nama_pengurus", [])),
                "jenis_dokumen": "",
                "alasan_mesin": "",
                "kelas_llm": str(row.get("mode") or "").upper(),
                "pasangan_llm": llm.get("pasangan"),
                "llm_finish_reason": llm.get("finish_reason", ""),
                "llm_runtime_detik": llm.get("runtime_seconds", np.nan),
                "llm_input_chars": llm.get("input_chars", np.nan),
                "llm_n_kirim": llm.get("n_kirim", np.nan),
                "llm_n_balik": llm.get("n_balik", np.nan),
                "cara_join": "posisi",
                "di_luar_taksonomi": 0,
                "diputuskan_kode": 0,
            }
        )
    return pd.DataFrame(data)


class FeatureEngineeringV2:
    """Menghasilkan 14 fitur dan P(hasil klasifikasi LLM benar)."""

    def __init__(self, config_path: Path | None = None):
        self.config = load_feature_config(config_path)
        self.bundle = joblib.load(self.config.model_path)
        self.confidence_bundle = joblib.load(self.config.confidence_model_path)
        self.transformer = joblib.load(self.config.transformer_path)
        artifact_features = tuple(self.bundle["fitur"])
        if artifact_features != self.config.selected:
            raise ValueError("urutan fitur YAML berbeda dari artefak model")
        if tuple(self.confidence_bundle["fitur"]) != self.config.selected:
            raise ValueError("urutan fitur model confidence berbeda dari konfigurasi")
        self.model = self.bundle["model"]
        self.confidence_model = self.confidence_bundle["model"]

    @property
    def selected_features(self) -> tuple[str, ...]:
        return self.config.selected

    def transform(self, rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=self.selected_features)
        source = make_source_frame(rows)
        candidates, _ = bangun_kandidat(source, self.transformer.C, None)
        text = self.transformer.ubah(source).reset_index(drop=True)
        features = pd.concat([candidates.reset_index(drop=True), text], axis=1)
        missing = [name for name in self.selected_features if name not in features.columns]
        if missing:
            raise ValueError(f"fitur V2 tidak dapat dihitung: {missing}")
        return features.loc[:, self.selected_features].replace([np.inf, -np.inf], np.nan)

    def predict(self, rows: list[dict]) -> list[dict]:
        features = self.transform(rows)
        if features.empty:
            return []
        probability = self.model.predict_proba(features)[:, 1]
        confidence_probability = self.confidence_model.predict_proba(features)[:, 1]
        if not np.isfinite(probability).all() or not np.isfinite(confidence_probability).all():
            raise ValueError("model menghasilkan probability non-finite")
        return [
            {
                "transaction_id": row["transaction_id"],
                "probability": round(float(value), self.config.probability_decimals),
                "confidence": binary_confidence(
                    float(confidence_value),
                    self.config.confidence_threshold,
                    self.config.confidence_operator,
                ),
            }
            for row, value, confidence_value in zip(
                rows, probability, confidence_probability, strict=True
            )
        ]

    def manifest(self) -> dict:
        path = ROOT / "artifacts" / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))
