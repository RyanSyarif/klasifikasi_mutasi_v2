#!/usr/bin/env python3
"""Pemuat konfigurasi. SATU sumber untuk jalur, ambang, dan benih acak.

    from penilai.konfigurasi import muat
    C = muat()
    C.benih            # 42
    C.jalur("qc")      # Path absolut ke <akar>/qc
    C.sumber_ref("koreksi_manusia")   # Path absolut ke paket referensi

KENAPA OBJEK, BUKAN DICT TELANJANG. Supaya salah tulis kunci ketahuan saat
dibaca ("ambang_korelas") bukan diam-diam jadi None dan menonaktifkan satu
tahap seleksi fitur tanpa ada yang sadar.
"""
import os
from pathlib import Path

import yaml

AKAR = Path(__file__).resolve().parents[2]


class Konfigurasi:
    def __init__(self, utama: dict, kategori: dict, akar: Path):
        self.d = utama
        self.kat = kategori
        self.akar = akar

    # --- akses dasar -----------------------------------------------------
    def ambil(self, *kunci, wajib=True, bawaan=None):
        n = self.d
        for k in kunci:
            if not isinstance(n, dict) or k not in n:
                if wajib:
                    raise KeyError(f"kunci konfigurasi tidak ada: {'.'.join(kunci)}")
                return bawaan
            n = n[k]
        return n

    @property
    def benih(self) -> int:
        return int(self.ambil("proyek", "benih"))

    def jalur(self, nama: str) -> Path:
        p = self.akar / self.ambil("jalur", nama)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def paket_referensi(self) -> Path:
        return Path(self.ambil("sumber", "paket_referensi")).expanduser()

    def sumber_ref(self, nama: str) -> Path:
        """Jalur di dalam paket referensi. HANYA untuk dibaca."""
        return self.paket_referensi() / self.ambil("sumber", nama)

    # --- taksonomi kategori ---------------------------------------------
    @property
    def kanonik(self):
        return list(self.kat["kanonik"])

    @property
    def alias(self):
        return dict(self.kat.get("alias") or {})

    @property
    def luar_taksonomi_llm(self):
        return list(self.kat.get("luar_taksonomi_llm") or [])

    @property
    def keluar_taksonomi(self):
        return list(self.kat.get("keluar_taksonomi") or [])

    def kanon_kategori(self, nilai):
        from penilai.kanon import kanon_kategori
        return kanon_kategori(nilai, self.alias, self.kanonik,
                              self.luar_taksonomi_llm, self.keluar_taksonomi)

    # --- kredensial ------------------------------------------------------
    def kredensial_siap(self):
        """-> (siap: bool, yang_kosong: list[str]).

        Nilainya TIDAK pernah dibaca atau dicatat — hanya ada/tidaknya. Nama
        env var ada di config, nilainya tidak boleh pernah masuk berkas mana pun.
        """
        butuh = self.ambil("llm", "env_kredensial", wajib=False, bawaan=[]) or []
        kosong = [k for k in butuh if not os.environ.get(k)]
        return (len(kosong) == 0, kosong)

    def ringkas(self) -> dict:
        """Salinan konfigurasi untuk manifest versi model. Kredensial tidak ada
        di sini karena tidak pernah masuk objek ini sejak awal."""
        return {"utama": self.d, "kategori": self.kat}


def muat(berkas_utama: str | None = None, berkas_kategori: str | None = None,
         akar: Path | None = None) -> Konfigurasi:
    akar = Path(akar) if akar else AKAR
    bu = Path(berkas_utama) if berkas_utama else akar / "config/config.yaml"
    bk = Path(berkas_kategori) if berkas_kategori else akar / "config/kategori.yaml"
    for b in (bu, bk):
        if not b.exists():
            raise FileNotFoundError(f"konfigurasi tidak ada: {b}")
    return Konfigurasi(yaml.safe_load(bu.read_text(encoding="utf-8")),
                       yaml.safe_load(bk.read_text(encoding="utf-8")), akar)
