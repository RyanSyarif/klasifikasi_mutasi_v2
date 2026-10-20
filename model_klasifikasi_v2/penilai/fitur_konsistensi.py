#!/usr/bin/env python3
"""Fitur V12 — konsistensi jawaban LLM dan kesepakatannya dengan leksikon.

    from penilai.fitur_konsistensi import fitur_konsistensi
    bagian.update(fitur_konsistensi(d))

Gagasannya satu: cara paling murah menebak apakah sebuah jawaban benar adalah
melihat apakah model MENJAWAB SAMA untuk pertanyaan yang sama.

Dalam satu dokumen sering ada beberapa baris berketerangan IDENTIK. Kalau LLM
memberi kategori berbeda untuk baris-baris itu, setidaknya satu jawabannya
salah — dan itu diketahui TANPA melihat ground truth sama sekali. Hitungannya
murni atas keluaran LLM dan teks masukan, keduanya tersedia saat inferensi.

Kelompok kedua: apakah jawaban LLM sejalan dengan leksikon kata kunci yang
deterministik. Bukan untuk MENIMPA jawaban LLM — melainkan mencatat kapan
keduanya berselisih, karena perselisihan itu sendiri adalah sinyal.

TIDAK ADA satu pun fitur di sini yang menyentuh `kategori_benar`,
`target_benar`, atau statistik yang dihitung dari label.
"""
import re

import numpy as np
import pandas as pd

# Leksikon deterministik: pola -> kategori yang disiratkannya. Dibatasi ke pola
# yang dukungan ground truth-nya sudah diperiksa dan tinggi.
LEKSIKON = [
    (re.compile(r"\bSPAN\b|PAYR|\bGAJI\b|\bSALARY\b|\bUPAH\b", re.I), "Gaji"),
    (re.compile(r"\bBONUS\b|\bTANTIEM\b", re.I), "Bonus"),
    (re.compile(r"\bTHR\b|HARI RAYA", re.I), "THR"),
    (re.compile(r"\bBUNGA\b|\bPAJAK\b|\bBIAYA\b|\bADM\b|ADMINISTRASI", re.I), "Non Business"),
    (re.compile(r"WARUNG|RESTO|\bRM\b|KOPI|COFFEE|BAKERY|\bNASI\b|SPBU|SHELL|"
                r"ALFAMART|INDOMARET|MINIMART|SALON|APOTEK", re.I), "Non Business"),
    (re.compile(r"KARTU KREDIT|\bPLN\b|PDAM|BPJS|TELKOM|PULSA|TOP ?UP", re.I), "Non Business"),
]


def _rapat(s):
    return " ".join(str(s or "").split()).upper()


def fitur_konsistensi(d: pd.DataFrame) -> dict:
    out = {}
    ket = d["keterangan"].fillna("").astype(str)
    kat = d["kategori_llm_mentah"].fillna("").astype(str) \
        if "kategori_llm_mentah" in d.columns else pd.Series([""] * len(d), index=d.index)
    dok = d["dokumen"].astype(str) if "dokumen" in d.columns else pd.Series(["_"] * len(d), index=d.index)

    # --- A. konsistensi LLM pada keterangan IDENTIK, dalam dokumen yang sama --
    kunci = pd.Series([f"{a}||{b}" for a, b in zip(dok, ket.map(_rapat))], index=d.index)
    g = kat.groupby(kunci)
    n_sama = g.transform("size")
    n_kategori = g.transform("nunique")
    modus = g.transform(lambda s: s.value_counts().idxmax() if len(s) else "")
    out["kons_n_keterangan_sama"] = n_sama.astype(int)
    out["kons_n_kategori_berbeda"] = n_kategori.astype(int)
    # >1 kategori untuk teks identik = setidaknya satu jawaban PASTI salah
    out["kons_llm_bertentangan_diri"] = (n_kategori > 1).astype(int)
    # baris ini ikut modus atau jadi yang ganjil? yang ganjil lebih mungkin salah
    out["kons_ikut_modus"] = (kat == modus).astype(int)
    out["kons_porsi_modus"] = g.transform(
        lambda s: (s.value_counts().iloc[0] / len(s)) if len(s) else 1.0).astype(float).round(4)

    # --- B. kesepakatan dengan leksikon deterministik ------------------------
    saran, n_kena = [], []
    for t in ket:
        hit = [lab for rx, lab in LEKSIKON if rx.search(t)]
        saran.append(hit[0] if hit else "")
        n_kena.append(len(set(hit)))
    saran = pd.Series(saran, index=d.index)
    out["leks_ada_saran"] = (saran != "").astype(int)
    out["leks_n_pola_kena"] = pd.Series(n_kena, index=d.index).astype(int)
    # ambigu: leksikon menyarankan lebih dari satu kategori berbeda
    out["leks_ambigu"] = (pd.Series(n_kena, index=d.index) > 1).astype(int)
    out["leks_sepakat_llm"] = ((saran != "") & (saran == kat)).astype(int)
    out["leks_bantah_llm"] = ((saran != "") & (saran != kat)).astype(int)

    # --- C. keyakinan yang dilaporkan model ---------------------------------
    if "confidence_llm" in d.columns:
        c = pd.to_numeric(d["confidence_llm"], errors="coerce")
        out["llm_confidence"] = c
        out["llm_confidence_rendah"] = (c < 0.7).astype("float")
        # Keyakinan hampir selalu sama nilainya di model ini; simpangannya dalam
        # dokumen memberi tahu apakah angka itu bermakna atau cuma konstanta.
        out["llm_confidence_beda_dari_dokumen"] = (
            c - c.groupby(dok).transform("mean")).round(4)

    # --- D. pointer pasangan yang dikembalikan LLM --------------------------
    if "pasangan_llm" in d.columns:
        p = pd.to_numeric(d["pasangan_llm"], errors="coerce")
        out["pasllm_ada"] = p.notna().astype(int)
        out["pasllm_nilai_valid"] = ((p.notna()) & (p >= 0)).astype(int)
    return out
