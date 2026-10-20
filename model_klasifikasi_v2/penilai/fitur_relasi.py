#!/usr/bin/env python3
"""Fitur V9 — hubungan ALASAN <-> KETERANGAN, dan penanda keraguan di alasan.

    from penilai.fitur_relasi import fitur_relasi
    bagian.update(fitur_relasi(d))

Kenapa kelompok ini ada. Fitur yang sudah ada menjelaskan transaksinya (nominal,
tanggal, merchant) dan BENTUK alasannya (panjang, jumlah angka). Yang belum ada:
apakah alasan itu benar-benar BERBICARA TENTANG baris di depannya.

Itu justru sinyal yang paling masuk akal untuk menebak LLM benar atau salah.
Alasan yang mengutip merchant yang memang ada di keterangan lebih layak dipercaya
daripada alasan yang menyebut merchant yang tidak ada di situ — dan perbedaan itu
tidak terlihat dari panjang teks.

SEMUA fitur di sini hanya memakai `keterangan` dan `alasan_llm`, dua kolom yang
tersedia saat inference. Tidak satu pun menyentuh `kategori_benar`,
`target_benar`, atau statistik apa pun yang dihitung dari label.
"""
import re

import numpy as np
import pandas as pd

_KATA = re.compile(r"[A-Za-z]{3,}")
_ANGKA = re.compile(r"\d[\d.,]{2,}")

# Penanda epistemik. Dipisah per jenis karena artinya berbeda: ragu = model tidak
# tahu; sangkal = model menolak sesuatu; lawan = model memegang dua kemungkinan
# sekaligus, dan yang terakhir itulah tanda paling kuat bahwa jawabannya rapuh.
_RAGU = re.compile(r"\b(mungkin|kemungkinan|sepertinya|tidak jelas|kurang jelas|"
                   r"tidak pasti|diduga|asumsi|possibly|probably|unclear|unsure|"
                   r"cannot|tidak dapat|tidak bisa|ambigu)\b", re.I)
_SANGKAL = re.compile(r"\b(bukan|tidak|tanpa|no|not|none|neither|belum)\b", re.I)
_LAWAN = re.compile(r"\b(tapi|tetapi|namun|but|however|walau|walaupun|meski|"
                    r"meskipun|padahal|sedangkan|although)\b", re.I)
_ATURAN = re.compile(r"\brules?\s*\d+[a-z]?|\baturan\s*\d+", re.I)
_KATEGORI = re.compile(r"\b(own|circling|kiting|non[\s-]business|business|gaji|"
                       r"thr|bonus|lembur|cuti|insentif|potongan|pinjaman)\b", re.I)
_CARI_PASANGAN = re.compile(r"\b(pasangan|pair|lawan|counterpart|dibalas|"
                            r"partner|matching|cocok)\b", re.I)


def _set_kata(s):
    return set(_KATA.findall(str(s or "").upper()))


def fitur_relasi(d: pd.DataFrame) -> dict:
    ket = d["keterangan"].fillna("").astype(str)
    als = d["alasan_llm"].fillna("").astype(str)
    out = {}

    tk = [_set_kata(x) for x in ket]
    ta = [_set_kata(x) for x in als]

    # --- D. hubungan input <-> alasan ------------------------------------
    # `jaccard` simetris; `cakupan_*` tidak, dan asimetrinya penting:
    #   cakupan_ket = berapa banyak ISI BARIS yang disentuh alasan (alasan yang
    #                 mengabaikan sebagian besar keterangan patut dicurigai);
    #   cakupan_als = berapa banyak ALASAN yang berpijak pada baris (alasan yang
    #                 sebagian besar katanya tidak ada di baris = mengarang).
    out["rel_jaccard"] = [round(len(a & b) / len(a | b), 4) if (a | b) else 0.0
                          for a, b in zip(tk, ta)]
    out["rel_cakupan_ket"] = [round(len(a & b) / max(len(a), 1), 4)
                              for a, b in zip(tk, ta)]
    out["rel_cakupan_als"] = [round(len(a & b) / max(len(b), 1), 4)
                              for a, b in zip(tk, ta)]
    out["rel_kata_bersama"] = [len(a & b) for a, b in zip(tk, ta)]
    out["rel_kata_als_asing"] = [len(b - a) for a, b in zip(tk, ta)]

    # Angka yang dikutip alasan tapi tidak ada di keterangan: paling sering
    # berarti alasannya merujuk baris LAIN, atau mengarang nominal.
    na = [set(_ANGKA.findall(x)) for x in ket]
    nb = [set(_ANGKA.findall(x)) for x in als]
    out["rel_angka_dikutip"] = [len(a & b) for a, b in zip(na, nb)]
    out["rel_angka_als_asing"] = [len(b - a) for a, b in zip(na, nb)]
    out["rel_ada_angka_asing"] = [int(len(b - a) > 0) for a, b in zip(na, nb)]

    # Token PANJANG yang sama biasanya nama merchant/orang — bukti terkuat bahwa
    # alasannya memang membaca baris ini.
    out["rel_token_panjang_bersama"] = [
        len({t for t in (a & b) if len(t) >= 6}) for a, b in zip(tk, ta)]

    # --- C. penanda epistemik di alasan ----------------------------------
    out["rel_n_ragu"] = als.map(lambda x: len(_RAGU.findall(x)))
    out["rel_n_sangkal"] = als.map(lambda x: len(_SANGKAL.findall(x)))
    out["rel_n_lawan"] = als.map(lambda x: len(_LAWAN.findall(x)))
    out["rel_n_sebut_aturan"] = als.map(lambda x: len(_ATURAN.findall(x)))
    out["rel_n_kategori_disebut"] = als.map(
        lambda x: len({m.lower().replace("-", " ") for m in _KATEGORI.findall(x)}))
    # Menyebut LEBIH DARI SATU kategori = model memegang dua kemungkinan.
    out["rel_kategori_ganda"] = (pd.Series(out["rel_n_kategori_disebut"]) > 1).astype(int).values
    out["rel_sebut_cari_pasangan"] = als.map(lambda x: int(bool(_CARI_PASANGAN.search(x))))

    # --- G. struktural ----------------------------------------------------
    out["rel_als_kosong"] = (als.str.strip() == "").astype(int)
    out["rel_ket_kosong"] = (ket.str.strip() == "").astype(int)
    out["rel_als_per_ket_kata"] = (als.str.split().map(len)
                                   / ket.str.split().map(len).replace(0, np.nan)
                                   ).fillna(0).round(4)
    return out
