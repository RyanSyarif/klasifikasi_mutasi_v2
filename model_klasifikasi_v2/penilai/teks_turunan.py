#!/usr/bin/env python3
"""Fitur teks yang HARUS dipelajari dari data: TF-IDF + target encoding OOF.

Dipisah dari `fitur.py` karena bedanya mendasar: fitur di `fitur.py` bisa
dihitung satu baris pada satu waktu tanpa melihat baris lain, sedangkan yang di
sini MEMPELAJARI sesuatu dari kumpulan data — kosakata, IDF, rata-rata target.
Segala sesuatu yang belajar dari data harus dipasang (fit) di LATIH saja, dan
objek yang sudah terpasang itulah yang dipakai ulang di uji dan di inferensi.

    T = TeksTurunan(C)
    Xl = T.pasang_ubah(d_latih, y_latih, grup_latih)   # fit + transform
    Xu = T.ubah(d_uji)                                 # transform saja

TARGET ENCODING OUT-OF-FOLD, DAN KENAPA REPOT
---------------------------------------------
`te_kategori_llm` adalah rata-rata target per kategori yang diprediksi LLM.
Sinyalnya kuat — LLM jauh lebih sering benar untuk sebagian kategori daripada
yang lain. Tapi kalau rata-rata itu dihitung dari SELURUH data latih termasuk
baris yang sedang dihitung, baris itu ikut menentukan nilainya sendiri, dan
model akan terlalu percaya pada fitur ini.

Jadi saat LATIH, nilainya diambil out-of-fold: untuk baris di lipatan k,
rata-ratanya dihitung dari lipatan SELAIN k. Lipatannya sadar-grup (GroupKFold
per dokumen), supaya baris dari dokumen yang sama tidak saling mengisi nilai.
Saat UJI dan INFERENSI, dipakai rata-rata dari SELURUH latih — di situ tidak ada
masalah, karena baris uji tidak pernah ikut menghitung.

Prior smoothing dipakai supaya kategori yang cuma muncul beberapa kali tidak
mendapat nilai 0 atau 1 yang ekstrem.
"""
import re

import numpy as np
import pandas as pd


def _bersih(s):
    """Normalisasi teks untuk TF-IDF: huruf kecil, angka panjang jadi penanda.

    Nomor rekening dan nomor referensi diganti penanda, bukan dibuang: KEHADIRAN
    nomor itu informatif, nilainya tidak — dan kalau dibiarkan apa adanya,
    setiap nomor jadi satu term sendiri dan kosakatanya meledak tanpa guna.
    """
    t = str(s or "").lower()
    t = re.sub(r"\d{8,}", " NOREK ", t)
    t = re.sub(r"\d{4,7}", " NOREF ", t)
    t = re.sub(r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?", " NOMINAL ", t)
    t = re.sub(r"\d+", " ANGKA ", t)
    t = re.sub(r"[^a-z ]+", " ", t)
    return " ".join(t.split())


class TeksTurunan:
    """Menyimpan vectorizer + peta target encoding supaya bisa dipakai ulang."""

    def __init__(self, C):
        self.C = C
        self.aktif_tfidf = bool(C.ambil("fitur", "aktifkan_tfidf"))
        self.aktif_te = bool(C.ambil("fitur", "aktifkan_target_encoding_oof"))
        self.vec_ket = None
        self.vec_als = None
        self.te_peta = {}
        self.te_prior = None
        self.nama = []

    # --- TF-IDF ----------------------------------------------------------
    def _buat_vec(self, maks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(
            max_features=int(maks),
            min_df=int(self.C.ambil("fitur", "tfidf_min_df")),
            ngram_range=tuple(self.C.ambil("fitur", "tfidf_ngram")),
            sublinear_tf=True, strip_accents="unicode")

    @staticmethod
    def _aman(term: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")

    def _tfidf_pasang(self, d):
        out = {}
        if not self.aktif_tfidf:
            return out
        self.vec_ket = self._buat_vec(self.C.ambil("fitur", "tfidf_ket_maks"))
        m = self.vec_ket.fit_transform(d["keterangan"].map(_bersih))
        for j, t in enumerate(self.vec_ket.get_feature_names_out()):
            out[f"tfidf_ket__{self._aman(t)}"] = np.asarray(m[:, j].todense()).ravel()
        self.vec_als = self._buat_vec(self.C.ambil("fitur", "tfidf_alasan_maks"))
        m = self.vec_als.fit_transform(d["alasan_llm"].map(_bersih))
        for j, t in enumerate(self.vec_als.get_feature_names_out()):
            out[f"tfidf_als__{self._aman(t)}"] = np.asarray(m[:, j].todense()).ravel()
        return out

    def _tfidf_ubah(self, d):
        out = {}
        if not self.aktif_tfidf:
            return out
        for vec, kol, pre in ((self.vec_ket, "keterangan", "tfidf_ket__"),
                              (self.vec_als, "alasan_llm", "tfidf_als__")):
            if vec is None:
                continue
            m = vec.transform(d[kol].map(_bersih))
            for j, t in enumerate(vec.get_feature_names_out()):
                out[f"{pre}{self._aman(t)}"] = np.asarray(m[:, j].todense()).ravel()
        return out

    # --- target encoding out-of-fold -------------------------------------
    def _te_pasang(self, d, y, grup):
        """Nilai OOF untuk latih, dan peta penuh untuk dipakai di uji."""
        out = {}
        if not self.aktif_te:
            return out
        from sklearn.model_selection import GroupKFold
        y = pd.Series(np.asarray(y, dtype=float), index=d.index)
        kunci = ["kategori_llm", "bank"]
        n_lipat = int(self.C.ambil("fitur", "oof_lipatan"))
        self.te_prior = float(y.mean())
        n_grup = pd.Series(grup).nunique()
        n_lipat = max(2, min(n_lipat, n_grup))
        gkf = GroupKFold(n_splits=n_lipat)
        for k in kunci:
            kol = d[k].fillna("(kosong)").astype(str)
            nilai = pd.Series(np.nan, index=d.index)
            for i_in, i_out in gkf.split(d, y, groups=grup):
                idx_in, idx_out = d.index[i_in], d.index[i_out]
                peta = self._peta_halus(kol.loc[idx_in], y.loc[idx_in])
                nilai.loc[idx_out] = kol.loc[idx_out].map(peta).fillna(self.te_prior)
            out[f"te_{k}"] = nilai.fillna(self.te_prior).values
            # peta penuh (dari SELURUH latih) untuk uji/inferensi
            self.te_peta[k] = self._peta_halus(kol, y)
        return out

    def _peta_halus(self, kol: pd.Series, y: pd.Series, m=10.0):
        """Rata-rata target per kategori dengan prior smoothing.

        m=10 berarti kategori dengan 10 baris ditarik setengah jalan ke prior.
        Tanpa ini, kategori yang muncul 2 kali bisa dapat nilai 0.0 atau 1.0 dan
        model memperlakukannya sebagai kepastian.
        """
        prior = float(y.mean())
        g = y.groupby(kol)
        n, rata = g.count(), g.mean()
        return ((rata * n + prior * m) / (n + m)).to_dict()

    def _te_ubah(self, d):
        out = {}
        if not self.aktif_te or self.te_prior is None:
            return out
        for k, peta in self.te_peta.items():
            kol = d[k].fillna("(kosong)").astype(str)
            out[f"te_{k}"] = kol.map(peta).fillna(self.te_prior).values
        return out

    # --- antarmuka -------------------------------------------------------
    def pasang_ubah(self, d, y, grup):
        b = {}
        b.update(self._tfidf_pasang(d))
        b.update(self._te_pasang(d, y, grup))
        X = pd.DataFrame(b, index=d.index)
        self.nama = list(X.columns)
        return X

    def ubah(self, d):
        b = {}
        b.update(self._tfidf_ubah(d))
        b.update(self._te_ubah(d))
        X = pd.DataFrame(b, index=d.index)
        # urutan kolom HARUS sama dengan saat latih; yang hilang diisi 0.
        for c in self.nama:
            if c not in X:
                X[c] = 0.0
        return X[self.nama]

    def sumber(self):
        return {**{n: "teks_turunan" for n in self.nama if n.startswith("tfidf_")},
                **{n: "target_encoding" for n in self.nama if n.startswith("te_")}}
