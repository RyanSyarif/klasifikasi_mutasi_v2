#!/usr/bin/env python3
"""Tahap 5 — rekayasa fitur KANDIDAT.

    from penilai.fitur import bangun_kandidat
    X, daftar = bangun_kandidat(d, C)

FITUR KANDIDAT BUKAN FITUR MODEL. Modul ini sengaja membuat BANYAK fitur —
ratusan — tanpa menilai mana yang berguna. Penyaringannya ada di
`pemilihan_fitur.py`, dan hanya boleh memakai data latih. Pemisahan ini penting:
kalau pembuatan dan pemilihan fitur bercampur, gampang sekali "membuat" fitur
karena dia kelihatan bagus di test.

ATURAN KETERSEDIAAN — DIPERIKSA UNTUK SETIAP FITUR
--------------------------------------------------
Sebuah fitur boleh ada di sini HANYA kalau nilainya bisa dihitung pada baris
BARU yang ground truth-nya belum ada. Praktisnya, hanya lima sumber ini:

  mentah     kolom transaksi dari statement: tanggal, keterangan, debet,
             kredit, saldo. Ada sebelum LLM dipanggil.
  kop        isi kop rekening + KK/Akta: nama_nasabah, no_rekening, bank,
             nama_keluarga, nama_pengurus. Ini isi `header` di prompt, jadi ada
             sebelum LLM dipanggil.
  llm_keluaran  kategori yang diprediksi LLM, dan `pasangan`/`kelas`.
  llm_alasan    penjelasan LLM. BOLEH DIPAKAI — dia dihasilkan bersamaan dengan
             prediksinya, jauh sebelum ada yang tahu prediksinya benar atau tidak.
  llm_metadata  finish_reason, runtime, input_chars, n_kirim, n_balik.
  aturan     `_alasan_mesin` dari mesin aturan paket referensi. Deterministik
             dari kolom mentah, tidak melihat ground truth.

YANG TIDAK PERNAH MASUK KE SINI — dan alasannya ada di daftar_fitur.py:
  kategori_benar / kategori_benar_mentah   ini targetnya sendiri
  target_benar                             ini targetnya sendiri
  _bocor_kategori_gabriel                  label MANUSIA. Tidak ada di data baru,
                                           dan berkorelasi dengan target karena
                                           dua-duanya penilaian manusia.
  _periksa_kategori_bedrock_csv            kolom pemeriksa silang, bukan masukan
  detail_kategori                          terikat ke kategori GT
  di_luar_taksonomi, catatan_peninjau      dibuat SETELAH peninjauan manusia
  alasan_dibuang, masuk_model              keputusan pipeline setelah target ada

FITUR AGREGAT DAN KEBOCORAN. Beberapa fitur di bawah butuh konteks seluruh
dokumen (mis. jumlah baris, apakah ada pasangan nominal). Itu SAH: pada data
baru, seluruh dokumen juga tersedia sekaligus — statement diproses per dokumen,
bukan per baris. Yang tidak sah adalah agregat yang memakai TARGET (mis.
"akurasi LLM di dokumen ini"), dan itu tidak ada satu pun di sini.
"""
import re

import numpy as np

from penilai.fitur_konsistensi import fitur_konsistensi
from penilai.fitur_relasi import fitur_relasi
import pandas as pd

from penilai.kanon import cocok_nama, hari_urut, nama_token, rapat, sen

# --- leksikon. Ditulis terbuka supaya bisa dibantah dan ditambah ------------
# Diangkat dari paket_ryan/usul_kategori.py supaya kedua paket memakai kamus
# yang sama; kalau di sana ditambah, di sini juga harus.
LEKS = {
    "gaji": r"\bSPAN\b|\bGAJI\b|\bSALARY\b|\bPAYROLL\b|\bPENGGAJIAN\b|\bUPAH\b",
    "thr": r"\bTHR\b|TUNJANGAN\s+HARI\s+RAYA|HARI\s+RAYA",
    "bonus": r"\bBONUS\b|\bTANTIEM\b|\bREWARD\b",
    "lembur": r"\bLEMBUR\b|\bOVERTIME\b",
    "cuti": r"\bCUTI\b|\bECUTI\b",
    "insentif": r"\bINSENTIF\b|\bINCENTIVE\b|\bKOMISI\b|\bCOMMISSION\b|\bTUNJANGAN\b",
    "biaya": r"\bBIAYA\b|\bADM\b|\bADMIN\b|ADMINISTRASI|\bFEE\b|\bKLIRING\b|\bMATERAI\b|\bPENALTI\b|\bDENDA\b",
    "pajak": r"\bPAJAK\b|\bPPH\b|\bPPN\b|\bTAX\b",
    "bunga": r"\bBUNGA\b|\bJASA\s*GIRO\b|\bINTEREST\b",
    "belanja": (r"QRIS|BPJS|PRCH|TSELDT|PUL-?SIM|INTERCHANGE"
                r"|\bOVO\b|\bGOPAY\b|\bSHOPEE\b|\bDANA\b|\bTOKOPEDIA\b|\bLINKAJA\b"
                r"|\bTOP\s*UP\b|\bPULSA\b|\bPLN\b|\bTELKOM\b|\bINDOMARET\b|\bALFAMART\b"
                r"|\bE-?COMMERCE\b|\bDEBIT\s*CARD\b|\bPURCHASE\b|\bPEMBELIAN\b"),
    "tunai": r"\bTARIK\s*TUNAI\b|\bTARIKAN\b|\bSETOR\s*TUNAI\b|\bSETORAN\s*TUNAI\b|\bATM\b|\bCASH\b",
    "transfer": r"\bTRANSFER\b|\bTRF\b|\bTRSF\b|\bBI\s*FAST\b|\bBIFAST\b|\bRTGS\b|\bLLG\b|\bSKN\b|\bKLIRING\b",
    "badan_usaha": r"\bPT\b|\bCV\b|\bUD\b|\bTBK\b|\bKOPERASI\b|\bTOKO\b|\bPERSERO\b",
    "nama_bank": r"\bBANK\b|\bBCA\b|\bBNI\b|\bBRI\b|\bMANDIRI\b|\bBSI\b|\bCIMB\b|\bPERMATA\b|\bDANAMON\b|\bBTN\b",
    "kekerabatan": r"\bISTRI\b|\bSUAMI\b|\bANAK\b|\bORANG\s*TUA\b|\bKELUARGA\b|\bIBU\b|\bBAPAK\b",
    "pengurus": r"\bPENGURUS\b|\bDIREKTUR\b|\bKOMISARIS\b|\bDEVIDEN\b|\bDIVIDEN\b|\bSETORAN\s*MODAL\b",
    "sendiri": r"REKENING\s*SENDIRI|OWN\s*ACCOUNT|PEMINDAHBUKUAN",
    "koreksi": r"\bCASHBACK\b|\bREFUND\b|\bREIMBURSE\b|\bKOREKSI\b|\bPROMO\b|\bRESTITUSI\b|\bREVERSAL\b|\bPEMBATALAN\b",
    "saldo_subtotal": r"SALDO\s*AWAL|SALDO\s*AKHIR|MUTASI\s*CR|MUTASI\s*DB|\bSUBTOTAL\b",
}
LEKS_RX = {k: re.compile(v, re.I) for k, v in LEKS.items()}

# Kata dalam ALASAN LLM. Ini bukan tentang isi transaksinya, tapi tentang
# BAGAIMANA LLM bicara — dan itu yang jadi sinyalnya.
RAGU = r"\bmungkin\b|\bkemungkinan\b|\bsepertinya\b|\bkelihatannya\b|\bdiduga\b|\bbisa\s+jadi\b|\btidak\s+jelas\b|\btidak\s+pasti\b|\bkurang\s+jelas\b|\bperlu\s+dicek\b|\bbelum\s+tentu\b|\bmaybe\b|\bpossibly\b|\bunclear\b|\bunknown\b|\bassum|\blikely\b|\bappears?\b|\bseems?\b"
YAKIN = r"\bjelas\b|\bpasti\b|\bmemang\b|\bterbukti\b|\bsesuai\b|\bcocok\b|\btepat\b|\bexplicit\b|\bclearly\b|\bconfirmed\b|\bdefinitely\b|\bmatches\b"
SANGKAL = r"\btidak\b|\bbukan\b|\btanpa\b|\bbelum\b|\bkecuali\b|\bnon\b|\bno\b|\bnot\b|\bwithout\b|\bneither\b"
LAWAN = r"\btapi\b|\btetapi\b|\bnamun\b|\bwalau\b|\bmeski\b|\bmeskipun\b|\bpadahal\b|\bsedangkan\b|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bdespite\b"
BUKTI = r"\bkarena\b|\bsebab\b|\bkata\s*kunci\b|\blabel\b|\bketerangan\b|\bnominal\b|\bpengirim\b|\bpenerima\b|\bnama\b|\brekening\b|\btanggal\b|\bpola\b|\bper\s*rule\b|\bmatch\b|\bcocok\b"
UMUM = r"^(?:transaksi|baris|kategori)\s+(?:selain|lain|lainnya)|tidak\s+ada\s+kata\s+kunci|bukan\s+uang\s+masuk|sesuai\s+aturan$|default"
RX = {k: re.compile(v, re.I) for k, v in
      (("ragu", RAGU), ("yakin", YAKIN), ("sangkal", SANGKAL), ("lawan", LAWAN),
       ("bukti", BUKTI), ("umum", UMUM))}

KANONIK_KATEGORI = ["Gaji", "THR", "Bonus", "Lembur", "Cuti", "Lainnya",
                    "Own", "Circling", "Kiting", "Non Business", "Business"]

# Sumber tiap prefiks fitur -> dipakai daftar_fitur.py untuk audit kebocoran.
SUMBER_PREFIKS = {
    "ket_": "mentah", "num_": "mentah", "tgl_": "mentah", "str_": "mentah",
    "kop_": "kop", "nama_": "kop", "dok_": "mentah",
    "llm_kat_": "llm_keluaran", "llm_pasangan": "llm_keluaran",
    "llm_kelas": "llm_keluaran",
    "als_": "llm_alasan", "meta_": "llm_metadata", "aturan_": "aturan",
    "tfidf_ket__": "mentah", "tfidf_als__": "llm_alasan",
    "te_": "llm_keluaran",
    "kanal_": "aturan+mentah", "merchant_": "mentah",
    "dok_porsi_": "mentah+llm_keluaran", "dok_kategori_": "llm_keluaran",
    # Kelompok V9: hubungan antara `keterangan` (masukan) dan `alasan_llm`
    # (keluaran). Sumbernya gabungan keduanya, dan keduanya sudah ada sebelum
    # ada yang tahu prediksinya benar atau salah.
    "rel_": "mentah+llm_alasan",
    # Kelompok V12: konsistensi jawaban LLM pada teks identik, kesepakatan
    # dengan leksikon, keyakinan yang dilaporkan model.
    "kons_": "mentah+llm_keluaran", "leks_": "mentah+llm_keluaran",
    "llm_confidence": "llm_keluaran", "pasllm_": "llm_keluaran",
}


# ==========================================================================
# A. Fitur teks umum — dipakai untuk keterangan, alasan LLM, alasan aturan
# ==========================================================================
def _fitur_teks(s: pd.Series, pre: str):
    t = s.fillna("").astype(str)
    out = {}
    out[pre + "panjang"] = t.str.len()
    out[pre + "jumlah_kata"] = t.str.split().map(len)
    out[pre + "jumlah_karakter_bukan_spasi"] = t.str.replace(r"\s", "", regex=True).str.len()
    out[pre + "jumlah_digit"] = t.str.count(r"\d")
    out[pre + "jumlah_huruf"] = t.str.count(r"[A-Za-z]")
    out[pre + "jumlah_huruf_besar"] = t.str.count(r"[A-Z]")
    out[pre + "jumlah_huruf_kecil"] = t.str.count(r"[a-z]")
    out[pre + "jumlah_tanda_baca"] = t.str.count(r"[.,;:!?/\\()\[\]{}'\"-]")
    out[pre + "jumlah_karakter_khusus"] = t.str.count(r"[^A-Za-z0-9\s.,;:!?/\\()\[\]{}'\"-]")
    out[pre + "jumlah_spasi"] = t.str.count(r" ")
    out[pre + "jumlah_baris_baru"] = t.str.count(r"\n")
    out[pre + "jumlah_kalimat"] = t.str.count(r"[.!?]+").clip(lower=0) + (t.str.len() > 0).astype(int)
    out[pre + "jumlah_angka"] = t.str.count(r"\b\d[\d.,]*\b")
    # token ~ 4 karakter: cukup untuk memban­dingkan biaya antar baris, dan
    # tidak berpura-pura jadi tokenizer model sungguhan.
    out[pre + "token_perkiraan"] = (t.str.len() / 4.0).round(2)
    out[pre + "kosong"] = (t.str.strip() == "").astype(int)
    pjg = out[pre + "panjang"].replace(0, np.nan)
    out[pre + "rasio_digit"] = (out[pre + "jumlah_digit"] / pjg).fillna(0)
    out[pre + "rasio_huruf_besar"] = (out[pre + "jumlah_huruf_besar"] / pjg).fillna(0)
    out[pre + "rasio_tanda_baca"] = (out[pre + "jumlah_tanda_baca"] / pjg).fillna(0)
    out[pre + "panjang_kata_rata"] = (out[pre + "jumlah_karakter_bukan_spasi"]
                                      / out[pre + "jumlah_kata"].replace(0, np.nan)).fillna(0)
    kata = t.str.lower().str.findall(r"[a-z]{2,}")
    out[pre + "kata_unik"] = kata.map(lambda x: len(set(x)))
    out[pre + "keragaman_leksikal"] = [
        round(len(set(x)) / len(x), 4) if x else 0.0 for x in kata]
    out[pre + "rasio_pengulangan"] = [
        round(1 - len(set(x)) / len(x), 4) if x else 0.0 for x in kata]
    out[pre + "kata_terpanjang"] = kata.map(lambda x: max((len(w) for w in x), default=0))
    return out


# ==========================================================================
# B. Fitur struktur numerik / tanggal / identifier
# ==========================================================================
def _fitur_angka(d: pd.DataFrame):
    out = {}
    deb = d["debet_mentah"].map(sen).astype("int64")
    kre = d["kredit_mentah"].map(sen).astype("int64")
    sal = d["saldo_mentah"].map(sen).astype("int64")
    out["num_debet"] = deb / 100.0
    out["num_kredit"] = kre / 100.0
    out["num_saldo"] = sal / 100.0
    out["num_nominal"] = (deb + kre) / 100.0
    out["num_ada_debet"] = (deb > 0).astype(int)
    out["num_ada_kredit"] = (kre > 0).astype(int)
    out["num_arah_kredit"] = (kre > deb).astype(int)
    out["num_dua_sisi_terisi"] = ((deb > 0) & (kre > 0)).astype(int)
    out["num_nol_semua"] = ((deb == 0) & (kre == 0)).astype(int)
    out["num_log_nominal"] = np.log1p((deb + kre) / 100.0)
    out["num_log_saldo"] = np.log1p(np.abs(sal) / 100.0) * np.sign(sal)
    out["num_saldo_negatif"] = (sal < 0).astype(int)
    # Nominal bulat sering menandai transfer antar-rekening, bukan belanja.
    n = deb + kre
    out["num_bulat_ribu"] = ((n % 100000 == 0) & (n > 0)).astype(int)
    out["num_bulat_juta"] = ((n % 100000000 == 0) & (n > 0)).astype(int)
    out["num_bulat_puluh_juta"] = ((n % 1000000000 == 0) & (n > 0)).astype(int)
    out["num_digit_nominal"] = n.map(lambda x: len(str(abs(x) // 100)) if x else 0)
    out["num_nol_ekor"] = n.map(lambda x: len(str(abs(x) // 100)) -
                                len(str(abs(x) // 100).rstrip("0")) if x else 0)
    # Konsistensi aritmetika baris. Ini ukuran mutu BACA per baris — kalau
    # angkanya salah baca, LLM juga menilai atas angka yang salah.
    out["num_saldo_konsisten"] = _konsisten(d, deb, kre, sal)
    return out


def _konsisten(d, deb, kre, sal):
    hasil = pd.Series(0, index=d.index, dtype=int)
    for _, g in d.groupby("dokumen", sort=False):
        prev = None
        for i in g.sort_values("baris_dokumen").index:
            s = sal.loc[i]
            if prev is not None and s:
                harap = prev - deb.loc[i] + kre.loc[i]
                hasil.loc[i] = int(abs(harap - s) <= 100)
            prev = s or prev
    return hasil


def _fitur_tanggal(d: pd.DataFrame):
    out = {}
    hu = d["tanggal_mentah"].map(hari_urut)
    out["tgl_ada_tahun"] = hu.notna().astype(int)
    tgl = d["tanggal_mentah"].fillna("").astype(str)
    out["tgl_ada_jam"] = tgl.str.contains(r"\d{1,2}[:.]\d{2}", regex=True).astype(int)
    out["tgl_panjang"] = tgl.str.len()
    # Konversi lewat datetime.date.fromordinal, BUKAN pd.to_datetime dengan
    # origin: origin tahun 1 di luar rentang datetime64[ns] pandas dan melempar
    # OutOfBoundsDatetime. Yang dibutuhkan cuma hari/bulan/hari-pekan, dan
    # fromordinal memberi itu tanpa menyentuh batas rentang pandas.
    import datetime as _dt
    hh = [(_dt.date.fromordinal(int(x)) if x is not None and not pd.isna(x) else None)
          for x in hu]
    out["tgl_hari_bulan"] = [x.day if x else -1 for x in hh]
    out["tgl_bulan"] = [x.month if x else -1 for x in hh]
    out["tgl_hari_pekan"] = [x.weekday() if x else -1 for x in hh]
    out["tgl_akhir_pekan"] = [int(x.weekday() >= 5) if x else 0 for x in hh]
    out["tgl_awal_bulan"] = [int(x.day <= 5) if x else 0 for x in hh]
    out["tgl_akhir_bulan"] = [int(x.day >= 25) if x else 0 for x in hh]
    # posisi baris di dalam dokumen: baris pertama/terakhir sering saldo/subtotal
    pos = d.groupby("dokumen")["baris_dokumen"].rank(method="first")
    n = d.groupby("dokumen")["baris_dokumen"].transform("count")
    out["dok_posisi_relatif"] = (pos / n).round(4)
    out["dok_baris_pertama"] = (pos == 1).astype(int)
    out["dok_baris_terakhir"] = (pos == n).astype(int)
    out["dok_jumlah_baris"] = n
    out["dok_jumlah_periode"] = d.groupby("dokumen")["periode_ke"].transform("nunique")
    return out


def _fitur_struktur(d: pd.DataFrame):
    """Penanda format di keterangan: nomor rekening, referensi, kode bank, dll."""
    out = {}
    t = d["keterangan"].fillna("").astype(str)
    out["str_jumlah_angka_panjang"] = t.str.count(r"\d{8,}")
    out["str_ada_nomor_rekening"] = t.str.contains(r"\d{8,}", regex=True).astype(int)
    out["str_jumlah_ref"] = t.str.count(r"[A-Z]{2,}\d{3,}|\d{3,}[A-Z]{2,}")
    out["str_jumlah_tanggal_dalam_ket"] = t.str.count(r"\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?")
    out["str_jumlah_nominal_dalam_ket"] = t.str.count(r"\d{1,3}(?:[.,]\d{3})+")
    out["str_jumlah_token_huruf_besar"] = t.str.count(r"\b[A-Z]{3,}\b")
    out["str_jumlah_token"] = t.str.split().map(len)
    out["str_semua_huruf_besar"] = (t.str.upper() == t).astype(int) * (t.str.len() > 0).astype(int)
    out["str_ada_titik_dua"] = t.str.contains(":").astype(int)
    out["str_ada_garis_miring"] = t.str.contains("/").astype(int)
    out["str_entitas_nama_perkiraan"] = t.map(lambda x: len(nama_token(x)))
    for nama, rx in LEKS_RX.items():
        out[f"ket_kunci_{nama}"] = t.map(lambda x, r=rx: int(bool(r.search(x))))
    out["ket_jumlah_kunci_kena"] = sum(out[f"ket_kunci_{k}"] for k in LEKS_RX)
    return out


# ==========================================================================
# C. Fitur kop rekening / nama — masukan `header` prompt
# ==========================================================================
def _fitur_kop(d: pd.DataFrame):
    out = {}
    ket = d["keterangan"].fillna("").astype(str)
    out["nama_pemilik_cocok"] = [
        int(cocok_nama(k, n)) for k, n in zip(ket, d["nama_nasabah"].fillna(""))]
    out["nama_rekening_cocok"] = [
        int(bool(r) and str(r).strip() in k) for k, r in zip(ket, d["nomor_rekening"].fillna(""))]
    kel = d["nama_keluarga"].fillna("").astype(str)
    peng = d["nama_pengurus"].fillna("").astype(str)
    out["nama_jumlah_keluarga_terdaftar"] = kel.map(lambda s: len([x for x in s.split("|") if x.strip()]))
    out["nama_jumlah_pengurus_terdaftar"] = peng.map(lambda s: len([x for x in s.split("|") if x.strip()]))
    out["nama_keluarga_cocok"] = [
        int(any(cocok_nama(k, x) for x in s.split("|") if x.strip()))
        for k, s in zip(ket, kel)]
    out["nama_pengurus_cocok"] = [
        int(any(cocok_nama(k, x) for x in s.split("|") if x.strip()))
        for k, s in zip(ket, peng)]
    out["kop_ada_nama_nasabah"] = (d["nama_nasabah"].fillna("").str.strip() != "").astype(int)
    out["kop_ada_nomor_rekening"] = (d["nomor_rekening"].fillna("").str.strip() != "").astype(int)
    out["kop_panjang_nama_nasabah"] = d["nama_nasabah"].fillna("").str.len()
    out["kop_token_nama_nasabah"] = d["nama_nasabah"].fillna("").map(lambda x: len(nama_token(x)))
    out["kop_kelengkapan_medan"] = (
        out["kop_ada_nama_nasabah"] + out["kop_ada_nomor_rekening"]
        + (d["bank"].fillna("").str.strip() != "").astype(int)
        + (out["nama_jumlah_keluarga_terdaftar"] > 0).astype(int)
        + (out["nama_jumlah_pengurus_terdaftar"] > 0).astype(int))
    # panjang medan: berapa banyak yang terisi, seberapa panjang
    medan = ["keterangan", "tanggal_mentah", "debet_mentah", "kredit_mentah",
             "saldo_mentah", "alasan_llm", "alasan_mesin"]
    pj = pd.DataFrame({m: d[m].fillna("").astype(str).str.len() for m in medan})
    out["kop_medan_terisi"] = (pj > 0).sum(axis=1)
    out["kop_medan_kosong"] = (pj == 0).sum(axis=1)
    out["kop_panjang_medan_rata"] = pj.mean(axis=1).round(2)
    out["kop_panjang_medan_maks"] = pj.max(axis=1)
    out["kop_panjang_medan_min"] = pj.min(axis=1)
    return out


# ==========================================================================
# D. Fitur keluaran LLM (kategori) — dan frekuensi kategori
# ==========================================================================
def _fitur_llm_keluaran(d: pd.DataFrame, frek_kategori: dict | None):
    out = {}
    kat = d["kategori_llm"].fillna("").astype(str)
    for k in KANONIK_KATEGORI:
        out[f"llm_kat_{k.lower().replace(' ', '_')}"] = (kat == k).astype(int)
    out["llm_kat_di_luar_taksonomi"] = (~kat.isin(KANONIK_KATEGORI)).astype(int)
    # Frekuensi kategori: DIHITUNG DARI LATIH SAJA (diserahkan lewat argumen).
    # Menghitungnya dari seluruh data akan memasukkan komposisi test ke fitur.
    if frek_kategori is not None:
        out["llm_kat_frekuensi"] = kat.map(frek_kategori).fillna(0.0)
        out["llm_kat_jarang"] = (out["llm_kat_frekuensi"] < 0.02).astype(int)
    out["llm_pasangan_ada"] = d["pasangan_llm"].notna().astype(int)
    out["llm_kelas_nfi"] = (d["kelas_llm"].fillna("").str.upper() == "NFI").astype(int)
    out["llm_kelas_fi"] = (d["kelas_llm"].fillna("").str.upper() == "FI").astype(int)
    out["llm_kelas_ada"] = (d["kelas_llm"].fillna("").str.strip() != "").astype(int)
    # Kategori yang diprediksi disebut di alasannya sendiri?
    out["llm_kat_disebut_di_alasan"] = [
        int(bool(k) and k.lower() in str(a or "").lower())
        for k, a in zip(kat, d["alasan_llm"])]
    # Berapa kategori LAIN yang disebut di alasan -> tanda LLM ragu antar pilihan
    out["llm_kat_alternatif_disebut"] = [
        sum(1 for kk in KANONIK_KATEGORI
            if kk.lower() != str(k).lower() and kk.lower() in str(a or "").lower())
        for k, a in zip(kat, d["alasan_llm"])]
    return out


def _fitur_alasan(d: pd.DataFrame):
    """Fitur dari CARA LLM menjelaskan, bukan dari isi transaksinya."""
    out = dict(_fitur_teks(d["alasan_llm"], "als_"))
    a = d["alasan_llm"].fillna("").astype(str)
    for nama, rx in RX.items():
        out[f"als_kata_{nama}"] = a.map(lambda x, r=rx: len(r.findall(x)))
    out["als_ragu_ada"] = (out["als_kata_ragu"] > 0).astype(int)
    out["als_yakin_ada"] = (out["als_kata_yakin"] > 0).astype(int)
    out["als_lawan_ada"] = (out["als_kata_lawan"] > 0).astype(int)
    out["als_generik"] = a.map(lambda x: int(bool(RX["umum"].search(x))))
    out["als_terlalu_pendek"] = (a.str.split().map(len) <= 3).astype(int)
    out["als_terpotong"] = a.str.strip().str.endswith(("dan", "atau", ",", "-", "dari", "ke")).astype(int)
    out["als_ada_angka"] = a.str.contains(r"\d").astype(int)
    out["als_ada_nominal"] = a.str.contains(r"\d{1,3}(?:[.,]\d{3})+|\bRp\b", regex=True).astype(int)
    out["als_ada_tanggal"] = a.str.contains(r"\d{1,2}[/\-]\d{1,2}", regex=True).astype(int)
    out["als_sebut_hari"] = a.str.contains(r"\bhari\b|\bjam\b|\bdays?\b|\bhours?\b", case=False).astype(int)
    out["als_sebut_persen"] = a.str.contains(r"%|\bpersen\b|\bselisih\b", case=False).astype(int)
    out["als_sebut_rule"] = a.str.contains(r"\brule\b|\baturan\b", case=False).astype(int)
    out["als_sebut_nama"] = a.str.contains(r"\bnama\b|\bpengirim\b|\bpenerima\b|\bsender\b", case=False).astype(int)
    # Kekhususan: banyak bukti, sedikit kata ragu.
    out["als_skor_kekhususan"] = (out["als_kata_bukti"] + out["als_ada_angka"]
                                 + out["als_ada_nominal"] + out["als_sebut_nama"]
                                 - out["als_kata_ragu"])
    # Tumpang tindih kata alasan dengan keterangan: alasan yang cuma menyalin
    # keterangan tidak membawa penalaran baru.
    ket = d["keterangan"].fillna("").astype(str)
    out["als_tumpang_ket"] = [
        round(len(set(re.findall(r"[a-z]{3,}", x.lower()))
                  & set(re.findall(r"[a-z]{3,}", y.lower())))
              / max(len(set(re.findall(r"[a-z]{3,}", x.lower()))), 1), 4)
        for x, y in zip(a, ket)]
    out["als_panjang_relatif_ket"] = (out["als_panjang"]
                                      / ket.str.len().replace(0, np.nan)).fillna(0).round(4)
    return out


def _fitur_metadata(d: pd.DataFrame):
    out = {}
    out["meta_runtime_detik"] = pd.to_numeric(d["llm_runtime_detik"], errors="coerce")
    out["meta_input_chars"] = pd.to_numeric(d["llm_input_chars"], errors="coerce")
    out["meta_n_kirim"] = pd.to_numeric(d["llm_n_kirim"], errors="coerce")
    out["meta_n_balik"] = pd.to_numeric(d["llm_n_balik"], errors="coerce")
    out["meta_balik_lengkap"] = (out["meta_n_balik"] == out["meta_n_kirim"]).astype(int)
    out["meta_finish_stop"] = (d["llm_finish_reason"].fillna("").str.lower() == "stop").astype(int)
    out["meta_finish_length"] = (d["llm_finish_reason"].fillna("").str.lower()
                                 .str.contains("len|max")).astype(int)
    out["meta_detik_per_baris"] = (out["meta_runtime_detik"]
                                   / out["meta_n_kirim"].replace(0, np.nan)).round(4)
    out["meta_chars_per_baris"] = (out["meta_input_chars"]
                                   / out["meta_n_kirim"].replace(0, np.nan)).round(2)
    # Beban konteks: makin banyak baris dalam satu panggilan, makin besar
    # kemungkinan LLM kehilangan perhatian di baris tengah.
    out["meta_bongkah_besar"] = (out["meta_n_kirim"] > 60).astype(int)
    return out


def _fitur_aturan(d: pd.DataFrame):
    """Fitur dari `_alasan_mesin` — keluaran mesin ATURAN paket referensi.

    Deterministik dari kolom mentah, jadi tersedia saat inferensi dan tidak
    pernah melihat ground truth.
    """
    out = dict(_fitur_teks(d["alasan_mesin"], "aturan_"))
    a = d["alasan_mesin"].fillna("").astype(str)
    out["aturan_ada"] = (a.str.strip() != "").astype(int)
    out["aturan_sebut_bukan_masuk"] = a.str.contains("bukan uang masuk", case=False).astype(int)
    out["aturan_sebut_tanpa_kunci"] = a.str.contains("tidak ada kata kunci", case=False).astype(int)
    out["aturan_sebut_pasangan"] = a.str.contains("keluar|masuk|selisih|rekening", case=False).astype(int)
    return out


def _fitur_pasangan(d: pd.DataFrame):
    """Apakah baris ini punya LAWAN nominal di dokumen yang sama?

    Inti definisi Circling/Kiting, dan kesalahan LLM paling mahal ada di situ.
    Dihitung dari kolom mentah saja, per dokumen — tersedia saat inferensi
    karena statement diproses per dokumen.
    """
    hasil = {k: pd.Series(0, index=d.index, dtype=float) for k in
             ("pas_ada_lawan_10pct", "pas_ada_lawan_persis", "pas_jumlah_lawan",
              "pas_jarak_hari_terdekat", "pas_selisih_pct_terdekat",
              "pas_lawan_nama_sama")}
    for _, g in d.groupby("dokumen", sort=False):
        deb = g["debet_mentah"].map(sen)
        kre = g["kredit_mentah"].map(sen)
        hu = g["tanggal_mentah"].map(hari_urut)
        tok = g["keterangan"].map(nama_token)
        for i in g.index:
            ni = deb.loc[i] or kre.loc[i]
            if not ni:
                continue
            sisi_i = "d" if deb.loc[i] else "k"
            n10 = npersis = 0
            jarak_min, sel_min, nama_sama = np.nan, np.nan, 0
            for j in g.index:
                if j == i:
                    continue
                nj = deb.loc[j] or kre.loc[j]
                if not nj:
                    continue
                sisi_j = "d" if deb.loc[j] else "k"
                if sisi_i == sisi_j:                       # pasangan wajib beda sisi
                    continue
                sel = abs(ni - nj) / max(ni, nj)
                if sel <= 0.10:
                    n10 += 1
                    if sel == 0:
                        npersis += 1
                    hi, hj = hu.loc[i], hu.loc[j]
                    jr = abs(hi - hj) if (hi is not None and hj is not None) else np.nan
                    if not np.isnan(jr) and (np.isnan(jarak_min) or jr < jarak_min):
                        jarak_min, sel_min = jr, round(sel, 4)
                        nama_sama = int(len(tok.loc[i] & tok.loc[j]) >= 2)
                    elif np.isnan(sel_min) or sel < sel_min:
                        sel_min = round(sel, 4)
                        nama_sama = int(len(tok.loc[i] & tok.loc[j]) >= 2)
            hasil["pas_ada_lawan_10pct"].loc[i] = int(n10 > 0)
            hasil["pas_ada_lawan_persis"].loc[i] = int(npersis > 0)
            hasil["pas_jumlah_lawan"].loc[i] = n10
            hasil["pas_jarak_hari_terdekat"].loc[i] = jarak_min
            hasil["pas_selisih_pct_terdekat"].loc[i] = sel_min
            hasil["pas_lawan_nama_sama"].loc[i] = nama_sama
    hasil["pas_lawan_dalam_3_hari"] = (hasil["pas_jarak_hari_terdekat"] <= 3).astype(int)
    # Apakah LLM dan struktur angka SEPAKAT? Ini bukan target — ini bandingan
    # dua masukan yang dua-duanya ada saat inferensi.
    kat = d["kategori_llm"].fillna("")
    hasil["pas_llm_bilang_pasangan"] = kat.isin(["Circling", "Kiting"]).astype(int)
    hasil["pas_llm_setuju_struktur"] = (
        hasil["pas_llm_bilang_pasangan"] == hasil["pas_ada_lawan_10pct"]).astype(int)
    return hasil



# ==========================================================================
# G. Sinyal dari mesin aturan kanal + kandidat pasangan + konteks dokumen
# ==========================================================================
def _fitur_kanal_dan_konteks(d: pd.DataFrame):
    """Fitur dari `aturan_kanal`, `merchant`, dan sebaran prediksi PER DOKUMEN.

    KENAPA FITUR KONTEKS DOKUMEN, DAN KENAPA TIDAK BOCOR
    ----------------------------------------------------
    Kesalahan model di bin teratas punya satu pola: LLM bilang `Non Business`,
    ground truth bilang `Business`, dan baris-baris itu ada di dokumen yang
    isinya justru mayoritas `Business`. Model belajar dari dokumen latih (yang
    mayoritas Non Business) bahwa prediksi Non Business itu andal — lalu salah
    di dokumen uji yang wataknya berbeda.

    Yang diberikan di sini adalah watak dokumen sebagaimana TERLIHAT DARI
    PREDIKSI DAN ATURAN, bukan dari ground truth:
      - porsi prediksi `Non Business` di dokumen ini
      - porsi baris yang aturan kanal TIDAK SEPAKAT dengan LLM
    Keduanya dihitung dari keluaran LLM + kolom mentah, yang dua-duanya ada saat
    inferensi (statement diproses per dokumen). TIDAK ADA target di dalamnya.

    Fitur kedua itu yang paling langsung: kalau di satu dokumen aturan kanal
    berkali-kali membantah LLM, itu tanda LLM sedang mengempiskan dokumen itu ke
    `Non Business` — dan prediksi `Non Business` di dokumen itu jadi kurang
    layak dipercaya.
    """
    out = {}
    ada_kanal = "diputuskan_kode" in d.columns
    out["kanal_diputuskan_kode"] = (d["diputuskan_kode"].fillna(0).astype(int)
                                    if ada_kanal else pd.Series(0, index=d.index))
    if "_kategori_llm_sebelum_aturan" in d.columns:
        beda = (d["_kategori_llm_sebelum_aturan"].fillna("").map(rapat)
                != d["kategori_llm_mentah"].fillna("").map(rapat))
        out["kanal_membantah_llm"] = beda.astype(int)
    else:
        out["kanal_membantah_llm"] = pd.Series(0, index=d.index)

    from penilai.merchant import jenis
    jn = d["keterangan"].map(jenis)
    out["merchant_tampak_konsumen"] = (jn != "").astype(int)
    for k in ("makanan_minuman", "bbm_otomotif", "kecantikan_kesehatan",
              "ritel_konsumen", "hiburan_langganan", "transportasi_parkir"):
        out[f"merchant_{k}"] = (jn == k).astype(int)

    # --- konteks dokumen, dari prediksi & aturan (bukan dari target) --------
    kat = d["kategori_llm"].fillna("").astype(str) if "kategori_llm" in d.columns         else d["kategori_llm_mentah"].fillna("").astype(str).map(rapat)
    g = pd.DataFrame({"dok": d["dokumen"], "kat": kat,
                      "bantah": out["kanal_membantah_llm"],
                      "kode": out["kanal_diputuskan_kode"]})
    out["dok_porsi_llm_non_business"] = g.groupby("dok")["kat"].transform(
        lambda s: (s == "Non Business").mean()).round(4)
    out["dok_porsi_llm_business"] = g.groupby("dok")["kat"].transform(
        lambda s: (s == "Business").mean()).round(4)
    out["dok_porsi_kanal_membantah"] = g.groupby("dok")["bantah"].transform("mean").round(4)
    out["dok_porsi_diputuskan_kode"] = g.groupby("dok")["kode"].transform("mean").round(4)
    out["dok_kategori_llm_unik"] = g.groupby("dok")["kat"].transform("nunique")
    # Seberapa jauh prediksi baris ini dari watak dokumennya. Baris `Non Business`
    # di dokumen yang 80% `Non Business` berbeda artinya dari baris `Non Business`
    # di dokumen yang cuma 20%.
    out["llm_kat_selaras_dokumen"] = [
        float(p) if k == "Non Business" else (1.0 - float(p))
        for k, p in zip(kat, out["dok_porsi_llm_non_business"])]
    return out


# ==========================================================================
# Perakitan
# ==========================================================================
def bangun_kandidat(d: pd.DataFrame, C, frek_kategori: dict | None = None):
    """-> (DataFrame fitur numerik, daftar nama fitur).

    `frek_kategori` HARUS dihitung dari data latih saja dan diserahkan dari luar.
    Kalau None, fitur frekuensi kategori tidak dibuat sama sekali — lebih baik
    tidak ada fitur daripada ada fitur yang diam-diam melihat test.
    """
    bagian = {}
    bagian.update(_fitur_teks(d["keterangan"], "ket_"))
    bagian.update(_fitur_angka(d))
    bagian.update(_fitur_tanggal(d))
    bagian.update(_fitur_struktur(d))
    bagian.update(_fitur_kop(d))
    bagian.update(_fitur_llm_keluaran(d, frek_kategori))
    bagian.update(_fitur_alasan(d))
    bagian.update(_fitur_metadata(d))
    bagian.update(_fitur_aturan(d))
    bagian.update(_fitur_pasangan(d))
    bagian.update(_fitur_kanal_dan_konteks(d))
    bagian.update(fitur_relasi(d))
    bagian.update(fitur_konsistensi(d))

    X = pd.DataFrame({k: pd.Series(v).values for k, v in bagian.items()}, index=d.index)
    X = X.apply(pd.to_numeric, errors="coerce")
    X.insert(0, "uid", d["uid"].values)
    return X, [c for c in X.columns if c != "uid"]


def sumber_fitur(nama: str) -> str:
    for pre, src in sorted(SUMBER_PREFIKS.items(), key=lambda x: -len(x[0])):
        if nama.startswith(pre):
            return src
    if nama.startswith("pas_"):
        return "mentah+llm_keluaran"
    if nama.startswith(("dok_", "llm_kat_selaras")):
        return "mentah+llm_keluaran"
    return "tak_diketahui"
