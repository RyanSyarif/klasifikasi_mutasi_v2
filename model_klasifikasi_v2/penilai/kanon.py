#!/usr/bin/env python3
"""Kanon SATU definisi untuk uang, tanggal, dan KATEGORI.

    from penilai.kanon import uang, sen, tanggal, kanon_kategori

Diangkat dari `paket_ryan/kanon.py` (uang/tanggal) supaya kedua paket tidak
pernah berbeda pendapat tentang nilai yang sama, lalu ditambahi normalisasi
KATEGORI yang jadi inti paket ini.

PRINSIPNYA SAMA: BACA APA ADANYA, NORMALKAN DENGAN KODE. Nilai tercetaknya
tetap disimpan utuh untuk jejak audit; yang di sini cuma turunannya.

NORMALISASI KATEGORI ADALAH PINTU BOCOR YANG PALING MUDAH DIBUKA TANPA SADAR.
`kanon_kategori` HANYA merapatkan spasi dan menyamakan besar-kecil huruf, lalu
menoleh ke peta alias yang ditulis TERBUKA di config/kategori.yaml. Dia tidak
punya kecerdasan sendiri: tidak ada stemming, tidak ada pencocokan mirip, tidak
ada fallback "kalau mengandung kata business maka Business". Sebab setiap
kelonggaran di sini menaikkan akurasi LLM yang dilaporkan tanpa ada satu pun
baris yang benar-benar diperbaiki.
"""
import re
import unicodedata

BULAN = {
    "jan": 1, "januari": 1, "january": 1, "feb": 2, "februari": 2, "february": 2,
    "mar": 3, "maret": 3, "march": 3, "apr": 4, "april": 4,
    "mei": 5, "may": 5, "jun": 6, "juni": 6, "june": 6,
    "jul": 7, "juli": 7, "july": 7, "agu": 8, "ags": 8, "agt": 8, "aug": 8,
    "agustus": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "okt": 10, "oct": 10, "oktober": 10, "october": 10,
    "nov": 11, "november": 11, "des": 12, "dec": 12, "desember": 12, "december": 12,
}

LUAR_TAKSONOMI = "__LUAR_TAKSONOMI__"
TAK_ADA = "__TAK_ADA__"


def _b(s):
    s = str(s).strip().lower()
    return BULAN.get(s) or BULAN.get(s[:3])


def _y(y):
    y = int(y)
    return y + 2000 if y < 100 else y


def uang(v):
    """'2.500,00' / '12,100,000.00' -> '2500.00'. Tanda minus hanya lewat kalau
    memang ada di masukan; pemanggil yang memutuskan boleh/tidaknya."""
    s = " ".join(str(v or "").split())
    if not s:
        return ""
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"[^0-9.,]", "", s)
    if not s:
        return ""
    if "." in s and "," in s:
        des = "." if s.rfind(".") > s.rfind(",") else ","
        s = s.replace("," if des == "." else ".", "").replace(des, ".")
    elif "." in s or "," in s:
        sep = "." if "." in s else ","
        ekor = s.rsplit(sep, 1)[1]
        s = s.replace(sep, "") if s.count(sep) > 1 or len(ekor) == 3 else s.replace(sep, ".")
    try:
        d = float(s)
    except ValueError:
        return ""
    return ("-" if neg else "") + f"{d:.2f}"


def sen(v):
    """Uang -> bilangan bulat SEN. Dipakai untuk membandingkan nominal.

    Bilangan bulat, bukan float: perbandingan pasangan keluar-masuk
    (Circling/Kiting) harus deterministik, dan 0.1 + 0.2 != 0.3 di float.
    """
    s = uang(v)
    if not s:
        return 0
    return int(round(float(s) * 100))


def tanggal(v):
    """-> 'DD/MM/YYYY', atau 'DD/MM' kalau tahunnya memang tidak tercetak, atau ''.

    Jam dibuang. Tahun TIDAK pernah dilengkapi dari tempat lain — aturan yang
    sama dengan prompt, karena BNI/BTN mencetak hari+bulan saja di tiap baris.
    """
    s = " ".join(str(v or "").split()).lower()
    if not s:
        return ""
    s = re.sub(r"\b\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\s*(wib|wita|wit)?\b", " ", s)
    s = " ".join(s.split())
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)                 # ISO
    if m:
        return f"{int(m[3]):02d}/{int(m[2]):02d}/{m[1]}"
    m = re.search(r"(\d{1,2})\s*[/\-. ]\s*([a-z]{3,9})\s*[/\-. ]?\s*(\d{2,4})?", s)
    if m and _b(m[2]):
        return (f"{int(m[1]):02d}/{_b(m[2]):02d}/{_y(m[3])}" if m[3]
                else f"{int(m[1]):02d}/{_b(m[2]):02d}")
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})(?:\s*[/\-.]\s*(\d{2,4}))?", s)
    if m and 1 <= int(m[2]) <= 12:
        return (f"{int(m[1]):02d}/{int(m[2]):02d}/{_y(m[3])}" if m[3]
                else f"{int(m[1]):02d}/{int(m[2]):02d}")
    return ""


def hari_urut(v):
    """Tanggal -> bilangan urut hari untuk mengurutkan/menghitung jarak.

    Kalau tahun tidak tercetak, tahun TIDAK dikarang: kembalikan None. Jarak
    hari antar baris yang tahunnya tidak diketahui memang tidak bisa dihitung,
    dan menebaknya akan membuat fitur 'jarak ke pasangan' bohong.
    """
    t = tanggal(v)
    if not t or len(t.split("/")) != 3:
        return None
    d, m, y = (int(x) for x in t.split("/"))
    try:
        import datetime
        return datetime.date(y, m, d).toordinal()
    except ValueError:
        return None


def rapat(s):
    """Rapatkan spasi + buang spasi tepi. Tidak mengubah huruf."""
    return " ".join(str(s if s is not None else "").split())


def _kunci_kategori(s):
    """Kunci pencocokan alias: NFKC, huruf kecil, spasi rapat, tanpa tanda baca
    pemisah yang cuma variasi tulisan (- dan _ jadi spasi)."""
    s = unicodedata.normalize("NFKC", str(s if s is not None else ""))
    s = " ".join(s.split()).lower()
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def kanon_kategori(nilai, alias, kanonik, luar_taksonomi_llm=(), keluar_taksonomi=()):
    """Kategori mentah -> (kategori_kanonik, status).

    status:
      'ok'              -> masuk taksonomi kanonik
      'keluar'          -> penanda "bukan kategori apa pun" (mis. '–'); baris dibuang
      'luar_taksonomi'  -> label yang SENGAJA tidak dialiaskan (mis. 'Insentif')
      'tak_dikenal'     -> label yang tidak dikenali sama sekali
      'kosong'          -> tidak ada nilai

    Tidak ada tebakan. Yang tidak ada di `alias` dan tidak sama dengan salah
    satu `kanonik` dikembalikan sebagai tak_dikenal, bukan dipaksa masuk.
    """
    mentah = rapat(nilai)
    if mentah == "" or mentah.lower() in ("none", "nan", "null"):
        return TAK_ADA, "kosong"
    k = _kunci_kategori(mentah)
    if any(_kunci_kategori(x) == k for x in keluar_taksonomi):
        return LUAR_TAKSONOMI, "keluar"
    if any(_kunci_kategori(x) == k for x in luar_taksonomi_llm):
        return mentah, "luar_taksonomi"
    for kk in kanonik:
        if _kunci_kategori(kk) == k:
            return kk, "ok"
    for a, target in alias.items():
        if _kunci_kategori(a) == k:
            return target, "ok"
    return mentah, "tak_dikenal"


# Token yang TIDAK boleh dihitung sebagai bagian nama orang.
#
# Gelar dan bentuk badan usaha sudah jelas. Yang ditambahkan kemudian: NAMA
# BANK dan kata kanal. Alasannya bug nyata: "Transfer dari BANK MANDIRI MUNIROH"
# lawan "Transfer ke BANK MANDIRI PHILIPUS RANTE TANDU" punya DUA token yang
# sama — BANK dan MANDIRI — sehingga dianggap orang yang sama. Akibatnya
# pasangan Kiting (rekening BEDA) salah dinilai jadi Circling (rekening SAMA),
# yang justru pembeda utama kedua kategori itu.
#
# Prompt v2 sudah memperingatkan hal setara: "BANK NAMES ARE NOT CORPORATE
# SENDERS ... Strings like `PT. BANK <X>` are the COUNTERPARTY'S BANK (routing
# info, not the sender)." Aturan itu berlaku juga untuk pencocokan nama.
BUANG_TOKEN = {
    # gelar & sapaan
    "BPK", "BAPAK", "IBU", "SDR", "SDRI", "TUAN", "NYONYA", "THE",
    # bentuk badan usaha
    "PT", "CV", "UD", "TBK", "PERSERO", "KOPERASI",
    # nama bank & kanal — routing, bukan nama orang
    "BANK", "MANDIRI", "BCA", "BNI", "BRI", "BSI", "CIMB", "NIAGA", "PERMATA",
    "DANAMON", "BTN", "BTPN", "MAYBANK", "PANIN", "OCBC", "UOB", "HSBC",
    "SYARIAH", "CENTRAL", "ASIA", "RAKYAT", "INDONESIA", "NEGARA", "TABUNGAN",
    "BIFAST", "FAST", "RTGS", "LLG", "SKN", "ATM", "TRF", "TRSF", "TRANSFER",
    "DARI", "KEPADA", "VIA", "REK", "REKENING",
}


def nama_token(s):
    """Nama -> himpunan token huruf >= 3 karakter, huruf besar.

    Dipakai untuk fitur 'nama pemilik/keluarga muncul di keterangan'. Token,
    bukan seluruh kalimat, karena bank memotong nama panjang
    ("NI PUTU ARI SUSANTI" -> "NI PUTU ARI").
    """
    t = re.sub(r"[^A-Za-z ]", " ", str(s or "")).upper()
    return {w for w in t.split() if len(w) >= 3 and w not in BUANG_TOKEN}


def cocok_nama(keterangan, nama, min_token=2):
    """Apakah `nama` muncul di `keterangan`? Aturan sama dengan paket_ryan:
    dua token cocok atau lebih = cocok; satu token umum ('MUHAMMAD') bukan."""
    kata = nama_token(nama)
    if len(kata) < min_token:
        return False
    teks = nama_token(keterangan)
    kena = len(kata & teks)
    return kena >= max(min_token, len(kata) - 1)
