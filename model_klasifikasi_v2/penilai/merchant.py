#!/usr/bin/env python3
"""Apakah lawan transaksi sebuah merchant KONSUMEN atau pemasok USAHA?

    from penilai.merchant import konsumen
    konsumen("Pembayaran QR ke WARTEG OTENS BAHARI 504538609896")   # True

KENAPA MODUL INI ADA
--------------------
Rule 9 prompt v2/v3 menyatakan SEMUA pembayaran QR/e-wallet/e-commerce sebagai
`Non Business`. Ground truth tidak sepakat: pada run 28 Sep 2026, 84 baris di
PK2509C83W ber-ground-truth `Business` dijawab `Non Business` oleh LLM dengan
alasan "per rule 9" — LLM menuruti prompt, prompt-nya yang salah.

Keputusan pemilik data (28 Sep 2026): pembayaran QR ke merchant yang BERNAMA
USAHA termasuk **Business**. Yang tetap `Non Business` adalah belanja KONSUMEN —
dan itu ditentukan oleh JENIS dagangannya, bukan bentuk badan hukumnya. Prompt
sendiri sudah memuat prinsipnya ("the KIND of spending decides, not the legal
form of the counterparty"); yang salah adalah daftar QR/e-wallet-nya yang
memborong semua.

LEKSIKON DITURUNKAN DARI DOKUMEN LATIH SAJA. Delapan dokumen latih (pembelahan
entitas manifest v20260926_184647) punya 122 baris QR ber-ground-truth
`Non Business`; kata-kata di bawah diambil dari situ. Tiga dokumen uji TIDAK
dipakai menyusunnya, supaya angka di dokumen uji tetap berarti.

BATAS KEGUNAANNYA, DITULIS TERBUKA. Ini leksikon, bukan pemahaman. Merchant baru
yang namanya tidak memuat kata mana pun di bawah akan dinilai "bukan konsumen"
dan jatuh ke `Business` — dan itu bisa salah. Karena itu keluaran modul ini
dipakai sebagai PETUNJUK yang diserahkan ke LLM (`petunjuk_merchant` di masukan
prompt v4), bukan sebagai keputusan akhir: menilai "toko ini jualan apa" memang
tugas yang butuh pengetahuan dunia, dan itu bagian LLM.
"""
import re

# Kelompok dipisah supaya bisa dibantah dan ditambah satu-satu, dan supaya
# keluaran `jenis()` bisa menjelaskan KENAPA sebuah merchant dianggap konsumen.
KELOMPOK = {
    "makanan_minuman": r"""
        WARUNG|WARUNK|WARTEG|\bRM\b|RUMAH\s*MAKAN|PONDOK|DAPUR|KULINER|SELERA|
        BAKMI|BAKSO|SATE|SOTO|NASI|AYAM|BEBEK|IKAN|SEAFOOD|STEAK|GRILL|
        MIE\b|RAMEN|SUSHI|GYU|KATSU|DIMSUM|BUBUR|SOP\b|KONRO|PUKIS|MARTABAK|
        RESTO|RESTAURANT|EATERY|FOOD|CATERING|KATERING|CANTEEN|KANTIN|
        CAFE|KOPI|COFFEE|SBUX|STARBUCKS|TOMORO|KEDAI|BAKERY|ROTI|DONAT|CAKE|
        CHOCOLATE|ICE\s*CREAM|JUICE|BOBA|MILK|\bTEA\b|MINUM|BAR\b|BISTRO|
        \bKFC\b|\bA&W\b|MCD|MCDONALD|PIZZA|BURGER|HOKBEN|YOSHINOYA|CHATERAISE|
        WAROENG|CHICKEN|SEGER|CAROK|LEKO|MINANG|MAKASSAR|CIANJU
    """,
    "bbm_otomotif": r"""
        SPBU|SHELL|PERTAMINA|\bVIVO\b|\bBP\b|BENSIN|PERTALITE|PERTAMAX|
        CAR\s*WASH|AUTOSPA|RODEO|BENGKEL|TAMBAL|OLI\b|SERVIS\s*MOBIL
    """,
    "kecantikan_kesehatan": r"""
        BEAUTY|SKINCARE|\bZAP\b|SALON|BARBER|\bSPA\b|CLINIC|KLINIK|APOTEK|
        OPTIK|DENTAL|NMW|\bWSJ\b|KECANTIKAN
    """,
    "ritel_konsumen": r"""
        ALFAMART|ALFAMIDI|\bALFA_|INDOMARET|SUPERINDO|HYPERMART|TRANSMART|
        CARREFOUR|GIANT|LOTTE|RANCH\s*MARKET|MINIMARKET|SUPERMARKET|TOSERBA|
        \bIBOX\b|\bOPPO\b|SAMSUNG|XIAOMI|ERAFONE|JDSPORTS|SPORT\s*STATION|
        UNIQLO|MATAHARI|RAMAYANA|ACE\s*HARDWARE|INFORMA|GRAMEDIA|
        \bMALL\b|PLAZA|CITY\b|SQUARE|CENTRAL\s*PARK|GANDARIA|KEMANG|BLOK\s*M
    """,
    "hiburan_langganan": r"""
        \bXXI\b|CGV|CINEPOLIS|CINEMA|KARAOKE|\bGYM\b|FITNESS|
        NETFLIX|SPOTIFY|DISNEY|YOUTUBE|STEAM\b|GAME
    """,
    "transportasi_parkir": r"""
        BLUEBIRD|BLUE\s*BIRD|GRAB|GOJEK|GOCAR|GOFOOD|GRABFOOD|SHOPEEFOOD|
        \bTAKSI\b|VALET|PARKIR|PARKING|\bTOL\b|\bKRL\b|TRANSJAKARTA
    """,
}
RX = {k: re.compile(v, re.I | re.X) for k, v in KELOMPOK.items()}
RX_SEMUA = re.compile("|".join(f"(?:{v})" for v in KELOMPOK.values()), re.I | re.X)


def jenis(keterangan):
    """-> nama kelompok konsumen yang kena, atau '' kalau tidak ada.

    Dipakai untuk menjelaskan keputusannya, bukan hanya menyatakannya —
    `petunjuk_merchant` di prompt v4 ikut membawa nama kelompoknya supaya LLM
    bisa menilai apakah petunjuknya masuk akal untuk baris itu.
    """
    t = str(keterangan or "")
    for nama, rx in RX.items():
        if rx.search(t):
            return nama
    return ""


def konsumen(keterangan):
    """True kalau lawan transaksi tampak merchant KONSUMEN."""
    return bool(RX_SEMUA.search(str(keterangan or "")))
