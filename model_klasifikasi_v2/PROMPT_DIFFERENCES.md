# Perbedaan prompt FI dan NFI

| Bagian | FI | NFI |
|---|---|---|
| Transaksi yang dikirim ke LLM | Kredit saja | Kredit dan debit |
| Kategori | `gaji`, `bonus`, `thr`, `cuti` | `own`, `circling`, `kiting`, `business`, `non_business` |
| Prioritas | Bonus → THR → Cuti → Gaji | Pasangan → Own → Business/Non Business |
| Kandidat pasangan | Tidak digunakan | Wajib diperiksa sebelum Business/Non Business |
| Konteks tambahan | Header rekening | Header, nama KK, dan nama pengurus Akta |
| Baris tidak sesuai | `null` | `null` hanya untuk saldo/subtotal/tidak layak klasifikasi |
| Field `pasangan` | Selalu `null` | Nomor baris lawan untuk Circling/Kiting |

Kedua prompt mempertahankan kontrak keluaran yang sama: JSON
`classifications[]` berisi `index`, `category`, `alasan`, dan `pasangan`.

## Perbedaan dari prompt runtime V11 asli

Prompt asli menghasilkan kategori gabungan tanpa memisahkan FI/NFI. Dalam V2:

1. FI dan NFI mempunyai whitelist kategori masing-masing.
2. Debit tidak pernah masuk klasifikasi FI.
3. Data KK/Akta hanya memperkaya NFI untuk Own dan Non Business.
4. `CUTI/ECUTI` dipetakan ke `cuti` karena kontrak FI meminta kategori Cuti.
   Pada prompt V11 asli, `ECUTI` berada dalam kelompok Insentif. Ini merupakan
   perubahan bisnis yang sudah ada pada `service_klasifikasi_mutasi`, bukan
   aturan asli prompt V11.
5. Confidence LLM tidak diminta. Probability dihitung oleh model 14 fitur.
