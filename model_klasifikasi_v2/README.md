# Model Klasifikasi V2

Folder ini adalah paket V2 terpisah. Service produksi yang sekarang tidak
berubah sampai `app/dependencies.py` diarahkan secara eksplisit ke paket ini.

## Isi

```text
model_klasifikasi_v2/
├── config/
│   ├── runtime.yaml
│   └── feature_engineering.yaml
├── prompts/
│   ├── fi.yaml
│   └── nfi.yaml
├── artifacts/
│   ├── model_logistic_regression.joblib
│   ├── teks_turunan.joblib
│   └── manifest.json
├── penilai/                    # runtime feature builder (vendored)
│   ├── fitur.py
│   ├── fitur_konsistensi.py
│   ├── fitur_relasi.py
│   ├── kanon.py
│   ├── konfigurasi.py
│   ├── merchant.py
│   └── teks_turunan.py
├── feature_engineering.py
├── prompt_loader.py
├── requirements.txt
└── PROMPT_DIFFERENCES.md
```

Folder ini mandiri dan tidak mengambil modul `penilai` dari service induk.
Install dependensi dengan `pip install -r model_klasifikasi_v2/requirements.txt`.

## Pengaturan

- Environment runtime ada di `.env`; template tanpa rahasia ada di
  `.env.example`. Credential AWS session bersifat sementara dan perlu
  diperbarui setelah kedaluwarsa.
- Edit prompt FI atau NFI pada `prompts/`.
- Edit threshold dan pembulatan pada `config/feature_engineering.yaml`.
- Pilih flow dan lokasi prompt pada `config/runtime.yaml`.
- Urutan 14 fitur tidak boleh diubah tanpa retraining karena menjadi kontrak
  input Logistic Regression.

Konfigurasi run GPT OSS yang direplikasi: Bedrock
`openai.gpt-oss-120b-1:0`, region `ap-southeast-3`, temperature `0`, maksimal
`8192` output token, timeout `300` detik, tiga total percobaan, dan batch `20`.

## Flow

```text
jenis_pinjaman
  ├─ FI  → prompt FI  → kategori gaji/bonus/thr/cuti
  └─ NFI → prompt NFI → kategori own/circling/kiting/business/non_business
                         ↓
              feature engineering 14 fitur
                         ↓
              Logistic Regression
                  ├─ probability
                  └─ bin 1–3 → confidence
```

`probability` merupakan skor Logistic Regression dan dibulatkan empat angka.
`confidence` memakai skor mentah Logistic Regression yang sama: bernilai `1`
jika skor minimal `0.9482552543312334` (bin 1–3), selain itu `0`.

Ambang dipilih dari prediksi group-OOF train: 187 dari 187 transaksi pada bin
1–3 benar. Ketika ambang tersebut diterapkan tanpa perubahan ke test, terdapat
6 dari 6 transaksi benar (presisi empiris 100%). Angka tersebut bukan jaminan
presisi data produksi baru.

## Pemakaian Python

```python
from model_klasifikasi_v2 import FeatureEngineeringV2, load_prompts

prompts = load_prompts()
scorer = FeatureEngineeringV2()
hasil = scorer.predict(records)
```

Struktur `records` sama dengan record internal `ClassificationService`: setiap
record membawa `transaction`, `account_context`, `document_context`, dan
`llm_result`.
