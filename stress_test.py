"""
STRESS TEST — STABILITAS DAN BATAS AMBANG TOLERANSI
====================================================
Menjawab rumusan masalah 3: seberapa jauh intensitas noise boleh naik
sebelum sinyal hasil denoising tidak lagi layak untuk lokalisasi PVC.

Data uji dibangkitkan ULANG pada beberapa level SNR (default 10, 5, 0, -5,
-10 dB) memakai prosedur yang sama dengan prapemrosesan: noise disuntikkan
di 9 elektroda lalu diturunkan ke 12 lead, dan potongan noise diambil dari
bagian rekaman NSTDB yang khusus untuk data uji.

Metrik per level, untuk ketiga model:
  * SNRimp (dB)        — kualitas rekonstruksi
  * MSE (mV^2)         — galat rekonstruksi
  * flip_stabil (%)    — Peak_index yang berubah, denyut dekat ambang dibuang
  * korelasi QRS       — kemiripan bentuk kompleks QRS

Batas ambang toleransi ditetapkan sebagai SNR input TERENDAH yang masih
memenuhi kriteria yang Anda tetapkan di CRIT (ubah sesuai justifikasi Bab 3).

Cara pakai di Colab:

    %run stress_test.py
    hasil = jalankan()                       # ~15-25 menit
    hasil = jalankan(levels=(5, 0, -5))      # lebih cepat untuk uji coba
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from prapemrosesan_final import CONFIG as PRE, load_noise_bank, inject_noise, scale_pair
from analisis_morfologi import (CFG as MOR, LEADS, qrs_windows, potong,
                                f_peak_index, f_qrs_int, f_peak, f_rms, f_var)
from pelatihan_model import PositionalEncoding

# Kriteria kelayakan — ubah sesuai justifikasi di Bab 3
CRIT = {
    "flip_stabil_maks": 5.0,     # %
    "korelasi_min": 0.95,
}

LEVELS = (10, 5, 0, -5, -10)


# =============================================================================
def siapkan(cfg_pre=PRE, cfg_mor=MOR):
    """Muat sinyal bersih uji (mV) beserta anotasinya, hanya segmen valid."""
    D = cfg_mor["data_dir"]
    Xc = np.load(D + "X_test_clean.npy")
    sc = np.load(D + "scale_test.npy").reshape(-1, 1, 1)
    keep = np.load(D + "valid_test.npy")
    rpk = np.load(D + "rpeak_test.npy", allow_pickle=True)[keep]
    sym = np.load(D + "symbol_test.npy", allow_pickle=True)[keep]

    Xc_mv = (Xc[keep] * sc[keep]).astype(np.float64)
    del Xc, sc

    semua = np.ones(len(Xc_mv), bool)
    seg, pos, w_clean = qrs_windows(Xc_mv, rpk, sym, semua, cfg_mor)
    print(f"Segmen valid: {len(Xc_mv):,} | denyut PVC: {len(seg):,}")

    # denyut yang rasio dominasinya jauh dari ambang 1.5 -> flip bukan efek ambang
    mx, mn = w_clean.max(-1), w_clean.min(-1)
    ratio = np.abs(np.where(np.abs(mx) > np.abs(mn),
                            mx / (mn - 1e-12), mn / (mx + 1e-12)))
    stabil = (ratio < 1.2) | (ratio > 1.9)
    print(f"Denyut-lead dekat ambang (dibuang dari flip_stabil): "
          f"{100*(~stabil).mean():.1f}%\n")

    return dict(clean=Xc_mv, seg=seg, pos=pos, w_clean=w_clean,
                pi_clean=f_peak_index(w_clean), stabil=stabil,
                W=w_clean.shape[-1])


def bangkitkan(clean_mv, bank, snr_db, cfg_pre, seed):
    """Suntik noise pada satu level SNR, lalu skalakan seperti prapemrosesan."""
    rng = np.random.default_rng(seed)
    n = len(clean_mv)
    Xn_s = np.empty(clean_mv.shape, np.float32)       # terskala (masukan model)
    Xc_s = np.empty(clean_mv.shape, np.float32)
    sc = np.empty(n, np.float32)
    for i in range(n):
        noisy = inject_noise(clean_mv[i], bank, cfg_pre["noise_kind"], snr_db, rng)
        Xn_s[i], Xc_s[i], sc[i] = scale_pair(noisy, clean_mv[i])
    return Xn_s, sc.reshape(-1, 1, 1)


def pooled_snr(ref, sig):
    r = ref - ref.mean(axis=1, keepdims=True)
    e = sig - ref
    e = e - e.mean(axis=1, keepdims=True)
    return 10 * np.log10((r ** 2).sum(axis=(0, 1)) / (e ** 2).sum(axis=(0, 1)))


def korelasi(w_clean, w_test):
    a = w_clean - w_clean.mean(-1, keepdims=True)
    b = w_test - w_test.mean(-1, keepdims=True)
    num = (a * b).sum(-1)
    den = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1)) + 1e-12
    return float((num / den).mean())


# =============================================================================
def jalankan(levels=LEVELS, cfg_pre=PRE, cfg_mor=MOR, crit=CRIT,
             seed=7, simpan=True):
    S = siapkan(cfg_pre, cfg_mor)
    clean = S["clean"]

    print("Memuat bank noise (bagian uji)...")
    _, bank_test = load_noise_bank(cfg_pre)

    print("\nMemuat model...")
    models = {}
    for core in cfg_mor["models"]:
        models[core] = tf.keras.models.load_model(
            cfg_mor["model_dir"] + f"{core}_seed{cfg_mor['seed']}_best.keras",
            custom_objects={"PositionalEncoding": PositionalEncoding})

    baris, per_lead = [], []
    for snr in levels:
        print(f"\n=== SNR input {snr:+.0f} dB ===")
        Xn_s, sc = bangkitkan(clean, bank_test, snr, cfg_pre, seed)
        Xn_mv = Xn_s * sc
        w_noisy = potong(Xn_mv, S["seg"], S["pos"], S["W"])

        snr_in = pooled_snr(clean, Xn_mv)
        fl = (f_peak_index(w_noisy) != S["pi_clean"])
        baris.append(dict(snr=snr, model="noisy", snrimp=0.0,
                          mse=float(np.mean((Xn_mv - clean) ** 2)),
                          flip_semua=float(fl.mean() * 100),
                          flip_stabil=float(fl[S["stabil"]].mean() * 100),
                          korelasi=korelasi(S["w_clean"], w_noisy)))
        print(f"  noisy      SNR terukur {snr_in.mean():5.2f} dB | "
              f"flip {baris[-1]['flip_stabil']:5.2f}% | "
              f"korelasi {baris[-1]['korelasi']:.3f}")

        for core, m in models.items():
            pr_mv = m.predict(Xn_s, batch_size=128, verbose=0) * sc
            imp = pooled_snr(clean, pr_mv) - snr_in
            w_pr = potong(pr_mv, S["seg"], S["pos"], S["W"])
            fl = (f_peak_index(w_pr) != S["pi_clean"])
            baris.append(dict(snr=snr, model=core, snrimp=float(imp.mean()),
                              mse=float(np.mean((pr_mv - clean) ** 2)),
                              flip_semua=float(fl.mean() * 100),
                              flip_stabil=float(fl[S["stabil"]].mean() * 100),
                              korelasi=korelasi(S["w_clean"], w_pr)))
            per_lead.append(dict(snr=snr, model=core,
                                 **{f"snrimp_{n}": float(v)
                                    for n, v in zip(LEADS, imp)}))
            print(f"  {core:10s} SNRimp {imp.mean():5.2f} dB | "
                  f"flip {baris[-1]['flip_stabil']:5.2f}% | "
                  f"korelasi {baris[-1]['korelasi']:.3f}")
            del pr_mv, w_pr
        del Xn_s, Xn_mv, w_noisy

    df = pd.DataFrame(baris)
    dfl = pd.DataFrame(per_lead)

    print("\n" + "=" * 70)
    print("RINGKASAN")
    print("=" * 70)
    for kol in ("snrimp", "flip_stabil", "korelasi"):
        print(f"\n{kol}:")
        print(df.pivot(index="snr", columns="model", values=kol)
                .round(3).sort_index(ascending=False).to_string())

    print("\n" + "=" * 70)
    print(f"BATAS AMBANG TOLERANSI "
          f"(flip_stabil <= {crit['flip_stabil_maks']}% dan "
          f"korelasi >= {crit['korelasi_min']})")
    print("=" * 70)
    for model in df.model.unique():
        d = df[df.model == model].sort_values("snr", ascending=False)
        ok = d[(d.flip_stabil <= crit["flip_stabil_maks"]) &
               (d.korelasi >= crit["korelasi_min"])]
        batas = f"{ok.snr.min():+.0f} dB" if len(ok) else "tidak terpenuhi"
        print(f"  {model:10s} : {batas}")

    if simpan:
        MD = cfg_mor["model_dir"]
        df.to_csv(MD + "stress_test.csv", index=False)
        dfl.to_csv(MD + "stress_test_per_lead.csv", index=False)
        print(f"\nTersimpan di {MD}")

    return {"ringkas": df, "per_lead": dfl}


def plot(df):
    """Tiga kurva: SNRimp, flip_stabil, korelasi terhadap SNR input."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for kol, a, judul in zip(("snrimp", "flip_stabil", "korelasi"), ax,
                             ("SNR improvement (dB)",
                              "Flip rate Peak_index (%)",
                              "Korelasi bentuk QRS")):
        p = df.pivot(index="snr", columns="model", values=kol).sort_index()
        for m in p.columns:
            if kol == "snrimp" and m == "noisy":
                continue
            a.plot(p.index, p[m], marker="o", label=m,
                   ls="--" if m == "noisy" else "-",
                   color="gray" if m == "noisy" else None)
        a.set_xlabel("SNR input (dB)"); a.set_title(judul)
        a.grid(alpha=.3); a.legend(fontsize=8)
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    print("Definisi dimuat. Jalankan: hasil = jalankan()")
