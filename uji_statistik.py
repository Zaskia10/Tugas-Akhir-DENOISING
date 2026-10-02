"""
UJI STATISTIK BERPASANGAN ANTAR MODEL
======================================
Menguatkan klaim stress test tanpa melatih ulang model.

Noise dibangkitkan ulang dengan SEED YANG SAMA seperti stress_test.py,
sehingga data yang diuji identik dengan yang menghasilkan kurva stress test.

Dua tingkat analisis, karena keduanya menjawab hal berbeda:

1. TINGKAT DENYUT (n = jumlah denyut PVC, ~3.200)
   - Flip Peak_index : uji McNemar eksak (pasangan denyut-lead, hanya yang stabil)
   - Korelasi QRS    : uji Wilcoxon signed-rank pada korelasi per denyut
   Kelemahan: denyut dari pasien yang sama tidak independen, sehingga
   p-value cenderung terlalu kecil. Laporkan bersama ukuran efeknya.

2. TINGKAT SUBJEK (n = 7 subjek uji)
   - Metrik dirata-rata per subjek, lalu dibandingkan antar model
   - Dilaporkan: berapa dari 7 subjek yang konsisten + Wilcoxon eksak
   Catatan: dengan n = 7, p-value dua sisi terkecil yang mungkin adalah
   2/128 = 0,0156. Jadi "7 dari 7 subjek" adalah bukti terkuat yang bisa
   diberikan data ini.

Semua p-value dikoreksi Holm untuk perbandingan berganda.

Cara pakai di Colab:

    %run uji_statistik.py
    hasil = jalankan()                       # ~15-25 menit (5 level SNR)
    hasil = jalankan(levels=(5, 0, -5))      # lebih cepat
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import binomtest, wilcoxon

from stress_test import siapkan, bangkitkan, PRE, MOR, LEVELS
from prapemrosesan_final import load_noise_bank
from analisis_morfologi import potong, f_peak_index
from pelatihan_model import PositionalEncoding

PASANGAN = [
    ("lstm", "attention"),
    ("lstm", "cdae"),
    ("attention", "cdae"),
    ("lstm", "noisy"),
    ("attention", "noisy"),
    ("cdae", "noisy"),
]
SEED_STRESS = 7      # HARUS sama dengan seed di stress_test.jalankan()


# =============================================================================
# STATISTIK
# =============================================================================
def mcnemar(flip_a, flip_b):
    """Uji McNemar eksak. flip_*: array bool berpasangan (True = berubah)."""
    n_a = int((flip_a & ~flip_b).sum())     # A berubah, B tidak
    n_b = int((~flip_a & flip_b).sum())     # B berubah, A tidak
    n = n_a + n_b
    p = 1.0 if n == 0 else binomtest(min(n_a, n_b), n, 0.5).pvalue
    return p, n_a, n_b


def wilcoxon_efek(x_a, x_b):
    """Wilcoxon signed-rank + rank-biserial berpasangan sebagai ukuran efek."""
    d = np.asarray(x_a) - np.asarray(x_b)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 1.0, 0.0
    p = wilcoxon(d).pvalue
    rk = pd.Series(np.abs(d)).rank().values
    w_pos, w_neg = rk[d > 0].sum(), rk[d < 0].sum()
    r_rb = (w_pos - w_neg) / (w_pos + w_neg)
    return float(p), float(r_rb)


def holm(p):
    """Koreksi Holm-Bonferroni."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def korelasi_per_denyut(w_clean, w_test):
    """Pearson per (denyut, lead) -> (n_beat, 12)."""
    a = w_clean - w_clean.mean(-1, keepdims=True)
    b = w_test - w_test.mean(-1, keepdims=True)
    num = (a * b).sum(-1)
    den = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1)) + 1e-12
    return num / den


# =============================================================================
# PIPELINE
# =============================================================================
def jalankan(levels=LEVELS, seed=SEED_STRESS, simpan=True):
    S = siapkan(PRE, MOR)
    D = MOR["data_dir"]
    keep = np.load(D + "valid_test.npy")
    subj_beat = np.load(D + "subject_test.npy")[keep][S["seg"]]
    subjek = np.unique(subj_beat)
    stabil = S["stabil"]
    print(f"Subjek uji: {len(subjek)} | denyut PVC: {len(subj_beat):,}\n")

    _, bank_test = load_noise_bank(PRE)
    models = {c: tf.keras.models.load_model(
                  MOR["model_dir"] + f"{c}_seed{MOR['seed']}_best.keras",
                  custom_objects={"PositionalEncoding": PositionalEncoding})
              for c in MOR["models"]}

    baris = []
    for snr in levels:
        print(f"=== SNR input {snr:+.0f} dB ===")
        Xn_s, sc = bangkitkan(S["clean"], bank_test, snr, PRE, seed)

        # flip & korelasi per denyut untuk tiap sumber
        flip, kor = {}, {}
        sumber = {"noisy": potong(Xn_s * sc, S["seg"], S["pos"], S["W"])}
        for c, m in models.items():
            pr = m.predict(Xn_s, batch_size=128, verbose=0) * sc
            sumber[c] = potong(pr, S["seg"], S["pos"], S["W"])
            del pr
        for k, w in sumber.items():
            flip[k] = f_peak_index(w) != S["pi_clean"]          # (n_beat, 12)
            kor[k] = korelasi_per_denyut(S["w_clean"], w)       # (n_beat, 12)

        for a, b in PASANGAN:
            # --- tingkat denyut ---
            p_flip, n_a, n_b = mcnemar(flip[a][stabil], flip[b][stabil])
            d_flip = 100 * (flip[a][stabil].mean() - flip[b][stabil].mean())
            p_kor, r_kor = wilcoxon_efek(kor[a].mean(1), kor[b].mean(1))
            d_kor = float(np.median(kor[a].mean(1) - kor[b].mean(1)))

            # --- tingkat subjek ---
            fa = np.array([flip[a][subj_beat == s][stabil[subj_beat == s]].mean()
                           for s in subjek])
            fb = np.array([flip[b][subj_beat == s][stabil[subj_beat == s]].mean()
                           for s in subjek])
            ka = np.array([kor[a][subj_beat == s].mean() for s in subjek])
            kb = np.array([kor[b][subj_beat == s].mean() for s in subjek])
            menang_flip = int((fa < fb).sum())                  # flip: makin kecil makin baik
            menang_kor = int((ka > kb).sum())                   # korelasi: makin besar
            ps_flip = wilcoxon_efek(fa, fb)[0] if len(subjek) > 1 else np.nan
            ps_kor = wilcoxon_efek(ka, kb)[0] if len(subjek) > 1 else np.nan

            baris.append(dict(
                snr=snr, A=a, B=b,
                flip_selisih_pp=d_flip, flip_A_saja=n_a, flip_B_saja=n_b,
                p_flip_denyut=p_flip,
                flip_menang_subjek=f"{menang_flip}/{len(subjek)}",
                p_flip_subjek=ps_flip,
                kor_selisih_median=d_kor, kor_rank_biserial=r_kor,
                p_kor_denyut=p_kor,
                kor_menang_subjek=f"{menang_kor}/{len(subjek)}",
                p_kor_subjek=ps_kor,
            ))
        del Xn_s, sumber, flip, kor

    df = pd.DataFrame(baris)
    for kol in ("p_flip_denyut", "p_kor_denyut", "p_flip_subjek", "p_kor_subjek"):
        df[kol + "_holm"] = holm(df[kol].fillna(1.0).values)

    def tanda(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

    print("\n" + "=" * 92)
    print("FLIP RATE Peak_index  (selisih = A - B, poin persen; negatif = A lebih baik)")
    print("=" * 92)
    for _, r in df.iterrows():
        print(f"{r.snr:+4.0f} dB  {r.A:9s} vs {r.B:9s} | selisih {r.flip_selisih_pp:+6.2f} pp | "
              f"denyut p={r.p_flip_denyut_holm:.1e} {tanda(r.p_flip_denyut_holm):3s} | "
              f"subjek {r.flip_menang_subjek} p={r.p_flip_subjek_holm:.3f} "
              f"{tanda(r.p_flip_subjek_holm)}")

    print("\n" + "=" * 92)
    print("KORELASI QRS  (selisih median = A - B; positif = A lebih baik)")
    print("=" * 92)
    for _, r in df.iterrows():
        print(f"{r.snr:+4.0f} dB  {r.A:9s} vs {r.B:9s} | selisih {r.kor_selisih_median:+.4f} "
              f"(r={r.kor_rank_biserial:+.2f}) | "
              f"denyut p={r.p_kor_denyut_holm:.1e} {tanda(r.p_kor_denyut_holm):3s} | "
              f"subjek {r.kor_menang_subjek} p={r.p_kor_subjek_holm:.3f} "
              f"{tanda(r.p_kor_subjek_holm)}")

    print("\nKeterangan: p sudah dikoreksi Holm. *** p<0,001  ** p<0,01  * p<0,05  ns tidak signifikan")
    print(f"Tingkat subjek: n={len(subjek)}, p dua sisi terkecil yang mungkin = "
          f"{2/2**len(subjek):.4f}")

    if simpan:
        df.to_csv(MOR["model_dir"] + "uji_statistik.csv", index=False)
        print(f"\nTersimpan: {MOR['model_dir']}uji_statistik.csv")
    return df


if __name__ == "__main__":
    print("Definisi dimuat. Jalankan: hasil = jalankan()")
