"""
ANALISIS PRESERVASI MORFOLOGI PVC
==================================
Menilai denoiser bukan dari SNR/MSE saja, melainkan dari apakah fitur yang
dipakai metode lokalisasi PVC tetap utuh setelah denoising.

Fitur mengikuti Wang dkk. (2024), "A two-step classification machine learning
method for non-invasive localization of PVC origins based on 12-lead ECG",
J Interv Card Electrophysiol 67(3):457-470. Kombinasi terbaik di paper itu:
QRS_Int, Peak, RMS, VAR, Peak_index, SampEn — dihitung per lead pada jendela
120 ms sejak awal QRS.

Yang dilaporkan:
  * flip rate Peak_index  : % (beat, lead) yang nilainya BERUBAH setelah denoising.
                            Peak_index kategorikal (-1/0/1), jadi perubahannya
                            diskret — masukan lokalisasi langsung salah.
  * error relatif fitur   : |f_denoised - f_clean| / mean|f_clean| per lead
  * korelasi QRS          : Pearson antara QRS bersih dan hasil denoising

Cara pakai di Colab:

    %run analisis_morfologi.py
    hasil = jalankan()                 # ketiga model sekaligus
    hasil = jalankan(sampen=True)      # + SampEn (lebih lambat)

CATATAN IMPLEMENTASI (tulis di Bab 3):
  1. Anotasi INCART menandai puncak R, bukan awal QRS. Awal QRS didekati
     sebagai R - 40 ms.
  2. Persamaan (1) di paper menulis kondisi Peak_index dengan "||" (OR).
     Dibaca harfiah, hampir semua jendela dengan Max >= 0 akan bernilai 1,
     sehingga fitur kehilangan daya beda. Di sini dipakai pembacaan yang
     sesuai Gambar 3 paper, yaitu berbasis rasio dominasi puncak.
  3. Persamaan VAR dan RMS di paper menempatkan 1/n di luar akar. Di sini
     dipakai definisi baku (1/n di dalam akar). Karena yang dilaporkan
     adalah SELISIH fitur antara sinyal bersih dan hasil denoising,
     pilihan ini tidak mengubah kesimpulan.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

CFG = {
    "data_dir":  "/content/drive/MyDrive/Dataset/PROCESSED_V3/",
    "model_dir": "/content/drive/MyDrive/Dataset/MODELS_V3/",
    "models": ("cdae", "lstm", "attention"),
    "seed": 42,
    "fs": 257,
    "qrs_onset_ms": 40,     # awal QRS = R - 40 ms
    "qrs_window_ms": 120,   # panjang jendela fitur (Wang dkk.)
    "pvc_symbol": "V",
    "ratio_thr": 1.5,       # ambang dominasi puncak untuk Peak_index
}

LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


# =============================================================================
# FITUR (Wang dkk.)
# =============================================================================
def f_peak(w):
    """Peak: nilai puncak dominan. Persamaan (3)."""
    mx, mn = w.max(-1), w.min(-1)
    return np.where(np.abs(mx) > np.abs(mn), mx, mn)


def f_peak_inv(w):
    """Peak': puncak lawan dari Peak. Persamaan (4)."""
    mx, mn = w.max(-1), w.min(-1)
    return np.where(np.abs(mx) > np.abs(mn), mn, mx)


def f_qrs_int(w, fs):
    """QRS_Int: integral sinyal sepanjang 120 ms. Persamaan (2)."""
    return np.trapezoid(w, dx=1.0 / fs, axis=-1)


def f_rms(w):
    return np.sqrt(np.mean(w ** 2, axis=-1))


def f_var(w):
    return np.var(w, axis=-1)


def f_peak_index(w, thr=1.5):
    """Peak_index: +1 puncak positif dominan, -1 negatif dominan, 0 seimbang."""
    mx, mn = w.max(-1), w.min(-1)
    eps = 1e-12
    pos = (mx > 0) & (np.abs(mx) > thr * np.abs(mn) + eps)
    neg = (mn < 0) & (np.abs(mn) > thr * np.abs(mx) + eps)
    return np.where(pos, 1, np.where(neg, -1, 0)).astype(np.int8)


def f_sampen(w, m=2, r_coef=0.2, chunk=4000):
    """Sample entropy per jendela. Lambat, jadi opsional.

    Jendela 120 ms hanya 31 sampel pada 257 Hz, sehingga SampEn di sini
    kurang stabil dibanding pemakaian umumnya. Laporkan dengan catatan itu.
    """
    shape = w.shape[:-1]
    x = w.reshape(-1, w.shape[-1])
    n, W = x.shape
    out = np.empty(n)
    for s in range(0, n, chunk):
        b = x[s:s + chunk]
        r = r_coef * b.std(axis=1, keepdims=True)
        cnt = []
        for mm in (m, m + 1):
            k = W - mm + 1
            idx = np.arange(k)[:, None] + np.arange(mm)[None, :]
            tem = b[:, idx]                                   # (B, k, mm)
            d = np.abs(tem[:, :, None, :] - tem[:, None, :, :]).max(-1)
            eye = np.eye(k, dtype=bool)[None, :, :]
            cnt.append((( d <= r[:, :, None]) & ~eye).sum(axis=(1, 2)))
        with np.errstate(divide='ignore', invalid='ignore'):
            out[s:s + chunk] = -np.log(cnt[1] / cnt[0])
    out[~np.isfinite(out)] = np.nan
    return out.reshape(shape)


# =============================================================================
# EKSTRAKSI JENDELA QRS
# =============================================================================
def qrs_windows(sig_mv, rpeaks, symbols, keep, cfg=CFG):
    """Potong jendela 120 ms dari tiap denyut PVC.

    Mengembalikan (indeks_segmen, posisi_r, array (n_beat, 12, W)).
    Denyut yang jendelanya keluar batas segmen dilewati.
    """
    fs = cfg["fs"]
    off = int(round(cfg["qrs_onset_ms"] * fs / 1000))
    W = int(round(cfg["qrs_window_ms"] * fs / 1000))
    L = sig_mv.shape[1]

    seg_idx, beat_pos = [], []
    for i in np.where(keep)[0]:
        for r, s in zip(rpeaks[i], symbols[i]):
            if s != cfg["pvc_symbol"]:
                continue
            a = int(r) - off
            if a < 0 or a + W > L:
                continue
            seg_idx.append(i); beat_pos.append(a)
    seg_idx = np.asarray(seg_idx); beat_pos = np.asarray(beat_pos)

    take = np.arange(W)[None, :] + beat_pos[:, None]          # (n_beat, W)
    win = sig_mv[seg_idx[:, None], take]                      # (n_beat, W, 12)
    return seg_idx, beat_pos, np.transpose(win, (0, 2, 1))    # -> (n_beat, 12, W)


def potong(sig_mv, seg_idx, beat_pos, W):
    take = np.arange(W)[None, :] + beat_pos[:, None]
    return np.transpose(sig_mv[seg_idx[:, None], take], (0, 2, 1))


# =============================================================================
# PERBANDINGAN FITUR
# =============================================================================
def bandingkan(w_clean, w_test, fs, sampen=False, thr=1.5):
    """Bandingkan fitur antara jendela bersih dan jendela uji.

    w_*: (n_beat, 12, W). Mengembalikan dict metrik per lead (panjang 12).
    """
    out = {}

    # Peak_index: kategorikal -> flip rate
    pi_c = f_peak_index(w_clean, thr)
    pi_t = f_peak_index(w_test, thr)
    out["flip_peak_index"] = (pi_c != pi_t).mean(axis=0) * 100

    # fitur kontinu: error relatif terhadap skala fitur bersih
    kont = {
        "QRS_Int": f_qrs_int(w_clean, fs),
        "Peak":    f_peak(w_clean),
        "Peak_inv": f_peak_inv(w_clean),
        "RMS":     f_rms(w_clean),
        "VAR":     f_var(w_clean),
    }
    kont_t = {
        "QRS_Int": f_qrs_int(w_test, fs),
        "Peak":    f_peak(w_test),
        "Peak_inv": f_peak_inv(w_test),
        "RMS":     f_rms(w_test),
        "VAR":     f_var(w_test),
    }
    if sampen:
        kont["SampEn"] = f_sampen(w_clean)
        kont_t["SampEn"] = f_sampen(w_test)

    for k in kont:
        c, t = kont[k], kont_t[k]
        skala = np.nanmean(np.abs(c), axis=0) + 1e-12
        out[f"err_{k}"] = np.nanmean(np.abs(t - c), axis=0) / skala * 100

    # korelasi bentuk QRS (Pearson, lag 0 — filtfilt zero-phase jadi tidak bergeser)
    a = w_clean - w_clean.mean(-1, keepdims=True)
    b = w_test - w_test.mean(-1, keepdims=True)
    num = (a * b).sum(-1)
    den = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1)) + 1e-12
    out["korelasi_QRS"] = (num / den).mean(axis=0)
    return out


# =============================================================================
# PIPELINE
# =============================================================================
def jalankan(cfg=CFG, sampen=False, simpan=True):
    D, MD = cfg["data_dir"], cfg["model_dir"]
    fs = cfg["fs"]
    W = int(round(cfg["qrs_window_ms"] * fs / 1000))

    Xn0 = np.load(D + "X_test_noisy.npy")
    Xc0 = np.load(D + "X_test_clean.npy")
    sc = np.load(D + "scale_test.npy").reshape(-1, 1, 1)
    keep = np.load(D + "valid_test.npy")
    rpk = np.load(D + "rpeak_test.npy", allow_pickle=True)
    sym = np.load(D + "symbol_test.npy", allow_pickle=True)

    Xn, Xc = Xn0 * sc, Xc0 * sc
    seg_idx, beat_pos, w_clean = qrs_windows(Xc, rpk, sym, keep, cfg)
    print(f"Denyut PVC yang dianalisis: {len(seg_idx):,} "
          f"dari {keep.sum():,} segmen valid\n")
    if len(seg_idx) == 0:
        raise ValueError("Tidak ada denyut PVC yang memenuhi syarat jendela.")

    w_noisy = potong(Xn, seg_idx, beat_pos, W)

    baris, per_lead = [], {}
    sumber = [("noisy", w_noisy)]
    for core in cfg["models"]:
        from pelatihan_model import PositionalEncoding
        m = tf.keras.models.load_model(
            MD + f"{core}_seed{cfg['seed']}_best.keras",
            custom_objects={"PositionalEncoding": PositionalEncoding})
        pr = m.predict(Xn0, batch_size=128, verbose=0) * sc
        sumber.append((core, potong(pr, seg_idx, beat_pos, W)))
        del pr

    for nama, w in sumber:
        r = bandingkan(w_clean, w, fs, sampen=sampen, thr=cfg["ratio_thr"])
        per_lead[nama] = r
        baris.append(dict(
            sumber=nama,
            flip_peak_index=r["flip_peak_index"].mean(),
            korelasi_QRS=r["korelasi_QRS"].mean(),
            **{k: v.mean() for k, v in r.items()
               if k.startswith("err_")}
        ))

    df = pd.DataFrame(baris).set_index("sumber")
    print("RATA-RATA 12 LEAD (nilai err_* dan flip dalam %, makin kecil makin baik)")
    print(df.round(2).to_string(), "\n")

    print("FLIP RATE Peak_index PER LEAD (%)")
    fl = pd.DataFrame({k: v["flip_peak_index"] for k, v in per_lead.items()},
                      index=LEADS)
    print(fl.round(1).to_string(), "\n")

    print("KORELASI BENTUK QRS PER LEAD")
    ko = pd.DataFrame({k: v["korelasi_QRS"] for k, v in per_lead.items()},
                      index=LEADS)
    print(ko.round(3).to_string())

    if simpan:
        df.to_csv(MD + "morfologi_ringkas.csv")
        fl.to_csv(MD + "morfologi_flip_peakindex.csv")
        ko.to_csv(MD + "morfologi_korelasi_qrs.csv")
        print(f"\nTersimpan di {MD}")

    return {"ringkas": df, "flip": fl, "korelasi": ko, "per_lead": per_lead,
            "n_beat": len(seg_idx)}


if __name__ == "__main__":
    print("Definisi dimuat. Jalankan: hasil = jalankan()")
