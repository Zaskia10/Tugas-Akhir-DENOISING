"""
PRAPEMROSESAN DATA — TUGAS AKHIR
Studi Komparatif Autoencoder untuk Reduksi Noise EKG 12-Lead (INCART + MIT-BIH NSTDB)
=====================================================================================

Menggantikan pipeline versi pertama. Setiap perbaikan diberi tanda [FIX n.n]
yang merujuk ke nomor checklist.

Versi 3: noise disuntikkan di level ELEKTRODA (lihat inject_noise), dan
penyaringan amplitudo sudah terintegrasi — tidak perlu sel _f terpisah.

Cara pakai di Google Colab (masing-masing di SEL TERPISAH):

    !pip install wfdb -q

    from google.colab import drive
    drive.mount('/content/drive')

    %run prapemrosesan_final.py
    sets = run(CONFIG)

Keluaran di CONFIG['save_dir']:
    X_{split}_noisy.npy   (N, 512, 12) float32  — input model, sudah diskalakan
    X_{split}_clean.npy   (N, 512, 12) float32  — target model, skala sama
    scale_{split}.npy     (N,)         float32  — faktor untuk kembali ke mV
    snr_{split}.npy       (N,)         float32  — SNR target yang disuntikkan
    snrmeas_{split}.npy   (N,)         float32  — SNR terukur (verifikasi)
    rpeak_{split}.npy     object               — posisi R relatif tiap segmen
    symbol_{split}.npy    object               — simbol beat tiap segmen
    pvc_{split}.npy       (N,)         bool    — segmen memuat >=1 beat 'V'
    record_{split}.npy    (N,)         str     — asal rekaman
    subject_{split}.npy   (N,)         str     — asal subjek
    meta.json                                  — konfigurasi + daftar subjek
"""

import os
import re
import gc
import glob
import json
from collections import Counter, defaultdict

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from sklearn.model_selection import GroupShuffleSplit


# =============================================================================
# KONFIGURASI  [FIX 5.4] semua angka terkumpul di sini, tidak tersebar
# =============================================================================
CONFIG = {
    # --- path ---
    "incart_dir": "/content/drive/MyDrive/Dataset/INCART/",
    "noise_dir":  "/content/drive/MyDrive/Dataset/NOISE/",
    # V3 = noise di level elektroda. Folder baru agar tidak tercampur
    # dengan data lama (PROCESSED_FINAL) yang memakai noise per lead.
    "save_dir":   "/content/drive/MyDrive/Dataset/PROCESSED_V3/",

    # --- penyaringan amplitudo (train & val saja, test TIDAK disaring) ---
    "amp_thresh": 5.0,

    # --- sinyal ---
    "fs": 257,                 # INCART
    "fs_noise": 360,           # NSTDB  [FIX 2.5] beda, wajib di-resample
    "window": 512,             # [FIX 2.17] genap: 512->256->128->256->512

    # --- filter ---
    "lowcut": 0.5,
    "highcut": 40.0,           # [FIX 2.1] bukan 50, agar tidak menyentuh jala-jala
    "notch_freq": 50.0,        # [FIX 2.2] INCART direkam di Rusia
    "notch_q": 30.0,

    # --- rekaman ---
    "n_records": None,         # None = semua 75; isi angka untuk membatasi
    "select_by_pvc": True,     # [FIX 1.3] kalau dibatasi, pilih yang PVC-nya banyak

    # --- noise ---
    "noise_kind": "mix",       # KEPUTUSAN 0.6 — 'mix' | 'bw' | 'ma' | 'em'
    "noise_split_ratio": 0.7,  # [FIX 2.9] 70% awal train/val, 30% akhir test
    "snr_train": (0.0, 10.0),  # KEPUTUSAN 0.5 — acak dalam rentang
    "snr_test": 5.0,           # tetap, untuk evaluasi baseline

    # --- morfologi PVC (fitur Wang dkk.) ---
    "qrs_onset_offset_ms": 40,  # KEPUTUSAN A — awal QRS = R - 40 ms
    "qrs_window_ms": 120,       # panjang jendela fitur Wang dkk.
    "beat_halfwidth_ms": 250,   # KEPUTUSAN B — jendela beat utuh untuk plot

    # --- split ---
    "test_frac": 0.2,
    "val_frac": 0.2,
    "seed": 42,                # [FIX 5.1]
}

PVC_SYMBOL = "V"
# Simbol yang dihitung sebagai "beat" (bukan penanda ritme seperti '+')
BEAT_SYMBOLS = set("NLRejAaJSVEFP/fQ")


# =============================================================================
# 1. FILTER
# =============================================================================
def build_filters(cfg):
    nyq = 0.5 * cfg["fs"]
    b_bp, a_bp = butter(2, [cfg["lowcut"] / nyq, cfg["highcut"] / nyq], btype="band")
    b_no, a_no = iirnotch(cfg["notch_freq"], cfg["notch_q"], cfg["fs"])
    return (b_bp, a_bp), (b_no, a_no)


def apply_filters(x, bp, no):
    """[FIX 2.3] diterapkan ke rekaman utuh, bukan per segmen.
       [FIX 2.4] filtfilt zero-phase, posisi puncak R tidak bergeser."""
    y = filtfilt(bp[0], bp[1], x, axis=0)
    y = filtfilt(no[0], no[1], y, axis=0)
    return y


# =============================================================================
# 2. BANK NOISE
# =============================================================================
def load_noise_bank(cfg):
    """[FIX 2.5] resample 360->257 Hz. [FIX 2.9] bagi temporal train/test."""
    train, test = {}, {}
    for kind in ("bw", "ma", "em"):
        path = os.path.join(cfg["noise_dir"], kind)
        sig, hdr = wfdb.rdsamp(path)

        if hdr["fs"] != cfg["fs_noise"]:
            print(f"  ! {kind}: fs={hdr['fs']}, bukan {cfg['fs_noise']}. Cek CONFIG.")

        sig = resample_poly(sig, cfg["fs"], hdr["fs"], axis=0)
        sig = sig - sig.mean(axis=0, keepdims=True)      # [FIX 2.8]

        cut = int(cfg["noise_split_ratio"] * len(sig))
        train[kind], test[kind] = sig[:cut], sig[cut:]
        print(f"  {kind}: {len(sig):,} sampel @ {cfg['fs']} Hz  "
              f"(train {cut:,} / test {len(sig)-cut:,})")
    return train, test


def draw_electrode_noise(bank, kind, length, rng, n_el=9):
    """Bangkitkan noise untuk n_el elektroda sekaligus.

    Dua jaminan yang dulu tidak ada:
      1. Jendela tiap elektroda diambil dari blok yang BERBEDA dan tidak
         tumpang tindih, sehingga antar elektroda benar-benar independen.
      2. Tiap elektroda dinormalisasi ke varians 1, sehingga rasio varians
         noise antar lead persis mengikuti rumus penurunan lead:
         ekstremitas 2.00 : augmented 1.50 : prekordial 1.33.
         Tanpa ini, rasio terukur meleset (0.83 vs 0.67 teoritis).
    """
    kinds = ("bw", "ma", "em") if kind == "mix" else (kind,)
    out = np.zeros((length, n_el))

    for k in kinds:
        src = bank[k]
        n_blocks = len(src) // length
        if n_blocks < n_el:
            raise ValueError(f"Rekaman noise '{k}' terlalu pendek: "
                             f"{n_blocks} blok, butuh {n_el}.")
        blocks = rng.choice(n_blocks, size=n_el, replace=False)   # tanpa tumpang tindih
        for e, b in enumerate(blocks):
            ch = rng.integers(0, src.shape[1])
            seg = src[b * length:(b + 1) * length, ch].astype(np.float64)
            seg = seg - seg.mean()
            sd = seg.std()
            if sd > 0:
                out[:, e] += rng.uniform(0.3, 1.0) * (seg / sd)   # [FIX 2.11]

    out -= out.mean(axis=0, keepdims=True)                        # [FIX 2.8]
    sd = out.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return out / sd                                               # varians 1 per elektroda


def inject_noise(clean_mv, bank, kind, snr_db, rng):
    """Noise disuntikkan di 9 ELEKTRODA, lalu diturunkan ke 12 lead.

    Alasan: di INCART, lead ekstremitas diturunkan dari 3 elektroda
    (terverifikasi: III = II - I tepat). Pada rekaman nyata noise masuk
    lewat elektroda, sehingga noise antar-lead ekstremitas terikat
    hubungan yang sama.
      - np.tile (versi lama)      : noise identik di 12 lead  -> terlalu berkorelasi
      - independen per lead (v2)  : lead ekstremitas tak terikat -> terlalu mudah
      - level elektroda (v3, ini) : struktur korelasi sesuai rekaman nyata

    Rumus penurunan lead:
      Einthoven  : I = LA-RA,  II = LL-RA,  III = LL-LA
      Goldberger : aVR = RA-(LA+LL)/2,  aVL = LA-(RA+LL)/2,  aVF = LL-(RA+LA)/2
      Wilson     : Vi = Vi_elektroda - WCT,  WCT = (RA+LA+LL)/3

    SNR didefinisikan GLOBAL atas 12 lead. Penskalaan per lead akan merusak
    struktur Einthoven, jadi satu faktor dipakai untuk semua lead dan SNR
    per lead dibiarkan bervariasi secara alami. Tulis definisi ini di Bab 3.
    """
    L = clean_mv.shape[0]
    e = draw_electrode_noise(bank, kind, L, rng, n_el=9)
    RA, LA, LL, V = e[:, 0], e[:, 1], e[:, 2], e[:, 3:]
    wct = (RA + LA + LL) / 3.0
    n = np.column_stack([
        LA - RA,                  # I
        LL - RA,                  # II
        LL - LA,                  # III
        RA - (LA + LL) / 2.0,     # aVR
        LA - (RA + LL) / 2.0,     # aVL
        LL - (RA + LA) / 2.0,     # aVF
        V - wct[:, None],         # V1..V6
    ])
    s = clean_mv - clean_mv.mean(axis=0, keepdims=True)
    p_s, p_n = np.mean(s ** 2), np.mean(n ** 2)
    if p_s <= 0 or p_n <= 0:
        return clean_mv.astype(np.float64).copy()
    return clean_mv + np.sqrt(p_s / (p_n * 10 ** (snr_db / 10.0))) * n


def measured_snr(clean_mv, noisy_mv):
    """SNR global atas 12 lead, konsisten dengan inject_noise. [FIX 6.2]"""
    s = clean_mv - clean_mv.mean(axis=0, keepdims=True)
    n = noisy_mv - clean_mv
    n = n - n.mean(axis=0, keepdims=True)
    p_s, p_n = np.mean(s ** 2), np.mean(n ** 2)
    return float(10 * np.log10(p_s / p_n)) if (p_s > 0 and p_n > 0) else np.nan


# =============================================================================
# 3. PENSKALAAN
# =============================================================================
def scale_pair(noisy_mv, clean_mv, eps=1e-8):
    """[FIX 2.12/2.13] SATU faktor untuk 12 lead — rasio antar-lead tetap utuh.
       Ini syarat mutlak agar fitur berdimensi (QRS_Int, Peak, RMS, VAR)
       milik Wang dkk. masih bisa dihitung.
       [FIX 2.14] faktor dihitung dari sinyal NOISY saja.
       [FIX 2.15] persentil, bukan max, agar spike EM tidak menenggelamkan sinyal.
       [FIX 2.16] faktor dikembalikan untuk disimpan."""
    s = float(np.percentile(np.abs(noisy_mv), 99.5)) + eps
    return (noisy_mv / s).astype(np.float32), (clean_mv / s).astype(np.float32), np.float32(s)


# =============================================================================
# 4. SUBJEK
# =============================================================================
def subject_of(header_comments):
    """Header INCART memuat 'patient N' — I01 dan I02 berasal dari pasien
    yang sama. Tanpa ini, split per subjek mustahil. [FIX 2.23]"""
    txt = " ".join(header_comments or [])
    m = re.search(r"patient\s*(\d+)", txt, re.IGNORECASE)
    return f"patient_{int(m.group(1)):02d}" if m else "UNKNOWN"


# =============================================================================
# 5. PEMBACAAN & SEGMENTASI
# =============================================================================
def list_records(cfg):
    """[FIX 5.3] sorted, bukan urutan glob yang acak."""
    files = sorted(glob.glob(os.path.join(cfg["incart_dir"], "*.hea")))
    assert files, f"Tidak ada .hea di {cfg['incart_dir']}"

    if cfg["n_records"] is None or cfg["n_records"] >= len(files):
        return files

    if not cfg["select_by_pvc"]:
        return files[: cfg["n_records"]]

    # [FIX 1.3] pilih rekaman terpadat PVC, tapi per SUBJEK utuh
    by_subj = defaultdict(list)
    for f in files:
        rec = f[:-4]
        pvc = Counter(wfdb.rdann(rec, "atr").symbol)[PVC_SYMBOL]
        by_subj[subject_of(wfdb.rdheader(rec).comments)].append((f, pvc))

    ranked = sorted(by_subj.items(), key=lambda kv: -sum(p for _, p in kv[1]))
    chosen = []
    for _, items in ranked:
        if len(chosen) >= cfg["n_records"]:
            break
        chosen += [f for f, _ in items]          # subjek diambil utuh
    return sorted(chosen)


def load_segments(cfg, files, bp, no):
    """Baca rekaman, filter, segmentasi, simpan anotasi. [FIX 2.18-2.21]"""
    W = cfg["window"]
    segs, subs, recs, rpks, syms, pvcs = [], [], [], [], [], []

    for i, path in enumerate(files, 1):
        rec = path[:-4]
        base = os.path.basename(rec)
        try:
            sig, hdr = wfdb.rdsamp(rec)          # [FIX 1.6] satuan mV, bukan ADU
            if hdr["fs"] != cfg["fs"]:
                print(f"  ! {base}: fs={hdr['fs']}, dilewati")
                continue

            ann = wfdb.rdann(rec, "atr")         # [FIX 2.18] sebelumnya tidak ada
            samp, symb = np.asarray(ann.sample), np.asarray(ann.symbol)
            beat = np.isin(symb, list(BEAT_SYMBOLS))
            samp, symb = samp[beat], symb[beat]  # buang '+' dan penanda non-beat

            filt = apply_filters(sig, bp, no)
            subj = subject_of(hdr["comments"])

            for k in range(len(filt) // W):
                a, b = k * W, (k + 1) * W
                m = (samp >= a) & (samp < b)
                if not m.any():                  # [FIX 2.22] segmen tanpa beat dibuang
                    continue
                segs.append(filt[a:b, :].astype(np.float32))
                subs.append(subj)
                recs.append(base)
                rpks.append((samp[m] - a).astype(np.int32))   # [FIX 2.19]
                syms.append(symb[m].astype("<U2"))            # [FIX 2.20]
                pvcs.append(bool((symb[m] == PVC_SYMBOL).any()))  # [FIX 2.21]

            del sig, filt
            gc.collect()
            print(f"  [{i}/{len(files)}] {base}  subjek={subj}")
        except Exception as e:
            print(f"  ! gagal {base}: {e}")

    return (np.stack(segs), np.array(subs), np.array(recs),
            np.array(rpks, dtype=object), np.array(syms, dtype=object),
            np.array(pvcs))


# =============================================================================
# 6. SPLIT PER SUBJEK
# =============================================================================
def split_by_subject(n, subjects, cfg):
    """[FIX 2.23] segmen dari subjek yang sama tidak boleh tersebar antar split."""
    idx = np.arange(n)
    g1 = GroupShuffleSplit(1, test_size=cfg["test_frac"], random_state=cfg["seed"])
    trv, te = next(g1.split(idx, groups=subjects))

    g2 = GroupShuffleSplit(1, test_size=cfg["val_frac"], random_state=cfg["seed"])
    a, b = next(g2.split(trv, groups=subjects[trv]))
    tr, va = trv[a], trv[b]

    # [FIX 2.24] assertion, bukan sekadar asumsi
    for name, x, y in (("train-test", tr, te), ("val-test", va, te), ("train-val", tr, va)):
        overlap = set(subjects[x]) & set(subjects[y])
        assert not overlap, f"Subjek bocor di {name}: {overlap}"
    return tr, va, te


# =============================================================================
# 7. PEMBENTUKAN SET
# =============================================================================
def build_set(X_mv, indices, bank, snr, kind, rng, label):
    n, W, C = len(indices), X_mv.shape[1], X_mv.shape[2]
    Xn = np.empty((n, W, C), np.float32)
    Xc = np.empty((n, W, C), np.float32)
    sc = np.empty(n, np.float32)
    st = np.empty(n, np.float32)
    sm = np.empty(n, np.float32)

    for j, k in enumerate(indices):
        target = rng.uniform(*snr) if isinstance(snr, tuple) else float(snr)
        clean = X_mv[k].astype(np.float64)
        noisy = inject_noise(clean, bank, kind, target, rng)
        sm[j] = measured_snr(clean, noisy)
        Xn[j], Xc[j], sc[j] = scale_pair(noisy, clean)
        st[j] = target

    print(f"  {label:5s} {Xn.shape}  SNR target {st.min():5.1f}..{st.max():5.1f} dB  "
          f"terukur {np.nanmean(sm):5.2f} dB")
    return Xn, Xc, sc, st, sm


# =============================================================================
# 8. SANITY CHECK
# =============================================================================
def sanity_checks(sets, cfg):
    print("\n" + "=" * 62)
    print("SANITY CHECK")
    print("=" * 62)

    for label, (Xn, Xc, sc, st, sm) in sets.items():
        # [FIX 6.2] SNR terukur harus mendekati target
        dev = np.nanmean(np.abs(sm - st))
        flag = "OK " if dev < 1.0 else "!! "
        print(f"{flag}{label:5s} |SNR terukur - target| = {dev:.2f} dB")

        # [FIX 6.3] struktur noise: lead ekstremitas harus mematuhi
        # Einthoven, lead prekordial tidak boleh identik satu sama lain
        nz = Xn[:200].astype(np.float64) - Xc[:200].astype(np.float64)
        scale_ref = np.abs(nz).max() + 1e-12
        ein = np.abs(nz[..., 2] - (nz[..., 1] - nz[..., 0])).max() / scale_ref
        pre = np.abs(nz[..., 6] - nz[..., 7]).max() / scale_ref
        ok = ein < 1e-3 and pre > 1e-2
        print(f"{'OK ' if ok else '!! '}{label:5s} struktur noise: "
              f"Einthoven {ein:.1e} (harus ~0) | V1 vs V2 {pre:.2f} (harus > 0)")

        # rasio varians noise per lead harus mengikuti rumus penurunan lead
        v = nz.var(axis=1).mean(axis=0)
        r_limb = float(v[:3].mean())
        r_aug = float(v[3:6].mean() / r_limb)
        r_pre = float(v[6:].mean() / r_limb)
        ok = abs(r_aug - 0.75) < 0.05 and abs(r_pre - 0.667) < 0.05
        print(f"{'OK ' if ok else '!! '}{label:5s} rasio varians noise: "
              f"ekstremitas 1.00 | augmented {r_aug:.3f} (0.75) | "
              f"prekordial {r_pre:.3f} (0.67)")

        # [FIX 6.5] NaN/Inf
        bad = (~np.isfinite(Xn)).sum() + (~np.isfinite(Xc)).sum()
        print(f"{'OK ' if bad == 0 else '!! '}{label:5s} NaN/Inf = {bad}")

        # [FIX 6.6] rentang amplitudo
        print(f"    {label:5s} |noisy| p99.5={np.percentile(np.abs(Xn),99.5):.3f}  "
              f"clean max={np.abs(Xc).max():.3f}  skala mV={sc.mean():.4f}")


# =============================================================================
# 9. MAIN
# =============================================================================
def run(cfg=CONFIG):
    rng = np.random.default_rng(cfg["seed"])          # [FIX 5.2]
    np.random.seed(cfg["seed"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    print("=" * 62); print("1. BANK NOISE"); print("=" * 62)
    bank_tr, bank_te = load_noise_bank(cfg)

    print("\n" + "=" * 62); print("2. PEMBACAAN REKAMAN"); print("=" * 62)
    files = list_records(cfg)
    print(f"{len(files)} rekaman terpilih\n")
    bp, no = build_filters(cfg)
    X_mv, subj, rec, rpk, sym, pvc = load_segments(cfg, files, bp, no)

    n_subj = len(np.unique(subj))
    print(f"\n{len(X_mv):,} segmen | {n_subj} subjek | {len(np.unique(rec))} rekaman")
    print(f"Segmen ber-PVC : {pvc.sum():,} ({100*pvc.mean():.1f}%)")
    assert "UNKNOWN" not in subj, "Ada rekaman tanpa 'patient N' di header."
    assert n_subj >= 5, f"Hanya {n_subj} subjek — terlalu sedikit untuk split per subjek."

    print("\n" + "=" * 62); print("3. SPLIT PER SUBJEK"); print("=" * 62)
    tr, va, te = split_by_subject(len(X_mv), subj, cfg)
    for nm, ix in (("train", tr), ("val", va), ("test", te)):
        print(f"  {nm:5s} {len(ix):6,} segmen | {len(set(subj[ix])):2d} subjek | "
              f"PVC {100*pvc[ix].mean():.1f}%")

    print("\n" + "=" * 62); print("4. INJEKSI NOISE"); print("=" * 62)
    sets = {
        "train": build_set(X_mv, tr, bank_tr, cfg["snr_train"], cfg["noise_kind"], rng, "train"),
        "val":   build_set(X_mv, va, bank_tr, cfg["snr_train"], cfg["noise_kind"], rng, "val"),
        "test":  build_set(X_mv, te, bank_te, cfg["snr_test"],  cfg["noise_kind"], rng, "test"),
    }

    # --- Penyaringan amplitudo: train & val saja. Test TIDAK disaring,
    #     karena menyaring data uji berdasarkan isinya = kebocoran. ---
    print(f"\nPenyaringan amplitudo (> {cfg['amp_thresh']}), train & val saja:")
    idx = {"train": tr, "val": va, "test": te}
    dropped = {}
    for name in ("train", "val"):
        keep = np.abs(sets[name][1]).max(axis=(1, 2)) <= cfg["amp_thresh"]
        dropped[name] = int((~keep).sum())
        sets[name] = tuple(a[keep] for a in sets[name])
        idx[name] = idx[name][keep]         # indeks ikut disaring -> tetap sejajar
        print(f"  {name:5s} buang {dropped[name]} -> sisa {keep.sum():,}")
    tr, va = idx["train"], idx["val"]

    sanity_checks(sets, cfg)

    print("\n" + "=" * 62); print("5. PENYIMPANAN"); print("=" * 62)
    d = cfg["save_dir"]
    for name, ix in (("train", tr), ("val", va), ("test", te)):
        Xn, Xc, sc, st, sm = sets[name]
        np.save(d + f"X_{name}_noisy.npy", Xn)
        np.save(d + f"X_{name}_clean.npy", Xc)
        np.save(d + f"scale_{name}.npy", sc)          # [FIX 2.16] wajib!
        np.save(d + f"snr_{name}.npy", st)
        np.save(d + f"snrmeas_{name}.npy", sm)
        np.save(d + f"rpeak_{name}.npy", rpk[ix], allow_pickle=True)
        np.save(d + f"symbol_{name}.npy", sym[ix], allow_pickle=True)
        np.save(d + f"pvc_{name}.npy", pvc[ix])
        np.save(d + f"record_{name}.npy", rec[ix])
        np.save(d + f"subject_{name}.npy", subj[ix])
        print(f"  {name} tersimpan")

    meta = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    meta.update({
        "n_segments": int(len(X_mv)),
        "n_subjects": int(n_subj),
        "n_records": int(len(np.unique(rec))),
        "subjects_train": sorted(set(subj[tr].tolist())),
        "subjects_val": sorted(set(subj[va].tolist())),
        "subjects_test": sorted(set(subj[te].tolist())),
        "wfdb_version": wfdb.__version__,
        "noise_model": "electrode-level (9 elektroda -> 12 lead), SNR global",
        "amp_dropped": dropped,
    })
    with open(d + "meta.json", "w") as f:
        json.dump(meta, f, indent=2)                  # [FIX 5.8]

    print(f"\nSelesai. Semua berkas ada di {d}")
    return sets


# =============================================================================
# 10. UTILITAS UNTUK TAHAP EVALUASI
# =============================================================================
def load_split(save_dir, name):
    """[FIX 5.7] memuat SEMUA berkas, termasuk test — cell reload lama tidak."""
    g = lambda s, p=False: np.load(save_dir + f"{s}_{name}.npy", allow_pickle=p) \
        if not s.startswith("X_") else np.load(save_dir + f"{s}_{name}.npy")
    return {
        "noisy":   np.load(save_dir + f"X_{name}_noisy.npy"),
        "clean":   np.load(save_dir + f"X_{name}_clean.npy"),
        "scale":   np.load(save_dir + f"scale_{name}.npy"),
        "snr":     np.load(save_dir + f"snr_{name}.npy"),
        "rpeak":   np.load(save_dir + f"rpeak_{name}.npy", allow_pickle=True),
        "symbol":  np.load(save_dir + f"symbol_{name}.npy", allow_pickle=True),
        "pvc":     np.load(save_dir + f"pvc_{name}.npy"),
        "record":  np.load(save_dir + f"record_{name}.npy"),
        "subject": np.load(save_dir + f"subject_{name}.npy"),
    }


def to_mv(x_scaled, scale):
    """Kembalikan ke mV. WAJIB sebelum menghitung SNR, PRD, dan fitur
    berdimensi (QRS_Int, Peak, RMS, VAR). [FIX 4.1]"""
    return x_scaled * scale.reshape(-1, 1, 1)


def qrs_windows(signal_mv, rpeaks, symbols, cfg=CONFIG, only_pvc=True):
    """Potong jendela 120 ms sejak awal QRS untuk fitur Wang dkk.

    KEPUTUSAN A: anotasi INCART menandai puncak R, bukan awal QRS, jadi
    awal QRS didekati sebagai R - 40 ms. Tulis asumsi ini di Bab 3.

    Mengembalikan array (n_beat, n_sampel, 12).
    """
    fs = cfg["fs"]
    off = int(round(cfg["qrs_onset_offset_ms"] * fs / 1000))
    wlen = int(round(cfg["qrs_window_ms"] * fs / 1000))
    L = signal_mv.shape[1]

    out = []
    for i, (rs, ss) in enumerate(zip(rpeaks, symbols)):
        for r, s in zip(rs, ss):
            if only_pvc and s != PVC_SYMBOL:
                continue
            a = int(r) - off
            if a < 0 or a + wlen > L:      # beat terpotong di batas segmen
                continue
            out.append(signal_mv[i, a:a + wlen, :])
    return np.stack(out) if out else np.empty((0, wlen, 12), np.float32)


if __name__ == "__main__":
    # %run hanya memuat definisi; pipeline dijalankan eksplisit dengan run(CONFIG)
    print("Definisi dimuat. Jalankan: sets = run(CONFIG)")
