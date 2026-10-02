"""
PELATIHAN MODEL — TUGAS AKHIR
Kerangka bersama untuk CDAE / LSTM-AE / Attention-AE
=====================================================

Satu skrip untuk ketiga model. Stem, decoder, output layer, dan protokol
pelatihan IDENTIK; hanya blok inti (core) yang ditukar. Dengan begitu
perbedaan hasil benar-benar berasal dari mekanisme yang dibandingkan.

    Input (512, 12)
      -> STEM     Conv1D(32,5,s=2) -> Conv1D(64,5,s=2)     512 -> 256 -> 128
      -> CORE     'cdae' | 'lstm' | 'attention'
      -> DECODER  Up(2) -> Conv1D(64,5) -> Up(2) -> Conv1D(32,5)
      -> OUTPUT   Conv1D(12,5, activation='linear')

Cara pakai di Colab:

    # Sel 1
    %run pelatihan_model.py

    # Sel 2 — latih CDAE
    hist = train('cdae', seed=42)

    # Sel 3 — nanti, model lain
    hist = train('lstm', seed=42)
    hist = train('attention', seed=42)

Keluaran per model, di CONFIG['model_dir']:
    {nama}_seed{n}_best.keras   bobot terbaik (val_loss terendah)
    {nama}_seed{n}_history.json riwayat loss untuk kurva di Bab 4
"""

import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# =============================================================================
# KONFIGURASI
# =============================================================================
CONFIG = {
    # V3 = noise di level elektroda
    "data_dir":  "/content/drive/MyDrive/Dataset/PROCESSED_V3/",
    "model_dir": "/content/drive/MyDrive/Dataset/MODELS_V3/",

    "window": 512,
    "n_leads": 12,

    # Di V3 penyaringan amplitudo sudah terintegrasi di prapemrosesan,
    # jadi tidak ada lagi berkas _f.
    "use_filtered": False,

    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "patience": 12,

    # Kalau Colab terlalu lambat, ambil sebagian data latih secara acak.
    # None = pakai semua. Test set SELALU utuh.
    "subsample_train": None,
}


# =============================================================================
# 1. DATA
# =============================================================================
def load_xy(split, cfg=CONFIG):
    """train/val memakai _f, test tanpa _f."""
    suf = "_f" if (cfg["use_filtered"] and split in ("train", "val")) else ""
    d = cfg["data_dir"]
    noisy = np.load(d + f"X_{split}_noisy{suf}.npy")
    clean = np.load(d + f"X_{split}_clean{suf}.npy")
    return noisy, clean


def load_all(cfg=CONFIG):
    Xtr_n, Xtr_c = load_xy("train", cfg)
    Xva_n, Xva_c = load_xy("val", cfg)

    if cfg["subsample_train"] and cfg["subsample_train"] < len(Xtr_n):
        rng = np.random.default_rng(0)          # tetap, agar semua model sama
        idx = rng.choice(len(Xtr_n), cfg["subsample_train"], replace=False)
        idx.sort()
        Xtr_n, Xtr_c = Xtr_n[idx], Xtr_c[idx]
        print(f"  subsample train -> {len(Xtr_n):,}")

    print(f"  train {Xtr_n.shape} | val {Xva_n.shape}")
    return Xtr_n, Xtr_c, Xva_n, Xva_c


# =============================================================================
# 2. BLOK INTI
# =============================================================================
def core_cdae(x, width=64):
    """Konvolusi berdilasi. Receptive field core ~100 sampel input (~390 ms),
    cukup untuk QRS PVC yang lebar. Versi lama hanya ~78 ms."""
    for d in (1, 2):
        x = layers.Conv1D(width, 5, padding="same", dilation_rate=d,
                          activation="relu")(x)
    return x


def core_lstm(x, units=32):
    """BiLSTM dengan aktivasi tanh bawaan.

    PENTING: jangan pakai activation='relu' di LSTM. Itu penyebab loss NaN
    di percobaan sebelumnya — state dijumlahkan sepanjang ratusan langkah
    waktu lalu meluap. Tanh juga mengaktifkan kernel cuDNN, jauh lebih cepat.

    return_sequences=True agar resolusi temporal tidak diperas jadi satu
    vektor. Pola RepeatVector cocok untuk deteksi anomali, bukan denoising.
    """
    for _ in range(2):
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    return x


class PositionalEncoding(layers.Layer):
    """Self-attention tidak mengenal urutan, jadi posisi harus disuntikkan."""

    def call(self, x):
        length = tf.shape(x)[1]
        depth = x.shape[-1]
        pos = tf.cast(tf.range(length)[:, None], tf.float32)
        i = tf.cast(tf.range(depth)[None, :], tf.float32)
        rate = 1.0 / tf.pow(10000.0, (2.0 * (i // 2)) / tf.cast(depth, tf.float32))
        ang = pos * rate
        pe = tf.where(tf.cast(tf.range(depth) % 2, tf.bool)[None, :],
                      tf.cos(ang), tf.sin(ang))
        return x + pe[None, ...]

    def compute_output_shape(self, s):
        return s


def core_attention(x, num_heads=4, key_dim=16, ff=128):
    """Satu blok Transformer encoder (pre-norm), dijalankan pada resolusi
    terdownsample (128), bukan 512 — kompleksitas attention kuadratik."""
    x = PositionalEncoding()(x)
    h = layers.LayerNormalization()(x)
    h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(h, h)
    x = layers.Add()([x, h])
    h = layers.LayerNormalization()(x)
    h = layers.Dense(ff, activation="relu")(h)
    h = layers.Dense(x.shape[-1])(h)
    return layers.Add()([x, h])


CORES = {"cdae": core_cdae, "lstm": core_lstm, "attention": core_attention}


# =============================================================================
# 3. PERAKITAN
# =============================================================================
def build_model(core_name, cfg=CONFIG):
    if core_name not in CORES:
        raise ValueError(f"core harus salah satu dari {list(CORES)}")

    inp = layers.Input((cfg["window"], cfg["n_leads"]), name="input_ecg")

    # STEM — identik untuk ketiga model
    x = layers.Conv1D(32, 5, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv1D(64, 5, strides=2, padding="same", activation="relu")(x)

    # CORE — satu-satunya bagian yang berbeda
    x = CORES[core_name](x)

    # DECODER — identik
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)

    # OUTPUT — 'linear', BUKAN 'tanh'. Tanh hanya mencapai +-1 secara
    # asimtotik sehingga puncak R cenderung terpotong, padahal puncak R
    # justru fitur yang ingin dijaga.
    out = layers.Conv1D(cfg["n_leads"], 5, padding="same",
                        activation="linear", name="output_ecg")(x)

    return Model(inp, out, name=f"{core_name.upper()}_AE")


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


# =============================================================================
# 4. PELATIHAN
# =============================================================================
def train(core_name, seed=42, cfg=CONFIG, data=None, verbose=1):
    os.makedirs(cfg["model_dir"], exist_ok=True)
    tag = f"{core_name}_seed{seed}"
    ckpt = os.path.join(cfg["model_dir"], f"{tag}_best.keras")

    print("=" * 62)
    print(f"PELATIHAN {core_name.upper()}  (seed {seed})")
    print("=" * 62)

    set_seed(seed)
    Xtr_n, Xtr_c, Xva_n, Xva_c = data if data else load_all(cfg)

    model = build_model(core_name, cfg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"], clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )
    print(f"  parameter: {model.count_params():,}")

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),          # hentikan segera bila NaN
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         patience=cfg["patience"],
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=5, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss",
                                           save_best_only=True),
    ]

    hist = model.fit(
        Xtr_n, Xtr_c,
        validation_data=(Xva_n, Xva_c),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=verbose,
    )

    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    h["core"] = core_name
    h["seed"] = seed
    h["params"] = int(model.count_params())
    h["n_train"] = int(len(Xtr_n))
    with open(os.path.join(cfg["model_dir"], f"{tag}_history.json"), "w") as f:
        json.dump(h, f, indent=2)

    best = min(h["val_loss"])
    print(f"\n  val_loss terbaik : {best:.6f} (epoch {h['val_loss'].index(best)+1})")
    print(f"  epoch berjalan   : {len(h['val_loss'])}")
    if len(h["val_loss"]) == cfg["epochs"]:
        print("  ! Mencapai batas epoch — model mungkin masih bisa membaik.")
    print(f"  tersimpan        : {ckpt}")
    return h


def train_all(cores=("cdae", "lstm", "attention"), seeds=(42,), cfg=CONFIG):
    """Muat data sekali, pakai untuk semua model — hemat RAM dan waktu."""
    data = load_all(cfg)
    out = {}
    for c in cores:
        for s in seeds:
            out[f"{c}_seed{s}"] = train(c, seed=s, cfg=cfg, data=data)
    return out


# =============================================================================
# 5. PEMERIKSAAN CEPAT
# =============================================================================
def compare_params(cfg=CONFIG):
    """Anggaran parameter ketiga model harus setara (dalam ~20%)."""
    print(f"{'model':16s} {'parameter':>12s}")
    for c in CORES:
        m = build_model(c, cfg)
        print(f"{m.name:16s} {m.count_params():12,}")


def quick_look(core_name, seed=42, sample=0, cfg=CONFIG):
    """Plot noisy vs denoised vs clean untuk beberapa lead."""
    import matplotlib.pyplot as plt

    model = tf.keras.models.load_model(
        os.path.join(cfg["model_dir"], f"{core_name}_seed{seed}_best.keras"),
        custom_objects={"PositionalEncoding": PositionalEncoding},
    )
    Xn, Xc = load_xy("test", cfg)
    pred = model.predict(Xn[sample:sample + 1], verbose=0)[0]

    leads = {0: "Lead I", 1: "Lead II", 6: "V1", 10: "V5"}
    fig, ax = plt.subplots(len(leads), 1, figsize=(13, 2.2 * len(leads)))
    for a, (li, nm) in zip(ax, leads.items()):
        a.plot(Xn[sample, :, li], color="red", alpha=.45, lw=.8, label="Noisy")
        a.plot(pred[:, li], color="green", lw=1.6, label="Denoised")
        a.plot(Xc[sample, :, li], color="blue", ls="--", lw=1.1, label="Clean")
        a.set_ylabel(nm, fontsize=9)
        a.legend(fontsize=7, loc="upper right")
    ax[0].set_title(f"{core_name.upper()} — sampel uji #{sample}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_params()
