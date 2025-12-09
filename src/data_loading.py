"""
Chargement des prédictions Vid2Seq à partir des CSV produits en sortie.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Charge des prédictions simples (une langue) depuis un dossier de CSV.
def load_predictions(results_dir, time_scale=1_000_000.0):
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(results_dir)

    predicted_segments = []
    predicted_captions = []
    keys = []

    for csv_file in sorted(results_dir.glob("*.csv")):
        key = csv_file.stem
        df = pd.read_csv(csv_file)

        # Segments [N, 2] en micro-secondes.
        segs_sec = df[["tc_start", "tc_end"]].values.astype(float)
        segs_us = (segs_sec * time_scale).astype(np.int64)

        # Légendes (colonne "caption").
        caps = [str(c).strip('"') for c in df["caption"].tolist()]

        predicted_segments.append(segs_us)
        predicted_captions.append(caps)
        keys.append(key)

    return predicted_segments, predicted_captions, keys


# Charge des prédictions avec légendes EN + FR (par exemple exports Excel).
def load_predictions_dual_language(csv_root, sep=";"):
    csv_root = Path(csv_root)
    if not csv_root.is_dir():
        raise FileNotFoundError(csv_root)

    pred_segments = []
    pred_caps_en = []
    pred_caps_fr = []
    keys = []

    for f in sorted(csv_root.rglob("*.csv")):
        df = pd.read_csv(f, sep=sep, quotechar='"', engine="python")

        # Conversion des timecodes en micro-secondes.
        segs_us = (
            df[["tc_start", "tc_end"]].to_numpy(dtype=float) * 1e6
        ).astype(np.int64)

        # Légendes en anglais et en français.
        en = df["caption"].fillna("").tolist()
        fr = df["caption_fr"].fillna("").tolist()

        pred_segments.append(segs_us)
        pred_caps_en.append(en)
        pred_caps_fr.append(fr)
        keys.append(f.stem)

    return pred_segments, pred_caps_en, pred_caps_fr, keys
