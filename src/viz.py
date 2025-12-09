"""
Fonctions de visualisation des statistiques et des scores d'évaluation.
"""

import numpy as np
import matplotlib.pyplot as plt


# Affiche les histogrammes du nombre de captions / vidéo et des durées vidéo.
def plot_dataset_overview(gt_segments, title_prefix=""):
    nb_captions = [len(segs) for segs in gt_segments]
    durees_sec = [
        (segs[-1, 1] - segs[0, 0]) / 1e6 for segs in gt_segments
    ]

    plt.figure()
    plt.hist(nb_captions, bins=15)
    plt.xlabel("Captions par vidéo")
    plt.ylabel("Nombre de vidéos")
    plt.title(f"{title_prefix}Répartition du nombre de captions par vidéo")
    plt.tight_layout()

    plt.figure()
    plt.hist(durees_sec, bins=15)
    plt.xlabel("Durée des vidéos (secondes)")
    plt.ylabel("Nombre de vidéos")
    plt.title(f"{title_prefix}Répartition des durées des vidéos")
    plt.tight_layout()


# Affiche les distributions de longueur de captions (GT vs inférence).
def plot_caption_lengths(gt_captions, pred_captions):
    len_caps_gt = [
        len(str(c).split()) for vid in gt_captions for c in vid
    ]
    len_caps_pred = [
        len(str(c).split()) for vid in pred_captions for c in vid
    ]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(len_caps_gt, bins=30, alpha=0.7, label="GT")
    plt.hist(len_caps_pred, bins=30, alpha=0.7, label="Inférence")
    plt.xlabel("Longueur des captions (mots)")
    plt.ylabel("Fréquence")
    plt.title("Distribution des longueurs de captions")
    plt.legend()
    plt.grid(ls="--", alpha=0.4)

    plt.subplot(1, 2, 2)
    plt.hist(len_caps_gt, bins=30, alpha=0.7, label="GT")
    plt.hist(len_caps_pred, bins=30, alpha=0.7, label="Inférence")
    plt.xlabel("Longueur des captions (mots)")
    plt.ylabel("Fréquence")
    plt.title("Distribution des longueurs de captions (zoom)")
    plt.legend()
    plt.grid(ls="--", alpha=0.4)

    plt.tight_layout()


# Affiche la distribution des durées de segments (GT vs inférence).
def plot_segment_durations(gt_segments, pred_segments):
    def normalize_seg(seg):
        # Normalise un segment sous forme (start, end) en secondes.
        if isinstance(seg, (list, tuple, np.ndarray)):
            start, end = map(float, seg[:2])
        else:
            raise TypeError("type de segment non géré")
        if start > end:
            start, end = end, start
        dur = end - start
        if dur > 1e4:
            start /= 1e6
            end /= 1e6
            dur = end - start
        return dur

    len_segs_gt = [
        normalize_seg(s)
        for vid in gt_segments
        for s in vid
        if len(vid) > 0
    ]
    len_segs_pred = [
        normalize_seg(s)
        for vid in pred_segments
        for s in vid
        if len(vid) > 0
    ]

    plt.figure(figsize=(10, 4))
    plt.hist(len_segs_gt, bins=40, alpha=0.7, label="GT")
    plt.hist(len_segs_pred, bins=40, alpha=0.7, label="Inférence")
    plt.xlabel("Durée des segments (s)")
    plt.ylabel("Fréquence")
    plt.title("Distribution des durées de segments")
    plt.legend()
    plt.grid(ls="--", alpha=0.4)
    plt.tight_layout()


# Barres horizontales CIDER / METEOR par vidéo.
def plot_cider_meteor(df_scores, keys, title="CIDEr vs METEOR"):
    cider = df_scores["CIDER"] * 100.0
    meteor = df_scores["METEOR"] * 100.0
    y = np.arange(len(keys))

    plt.figure(figsize=(8, max(6, len(keys) * 0.3)))
    plt.barh(y - 0.15, cider, height=0.3, label="CIDEr")
    plt.barh(y + 0.15, meteor, height=0.3, label="METEOR")
    plt.yticks(y, keys)
    plt.xlabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
