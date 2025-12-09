#!/usr/bin/env python
"""
Script principal pour charger GT + prédictions et lancer l'évaluation Vid2Seq.
"""

import argparse

from src.gt_parsing import load_ground_truth
from src.data_loading import load_predictions
from src.metrics_eval import eval_corpus, compute_text_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-root",
        type=str,
        required=True,
        help="Racine des annotations INA (un dossier par vidéo).",
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        required=True,
        help="Dossier contenant les CSV de prédiction.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Chemin du CSV de sortie pour les scores par vidéo.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Chargement du ground truth.
    gt = load_ground_truth(args.gt_root)
    gt_segments = gt.segments_us
    gt_captions = gt.captions
    splits = gt.splits
    keys = gt.keys

    # Chargement des prédictions.
    pred_segments, pred_caps, pred_keys = load_predictions(args.pred_root)

    # L'évaluation suppose le même ordre de clés.
    assert pred_keys == keys, "Clés GT / prédictions désalignées"

    metrics, df = eval_corpus(
        pred_segments=pred_segments,
        pred_captions=pred_caps,
        gt_segments=gt_segments,
        gt_captions=gt_captions,
        splits=splits,
        keys=keys,
    )

    # Calcul des métriques textuelles supplémentaires.
    text_metrics = compute_text_metrics(pred_caps, gt_captions)

    print("=== Résumé corpus (moyennes) ===")
    for k, v in metrics.items():
        # metrics[k] est souvent un tableau par vidéo ; on prend la moyenne.
        if hasattr(v, "__len__"):
            val = float(sum(v) / len(v))
        else:
            val = float(v)
        print(f"{k}: {val * 100:.2f}")

    print("\n=== BLEU / ROUGE-L ===")
    for k, v in text_metrics.items():
        print(f"{k}: {v:.2f}")

    if args.out is not None:
        df.to_csv(args.out, index=False)
        print(f"\nScores détaillés sauvegardés dans {args.out}")


if __name__ == "__main__":
    main()
