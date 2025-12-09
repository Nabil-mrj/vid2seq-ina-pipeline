#!/usr/bin/env python
"""
Compare plusieurs runs (baseline, nettoyé, corrigé) en réutilisant le pipeline.
"""

import argparse
import pandas as pd

from src.gt_parsing import load_ground_truth
from src.data_loading import load_predictions
from src.metrics_eval import eval_corpus


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-root",
        type=str,
        required=True,
        help="Racine des annotations INA.",
    )
    parser.add_argument(
        "--pred-root-baseline",
        type=str,
        required=True,
        help="Dossier des prédictions baseline.",
    )
    parser.add_argument(
        "--pred-root-clean",
        type=str,
        default=None,
        help="Dossier des prédictions nettoyées (optionnel).",
    )
    parser.add_argument(
        "--pred-root-corrected",
        type=str,
        default=None,
        help="Dossier des prédictions corrigées (optionnel).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="CSV de sortie avec comparaison des scores.",
    )
    return parser.parse_args()


def run_single(name, gt, pred_root):
    pred_segments, pred_caps, pred_keys = load_predictions(pred_root)
    assert pred_keys == gt.keys, f"Clés GT / {name} désalignées"

    metrics, _ = eval_corpus(
        pred_segments=pred_segments,
        pred_captions=pred_caps,
        gt_segments=gt.segments_us,
        gt_captions=gt.captions,
        splits=gt.splits,
        keys=gt.keys,
    )

    # On résume chaque métrique par une moyenne sur les vidéos.
    summary = {f"{name}_{k}": float(sum(v) / len(v)) for k, v in metrics.items()}
    return summary


def main():
    args = parse_args()

    gt = load_ground_truth(args.gt_root)

    rows = []

    base_summary = run_single("baseline", gt, args.pred_root_baseline)
    rows.append(base_summary)

    if args.pred_root_clean:
        clean_summary = run_single("clean", gt, args.pred_root_clean)
        rows.append(clean_summary)

    if args.pred_root_corrected:
        corr_summary = run_single("corr2x", gt, args.pred_root_corrected)
        rows.append(corr_summary)

    df = pd.DataFrame(rows).T
    print(df)

    if args.out:
        df.to_csv(args.out)
        print(f"Comparaison sauvegardée dans {args.out}")


if __name__ == "__main__":
    main()
