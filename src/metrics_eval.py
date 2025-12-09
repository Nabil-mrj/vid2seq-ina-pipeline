"""
Fonctions de calcul des métriques d'évaluation (Vid2Seq + métriques texte).
"""

import numpy as np
import pandas as pd

from scenic.projects.vid2seq import dvc_eval
from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# Évalue un corpus avec la fonction officielle dvc_eval.evaluate_dense_captions.
def eval_corpus(
    pred_segments,
    pred_captions,
    gt_segments,
    gt_captions,
    splits,
    keys,
    iou_thresholds=(0.3, 0.5, 0.7, 0.9),
    soda=True,
    tmponly=False,
):
    # Dictionnaires GT / prédictions indexés par clé vidéo.
    gt_by_key = {
        keys[i]: (gt_segments[i], gt_captions[i], splits[i])
        for i in range(len(keys))
    }
    pred_by_key = {
        k: (s, c) for k, s, c in zip(keys, pred_segments, pred_captions)
    }

    sel_gt_segments = []
    sel_gt_captions = []
    sel_splits = []
    sel_pred_segments = []
    sel_pred_captions = []
    sel_keys = []

    for k, (seg_pred, cap_pred) in pred_by_key.items():
        if k not in gt_by_key:
            # On ignore les vidéos sans GT.
            continue
        seg_gt, cap_gt, split_gt = gt_by_key[k]
        sel_pred_segments.append(seg_pred)
        sel_pred_captions.append(cap_pred)
        sel_gt_segments.append(seg_gt)
        sel_gt_captions.append(cap_gt)
        sel_splits.append(split_gt)
        sel_keys.append(k)

    metrics = dvc_eval.evaluate_dense_captions(
        predicted_segments=sel_pred_segments,
        gt_segments=sel_gt_segments,
        predicted_captions=sel_pred_captions,
        gt_captions=sel_gt_captions,
        splits=sel_splits,
        keys=sel_keys,
        iou_thresholds=iou_thresholds,
        soda=soda,
        tmponly=tmponly,
    )

    # On renvoie à la fois le dict brut et un DataFrame pour analyse.
    df = pd.DataFrame({k: pd.Series(v) for k, v in metrics.items()})
    return metrics, df


# Calcule BLEU-4, BLEU-3 et ROUGE-L sur l'ensemble du corpus.
def compute_text_metrics(pred_captions, gt_captions):
    smooth = SmoothingFunction().method3

    hypotheses = []
    references = []

    # On parcourt vidéo par vidéo.
    for preds_video, refs_video in zip(pred_captions, gt_captions):
        for pred_seg, ref_seg in zip(preds_video, refs_video):
            # Hypothèse.
            hypotheses.append(
                wordpunct_tokenize(str(pred_seg).lower())
            )

            # Références (liste de listes).
            if isinstance(ref_seg, list):
                references.append(
                    [wordpunct_tokenize(str(r).lower()) for r in ref_seg]
                )
            else:
                references.append(
                    [wordpunct_tokenize(str(ref_seg).lower())]
                )

    # BLEU-4.
    bleu4 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )

    # BLEU-3.
    bleu3 = corpus_bleu(
        references,
        hypotheses,
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smooth,
    )

    # ROUGE-L (F-mesure).
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL_vals = []
    for refs, hyp in zip(references, hypotheses):
        ref_str = " ".join(refs[0])
        hyp_str = " ".join(hyp)
        score = scorer.score(ref_str, hyp_str)["rougeL"].fmeasure
        rougeL_vals.append(score)

    rougeL = float(np.mean(rougeL_vals)) if rougeL_vals else 0.0

    return {
        "BLEU4": bleu4 * 100.0,
        "BLEU3": bleu3 * 100.0,
        "ROUGEL": rougeL * 100.0,
    }
