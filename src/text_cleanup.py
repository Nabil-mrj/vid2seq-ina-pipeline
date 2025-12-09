"""
Pipeline de nettoyage et de correction des légendes de prédiction.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Génère des sorties (traduction ou correction) par lots.
@torch.no_grad()
def batch_generate(texts, tok, model, bs=32, max_len=128):
    out = []
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        if not batch:
            continue
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        gen = model.generate(**enc, max_length=max_len)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


# Charge les modèles de correction EN / FR et la traduction EN→FR.
def load_correction_models():
    # Modèle de correction grammaticale anglais.
    tok_en = AutoTokenizer.from_pretrained(
        "vennify/t5-base-grammar-correction"
    )
    gec_en = AutoModelForSeq2SeqLM.from_pretrained(
        "vennify/t5-base-grammar-correction"
    )

    # Modèle de traduction EN→FR.
    tok_mt = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-en-fr"
    )
    mt_enfr = AutoModelForSeq2SeqLM.from_pretrained(
        "Helsinki-NLP/opus-mt-en-fr"
    )

    # Modèle de correction orthographique / grammaticale français.
    tok_fr = AutoTokenizer.from_pretrained(
        "fdemelo/t5-base-spell-correction-fr"
    )
    gec_fr = AutoModelForSeq2SeqLM.from_pretrained(
        "fdemelo/t5-base-spell-correction-fr"
    )

    return {
        "tok_en": tok_en,
        "gec_en": gec_en,
        "tok_mt": tok_mt,
        "mt_enfr": mt_enfr,
        "tok_fr": tok_fr,
        "gec_fr": gec_fr,
    }


# Applique un masque d'erreurs pour construire une version "nettoyée".
def apply_error_mask(pred_segments, pred_caps_fr, err_masks):
    clean_segments = []
    clean_caps = []
    removed = 0

    for segs, caps_fr_raw, mask in zip(
        pred_segments, pred_caps_fr, err_masks
    ):
        # mask est une liste de booléens : True = à supprimer.
        keep = [not m for m in mask]
        removed += sum(mask)

        if any(keep):
            clean_segments.append(segs[keep])
            clean_caps.append(
                [c for c, k in zip(caps_fr_raw, keep) if k]
            )
        else:
            # Cas où tout est supprimé : on garde des tableaux vides.
            clean_segments.append(
                np.empty((0, 2), dtype=np.int64)
            )
            clean_caps.append([])

    return clean_segments, clean_caps, removed


# Construit une version corrigée FR à partir des légendes EN.
def build_corrected_caps(
    pred_caps_en,
    tok_en,
    gec_en,
    tok_mt,
    mt_enfr,
    tok_fr,
    gec_fr,
):
    corrected_caps = []
    corr_en_cnt = 0
    corr_fr_cnt = 0

    for caps_en_raw in pred_caps_en:
        # Correction grammaticale en anglais.
        caps_en_corr = batch_generate(caps_en_raw, tok_en, gec_en)
        corr_en_cnt += sum(
            c0.strip() != c1.strip()
            for c0, c1 in zip(caps_en_raw, caps_en_corr)
        )

        # Traduction EN→FR.
        caps_fr_mt = batch_generate(caps_en_corr, tok_mt, mt_enfr)

        # Correction en français.
        caps_fr_corr = batch_generate(caps_fr_mt, tok_fr, gec_fr)
        corr_fr_cnt += sum(
            c0.strip() != c1.strip()
            for c0, c1 in zip(caps_fr_mt, caps_fr_corr)
        )

        corrected_caps.append(caps_fr_corr)

    return corrected_caps, corr_en_cnt, corr_fr_cnt


# Fonction utilitaire à compléter si besoin pour reconstruire un masque d'erreurs.
def build_error_mask_from_rules(pred_caps_fr):
    """
    Cette fonction est volontairement simple. Elle peut être complétée avec
    les règles utilisées pendant le stage (détection d'artefacts de traduction,
    phrases clairement hors sujet, caractères illisibles, etc.).

    Pour l'instant, on renvoie un masque "tout garder".
    """
    masks = []
    for caps in pred_caps_fr:
        masks.append([False for _ in caps])
    return masks
