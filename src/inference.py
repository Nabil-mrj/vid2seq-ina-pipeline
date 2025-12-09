#!/usr/bin/env python3
"""
Inférence Vid2Seq : features → segments + JSON (+ sous-titres optionnels).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from dmvr import tokenizers as dmvr_tokenizers
from flax.training import checkpoints
from scenic.projects.vid2seq import models
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

# Charge la config Scenic/Flax à partir d'un fichier Python.
def load_cfg(path):
    spec = importlib.util.spec_from_file_location("cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod.get_config()


# Retourne la durée en secondes via ffprobe.
def ffprobe_duration(path):
    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(res.stdout.strip())


# Formate un timestamp hh:mm:ss,mmm pour SRT.
def srt_ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# Convertit deux time-tokens en secondes (début, fin ou durée).
def bins_to_sec(t0, t1, dur_us, vocab, num_bins, abs_time):
    i0, i1 = t0 - vocab, t1 - vocab
    if abs_time or i0 >= num_bins or i1 >= num_bins:
        s_us = i0 * 1_000_000
        e_us = i1 * 1_000_000
    else:
        s_us = i0 * dur_us // (num_bins - 1)
        e_us = i1 * dur_us // (num_bins - 1)
    return s_us / 1e6, e_us / 1e6


# --------------------------------------------------------------------------- #
# 1) Vid2Seq inference → segments                                             #
# --------------------------------------------------------------------------- #

def vid2seq_segments(args):
    cfg = load_cfg(args.config)
    model = models.DenseVideoCaptioningModel(cfg, {})

    state = checkpoints.restore_checkpoint(args.checkpoint, None)
    if state is None:
        raise RuntimeError(f"Aucun checkpoint dans {args.checkpoint}")
    variables = {
        "params": state["optimizer"]["target"],
        **state.get("model_state", {}),
    }

    # ------------------------------------------------------------------- #
    # Chargement des features                                            #
    # ------------------------------------------------------------------- #
    feats = np.load(args.video_feats).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-7
    N = cfg.dataset_configs.num_frames

    if feats.shape[0] < N:
        pad = np.zeros((N - feats.shape[0], feats.shape[1]), feats.dtype)
        feats = np.concatenate([feats, pad], 0)

    feats = feats[:N][None]  # [1, N, D]
    encoder_inputs = {"features": jnp.asarray(feats)}

    # ------------------------------------------------------------------- #
    # ASR optionnel                                                      #
    # ------------------------------------------------------------------- #
    if args.asr:
        import pickle

        with open(args.asr, "rb") as fh:
            asr_obj = pickle.load(fh)

        if isinstance(asr_obj, str):
            asr_txt = asr_obj
        elif isinstance(asr_obj, dict) and "segments" in asr_obj:
            asr_txt = " ".join(s["text"] for s in asr_obj["segments"])
        elif isinstance(asr_obj, list) and asr_obj and "text" in asr_obj[0]:
            asr_txt = " ".join(s["text"] for s in asr_obj)
        else:
            asr_txt = str(asr_obj)

        spm = (
            getattr(cfg.dataset_configs, "vocabulary_path", None)
            or "scenic/projects/vid2seq/tokenizers/cc_all.32000/sentencepiece.model"
        )
        tok = dmvr_tokenizers.SentencePieceTokenizer(model_path=spm)
        ids = [0] + tok.string_to_indices(asr_txt)[
            : cfg.dataset_configs.max_num_input_words - 2
        ] + [1]
        encoder_inputs["text"] = jnp.asarray(ids, jnp.int32)[None, :]

    # ------------------------------------------------------------------- #
    # Décodage                                                           #
    # ------------------------------------------------------------------- #
    L = cfg.dataset_configs.max_num_output_words
    dummy = jnp.ones((1, L), jnp.int32)
    batch = {
        "encoder_inputs": encoder_inputs,
        "decoder_inputs": {
            "decoder_input_tokens": dummy,
            "decoder_target_tokens": dummy,
        },
    }

    decode_fn = (
        models.beam_search
        if cfg.decoding.decoding_method.lower() == "beamsearch"
        else models.temperature_sample
    )

    decodes, _ = model.predict_batch_with_aux(
        variables,
        batch,
        decode_fn,
        return_all_decodes=True,
        num_decodes=cfg.decoding.num_decodes,
        alpha=cfg.decoding.alpha,
        decoding_method=cfg.decoding.decoding_method,
        temperature=cfg.decoding.temperature,
        eos_id=1,
    )

    # ------------------------------------------------------------------- #
    # Post-traitement Vid2Seq → segments                                 #
    # ------------------------------------------------------------------- #
    tok = dmvr_tokenizers.SentencePieceTokenizer(
        model_path=(
            getattr(cfg.dataset_configs, "vocabulary_path", None)
            or "scenic/projects/vid2seq/tokenizers/cc_all.32000/sentencepiece.model"
        )
    )
    vocab_size = tok.vocab_size
    num_bins = cfg.dataset_configs.num_bins
    abs_time = cfg.dataset_configs.abs_time_token
    time_fmt = cfg.dataset_configs.time_format

    # Durée de la séquence traitée
    if args.duration is not None:
        dur_s = float(args.duration)
    elif args.video:
        dur_s = ffprobe_duration(args.video)
    else:
        raise ValueError("Spécifie --duration ou --video pour déterminer la durée")
    dur_us = int(dur_s * 1e6)

    seqs = np.asarray(decodes)[0]
    segments = []

    for seq in seqs:
        ids = seq.tolist()

        # On se place à partir du premier token de temps.
        while ids and ids[0] < vocab_size:
            ids.pop(0)

        # Recherche des paires de time-tokens avec deux schémas :
        # - time time
        # - time X time (token intermédiaire).
        pairs = []
        k = 0
        while k < len(ids) - 1:
            if ids[k] >= vocab_size:
                if ids[k + 1] >= vocab_size:  # time time
                    pairs.append((k, k + 1, ids[k], ids[k + 1]))
                    k += 2
                    continue
                if k + 2 < len(ids) and ids[k + 2] >= vocab_size:  # time X time
                    pairs.append((k, k + 2, ids[k], ids[k + 2]))
                    k += 3
                    continue
            k += 1

        if not pairs:
            continue

        tmp = []
        for p0, p1, t0, t1 in pairs:
            s, e = bins_to_sec(t0, t1, dur_us, vocab_size, num_bins, abs_time)
            if time_fmt == "cd":  # format début + durée
                e = s + e
                if e > dur_s:
                    e = dur_s
            tmp.append((p1, s, e))

        offset = tmp[0][1] if (abs_time or tmp[0][1] > dur_s) else 0.0

        for j, (anchor, s_abs, e_abs) in enumerate(tmp):
            s = max(0.0, s_abs - offset)
            e = max(s, min(e_abs - offset, dur_s))

            start_txt = anchor + 1
            end_txt = pairs[j + 1][0] if j + 1 < len(pairs) else len(ids)
            words_ids = [t for t in ids[start_txt:end_txt] if t < vocab_size]
            caption = tok.indices_to_string(words_ids).strip()

            MIN_DUR = 0.5
            if caption and (e - s) >= MIN_DUR and s < dur_s:
                segments.append((s, e, caption))

    # ------------------------------------------------------------------- #
    # NMS temporel + fusion sémantique                                   #
    # (version finale, avec traces de la logique d'inclusion)            #
    # ------------------------------------------------------------------- #
    IOU_THR = 0.3
    SIM_THR = 0.6
    WIN = 3  # secondes

    # Tri par début croissant.
    segments.sort(key=lambda x: x[0])

    # NMS temporel simple.
    kept = []
    for s, e, cap in segments:
        keep = True
        for ks, ke, _ in kept:
            inter = max(0.0, min(e, ke) - max(s, ks))
            if inter == 0:
                continue
            union = (e - s) + (ke - ks) - inter
            if union > 0 and inter / union > IOU_THR:
                keep = False
                break
        if keep:
            kept.append((s, e, cap))

    # Similarité de contenu (ancienne approche : fusion uniquement par fenêtre)
    # Ici on garde la même idée mais on ajoute un test d'inclusion complète.
    sb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    caps = [cap for _, _, cap in kept]
    embs = sb_model.encode(caps, convert_to_numpy=True, normalize_embeddings=True)

    final = []  # [{s,e,cap,emb}]
    ptr = 0
    for idx, (s, e, cap) in enumerate(kept):
        # On ne compare qu'avec une fenêtre temporelle glissante.
        while ptr < len(final) and s - final[ptr]["s"] > WIN:
            ptr += 1

        merged = False
        for j in range(ptr, len(final)):
            sim = float(np.dot(embs[idx], final[j]["emb"]))
            if sim >= SIM_THR:
                final[j]["s"] = min(final[j]["s"], s)
                final[j]["e"] = max(final[j]["e"], e)
                if len(cap) > len(final[j]["cap"]):
                    final[j]["cap"] = cap
                merged = True
                break

        if not merged:
            # Variante testée pendant le stage : fusionner aussi si un segment
            # est entièrement inclus dans un autre et suffisamment similaire.
            for j in range(len(final)):
                fs, fe = final[j]["s"], final[j]["e"]
                contained = ((s >= fs and e <= fe) or (fs >= s and fe <= e))
                if contained and float(np.dot(embs[idx], final[j]["emb"])) >= SIM_THR:
                    final[j]["s"] = min(fs, s)
                    final[j]["e"] = max(fe, e)
                    if len(cap) > len(final[j]["cap"]):
                        final[j]["cap"] = cap
                    merged = True
                    break

        if not merged:
            final.append({"s": s, "e": e, "cap": cap, "emb": embs[idx]})

    filtered_segments = [
        (d["s"], d["e"], d["cap"]) for d in sorted(final, key=lambda x: x["s"])
    ]

    return filtered_segments, dur_s


# --------------------------------------------------------------------------- #
# 2) SRT writer                                                               #
# --------------------------------------------------------------------------- #

def write_srt(segs, path):
    with open(path, "w", encoding="utf-8") as fh:
        for i, (s, e, txt) in enumerate(segs, 1):
            tc = f"({s:.2f}s : {e:.2f}s)"
            fh.write(f"{i}\n{srt_ts(s)} --> {srt_ts(e)}\n{tc} {txt}\n\n")


# --------------------------------------------------------------------------- #
# 3) FFmpeg overlay                                                           #
# --------------------------------------------------------------------------- #

def burn_ffmpeg(src, dst, srt):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"subtitles={srt}",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "fast",
            "-c:a",
            "copy",
            str(dst),
        ],
        check=True,
    )


# --------------------------------------------------------------------------- #
# 4) Main                                                                     #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser("Vid2Seq vers JSON (MP4 facultatif)")
    p.add_argument("video_feats")
    p.add_argument("checkpoint")
    p.add_argument("config")
    p.add_argument("--video", help="Chemin du MP4 original (optionnel)")
    p.add_argument(
        "--duration",
        type=float,
        help="Durée du clip en secondes si --video n'est pas fourni",
    )
    p.add_argument("--asr", help="Fichier pickle ou texte avec transcription ASR")
    p.add_argument(
        "--out_video",
        default="./outputs",
        help="Dossier de sortie pour JSON et MP4 sous-titré",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Garde le .srt et affiche les segments",
    )
    args = p.parse_args()

    segments, dur = vid2seq_segments(args)
    if not segments:
        raise RuntimeError("Aucune légende détectée.")

    if args.debug:
        print(f"[DEBUG] Durée clip : {dur:.2f} s | {len(segments)} segments")
        for i, (s, e, txt) in enumerate(segments[:10], 1):
            print(f"  {i:02d} | {s:.2f}s - {e:.2f}s | {e - s:.2f}s | {txt[:120]}")

    out_dir = Path(args.out_video)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 4.1) JSON                                                         #
    # ------------------------------------------------------------------ #
    video_id = Path(args.video).stem if args.video else Path(args.video_feats).stem

    json_dict = {
        "video_id": video_id,
        "timestamps": [[float(s), float(e)] for s, e, _ in segments],
        "sentences": [txt for _, _, txt in segments],
    }
    json_path = out_dir / f"{video_id}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_dict, fh, ensure_ascii=False, indent=2)
    print("[OK] JSON créé :", json_path)

    # Sortie texte brute pour les scripts batch.
    for s, e, txt in segments:
        print(f"{s:.2f}s → {e:.2f}s : [\"{txt}\"]")

    # ------------------------------------------------------------------ #
    # 4.2) MP4 + sous-titres (si MP4 dispo)                             #
    # ------------------------------------------------------------------ #
    if args.video:
        dst_mp4 = out_dir / f"{video_id}_subs.mp4"
        if args.debug:
            srt_path = out_dir / f"{video_id}_debug.srt"
        else:
            # On laisse ffmpeg écrire dans un fichier temporaire.
            tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
            tmp.close()
            srt_path = Path(tmp.name)

        write_srt(segments, srt_path)
        print("[Info] ffmpeg :", dst_mp4)
        burn_ffmpeg(args.video, dst_mp4, srt_path)

        if not args.debug:
            os.remove(srt_path)

        print("[OK] Fichiers générés :", dst_mp4, "|", json_path)
    else:
        print("[OK] Fichier JSON généré (pas de MP4) :", json_path)


if __name__ == "__main__":
    main()
