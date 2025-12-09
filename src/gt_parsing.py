"""
Utilitaires pour charger le ground truth INA à partir de fichiers CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import logging
import re

import numpy as np
import pandas as pd


logger = logging.getLogger("dvc_gt")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)7s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GroundTruth:
    segments_us: list[np.ndarray]
    captions: list[list[list[str]]]
    splits: list[np.ndarray]
    keys: list[str]


# Devine le séparateur CSV le plus probable à partir d'un échantillon.
def guess_delimiter(buf, default=","):
    try:
        sniff = csv.Sniffer().sniff(
            buf.decode("latin1", errors="ignore"),
            delimiters=[",", ";", "\t", "|"],
        )
        return sniff.delimiter
    except Exception:
        return default


# Essaie de détecter l'encodage d'un fichier si chardet est disponible.
def detect_encoding(path):
    try:
        import chardet  # type: ignore

        with open(path, "rb") as fh:
            res = chardet.detect(fh.read(4096))
        return res.get("encoding")
    except Exception:
        return None


# Convertit un timecode HH:MM:SS:FF en micro-secondes.
def timecode_to_us(tc, fps=25):
    if pd.isna(tc):
        return None

    if isinstance(tc, (int, float)):
        return int(tc)

    m = re.match(r"(\d+):(\d+):(\d+):(\d+)", str(tc))
    if not m:
        logger.debug("timecode non conforme « %s »", tc)
        return None

    h, m_, s, f = map(int, m.groups())
    seconds = h * 3600 + m_ * 60 + s + f / fps
    return int(seconds * 1_000_000)


# Lecture d'un CSV en testant plusieurs encodages et séparateurs.
def _read_csv_any(path):
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252", "iso-8859-1"]

    first_bytes = path.read_bytes()[:4096]
    sep_hint = guess_delimiter(first_bytes)

    detected = detect_encoding(path)
    if detected and detected not in encodings:
        encodings.append(detected)

    for enc in encodings:
        for sep in {sep_hint, ",", ";", "\t", "|"}:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    engine="python",
                    encoding=enc,
                    on_bad_lines="skip",
                )

                # Certains exports utilisent tc_start_scene / tc_end_scene.
                if {"tc_start_scene", "tc_end_scene", "caption"}.issubset(df.columns):
                    df = df[["tc_start_scene", "tc_end_scene", "caption"]].rename(
                        columns={
                            "tc_start_scene": "start",
                            "tc_end_scene": "end",
                        }
                    )

                if {"start", "end", "caption"}.issubset(df.columns):
                    return df

            except Exception:
                # Si la combinaison échoue, on passe à la suivante.
                continue

    logger.warning("échec de lecture du CSV %s", path)
    return None


# Normalise le champ "caption" en liste de chaînes.
def _parse_caption_field(raw):
    text = str(raw)

    # Certains champs contiennent une liste sérialisée.
    if text.strip().startswith("["):
        try:
            value = eval(text, {"__builtins__": {}})  # type: ignore[arg-type]
            if isinstance(value, list):
                return [str(x) for x in value]
        except Exception:
            pass

    return [text]


# Parcourt un répertoire structuré par vidéo et charge toutes les annotations.
def load_ground_truth(root_dir, annotator_id=1):
    root = Path(root_dir)

    segments_all = []
    captions_all = []
    splits_all = []
    keys_all = []

    for csv_dir in sorted(root.iterdir()):
        if not csv_dir.is_dir():
            continue

        video_key = csv_dir.name
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("%s : aucun CSV trouvé", video_key)
            continue

        df = _read_csv_any(csv_files[0])
        if df is None or df.empty:
            logger.error("%s : échec lecture CSV", video_key)
            continue

        segs = []
        caps = []
        split_vec = []

        for _, row in df.iterrows():
            s_us = timecode_to_us(row["start"])
            e_us = timecode_to_us(row["end"])

            # On ignore les lignes sans timecode valide.
            if s_us is None or e_us is None:
                continue

            segs.append((s_us, e_us))
            caps.append(_parse_caption_field(row["caption"]))
            split_vec.append(annotator_id)

        if not segs:
            logger.warning("%s : aucun segment valide après filtrage", video_key)
            continue

        segments_all.append(np.array(segs, dtype=np.int64))
        captions_all.append(caps)
        splits_all.append(np.array(split_vec, dtype=np.int64))
        keys_all.append(video_key)

        logger.info("%s : %d segments retenus", video_key, len(segs))

    logger.info(
        "ground truth chargé : %d vidéos, %d segments",
        len(keys_all),
        sum(len(x) for x in segments_all),
    )

    return GroundTruth(
        segments_us=segments_all,
        captions=captions_all,
        splits=splits_all,
        keys=keys_all,
    )


__all__ = [
    "GroundTruth",
    "load_ground_truth",
    "timecode_to_us",
]
