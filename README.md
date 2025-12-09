# Vid2Seq on INA Archives — Dense Video Captioning and Evaluation Pipeline

This repository contains the code used to run Vid2Seq on INA (French National Audiovisual Institute) archival videos and to evaluate its performance against existing dense video captioning baselines.

The project implements a complete, modular pipeline:

1. **Feature-based inference** using Vid2Seq on pre-computed visual embeddings.  
2. **Optional multimodal decoding** with ASR transcripts.  
3. **Post-processing of raw predictions** (temporal non-maximum suppression and semantic fusion).  
4. **Evaluation** using the official Vid2Seq dense-captioning metrics (`dvc_eval`) and additional text metrics.  
5. **Comparison to state-of-the-art baselines** on several benchmarks and on the INA dataset (original and re-annotated).

INA data and pre-trained checkpoints are not included in this repository due to licensing constraints, but the code is designed to be reusable on other corpora (e.g. ActivityNet, YouCook2) with compatible formats.

---

## Objectives

- Run **Vid2Seq** on heterogeneous archival content (news, documentaries, historical footage).
- Build a **robust evaluation pipeline** that handles:
  - noisy CSV annotations (encodings, separators, timecodes),
  - multiple annotators and segment styles,
  - English predictions and French ground truth.
- Design a **post-processing layer** that turns raw Vid2Seq outputs into a set of clean, non-redundant segments suitable for indexing.
- Compare Vid2Seq to existing **dense video captioning baselines** in a consistent setting.

---

## Project Architecture

```text
.
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── gt_parsing.py          # Robust loading of INA ground truth from CSV
│   ├── data_loading.py        # Loading of Vid2Seq predictions from CSV
│   ├── metrics_eval.py        # Wrapper around dvc_eval + text metrics
│   ├── text_cleanup.py        # Optional caption cleaning / correction pipeline
│   ├── viz.py                 # Utility plots for analysis
└── scripts
    ├── vid2seq_inference.py   # Inference from CLIP features → JSON (+ SRT/MP4)
    ├── run_eval.py            # End-to-end evaluation (GT + predictions)
    └── compare_runs.py        # Aggregate comparison of multiple runs
````

### Inference (`scripts/vid2seq_inference.py`)

* Loads a Scenic/Flax Vid2Seq checkpoint and configuration.
* Accepts **pre-computed video features** (e.g. CLIP features) as input.
* Optionally incorporates **ASR transcripts** as an additional modality.
* Decodes Vid2Seq output sequences into:

  * temporal segments in seconds,
  * natural language captions.

A temporal post-processing stage is applied:

* **Temporal Non-Maximum Suppression (NMS)** to remove redundant overlapping segments.
* **Semantic fusion** based on SentenceTransformer embeddings to merge near-duplicate captions within a local temporal window.

The script can export:

* a **JSON** file with timestamps and sentences (Vid2Seq format),
* an optional **subtitled MP4** created via `ffmpeg` using a temporary SRT file.

### Ground Truth and Predictions (`src/gt_parsing.py`, `src/data_loading.py`)

* `gt_parsing.py`:

  * loads per-video CSV annotations,
  * performs robust delimiter and encoding detection,
  * converts timecodes (`HH:MM:SS:FF`) to microseconds,
  * stores segments, captions and annotator IDs per video.

* `data_loading.py`:

  * loads Vid2Seq prediction CSVs,
  * converts predicted start/end times to microseconds,
  * supports single-language (EN) or dual-language (EN/FR) captions.

### Evaluation (`src/metrics_eval.py`, `scripts/run_eval.py`)

* Wrapper around the official **Vid2Seq dense-captioning evaluation**:

  * SODA_c,
  * CIDEr,
  * METEOR,
  * temporal precision and recall across multiple IoU thresholds.
* Additional corpus-level metrics:

  * BLEU-3, BLEU-4,
  * ROUGE-L.

`run_eval.py` ties everything together:

* loads INA ground truth and predictions,
* runs the full evaluation,
* prints corpus-level averages,
* optionally exports per-video scores to CSV.

`compare_runs.py` allows comparison of several prediction folders (e.g. baseline vs cleaned vs corrected captions) by reporting aggregated metrics for each configuration.

### Caption Cleaning (Optional) (`src/text_cleanup.py`)

This module implements an optional correction pipeline:

1. English grammar correction with a T5-based model.
2. EN→FR translation using Marian.
3. French spell/grammar correction with a T5-based model.

It was used during the internship to verify whether low metrics were caused by grammatical noise or translation artefacts. Results show that caption correction alone does not close the gap with benchmarks; annotation style and segmentation granularity play a much larger role.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # or Scripts/activate on Windows

pip install -r requirements.txt
```

You will also need:

* a Vid2Seq checkpoint and configuration compatible with `scenic.projects.vid2seq`,
* `ffmpeg` and `ffprobe` available in your `PATH` if you want MP4 + subtitles,
* access to the target dataset (INA is not distributed here).

---

## Usage

### 1. Run Vid2Seq inference on a feature file

```bash
python scripts/vid2seq_inference.py \
  path/to/features.npy \
  path/to/checkpoint_dir \
  path/to/config.py \
  --video path/to/video.mp4 \
  --out_video outputs/
```

This will:

* read `features.npy`,
* run Vid2Seq decoding,
* apply NMS + semantic fusion,
* write `outputs/<video_id>.json`,
* and, if `--video` is provided, produce `outputs/<video_id>_subs.mp4` with burned-in subtitles.

If the original MP4 is not available, you can provide only the duration:

```bash
python scripts/vid2seq_inference.py \
  path/to/features.npy \
  path/to/checkpoint_dir \
  path/to/config.py \
  --duration 97.3 \
  --out_video outputs/
```

### 2. Evaluate a set of predictions

Ground truth is expected to be organized as:

```text
DVC_annotation/
  video_001/
    annotations.csv
  video_002/
    annotations.csv
  ...
```

Predictions are a folder of CSVs with at least `tc_start`, `tc_end`, `caption`.

```bash
python scripts/run_eval.py \
  --gt-root DVC_annotation \
  --pred-root results_vid2seq \
  --out scores_vid2seq.csv
```

### 3. Compare multiple runs

```bash
python scripts/compare_runs.py \
  --gt-root DVC_annotation \
  --pred-root-baseline results_vid2seq \
  --pred-root-clean results_vid2seq_clean \
  --pred-root-corrected results_vid2seq_corrected \
  --out comparison.csv
```

---

## Dense Video Captioning Results

The tables below summarize dense-captioning and event localization results across several benchmarks.
INA and INA adjusted correspond to the original and re-annotated INA datasets, respectively.

### Dense-captioning metrics (SODA_c / CIDEr / METEOR)

| Backbone             | YouCook2 (val) SODA_c | YouCook2 (val) CIDEr | YouCook2 (val) METEOR | ViTT (test) SODA_c | ViTT (test) CIDEr | ViTT (test) METEOR | ActivityNet (val) SODA_c | ActivityNet (val) CIDEr | ActivityNet (val) METEOR | INA SODA_c | INA CIDEr | INA METEOR | INA adjusted SODA_c | INA adjusted CIDEr | INA adjusted METEOR |
| -------------------- | --------------------- | -------------------- | --------------------- | ------------------ | ----------------- | ------------------ | ------------------------ | ----------------------- | ------------------------ | ---------- | --------- | ---------- | ------------------- | ------------------ | ------------------- |
| MT (TSN)             | –                     | 6.1                  | 3.2                   | –                  | –                 | –                  | –                        | 9.3                     | 5.0                      | –          | –         | –          | –                   | –                  | –                   |
| ECHR (C3D)           | –                     | –                    | 3.8                   | –                  | –                 | –                  | 3.2                      | 14.7                    | 7.2                      | –          | –         | –          | –                   | –                  | –                   |
| PDVC (TSN)           | 4.4                   | 22.7                 | 4.7                   | –                  | –                 | –                  | 5.4                      | 29.0                    | 8.0                      | –          | –         | –          | –                   | –                  | –                   |
| PDVC* (CLIP)         | 4.9                   | 28.9                 | 5.7                   | –                  | –                 | –                  | 6.0                      | 29.3                    | 7.6                      | –          | –         | –          | –                   | –                  | –                   |
| UEDVC (TSN)          | –                     | –                    | –                     | –                  | –                 | –                  | 5.5                      | –                       | –                        | –          | –         | –          | –                   | –                  | –                   |
| E2ESG (C3D)          | 25.0                  | –                    | 3.5                   | 25.0               | 8.1               | –                  | –                        | –                       | –                        | –          | –         | –          | –                   | –                  | –                   |
| Vid2Seq (Ours, CLIP) | 7.9                   | 47.1                 | 9.3                   | 13.5               | 43.5              | 8.5                | 5.8                      | 30.1                    | 8.5                      | 8.2        | 5.0       | 3.6        | 10.6                | 5.2                | 3.9                 |

### Event localization metrics (Recall / Precision)

| Method         | Backbone | YouCook2 (val) Recall | YouCook2 (val) Precision | ViTT (test) Recall | ViTT (test) Precision | ActivityNet (val) Recall | ActivityNet (val) Precision | INA Recall | INA Precision | INA adjusted Recall | INA adjusted Precision |
| -------------- | -------- | --------------------- | ------------------------ | ------------------ | --------------------- | ------------------------ | --------------------------- | ---------- | ------------- | ------------------- | ---------------------- |
| PDVC           | TSN      | –                     | –                        | –                  | –                     | 55.4                     | 58.1                        | –          | –             | –                   | –                      |
| PDVC*          | CLIP     | –                     | –                        | –                  | –                     | 53.2                     | 54.7                        | –          | –             | –                   | –                      |
| UEDVC          | TSN      | –                     | –                        | –                  | –                     | 59.0                     | 60.3                        | –          | –             | –                   | –                      |
| E2ESG          | C3D      | 20.7                  | 20.6                     | 32.2               | 32.1                  | –                        | –                           | –          | –             | –                   | –                      |
| Vid2Seq (Ours) | CLIP     | 27.9                  | 27.8                     | 42.6               | 46.2                  | 52.7                     | 53.9                        | 23.4       | 30.3          | 32.0                | 31.8                   |

On the INA datasets, Vid2Seq with CLIP features:

* reaches **SODA_c 8.2 / 5.0 CIDEr / 3.6 METEOR** on the original annotations,
* improves to **SODA_c 10.6 / 5.2 CIDEr / 3.9 METEOR** on the adjusted ground truth,
* and achieves **32.0% recall / 31.8% precision** for event localization in the adjusted setting.

These results are consistent with Vid2Seq’s strong performance on public benchmarks while illustrating the specific challenges of heterogeneous archival data (annotation style, segmentation density, multilingual captions).
