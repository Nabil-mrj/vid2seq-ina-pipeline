# Vid2Seq on INA Archives — Dense Video Captioning and Evaluation Pipeline

This repository contains the code used to run Vid2Seq on INA (French National Audiovisual Institute) archival videos and to evaluate its performance against existing dense video captioning baselines.

The project implements a complete, modular pipeline:

1. **Feature-based inference** using Vid2Seq on pre-computed visual embeddings.  
2. **Optional multimodal decoding** with ASR transcripts.  
3. **Post-processing of raw predictions** (temporal non-maximum suppression and semantic fusion).  
4. **Evaluation** using the official Vid2Seq dense-captioning metrics (`dvc_eval`) and additional text metrics.  
5. **Comparison to state-of-the-art baselines** on several benchmarks and on the INA dataset (original and re-annotated).

INA data and pre-trained checkpoints are not included in this repository, but the code is designed to be reusable on other corpora (e.g. ActivityNet, YouCook2) with compatible formats.

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



## Dense Video Captioning Results

The tables below report Vid2Seq performance against state-of-the-art dense video captioning baselines.
INA and INA adjusted correspond to the original and re-annotated INA datasets.

---

## Dense-captioning metrics

### SODA_c

<table>
<tr><th>Model</th><th>YouCook2</th><th>ViTT</th><th>ActivityNet</th><th>INA</th><th>INA (adjusted)</th></tr>

<tr><td>MT (TSN)</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td>ECHR (C3D)</td><td>–</td><td>–</td><td>3.2</td><td>–</td><td>–</td></tr>
<tr><td>PDVC (TSN)</td><td>4.4</td><td>–</td><td>5.4</td><td>–</td><td>–</td></tr>
<tr><td>PDVC* (CLIP)</td><td>4.9</td><td>–</td><td>6.0</td><td>–</td><td>–</td></tr>
<tr><td>UEDVC (TSN)</td><td>–</td><td>–</td><td>5.5</td><td>–</td><td>–</td></tr>
<tr><td>E2ESG (C3D)</td><td>25.0</td><td>25.0</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td><b>Vid2Seq (Ours, CLIP)</b></td><td>7.9</td><td>13.5</td><td>5.8</td><td>8.2</td><td>10.6</td></tr>

</table>

---

### CIDEr

<table>
<tr><th>Model</th><th>YouCook2</th><th>ViTT</th><th>ActivityNet</th><th>INA</th><th>INA (adjusted)</th></tr>

<tr><td>MT (TSN)</td><td>6.1</td><td>–</td><td>9.3</td><td>–</td><td>–</td></tr>
<tr><td>ECHR (C3D)</td><td>–</td><td>–</td><td>14.7</td><td>–</td><td>–</td></tr>
<tr><td>PDVC (TSN)</td><td>22.7</td><td>–</td><td>29.0</td><td>–</td><td>–</td></tr>
<tr><td>PDVC* (CLIP)</td><td>28.9</td><td>–</td><td>29.3</td><td>–</td><td>–</td></tr>
<tr><td>UEDVC (TSN)</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td>E2ESG (C3D)</td><td>–</td><td>8.1</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td><b>Vid2Seq (Ours, CLIP)</b></td><td>47.1</td><td>43.5</td><td>30.1</td><td>5.0</td><td>5.2</td></tr>

</table>

---

### METEOR

<table>
<tr><th>Model</th><th>YouCook2</th><th>ViTT</th><th>ActivityNet</th><th>INA</th><th>INA (adjusted)</th></tr>

<tr><td>MT (TSN)</td><td>3.2</td><td>–</td><td>5.0</td><td>–</td><td>–</td></tr>
<tr><td>ECHR (C3D)</td><td>3.8</td><td>–</td><td>7.2</td><td>–</td><td>–</td></tr>
<tr><td>PDVC (TSN)</td><td>4.7</td><td>–</td><td>8.0</td><td>–</td><td>–</td></tr>
<tr><td>PDVC* (CLIP)</td><td>5.7</td><td>–</td><td>7.6</td><td>–</td><td>–</td></tr>
<tr><td>UEDVC (TSN)</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td>E2ESG (C3D)</td><td>3.5</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td><b>Vid2Seq (Ours, CLIP)</b></td><td>9.3</td><td>8.5</td><td>8.5</td><td>3.6</td><td>3.9</td></tr>

</table>

---

## Event localization metrics

### Recall and Precision

<table>
<tr>
<th>Model</th>
<th>Backbone</th>
<th>YouCook2 Recall</th>
<th>YouCook2 Precision</th>
<th>ViTT Recall</th>
<th>ViTT Precision</th>
<th>ActivityNet Recall</th>
<th>ActivityNet Precision</th>
<th>INA Recall</th>
<th>INA Precision</th>
<th>INA adj. Recall</th>
<th>INA adj. Precision</th>
</tr>

<tr><td>PDVC</td><td>TSN</td><td>–</td><td>–</td><td>–</td><td>–</td><td>55.4</td><td>58.1</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td>PDVC*</td><td>CLIP</td><td>–</td><td>–</td><td>–</td><td>–</td><td>53.2</td><td>54.7</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td>UEDVC</td><td>TSN</td><td>–</td><td>–</td><td>–</td><td>–</td><td>59.0</td><td>60.3</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td>E2ESG</td><td>C3D</td><td>20.7</td><td>20.6</td><td>32.2</td><td>32.1</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td><b>Vid2Seq (Ours)</b></td><td>CLIP</td><td>27.9</td><td>27.8</td><td>42.6</td><td>46.2</td><td>52.7</td><td>53.9</td><td>23.4</td><td>30.3</td><td>32.0</td><td>31.8</td></tr>

</table>


On the INA datasets, Vid2Seq with CLIP features:

* reaches **SODA_c 8.2 / 5.0 CIDEr / 3.6 METEOR** on the original annotations,
* improves to **SODA_c 10.6 / 5.2 CIDEr / 3.9 METEOR** on the adjusted ground truth,
* and achieves **32.0% recall / 31.8% precision** for event localization in the adjusted setting.

These results are consistent with Vid2Seq’s strong performance on public benchmarks while illustrating the specific challenges of heterogeneous archival data (annotation style, segmentation density, multilingual captions).

